import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from scipy.spatial.transform import Rotation

from .base_operator import BaseOperator


class SpiritV1p5Operator(BaseOperator):
    """
    Operator for Spirit-VLA that handles:
    - Image preprocessing
    - State preprocessing (robot-specific transformations)
    - Action postprocessing (delta to absolute conversion)
    - Interaction signal management
    """
    
    # Supported robot types
    SUPPORTED_ROBOTS = ["ARX5", "UR5", "Franka", "aloha"]
    
    # Camera key mappings
    CAMERA_KEY_MAPPING = {
        "main": "observation.images.cam_high",
        "main_view": "observation.images.cam_high",
        "cam_high": "observation.images.cam_high",
        "wrist": "observation.images.cam_left_wrist",
        "wrist_view": "observation.images.cam_left_wrist",
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    }
    
    REQUIRED_IMAGE_KEYS = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    
    def __init__(
        self,
        raw_embodiment_stats: Optional[Dict] = None,
        target_image_size: tuple = (320, 240),
        state_dim: int = 14,
        operation_types: Optional[List[str]] = None,
    ):
        """
        Initialize Spirit-VLA Operator.
        
        Args:
            raw_embodiment_stats: Dictionary containing normalization statistics
            target_image_size: Target (width, height) for image resizing
            state_dim: Dimension of unified state representation
            operation_types: Types of operations supported
        """
        operation_types = operation_types or [
            "visual_instruction",
            "textual_instruction",
            "action_instruction",
        ]
        super().__init__(operation_types=operation_types)
        
        self.raw_embodiment_stats = raw_embodiment_stats
        self.target_image_size = target_image_size
        self.state_dim = state_dim
        self.device = "cpu"
        
        # Parse normalization statistics
        self._parse_norm_stats()
        
        # Initialize interaction templates
        self.interaction_template = self.SUPPORTED_ROBOTS
        self.interaction_template_init()
    
    def _parse_norm_stats(self):
        """Parse normalization statistics from raw embodiment stats."""
        if self.raw_embodiment_stats is None:
            self.gripper_stats = {
                "left": {"min": 0.0, "max": 1.0},
                "right": {"min": 0.0, "max": 1.0},
            }
            self.action_gripper_stats = {
                "left": {"min": 0.0, "max": 1.0},
                "right": {"min": 0.0, "max": 1.0},
            }
            return
        
        norm_stats = self.raw_embodiment_stats.get("norm_stats", self.raw_embodiment_stats)
        
        # State gripper stats
        state_stats = norm_stats.get("state", {})
        q01 = state_stats.get("q01", [])
        q99 = state_stats.get("q99", [])
        
        self.gripper_stats = {
            "left": {
                "min": q01[7] if len(q01) > 7 else 0.0,
                "max": q99[7] if len(q99) > 7 else 1.0,
            },
            "right": {
                "min": q01[15] if len(q01) > 15 else 0.0,
                "max": q99[15] if len(q99) > 15 else 1.0,
            },
        }
        
        # Action gripper stats
        action_stats = norm_stats.get("actions", {})
        a_q01 = action_stats.get("q01", [])
        a_q99 = action_stats.get("q99", [])
        
        self.action_gripper_stats = {
            "left": {
                "min": a_q01[6] if len(a_q01) > 6 else 0.0,
                "max": a_q99[6] if len(a_q99) > 6 else 1.0,
            },
            "right": {
                "min": a_q01[13] if len(a_q01) > 13 else 0.0,
                "max": a_q99[13] if len(a_q99) > 13 else 1.0,
            },
        }
    
    def to(self, device: Union[str, torch.device]):
        """Move operator to specified device."""
        self.device = device
        return self
    
    # ==================== Interaction Handling ====================
    
    def interaction_template_init(self):
        """Initialize and validate interaction template."""
        if not isinstance(self.interaction_template, list):
            raise ValueError("interaction_template should be a list")
    
    def check_interaction(self, interaction: Any) -> bool:
        """
        Validate an interaction input.
        
        Args:
            interaction: Interaction to validate (robot_type or task string)
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If interaction is invalid
        """
        if isinstance(interaction, dict):
            robot_type = interaction.get("robot_type")
            if robot_type and robot_type not in self.SUPPORTED_ROBOTS:
                raise ValueError(f"Unsupported robot type: {robot_type}")
        elif isinstance(interaction, str):
            # Accept any task string
            pass
        return True
    
    def get_interaction(self, interaction: Any):
        """
        Update interaction list with new interaction.
        
        Args:
            interaction: New interaction to add
        """
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)
    
    def process_interaction(
        self,
        task: str,
        raw_state: List[float],
        robot_type: str,
    ) -> Dict[str, Any]:
        """
        Process interaction signals (task, state, robot type).
        
        Args:
            task: Task description text
            raw_state: Raw robot state vector
            robot_type: Robot type identifier
        
        Returns:
            Dictionary containing processed state tensor and metadata
        """
        # Validate robot type
        if robot_type not in self.SUPPORTED_ROBOTS:
            raise ValueError(f"Unsupported robot type: {robot_type}. Supported: {self.SUPPORTED_ROBOTS}")
        
        # Record interaction
        self.get_interaction({"task": task, "robot_type": robot_type})
        
        # Preprocess state
        state_tensor = self._preprocess_state(raw_state, robot_type)
        state_before_norm = state_tensor.clone()
        
        return {
            "state_tensor": state_tensor,
            "state_before_norm": state_before_norm,
            "task": task,
            "robot_type": robot_type,
        }
    
    # ==================== Perception Processing ====================
    
    def process_perception(
        self,
        images: Dict[str, Union[str, Path, Image.Image, np.ndarray, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Process visual inputs (camera images).
        
        Args:
            images: Dictionary mapping camera names to images
        
        Returns:
            Dictionary mapping standardized keys to image tensors
        """
        image_tensors = {}
        
        for key, img in images.items():
            target_key = self.CAMERA_KEY_MAPPING.get(key, f"observation.images.{key}")
            
            if isinstance(img, torch.Tensor):
                image_tensors[target_key] = img
            else:
                image_tensors[target_key] = self._img_to_tensor(img)
        
        # Get reference shape for padding missing images
        if image_tensors:
            ref_shape = next(iter(image_tensors.values())).shape
        else:
            ref_shape = (3, self.target_image_size[1], self.target_image_size[0])
        
        # Fill missing required images with zeros
        for key in self.REQUIRED_IMAGE_KEYS:
            if key not in image_tensors:
                image_tensors[key] = torch.zeros(ref_shape)
        
        return image_tensors
    
    def _img_to_tensor(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> torch.Tensor:
        """
        Convert image input to tensor.
        
        Args:
            image_input: Image as path, PIL Image, or numpy array
        
        Returns:
            Tensor of shape (C, H, W) with values in [0, 1]
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")
        
        # Resize to target size
        img = img.resize(self.target_image_size, Image.BILINEAR)
        
        # Convert to tensor
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).permute(2, 0, 1)
    
    # ==================== State Processing ====================
    
    def _preprocess_state(
        self,
        raw_state: List[float],
        robot_type: str,
    ) -> torch.Tensor:
        """
        Convert raw robot state to unified 14-dim format.
        
        Args:
            raw_state: Raw robot state vector
            robot_type: Robot type identifier
        
        Returns:
            14-dimensional unified state tensor
        """
        state = torch.zeros(self.state_dim, dtype=torch.float32)
        eps = 1e-8
        
        left_min = self.gripper_stats["left"]["min"]
        left_max = self.gripper_stats["left"]["max"]
        right_min = self.gripper_stats["right"]["min"]
        right_max = self.gripper_stats["right"]["max"]
        
        if robot_type == "ARX5":
            # xyz + euler -> xyz + rotvec
            state[:3] = torch.tensor(raw_state[:3], dtype=torch.float32)
            state[3:6] = torch.tensor(
                Rotation.from_euler("xyz", raw_state[3:6], degrees=False).as_rotvec(),
                dtype=torch.float32
            )
            if self.raw_embodiment_stats is not None:
                gripper = (raw_state[6] - left_min) / (left_max - left_min + eps) * 0.1
            else:
                gripper = raw_state[6]
            state[6] = torch.tensor(gripper, dtype=torch.float32)
            
        elif robot_type == "UR5":
            assert len(raw_state) == 7
            state[:6] = torch.tensor(raw_state[:6], dtype=torch.float32)
            if self.raw_embodiment_stats is not None:
                gripper = 0.1 - (raw_state[6] - left_min) / (left_max - left_min + eps) * 0.1
            else:
                gripper = 0.1 - raw_state[6] / 255 * 0.1
            state[6] = torch.tensor(gripper, dtype=torch.float32)
            
        elif robot_type == "Franka":
            assert len(raw_state) == 8
            state[:3] = torch.tensor(raw_state[:3], dtype=torch.float32)
            # quat -> rotvec
            state[3:6] = torch.tensor(
                Rotation.from_quat(raw_state[3:7]).as_rotvec(),
                dtype=torch.float32
            )
            if self.raw_embodiment_stats is not None:
                gripper = (raw_state[7] - left_min) / (left_max - left_min + eps) * 0.1
            else:
                gripper = raw_state[7]
            state[6] = torch.tensor(gripper, dtype=torch.float32)
            
        elif robot_type == "aloha":
            assert len(raw_state) == 16, "Expected 16-dim state for aloha"
            # Left arm: xyz + quat + gripper
            state[:3] = torch.tensor(raw_state[:3], dtype=torch.float32)
            state[3:6] = torch.tensor(
                Rotation.from_quat(raw_state[3:7]).as_rotvec(),
                dtype=torch.float32
            )
            if self.raw_embodiment_stats is not None:
                gripper = (raw_state[7] - left_min) / (left_max - left_min + eps) * 0.1
            else:
                gripper = raw_state[7]
            state[6] = torch.tensor(gripper, dtype=torch.float32)
            
            # Right arm: xyz + quat + gripper
            state[7:10] = torch.tensor(raw_state[8:11], dtype=torch.float32)
            state[10:13] = torch.tensor(
                Rotation.from_quat(raw_state[11:15]).as_rotvec(),
                dtype=torch.float32
            )
            if self.raw_embodiment_stats is not None:
                gripper = (raw_state[15] - right_min) / (right_max - right_min + eps) * 0.1
            else:
                gripper = raw_state[15]
            state[13] = torch.tensor(gripper, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")
        
        return state
    
    # ==================== Output Processing ====================
    
    def process_output(
        self,
        action_np: np.ndarray,
        state_np: np.ndarray,
        robot_type: str,
        used_chunk_size: int,
    ) -> List[List[float]]:
        """
        Convert model output (delta actions) to absolute target actions.
        
        Args:
            action_np: Model output actions of shape (T, action_dim)
            state_np: Preprocessed 14-dim state
            robot_type: Robot type identifier
            used_chunk_size: Number of action steps to use
        
        Returns:
            List of absolute target actions
        """
        result_list = []
        eps = 1e-8
        
        left_min = self.action_gripper_stats["left"]["min"]
        left_max = self.action_gripper_stats["left"]["max"]
        right_min = self.action_gripper_stats["right"]["min"]
        right_max = self.action_gripper_stats["right"]["max"]
        
        for i in range(min(action_np.shape[0], used_chunk_size)):
            action_i = action_np[i]
            
            if robot_type == "ARX5":
                target_xyz = action_i[:3] + state_np[:3]
                target_rot = (
                    Rotation.from_rotvec(action_i[3:6]) * 
                    Rotation.from_rotvec(state_np[3:6])
                ).as_rotvec()
                target_euler = Rotation.from_rotvec(target_rot).as_euler("xyz", degrees=False)
                target_gripper = action_i[6].item()
                if self.raw_embodiment_stats is not None:
                    target_gripper = target_gripper / 0.1 * (left_max - left_min + eps) + left_min
                list_i = target_xyz.tolist() + target_euler.tolist() + [target_gripper]
                
            elif robot_type == "UR5":
                target_joint = action_i[:6] + state_np[:6]
                target_gripper = 0.1 - action_i[6].item()
                if self.raw_embodiment_stats is not None:
                    target_gripper = target_gripper / 0.1 * (left_max - left_min + eps) + left_min
                else:
                    target_gripper = target_gripper / 0.1 * 255
                list_i = target_joint.tolist() + [target_gripper]
                
            elif robot_type == "Franka":
                target_xyz = action_i[:3] + state_np[:3]
                target_rot = (
                    Rotation.from_rotvec(action_i[3:6]) * 
                    Rotation.from_rotvec(state_np[3:6])
                ).as_rotvec()
                target_quat = Rotation.from_rotvec(target_rot).as_quat()
                target_gripper = action_i[6].item()
                if self.raw_embodiment_stats is not None:
                    target_gripper = target_gripper / 0.1 * (left_max - left_min + eps) + left_min
                list_i = target_xyz.tolist() + target_quat.tolist() + [target_gripper]
                
            elif robot_type == "aloha":
                # Left arm
                target_left_xyz = action_i[:3] + state_np[:3]
                target_left_rot = (
                    Rotation.from_rotvec(action_i[3:6]) * 
                    Rotation.from_rotvec(state_np[3:6])
                ).as_rotvec()
                target_left_quat = Rotation.from_rotvec(target_left_rot).as_quat()
                target_left_gripper = action_i[6].item()
                if self.raw_embodiment_stats is not None:
                    target_left_gripper = target_left_gripper / 0.1 * (left_max - left_min + eps) + left_min
                
                # Right arm
                target_right_xyz = action_i[7:10] + state_np[7:10]
                target_right_rot = (
                    Rotation.from_rotvec(action_i[10:13]) * 
                    Rotation.from_rotvec(state_np[10:13])
                ).as_rotvec()
                target_right_quat = Rotation.from_rotvec(target_right_rot).as_quat()
                target_right_gripper = action_i[13].item()
                if self.raw_embodiment_stats is not None:
                    target_right_gripper = target_right_gripper / 0.1 * (right_max - right_min + eps) + right_min
                
                list_i = (
                    target_left_xyz.tolist() + target_left_quat.tolist() + [target_left_gripper] +
                    target_right_xyz.tolist() + target_right_quat.tolist() + [target_right_gripper]
                )
            else:
                raise ValueError(f"Unsupported robot type: {robot_type}")
            
            result_list.append(list_i)
        
        return result_list
