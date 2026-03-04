import os
import cv2
import numpy as np
import torch
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from .base_operator import BaseOperator


class CUT3ROperator(BaseOperator):
    """Operator for CUT3R pipeline utilities."""
    
    def __init__(
        self,
        operation_types=["visual_instruction"],
        interaction_template=[
            "image_3d", 
            "video_3d", 
            "point_cloud", 
            "depth_map", 
            "camera_pose",
            "move_left",
            "move_right",
            "move_up",
            "move_down",
            "zoom_in",
            "zoom_out",
            "rotate_left",
            "rotate_right"
        ]
    ):
        """
        Initialize CUT3R operator.
        
        Args:
            operation_types: List of operation types
            interaction_template: List of valid interaction types
                - "image_3d": Process single image for 3D reconstruction
                - "video_3d": Process video for 3D reconstruction
                - "point_cloud": Generate point cloud output
                - "depth_map": Generate depth map output
                - "camera_pose": Estimate camera poses
                - "move_left/right/up/down": Camera movement controls
                - "zoom_in/out": Camera zoom controls
                - "rotate_left/right": Camera rotation controls
        """
        super(CUT3ROperator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()
    
    def collect_paths(self, path: Union[str, Path]) -> List[str]:
        """
        Collect file paths from a file, directory, or txt list.
        
        Args:
            path: File path, directory path, or txt file containing paths
            
        Returns:
            List of file paths
        """
        path = str(path)
        if os.path.isfile(path):
            if path.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8") as handle:
                    files = [line.strip() for line in handle.readlines() if line.strip()]
            else:
                files = [path]
        else:
            files = [
                os.path.join(path, name)
                for name in os.listdir(path)
                if not name.startswith(".") and name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            files.sort()
        return files
    
    def process_perception(
        self,
        input_signal: Union[str, np.ndarray, torch.Tensor, List[str], List[np.ndarray]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Process visual signal (image/video) for real-time interactive updates.
        This function handles loading and preprocessing of images from various input types.
        
        Args:
            input_signal: Visual input signal - can be:
                - Image file path (str)
                - List of image file paths (List[str])
                - Numpy array (H, W, 3) in RGB or BGR format
                - List of numpy arrays
                - Torch tensor (C, H, W) or (1, C, H, W) in CHW format
                
        Returns:
            Preprocessed RGB image array(s) (normalized to [0, 1]) with shape (H, W, 3)
            or list of such arrays
            
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        # Handle list inputs
        if isinstance(input_signal, list):
            return [self.process_perception(item) for item in input_signal]
        
        # Handle single input
        if isinstance(input_signal, torch.Tensor):
            # Assume tensor is in CHW format, convert to numpy
            if input_signal.dim() == 3:
                image_rgb = input_signal.permute(1, 2, 0).cpu().numpy()
            else:
                image_rgb = input_signal[0].permute(1, 2, 0).cpu().numpy()
            if image_rgb.max() > 1.0:
                image_rgb = image_rgb / 255.0
        elif isinstance(input_signal, np.ndarray):
            image_rgb = input_signal / 255.0 if input_signal.max() > 1.0 else input_signal
            # Convert BGR to RGB if needed (heuristic: if first channel mean > last channel mean)
            if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
                if image_rgb[..., 0].mean() > image_rgb[..., 2].mean():
                    image_rgb = image_rgb[..., ::-1]
        else:
            # String path: support single image, directory, or txt list.
            if isinstance(input_signal, (str, Path)):
                input_path = str(input_signal)
                if os.path.isdir(input_path) or (os.path.isfile(input_path) and input_path.lower().endswith(".txt")):
                    file_list = self.collect_paths(input_path)
                    if len(file_list) == 0:
                        raise ValueError(f"No valid image files found in {input_path}")
                    return [self.process_perception(p) for p in file_list]

                raw_image = cv2.imread(input_path)
                if raw_image is None:
                    raise ValueError(f"Could not read image from {input_signal}")
                image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            else:
                raise ValueError(f"Unsupported input type for process_perception: {type(input_signal)}")
        
        return image_rgb
    
    def check_interaction(self, interaction):
        """
        Check if interaction is in the interaction template.
        
        Args:
            interaction: Interaction string to check
            
        Returns:
            True if interaction is valid
            
        Raises:
            ValueError: If interaction is not in template
        """
        if interaction not in self.interaction_template:
            raise ValueError(f"Interaction '{interaction}' not in interaction_template. "
                           f"Available interactions: {self.interaction_template}")
        return True
    
    def get_interaction(self, interaction):
        """
        Add interaction to current_interaction list after validation.
        
        Args:
            interaction: Interaction string to add
        """
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)
    
    def process_interaction(self, num_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Process current interactions and convert to features for representation/synthesis.
        
        Args:
            num_frames: Number of frames (for video processing, optional)
            
        Returns:
            Dictionary containing processed interaction features:
                - data_type: "image" or "video"
                - output_type: "point_cloud", "depth_map", "camera_pose", or "all"
                - camera_control: Dict with camera movement parameters (if applicable)
        """
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")
        
        # Get the latest interaction
        latest_interaction = self.current_interaction[-1]
        self.interaction_history.append(latest_interaction)
        
        # Process interaction based on type
        result = {
            "data_type": "image",
            "output_type": "all",  # point_cloud, depth_map, camera_pose, or all
            "camera_control": None
        }
        
        # Data type interactions
        if latest_interaction == "image_3d":
            result["data_type"] = "image"
            result["output_type"] = "all"
        elif latest_interaction == "video_3d":
            result["data_type"] = "video"
            result["output_type"] = "all"
        elif latest_interaction == "point_cloud":
            result["data_type"] = "image"
            result["output_type"] = "point_cloud"
        elif latest_interaction == "depth_map":
            result["data_type"] = "image"
            result["output_type"] = "depth_map"
        elif latest_interaction == "camera_pose":
            result["data_type"] = "image"
            result["output_type"] = "camera_pose"
        
        # Camera control interactions
        elif latest_interaction in ["move_left", "move_right", "move_up", "move_down"]:
            direction_map = {
                "move_left": {"x": -0.1, "y": 0, "z": 0},
                "move_right": {"x": 0.1, "y": 0, "z": 0},
                "move_up": {"x": 0, "y": 0.1, "z": 0},
                "move_down": {"x": 0, "y": -0.1, "z": 0},
            }
            result["camera_control"] = direction_map[latest_interaction]
        elif latest_interaction in ["zoom_in", "zoom_out"]:
            zoom_map = {
                "zoom_in": {"scale": 1.1},
                "zoom_out": {"scale": 0.9},
            }
            result["camera_control"] = zoom_map[latest_interaction]
        elif latest_interaction in ["rotate_left", "rotate_right"]:
            rotation_map = {
                "rotate_left": {"angle": -10},
                "rotate_right": {"angle": 10},
            }
            result["camera_control"] = rotation_map[latest_interaction]
        
        # Add num_frames if provided (for video processing)
        if num_frames is not None:
            result["num_frames"] = num_frames
        
        return result
    
    def delete_last_interaction(self):
        """Delete the last interaction from current_interaction list."""
        if len(self.current_interaction) > 0:
            self.current_interaction = self.current_interaction[:-1]
        else:
            raise ValueError("No interaction to delete.")


