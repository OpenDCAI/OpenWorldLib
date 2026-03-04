import os
import numpy as np
import torch
from typing import List, Optional, Union, Dict, Any
from PIL import Image
import base64
import io

from .base_operator import BaseOperator


class FlashWorldOperator(BaseOperator):
    """Operator for FlashWorld pipeline utilities."""
    
    def __init__(
        self,
        operation_types=["textual_instruction", "action_instruction", "visual_instruction"],
        interaction_template=[
            "text_prompt",
            "forward", "backward", "left", "right",
            "camera_up", "camera_down", "camera_l", "camera_r",
            "camera_zoom_in", "camera_zoom_out"
        ]
    ):
        """
        Initialize FlashWorld operator.
        
        Args:
            operation_types: List of operation types
            interaction_template: List of valid interaction types
                - "text_prompt": Text description for scene generation
                - "forward/backward/left/right/up/down": Camera movement
                - "camera_l/r": Camera rotation (left/right)
                - "camera_zoom_in/out": Camera zoom
        """
        super(FlashWorldOperator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()
    
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
            raise ValueError(f"{interaction} not in template. Available: {self.interaction_template}")
        return True
    
    def get_interaction(self, interaction):
        """
        Add interaction to current_interaction list after validation.
        
        Args:
            interaction: Interaction string to add
        """
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)
    
    def process_interaction(
        self, 
        num_frames: Optional[int] = None,
        image_width: int = 704,
        image_height: int = 480
    ) -> Dict[str, Any]:
        """
        Process current interactions and convert to features for representation/synthesis.
        Converts camera actions to actual camera parameters that can be used by representation.
        
        Args:
            num_frames: Number of frames for video generation (optional)
            image_width: Image width for camera intrinsics
            image_height: Image height for camera intrinsics
            
        Returns:
            Dictionary containing processed interaction features:
                - text_prompt: str, text description (if provided)
                - cameras: List[Dict], camera parameters for each frame
                - num_frames: int, number of frames
        """
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")
        
        # Get the latest interaction
        latest_interaction = self.current_interaction[-1]
        self.interaction_history.append(latest_interaction)
        
        num_frames = num_frames or 16
        
        # Extract text prompts
        text_prompt = ""
        camera_actions = []
        for interaction in self.current_interaction:
            if interaction == "text_prompt":
                # Text prompt should be passed separately via data
                pass
            elif interaction.startswith("camera_"):
                camera_actions.append(interaction)
        
        # Convert camera actions to camera parameters
        cameras = self._camera_actions_to_cameras(
            camera_actions=camera_actions,
            num_frames=num_frames,
            image_width=image_width,
            image_height=image_height
        )
        
        result = {
            "text_prompt": text_prompt,
            "cameras": cameras,
            "num_frames": num_frames,
        }
        
        return result
    
    def _camera_actions_to_cameras(
        self,
        camera_actions: List[str],
        num_frames: int,
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        """
        Convert camera action strings to camera parameter dictionaries.
        
        Args:
            camera_actions: List of camera action strings
            num_frames: Number of frames
            image_width: Image width
            image_height: Image height
            
        Returns:
            List of camera dictionaries with position, quaternion, and intrinsics
        """
        if not camera_actions:
            # Default circular camera path
            return self._create_default_cameras(num_frames, image_width, image_height)
        
        # Process camera actions to generate camera trajectory
        cameras = []
        radius = 2.0
        base_position = np.array([0.0, 0.5, 2.0])  # Default position
        
        for i in range(num_frames):
            # Apply camera actions sequentially
            position = base_position.copy()
            angle = 2 * np.pi * i / num_frames
            
            # Process each camera action
            for action in camera_actions:
                if action == "forward":
                    position[2] -= 0.1 * (i / num_frames)
                elif action == "backward":
                    position[2] += 0.1 * (i / num_frames)
                elif action == "left":
                    position[0] -= 0.1 * (i / num_frames)
                elif action == "right":
                    position[0] += 0.1 * (i / num_frames)
                elif action == "camera_up":
                    position[1] += 0.1 * (i / num_frames)
                elif action == "camera_down":
                    position[1] -= 0.1 * (i / num_frames)
                elif action == "camera_l":
                    angle -= np.pi / 4 * (i / num_frames)
                elif action == "camera_r":
                    angle += np.pi / 4 * (i / num_frames)
                # zoom_in/out affects intrinsics, handled separately
            
            # Calculate position based on angle (circular path with modifications)
            x = radius * np.cos(angle) + position[0]
            z = radius * np.sin(angle) + position[2]
            y = position[1]
            
            # Calculate quaternion (look at origin)
            direction = np.array([-x, -y, -z])
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Simple quaternion (identity for now, can be improved)
            quat = [1.0, 0.0, 0.0, 0.0]
            
            # Handle zoom
            zoom_factor = 1.0
            for action in camera_actions:
                if action == "camera_zoom_in":
                    zoom_factor *= 1.1
                elif action == "camera_zoom_out":
                    zoom_factor *= 0.9
            
            camera = {
                'position': [float(x), float(y), float(z)],
                'quaternion': quat,
                'fx': image_width * 0.7 * zoom_factor,
                'fy': image_height * 0.7 * zoom_factor,
                'cx': image_width * 0.5,
                'cy': image_height * 0.5,
            }
            cameras.append(camera)
        
        return cameras
    
    def _create_default_cameras(
        self,
        num_frames: int,
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        """
        Create default camera trajectory (circular path).
        
        Args:
            num_frames: Number of frames
            image_width: Image width
            image_height: Image height
            
        Returns:
            List of camera dictionaries
        """
        cameras = []
        radius = 2.0
        
        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            
            # Circular camera path
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0.5
            
            # Look at origin
            direction = np.array([-x, -y, -z])
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Simple quaternion (identity rotation)
            quat = [1.0, 0.0, 0.0, 0.0]
            
            camera = {
                'position': [float(x), float(y), float(z)],
                'quaternion': quat,
                'fx': image_width * 0.7,
                'fy': image_height * 0.7,
                'cx': image_width * 0.5,
                'cy': image_height * 0.5,
            }
            cameras.append(camera)
        
        return cameras
    
    def process_perception(
        self,
        input_signal: Union[str, np.ndarray, torch.Tensor, Image.Image, bytes]
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Process visual signal (image) for real-time interactive updates.
        
        Args:
            input_signal: Visual input signal - can be:
                - Image file path (str)
                - Numpy array (H, W, 3) in RGB format
                - Torch tensor (C, H, W) or (1, C, H, W) in CHW format
                - PIL Image
                - Base64 encoded image string
                - Bytes of image data
                
        Returns:
            PIL Image in RGB format
            
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        if isinstance(input_signal, Image.Image):
            # Already a PIL Image, convert to RGB
            return input_signal.convert('RGB')
        
        elif isinstance(input_signal, str):
            # Check if it's a file path or base64
            if os.path.exists(input_signal):
                # File path
                image = Image.open(input_signal)
                return image.convert('RGB')
            elif input_signal.startswith('data:image'):
                # Base64 encoded image
                if ',' in input_signal:
                    image_data = input_signal.split(',', 1)[1]
                else:
                    image_data = input_signal
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                return image.convert('RGB')
            else:
                raise ValueError(f"Invalid input: {input_signal}")
        
        elif isinstance(input_signal, bytes):
            # Bytes data
            image = Image.open(io.BytesIO(input_signal))
            return image.convert('RGB')
        
        elif isinstance(input_signal, np.ndarray):
            # Numpy array
            if input_signal.max() <= 1.0:
                input_signal = (input_signal * 255).astype(np.uint8)
            else:
                input_signal = input_signal.astype(np.uint8)
            
            # Convert BGR to RGB if needed
            if len(input_signal.shape) == 3 and input_signal.shape[2] == 3:
                if input_signal[..., 0].mean() > input_signal[..., 2].mean():
                    input_signal = input_signal[..., ::-1]
            
            image = Image.fromarray(input_signal)
            return image.convert('RGB')
        
        elif isinstance(input_signal, torch.Tensor):
            # Torch tensor
            if input_signal.dim() == 3:
                image_array = input_signal.permute(1, 2, 0).cpu().numpy()
            else:
                image_array = input_signal[0].permute(1, 2, 0).cpu().numpy()
            
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
            
            image = Image.fromarray(image_array)
            return image.convert('RGB')
        
        else:
            raise ValueError(f"Unsupported input type: {type(input_signal)}")
    
    def delete_last_interaction(self):
        """Delete the last interaction from current_interaction list."""
        if len(self.current_interaction) > 0:
            self.current_interaction = self.current_interaction[:-1]
        else:
            raise ValueError("No interaction to delete.")

