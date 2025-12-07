import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from .base_operator import BaseOperator


class DepthAnythingOperator(BaseOperator):
    """Operator for DepthAnything pipeline utilities."""
    
    def __init__(
        self,
        operation_types=["visual_instruction"],
        interaction_template=["image_depth", "video_depth", "grayscale_depth", "color_depth"]
    ):
        """
        Initialize DepthAnything operator.
        
        Args:
            operation_types: List of operation types
            interaction_template: List of valid interaction types
                - "image_depth": Process single image for depth estimation
                - "video_depth": Process video for depth estimation
                - "grayscale_depth": Generate grayscale depth map
                - "color_depth": Generate color-mapped depth map
        """
        super(DepthAnythingOperator, self).__init__(operation_types=operation_types)
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
                if not name.startswith(".")
            ]
            files.sort()
        return files
    
    def normalize_depth(self, prediction: torch.Tensor) -> np.ndarray:
        """
        Normalize depth prediction to uint8 for visualization.
        
        Args:
            prediction: Depth tensor
            
        Returns:
            Normalized depth array as uint8
        """
        prediction = (prediction - prediction.min()) / (
            prediction.max() - prediction.min() + 1e-8
        )
        return (prediction * 255.0).cpu().numpy().astype(np.uint8)
    
    def prepare_depth_visualization(
        self, 
        depth: np.ndarray, 
        grayscale: bool = False
    ) -> np.ndarray:
        """
        Prepare depth map for visualization.
        
        Args:
            depth: Normalized depth array (uint8)
            grayscale: If True, return grayscale, else return color map
            
        Returns:
            Visualization-ready depth image
        """
        if grayscale:
            return np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            return cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    
    def interpolate_depth(
        self, 
        depth: torch.Tensor, 
        target_size: tuple
    ) -> torch.Tensor:
        """
        Interpolate depth map to target size.
        
        Args:
            depth: Depth tensor of shape (H, W)
            target_size: Target (height, width)
            
        Returns:
            Interpolated depth tensor
        """
        return F.interpolate(
            depth[None], target_size, mode="bilinear", align_corners=False
        )[0, 0]
    
    def process_perception(
        self,
        input_signal: Union[str, np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Process visual signal (image) for real-time interactive updates.
        This function handles loading and preprocessing of images from various input types.
        
        Args:
            input_signal: Visual input signal - can be:
                - Image file path (str)
                - Numpy array (H, W, 3) in RGB or BGR format
                - Torch tensor (C, H, W) or (1, C, H, W) in CHW format
                
        Returns:
            Preprocessed RGB image array (normalized to [0, 1]) with shape (H, W, 3)
            
        Raises:
            ValueError: If image cannot be loaded or processed
        """
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
            # Assume it's a file path
            raw_image = cv2.imread(input_signal)
            if raw_image is None:
                raise ValueError(f"Could not read image from {input_signal}")
            image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
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
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)
    
    def process_interaction(self, num_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Process current interactions and convert to features for representation/synthesis.
        
        Args:
            num_frames: Number of frames (for video processing, optional)
            
        Returns:
            Dictionary containing processed interaction features:
                - data_type: "image" or "video"
                - grayscale: bool, whether to use grayscale depth
                - output_format: str, output format specification
        """
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")
        
        # Get the latest interaction
        latest_interaction = self.current_interaction[-1]
        self.interaction_history.append(latest_interaction)
        
        # Process interaction based on type
        result = {
            "data_type": "image",
            "grayscale": False,
            "output_format": "color_map"
        }
        
        if latest_interaction == "image_depth":
            result["data_type"] = "image"
            result["grayscale"] = False
            result["output_format"] = "color_map"
        elif latest_interaction == "video_depth":
            result["data_type"] = "video"
            result["grayscale"] = False
            result["output_format"] = "color_map"
        elif latest_interaction == "grayscale_depth":
            result["data_type"] = "image"
            result["grayscale"] = True
            result["output_format"] = "grayscale"
        elif latest_interaction == "color_depth":
            result["data_type"] = "image"
            result["grayscale"] = False
            result["output_format"] = "color_map"
        
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

