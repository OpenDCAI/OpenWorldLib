import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Union
from pathlib import Path

from .base_operator import BaseOperator


class DepthAnythingOperator(BaseOperator):
    """Operator for DepthAnything pipeline utilities."""
    
    def __init__(self, operation_types=[]):
        super(DepthAnythingOperator, self).__init__(operation_types=operation_types)
    
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
    
    def load_and_preprocess_image(
        self,
        input_image: Union[str, np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Load and preprocess image from various input types.
        
        Args:
            input_image: Image path, numpy array, or torch tensor
            
        Returns:
            Preprocessed RGB image array (normalized to [0, 1])
        """
        if isinstance(input_image, torch.Tensor):
            # Assume tensor is in CHW format, convert to numpy
            if input_image.dim() == 3:
                image_rgb = input_image.permute(1, 2, 0).cpu().numpy()
            else:
                image_rgb = input_image[0].permute(1, 2, 0).cpu().numpy()
            if image_rgb.max() > 1.0:
                image_rgb = image_rgb / 255.0
        elif isinstance(input_image, np.ndarray):
            image_rgb = input_image / 255.0 if input_image.max() > 1.0 else input_image
            # Convert BGR to RGB if needed (heuristic: if first channel mean > last channel mean)
            if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
                if image_rgb[..., 0].mean() > image_rgb[..., 2].mean():
                    image_rgb = image_rgb[..., ::-1]
        else:
            # Assume it's a file path
            raw_image = cv2.imread(input_image)
            if raw_image is None:
                raise ValueError(f"Could not read image from {input_image}")
            image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        return image_rgb

