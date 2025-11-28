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
    
    def __init__(self):
        super().__init__()
    
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

