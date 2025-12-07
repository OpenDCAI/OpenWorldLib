import os
from typing import List, Optional, Union, Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ...operators.depth_anything_operator import DepthAnythingOperator
from ...representations.depth_generation.depth_anything.depth_anything_v1_representation import (
    DepthAnything1Representation,
)


class DepthResult:
    """Container class for depth estimation results."""
    
    def __init__(self, results: List[Dict], data_type: str):
        """
        Initialize depth result container.
        
        Args:
            results: List of dictionaries containing:
                - For images: {'image': np.ndarray, 'filename': str, 'stem': str}
                - For videos: {'frames': List[np.ndarray], 'filename': str, 'stem': str, 
                              'frame_rate': float, 'frame_width': int, 'frame_height': int}
            data_type: Type of data ('image' or 'video')
        """
        self.results = results
        self.data_type = data_type
    
    def save(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Save depth results to files.
        
        Args:
            output_dir: Output directory. If None, uses default based on data_type.
            
        Returns:
            List of saved file paths
        """
        if output_dir is None:
            output_dir = "./vis_depth" if self.data_type == "image" else "./vis_video_depth"
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files: List[str] = []
        
        if self.data_type == "image":
            for result in self.results:
                output_path = os.path.join(output_dir, f"{result['stem']}_depth.png")
                cv2.imwrite(output_path, result['image'])
                saved_files.append(output_path)
        else:  # video
            for result in self.results:
                output_path = os.path.join(output_dir, f"{result['stem']}_depth.mp4")
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    result['frame_rate'],
                    (result['frame_width'], result['frame_height']),
                )
                for frame in result['frames']:
                    writer.write(frame)
                writer.release()
                saved_files.append(output_path)
        
        return saved_files
    
    def __len__(self):
        return len(self.results)
    
    def __getitem__(self, idx):
        return self.results[idx]


class DepthAnything1Pipeline:
    """Pipeline for Depth Anything (V1) depth estimation."""
    
    def __init__(
        self,
        representation: Optional[DepthAnything1Representation] = None,
        operator: Optional[DepthAnythingOperator] = None,
        encoder: str = "vitl",
        device: Optional[str] = None,
        data_type: str = "image",
    ) -> None:
        """
        Args:
            representation: Pre-loaded DepthAnything1Representation instance (optional)
            operator: DepthAnythingOperator instance (optional)
            encoder: Encoder type ('vits', 'vitb', 'vitl')
            device: Device to run on ('cuda' or 'cpu')
            data_type: Type of data to process ('image' or 'video')
        """
        if data_type not in {"image", "video"}:
            raise ValueError("data_type must be either 'image' or 'video'")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder
        self.data_type = data_type
        self.representation = representation
        self.operator = operator or DepthAnythingOperator()
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        encoder: str = "vitl",
        device: Optional[str] = None,
        data_type: str = "image",
        **kwargs
    ) -> 'DepthAnything1Pipeline':
        """
        Args:
            pretrained_model_path: Path to local checkpoint or HuggingFace repo ID
            encoder: Encoder type ('vits', 'vitb', 'vitl')
            device: Device to run on
            data_type: Type of data to process ('image' or 'video')
            **kwargs: Additional arguments
        """
        # Load representation from pretrained model
        representation = DepthAnything1Representation.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            encoder=encoder,
            device=device,
            **kwargs
        )
        
        return cls(
            representation=representation,
            encoder=encoder,
            device=device,
            data_type=data_type,
        )
    
    def process(
        self,
        input_image: Union[str, np.ndarray, torch.Tensor],
        return_visualization: bool = False,
        grayscale: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Process input image and return depth map.
        
        Args:
            input_image: Image path, numpy array, or torch tensor
            return_visualization: If True, return visualized depth (uint8), else return raw depth tensor
            grayscale: For visualization, whether to use grayscale (ignored if return_visualization=False)
            
        Returns:
            Depth map as tensor (if return_visualization=False) or visualized array (if return_visualization=True)
        """
        if self.representation is None:
            raise RuntimeError("Representation not loaded. Use from_pretrained() first.")
        
        # Load and preprocess image using operator's process_perception method
        image_rgb = self.operator.process_perception(input_image)
        
        # Get depth representation
        data = {
            'image': image_rgb,
            'return_visualization': return_visualization,
            'grayscale': grayscale,
        }
        result = self.representation.get_representation(data)
        
        # Return based on request
        if return_visualization:
            return result['depth_visualization']
        else:
            return result['depth']
    
    def run_image(
        self,
        img_path: str,
        grayscale: bool = False,
    ) -> DepthResult:
        """
        Process images and return depth maps.
        
        Args:
            img_path: Image file, directory, or txt file with paths
            grayscale: If True, return grayscale depth, else color map
            
        Returns:
            DepthResult object containing processed depth images
        """
        results: List[Dict] = []
        
        for filename in tqdm(self.operator.collect_paths(img_path), desc="DepthAnything-Image"):
            try:
                depth_vis = self.process(filename, return_visualization=True, grayscale=grayscale)
                
                basename = os.path.basename(filename)
                stem = basename[:basename.rfind(".")] if "." in basename else basename
                
                results.append({
                    'image': depth_vis,
                    'filename': filename,
                    'stem': stem,
                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        return DepthResult(results, data_type="image")
    
    def run_video(
        self,
        video_path: str,
    ) -> DepthResult:
        """
        Process videos and return depth video frames.
        
        Args:
            video_path: Video file, directory, or txt file with paths
            
        Returns:
            DepthResult object containing processed depth video frames
        """
        results: List[Dict] = []
        
        for k, filename in enumerate(self.operator.collect_paths(video_path), start=1):
            raw_video = cv2.VideoCapture(filename)
            if not raw_video.isOpened():
                continue
            
            frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = raw_video.get(cv2.CAP_PROP_FPS) or 30
            
            basename = os.path.basename(filename)
            stem = basename[:basename.rfind(".")] if "." in basename else basename
            
            frames: List[np.ndarray] = []
            with tqdm(total=raw_video.get(cv2.CAP_PROP_FRAME_COUNT) or 0, desc=f"Video {k}", unit="frame") as pbar:
                while raw_video.isOpened():
                    ret, raw_frame = raw_video.read()
                    if not ret:
                        break
                    
                    depth_color = self.process(raw_frame, return_visualization=True, grayscale=False)
                    frames.append(depth_color)
                    pbar.update(1)
            
            raw_video.release()
            
            results.append({
                'frames': frames,
                'filename': filename,
                'stem': stem,
                'frame_rate': frame_rate,
                'frame_width': frame_width,
                'frame_height': frame_height,
            })
        
        return DepthResult(results, data_type="video")
    
    def __call__(
        self,
        data_path: str,
        grayscale: bool = False,
        **kwargs
    ) -> DepthResult:
        """
        Main call interface for the pipeline.
        
        Args:
            data_path: Path to image/video file, directory, or txt file
            grayscale: For image mode, whether to use grayscale (ignored for video)
            **kwargs: Additional arguments (ignored for now)
            
        Returns:
            DepthResult object containing processed depth results
        """
        if self.data_type == "image":
            return self.run_image(data_path, grayscale=grayscale)
        else:
            return self.run_video(data_path)


__all__ = ["DepthAnything1Pipeline", "DepthResult"]

