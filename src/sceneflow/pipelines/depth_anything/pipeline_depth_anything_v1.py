import os
from pathlib import Path
from typing import Iterable, List, Optional, Union, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from ...operators.depth_anything_operator import DepthAnythingOperator
from ...representations.models.depth_anything.depth_anything_v1.dpt import DepthAnything
from ...representations.models.depth_anything.depth_anything_v1.util.transform import (
    NormalizeImage,
    PrepareForNet,
    Resize,
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


class DepthAnythingPipeline:
    """Pipeline for Depth Anything depth estimation."""
    
    def __init__(
        self,
        model: Optional[DepthAnything] = None,
        operator: Optional[DepthAnythingOperator] = None,
        encoder: str = "vitl",
        device: Optional[str] = None,
        data_type: str = "image",
    ) -> None:
        """
        Args:
            model: Pre-loaded DepthAnything model (optional)
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
        self.model = model
        self.operator = operator or DepthAnythingOperator()
        
        # Initialize transform if model is provided
        if self.model is not None:
            self._init_transform()
    
    def _init_transform(self):
        """Initialize image transformation pipeline."""
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def _prepare_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image tensor for model inference."""
        tensor = self.transform({"image": image})["image"]
        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        encoder: str = "vitl",
        device: Optional[str] = None,
        data_type: str = "image",
        **kwargs
    ) -> 'DepthAnythingPipeline':
        """
        Args:
            pretrained_model_path: Path to local checkpoint or HuggingFace repo ID
            encoder: Encoder type ('vits', 'vitb', 'vitl')
            device: Device to run on
            data_type: Type of data to process ('image' or 'video')
            **kwargs: Additional arguments
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model from local path or HuggingFace repo
        if pretrained_model_path and Path(pretrained_model_path).exists():
            model = cls._load_from_local(pretrained_model_path, encoder, device)
        else:
            model = cls._load_from_huggingface(pretrained_model_path, encoder, device)
        
        model = model.to(device).eval()
        
        return cls(
            model=model,
            encoder=encoder,
            device=device,
            data_type=data_type,
        )
    
    @staticmethod
    def _load_from_local(
        pretrained_model_path: str,
        encoder: str,
        device: str
    ) -> DepthAnything:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
            checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        detected_encoder = encoder
        if 'pretrained.cls_token' in state_dict:
            embed_dim = state_dict['pretrained.cls_token'].shape[-1]
            if embed_dim == 384:
                detected_encoder = 'vits'
            elif embed_dim == 768:
                detected_encoder = 'vitb'
            elif embed_dim == 1024:
                detected_encoder = 'vitl'
        elif 'pretrained.pos_embed' in state_dict:
            embed_dim = state_dict['pretrained.pos_embed'].shape[-1]
            if embed_dim == 384:
                detected_encoder = 'vits'
            elif embed_dim == 768:
                detected_encoder = 'vitb'
            elif embed_dim == 1024:
                detected_encoder = 'vitl'
        
        detected_out_channels = None
        if 'depth_head.projects.0.weight' in state_dict:
            detected_out_channels = [
                state_dict['depth_head.projects.0.weight'].shape[0],
                state_dict['depth_head.projects.1.weight'].shape[0],
                state_dict['depth_head.projects.2.weight'].shape[0],
                state_dict['depth_head.projects.3.weight'].shape[0],
            ]
        
        # Model configurations
        encoder_configs = {
            'vitl': {'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            if detected_encoder != encoder:
                model_config['encoder'] = detected_encoder
            if detected_out_channels:
                model_config['out_channels'] = detected_out_channels
                if detected_encoder in encoder_configs:
                    model_config['features'] = encoder_configs[detected_encoder]['features']
        else:
            if detected_encoder in encoder_configs:
                base_config = encoder_configs[detected_encoder].copy()
                model_config = {
                    'encoder': detected_encoder,
                    'features': base_config['features'],
                    'out_channels': detected_out_channels or base_config['out_channels'],
                    'use_bn': False,
                    'use_clstoken': False,
                    'localhub': True,
                }
            else:
                model_config = {
                    'encoder': detected_encoder,
                    'features': 256,
                    'out_channels': detected_out_channels or [256, 512, 1024, 1024],
                    'use_bn': False,
                    'use_clstoken': False,
                    'localhub': True,
                }
        
        model = DepthAnything(model_config)
        model.load_state_dict(state_dict, strict=False)
        return model
    
    @staticmethod
    def _load_from_huggingface(
        pretrained_model_path: Optional[str],
        encoder: str,
        device: str
    ) -> DepthAnything:
        """Load model from HuggingFace repository."""
        model_source = pretrained_model_path or f"LiheYoung/depth_anything_{encoder}14"
        return DepthAnything.from_pretrained(model_source)
    
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
        # Load and preprocess image using operator
        image_rgb = self.operator.load_and_preprocess_image(input_image)
        h, w = image_rgb.shape[:2]
        
        # Prepare tensor and run inference
        tensor = self._prepare_tensor(image_rgb)
        with torch.no_grad():
            depth = self.model(tensor)
        
        # Interpolate to original size
        depth = self.operator.interpolate_depth(depth, (h, w))
        
        # Return raw depth tensor or visualization
        if return_visualization:
            depth_norm = self.operator.normalize_depth(depth)
            depth_vis = self.operator.prepare_depth_visualization(depth_norm, grayscale)
            return depth_vis
        else:
            return depth
    
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


__all__ = ["DepthAnythingPipeline", "DepthResult"]

