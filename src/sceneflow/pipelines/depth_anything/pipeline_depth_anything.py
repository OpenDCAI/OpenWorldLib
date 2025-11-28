"""
DepthAnything Pipeline for depth estimation from images and videos.
"""
import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from ...operators.depth_anything_operator import DepthAnythingOperator
from ...representations.models.depth_anything.dpt import DepthAnything
from ...representations.models.depth_anything.util.transform import (
    NormalizeImage,
    PrepareForNet,
    Resize,
)


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
        Initialize DepthAnything pipeline.
        
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
        Load pipeline from pretrained model.
        
        Args:
            pretrained_model_path: Path to local checkpoint or HuggingFace repo ID
            encoder: Encoder type ('vits', 'vitb', 'vitl')
            device: Device to run on
            data_type: Type of data to process ('image' or 'video')
            **kwargs: Additional arguments
            
        Returns:
            Initialized DepthAnythingPipeline instance
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
        """Load model from local checkpoint file."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
            checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
        
        # Extract state_dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Infer encoder type from checkpoint dimensions
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
        
        # Infer out_channels from checkpoint
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
    
    def process_image(
        self,
        image_path: Union[str, np.ndarray],
        grayscale: bool = False,
    ) -> np.ndarray:
        """
        Process a single image and return depth map.
        
        Args:
            image_path: Path to image file or numpy array
            grayscale: If True, return grayscale depth, else color map
            
        Returns:
            Depth visualization as numpy array
        """
        if isinstance(image_path, np.ndarray):
            image_rgb = image_path / 255.0 if image_path.max() > 1.0 else image_path
        else:
            raw_image = cv2.imread(image_path)
            if raw_image is None:
                raise ValueError(f"Could not read image from {image_path}")
            image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image_rgb.shape[:2]
        
        tensor = self._prepare_tensor(image_rgb)
        with torch.no_grad():
            depth = self.model(tensor)
        
        depth = self.operator.interpolate_depth(depth, (h, w))
        depth_norm = self.operator.normalize_depth(depth)
        depth_vis = self.operator.prepare_depth_visualization(depth_norm, grayscale)
        
        return depth_vis
    
    def process_video_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single video frame and return depth map.
        
        Args:
            frame: Video frame as numpy array (BGR format)
            
        Returns:
            Depth visualization as numpy array
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        h, w = frame_rgb.shape[:2]
        
        tensor = self._prepare_tensor(frame_rgb)
        with torch.no_grad():
            depth = self.model(tensor)
        
        depth = self.operator.interpolate_depth(depth, (h, w))
        depth_norm = self.operator.normalize_depth(depth)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        
        return depth_color
    
    def run_image(
        self,
        img_path: str,
        outdir: str = "./vis_depth",
        grayscale: bool = False,
    ) -> Iterable[str]:
        """
        Process images and save depth maps.
        
        Args:
            img_path: Image file, directory, or txt file with paths
            outdir: Output directory
            grayscale: If True, save grayscale depth, else color map
            
        Returns:
            List of generated file paths
        """
        os.makedirs(outdir, exist_ok=True)
        generated_files: List[str] = []
        
        for filename in tqdm(self.operator.collect_paths(img_path), desc="DepthAnything-Image"):
            try:
                depth_vis = self.process_image(filename, grayscale)
                
                basename = os.path.basename(filename)
                stem = basename[:basename.rfind(".")] if "." in basename else basename
                output_path = os.path.join(outdir, f"{stem}_depth.png")
                cv2.imwrite(output_path, depth_vis)
                generated_files.append(output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        return generated_files
    
    def run_video(
        self,
        video_path: str,
        outdir: str = "./vis_video_depth",
    ) -> Iterable[str]:
        """
        Process videos and save depth videos.
        
        Args:
            video_path: Video file, directory, or txt file with paths
            outdir: Output directory
            
        Returns:
            List of generated file paths
        """
        os.makedirs(outdir, exist_ok=True)
        generated_files: List[str] = []
        
        for k, filename in enumerate(self.operator.collect_paths(video_path), start=1):
            raw_video = cv2.VideoCapture(filename)
            if not raw_video.isOpened():
                continue
            
            frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = raw_video.get(cv2.CAP_PROP_FPS) or 30
            
            basename = os.path.basename(filename)
            stem = basename[:basename.rfind(".")] if "." in basename else basename
            output_path = os.path.join(outdir, f"{stem}_depth.mp4")
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                frame_rate,
                (frame_width, frame_height),
            )
            
            with tqdm(total=raw_video.get(cv2.CAP_PROP_FRAME_COUNT) or 0, desc=f"Video {k}", unit="frame") as pbar:
                while raw_video.isOpened():
                    ret, raw_frame = raw_video.read()
                    if not ret:
                        break
                    
                    depth_color = self.process_video_frame(raw_frame)
                    writer.write(depth_color)
                    pbar.update(1)
            
            raw_video.release()
            writer.release()
            generated_files.append(output_path)
        
        return generated_files
    
    def __call__(
        self,
        data_path: str,
        outdir: Optional[str] = None,
        grayscale: bool = False,
        **kwargs
    ) -> Iterable[str]:
        """
        Main call interface for the pipeline.
        
        Args:
            data_path: Path to image/video file, directory, or txt file
            outdir: Output directory (default based on data_type)
            grayscale: For image mode, whether to use grayscale (ignored for video)
            **kwargs: Additional arguments
            
        Returns:
            List of generated file paths
        """
        if self.data_type == "image":
            target_dir = outdir or "./vis_depth"
            return self.run_image(data_path, outdir=target_dir, grayscale=grayscale)
        else:
            target_dir = outdir or "./vis_video_depth"
            return self.run_video(data_path, outdir=target_dir)


__all__ = ["DepthAnythingPipeline"]

