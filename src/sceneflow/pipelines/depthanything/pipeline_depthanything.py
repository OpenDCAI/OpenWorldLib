import os
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from sceneflow.representations.models.depth_anything.dpt import DepthAnything
from sceneflow.representations.models.depth_anything.util.transform import (
    NormalizeImage,
    PrepareForNet,
    Resize,
)


class DepthAnythingPipeline:
    """Utility wrapper that exposes Depth Anything for image and video inputs."""

    def __init__(
        self,
        encoder: str = "vitl",
        device: Optional[str] = None,
        pretrained_model_path: Optional[str] = None,
        data_type: str = "image",
    ) -> None:
        if data_type not in {"image", "video"}:
            raise ValueError("data_type must be either 'image' or 'video'")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder
        self.data_type = data_type
        
        # Load model from local path or HuggingFace repo
        if pretrained_model_path and Path(pretrained_model_path).exists():
            # Load from local checkpoint file
            # Note: weights_only=False is needed for loading full model checkpoints
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
            # Check pretrained.cls_token or pretrained.pos_embed dimension
            detected_encoder = encoder  # Default to provided encoder
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
            
            # Infer out_channels from checkpoint if available
            detected_out_channels = None
            if 'depth_head.projects.0.weight' in state_dict:
                # Extract out_channels from projects weights
                detected_out_channels = [
                    state_dict['depth_head.projects.0.weight'].shape[0],
                    state_dict['depth_head.projects.1.weight'].shape[0],
                    state_dict['depth_head.projects.2.weight'].shape[0],
                    state_dict['depth_head.projects.3.weight'].shape[0],
                ]
            
            # Model configurations according to Depth Anything README
            encoder_configs = {
                'vitl': {'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitb': {'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vits': {'features': 64, 'out_channels': [48, 96, 192, 384]},
            }
            
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                # Override encoder if detected
                if detected_encoder != encoder:
                    model_config['encoder'] = detected_encoder
                # Override out_channels if detected from checkpoint
                if detected_out_channels:
                    model_config['out_channels'] = detected_out_channels
                    # Also update features if needed
                    if detected_encoder in encoder_configs:
                        model_config['features'] = encoder_configs[detected_encoder]['features']
            else:
                # Infer config from detected encoder
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
                    # Fallback to default vitl config
                    model_config = {
                        'encoder': detected_encoder,
                        'features': 256,
                        'out_channels': detected_out_channels or [256, 512, 1024, 1024],
                        'use_bn': False,
                        'use_clstoken': False,
                        'localhub': True,
                    }
            
            # Update self.encoder to match detected encoder
            self.encoder = detected_encoder
            
            self.model = DepthAnything(model_config)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # Load from HuggingFace repo
            model_source = pretrained_model_path or f"LiheYoung/depth_anything_{encoder}14"
            self.model = DepthAnything.from_pretrained(model_source)
        
        self.model = self.model.to(self.device).eval()
        self.transform = Compose(
            [
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
            ]
        )

    def _normalize_depth(self, prediction: torch.Tensor) -> np.ndarray:
        prediction = (prediction - prediction.min()) / (
            prediction.max() - prediction.min() + 1e-8
        )
        return (prediction * 255.0).cpu().numpy().astype(np.uint8)

    def _collect_paths(self, path: str) -> List[str]:
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

    def _prepare_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = self.transform({"image": image})["image"]
        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)

    def run_image(
        self,
        img_path: str,
        outdir: str = "./vis_depth",
        grayscale: bool = False,
    ) -> Iterable[str]:
        os.makedirs(outdir, exist_ok=True)

        generated_files: List[str] = []
        for filename in tqdm(self._collect_paths(img_path), desc="DepthAnything-Image"):
            raw_image = cv2.imread(filename)
            if raw_image is None:
                continue

            image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            h, w = image_rgb.shape[:2]

            tensor = self._prepare_tensor(image_rgb)
            with torch.no_grad():
                depth = self.model(tensor)

            depth = F.interpolate(
                depth[None], (h, w), mode="bilinear", align_corners=False
            )[0, 0]
            depth = self._normalize_depth(depth)

            if grayscale:
                depth_vis = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth_vis = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

            basename = os.path.basename(filename)
            stem = basename[: basename.rfind(".")]
            output_path = os.path.join(outdir, f"{stem}_depth.png")
            cv2.imwrite(output_path, depth_vis)
            generated_files.append(output_path)

        return generated_files

    def run_video(
        self,
        video_path: str,
        outdir: str = "./vis_video_depth",
    ) -> Iterable[str]:
        os.makedirs(outdir, exist_ok=True)
        generated_files: List[str] = []

        for k, filename in enumerate(self._collect_paths(video_path), start=1):
            raw_video = cv2.VideoCapture(filename)
            if not raw_video.isOpened():
                continue

            frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = raw_video.get(cv2.CAP_PROP_FPS) or 30

            basename = os.path.basename(filename)
            stem = basename[: basename.rfind(".")]
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

                    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
                    tensor = self._prepare_tensor(frame)
                    with torch.no_grad():
                        depth = self.model(tensor)

                    depth = F.interpolate(
                        depth[None],
                        (frame_height, frame_width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    depth = self._normalize_depth(depth)
                    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

                    writer.write(depth_color)
                    pbar.update(1)

            raw_video.release()
            writer.release()
            generated_files.append(output_path)

        return generated_files


    def __call__(self, data_path: str, **kwargs) -> Iterable[str]:
        """Run the configured data type pipeline."""

        if self.data_type == "image":
            return self.run_image(img_path=data_path, **kwargs)

        video_kwargs = {k: v for k, v in kwargs.items() if k in {"outdir"}}
        return self.run_video(video_path=data_path, **video_kwargs)


def run_depthanything(
    mode: str,
    path: str,
    outdir: Optional[str] = None,
    encoder: str = "vitl",
    pretrained_model_path: Optional[str] = None,
    data_type: Optional[str] = None,
    **kwargs,
) -> Iterable[str]:
    """Convenience function to trigger the pipeline with minimal setup."""

    selected_type = data_type or mode
    if selected_type not in {"image", "video"}:
        raise ValueError("mode/data_type must be either 'image' or 'video'")

    pipeline = DepthAnythingPipeline(
        encoder=encoder,
        pretrained_model_path=pretrained_model_path,
        data_type=selected_type,
    )
    default_outdir = "./vis_depth" if selected_type == "image" else "./vis_video_depth"
    target_dir = outdir or default_outdir
    return pipeline(path, outdir=target_dir, **kwargs)


__all__ = ["DepthAnythingPipeline", "run_depthanything"]

