"""
InfiniteVGGT (StreamVGGT) representation for SceneFlow: streaming 3D reconstruction from images.
Input: operator-processed features. Output: 3D scene, depth map, point cloud (static assets).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import snapshot_download

from .infinite_vggt import (
    StreamVGGT,
    load_and_preprocess_images,
    pose_encoding_to_extri_intri,
)


class InfiniteVGGTRepresentation:
    """InfiniteVGGT representation: operator features -> 3D scene, depth map, visual embedding."""

    def __init__(
        self,
        model: Optional[StreamVGGT] = None,
        device: Optional[str] = None,
        total_budget: int = 1200000,
    ):
        """Initialize representation model. self.model = model."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.total_budget = total_budget
        self.preprocess_mode = "crop"
        self.resolution = 518

        if self.model is not None:
            self.model = self.model.to(self.device).eval()
            if self.device == "cuda" and torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()[0]
                self.dtype = torch.bfloat16 if cap >= 8 else torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = torch.float32

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[str] = None,
        total_budget: int = 1200000,
        **kwargs,
    ) -> "InfiniteVGGTRepresentation":
        """
        Load representation from local path or HuggingFace repo.
        input: pretrained_model_path (local dir/file or HuggingFace repo_id)
        output: InfiniteVGGTRepresentation instance
        """
        path = pretrained_model_path
        if os.path.isdir(path):
            model_root = path
            pth = list(Path(model_root).glob("*.pth"))
            if not pth:
                raise ValueError(f"No .pth file in {model_root}")
            path = str(pth[0])
        elif os.path.isfile(path):
            model_root = str(Path(path).parent)
        else:
            print(f"Downloading weights from HuggingFace repo: {pretrained_model_path}")
            model_root = snapshot_download(pretrained_model_path)
            print(f"Model downloaded to: {model_root}")
            pth = list(Path(model_root).glob("*.pth"))
            if not pth:
                raise ValueError(f"No .pth file in downloaded repo: {model_root}")
            path = str(pth[0])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        model = StreamVGGT(total_budget=total_budget)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt, strict=True)
        del ckpt

        instance = cls(model=model, device=device, total_budget=total_budget)
        instance.preprocess_mode = kwargs.get("preprocess_mode", "crop")
        instance.resolution = kwargs.get("resolution", 518)
        return instance

    def api_init(self, api_key: str, endpoint: str) -> None:
        """Placeholder for API-based init."""
        pass

    def _prepare_images(self, images_input: Any) -> torch.Tensor:
        """Build batched image tensor (N, C, H, W) from paths or arrays."""
        if isinstance(images_input, list):
            image_list = images_input
        elif isinstance(images_input, np.ndarray):
            if images_input.ndim == 3:
                image_list = [images_input]
            elif images_input.ndim == 4:
                image_list = [images_input[i] for i in range(images_input.shape[0])]
            else:
                image_list = [images_input]
        else:
            image_list = [images_input] if not isinstance(images_input, list) else images_input

        has_paths = any(isinstance(x, str) for x in image_list)
        if has_paths:
            images = load_and_preprocess_images(image_list, mode=self.preprocess_mode)
        else:
            tensors = []
            for arr in image_list:
                if not isinstance(arr, np.ndarray):
                    raise ValueError(f"Unsupported image type: {type(arr)}")
                if arr.max() > 1.0:
                    arr = arr.astype(np.float32) / 255.0
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                t = torch.from_numpy(arr).permute(2, 0, 1).float()
                tensors.append(t)
            images = torch.stack(tensors)
            # Resize to resolution divisible by 14 (e.g. 518)
            h, w = images.shape[-2], images.shape[-1]
            target = self.resolution
            if w != target or h != target:
                new_w = target
                new_h = round(h * (target / w) / 14) * 14
                if new_h > target and self.preprocess_mode == "crop":
                    new_h = target
                images = F.interpolate(
                    images, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
            if images.shape[-2] > target or images.shape[-1] > target:
                start_h = (images.shape[-2] - target) // 2
                start_w = (images.shape[-1] - target) // 2
                images = images[:, :, start_h : start_h + target, start_w : start_w + target]

        return images

    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run streaming inference and return point maps, depth, and cameras.

        data['images']: list of image paths or list of numpy arrays (H,W,3) in [0,1] or [0,255].
        data['predict_cameras'] / 'predict_depth' / 'predict_points': optional bools (default True).
        data['preprocess_mode']: 'crop' or 'pad' (default 'crop').
        data['resolution']: int (default 518).

        Returns dict with:
          point_map: (N, H, W, 3) world points
          point_conf: (N, H, W)
          depth_map: (N, H, W) or (N, H, W, 1)
          depth_conf: (N, H, W)
          extrinsic: (N, 3, 4) world-to-cam
          intrinsic: (N, 3, 3)
          point_cloud: list of (H*W, 3) per frame (for pipeline compatibility)
          colors: list of (H, W, 3) per frame
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() first.")

        images_input = data["images"]
        predict_cameras = data.get("predict_cameras", True)
        predict_depth = data.get("predict_depth", True)
        predict_points = data.get("predict_points", True)
        self.preprocess_mode = data.get("preprocess_mode", self.preprocess_mode)
        self.resolution = data.get("resolution", self.resolution)

        images = self._prepare_images(images_input).to(self.device)
        if images.dim() == 3:
            images = images.unsqueeze(0)

        frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]

        with torch.no_grad():
            # Use legacy autocast API for compatibility with older PyTorch (no device_type)
            with torch.cuda.amp.autocast(dtype=self.dtype, enabled=(self.device == "cuda")):
                output = self.model.inference(
                    frames,
                    frame_writer=None,
                    cache_results=True,
                )

        if output.ress is None or len(output.ress) == 0:
            raise RuntimeError("InfiniteVGGT inference returned no results.")

        all_pts3d = [r["pts3d_in_other_view"].squeeze(0) for r in output.ress]
        all_conf = [r["conf"].squeeze(0) for r in output.ress]
        all_depth = [r["depth"].squeeze(0) for r in output.ress]
        all_depth_conf = [r["depth_conf"].squeeze(0) for r in output.ress]
        all_camera_pose = [r["camera_pose"].squeeze(0) for r in output.ress]

        world_points = torch.stack(all_pts3d, dim=0)
        point_conf = torch.stack(all_conf, dim=0)
        depth = torch.stack(all_depth, dim=0)
        depth_conf = torch.stack(all_depth_conf, dim=0)
        pose_enc = torch.stack(all_camera_pose, dim=0)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc.unsqueeze(0), images.shape[-2:]
        )
        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy() if intrinsic is not None else None

        results = {
            "point_map": world_points.cpu().numpy(),
            "point_conf": point_conf.cpu().numpy(),
            "depth_map": depth.squeeze(-1).cpu().numpy() if depth.dim() > 3 else depth.cpu().numpy(),
            "depth_conf": depth_conf.cpu().numpy(),
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        }

        # Optional: list-style point_cloud and colors for pipelines that expect them
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()
        results["point_cloud"] = [world_points[i].cpu().numpy().reshape(-1, 3) for i in range(world_points.shape[0])]
        results["colors"] = [images_np[i] for i in range(images_np.shape[0])]
        results["confidence"] = [results["point_conf"][i] for i in range(results["point_conf"].shape[0])]

        return results
