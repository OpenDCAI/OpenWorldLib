import os
import cv2
import numpy as np
import torch
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from .base_operator import BaseOperator


class Pi3Operator(BaseOperator):

    PATCH_SIZE = 14
    PIXEL_LIMIT = 255000

    def __init__(
        self,
        operation_types=["visual_instruction", "action_instruction"],
        interaction_template=[
            "3d_reconstruction",
            "point_cloud_generation",
            "depth_estimation",
            "camera_pose_estimation",
            "multi_view_reconstruction",
            "conditional_reconstruction",
        ]
    ):
        super(Pi3Operator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()

    def collect_paths(self, path: Union[str, Path]) -> List[str]:
        """Collect image file paths from a file, directory, or txt list."""
        path = str(path)
        SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        if os.path.isfile(path):
            if path.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8") as handle:
                    files = [line.strip() for line in handle.readlines() if line.strip()]
            else:
                files = [path]
        elif os.path.isdir(path):
            files = [
                os.path.join(path, name)
                for name in sorted(os.listdir(path))
                if not name.startswith(".") and os.path.splitext(name)[1].lower() in SUPPORTED_EXTS
            ]
        else:
            raise ValueError(f"Path does not exist: {path}")
        return files

    @staticmethod
    def _is_video(path: str) -> bool:
        """Check if the given path is a video file."""
        VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        return os.path.splitext(path)[1].lower() in VIDEO_EXTS

    def _load_video_frames(self, video_path: str, interval: int = 10) -> List[np.ndarray]:
        # Load frames from a video file at the given sampling interval.

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")
        return frames

    def _load_single_image(self, image_path: str) -> np.ndarray:
        """Load a single image file as uint8 RGB array.

        Returns:
            uint8 numpy array with shape (H, W, 3).
        """
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise ValueError(f"Could not read image from {image_path}")
        return cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _compute_target_size(
        W_orig: int, H_orig: int, patch_size: int = 14, pixel_limit: int = 255000,
    ) -> tuple:
        """Compute target (W, H) aligned to patch_size multiples under pixel_limit."""
        import math
        scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / patch_size), round(H_target / patch_size)
        while (k * patch_size) * (m * patch_size) > pixel_limit:
            if k / m > W_target / H_target:
                k -= 1
            else:
                m -= 1
        return max(1, k) * patch_size, max(1, m) * patch_size

    def images_to_tensor(self, images: List[np.ndarray], device: str = "cuda") -> torch.Tensor:
        """
        Convert a list of numpy images to a batched tensor for Pi3.
        Applies resize (LANCZOS, patch-aligned) and ToTensor conversion.
        """
        if len(images) == 0:
            raise ValueError("No images provided")

        h, w = images[0].shape[:2]
        target_w, target_h = self._compute_target_size(w, h, self.PATCH_SIZE, self.PIXEL_LIMIT)
        if target_h == 0 or target_w == 0:
            raise ValueError(f"Image too small ({h}x{w}) for patch_size={self.PATCH_SIZE}")

        from PIL import Image as PILImage
        from torchvision import transforms
        to_tensor = transforms.ToTensor()

        tensors = []
        for img in images:
            if img.dtype == np.uint8:
                img_uint8 = img
            else:
                img_uint8 = np.round(img.astype(np.float64) * 255.0).clip(0, 255).astype(np.uint8)
            if img_uint8.ndim == 2:
                img_uint8 = np.stack([img_uint8] * 3, axis=-1)
            pil_img = PILImage.fromarray(img_uint8)
            resized = pil_img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)
            tensors.append(to_tensor(resized))

        return torch.stack(tensors, dim=0).unsqueeze(0).to(device)

    def process_perception(
        self,
        input_signal: Union[str, np.ndarray, torch.Tensor, List[str], List[np.ndarray]],
        interval: int = -1,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Process visual signal (image/images/video) for Pi3 inference.
        Returns: List of numpy arrays (H, W, 3), uint8 or float [0,1].
        """
        if isinstance(input_signal, (str, Path)):
            input_signal = str(input_signal)
            if self._is_video(input_signal):
                if interval < 0:
                    interval = 10
                return self._load_video_frames(input_signal, interval=interval)
            elif os.path.isdir(input_signal) or input_signal.lower().endswith(".txt"):
                image_paths = self.collect_paths(input_signal)
                if len(image_paths) == 0:
                    raise ValueError(f"No images found in: {input_signal}")
                if interval < 0:
                    interval = 1
                return [
                    self._load_single_image(image_paths[i])
                    for i in range(0, len(image_paths), interval)
                ]
            else:
                return [self._load_single_image(input_signal)]
        elif isinstance(input_signal, list):
            if len(input_signal) == 0:
                raise ValueError("Empty input list")
            if isinstance(input_signal[0], str):
                return [self._load_single_image(p) for p in input_signal]
            elif isinstance(input_signal[0], np.ndarray):
                return list(input_signal)
            else:
                raise ValueError(f"Unsupported list element type: {type(input_signal[0])}")
        elif isinstance(input_signal, torch.Tensor):
            if input_signal.dim() == 4:
                imgs = input_signal
            elif input_signal.dim() == 3:
                imgs = input_signal.unsqueeze(0)
            elif input_signal.dim() == 5:
                imgs = input_signal[0]
            else:
                raise ValueError(f"Unsupported tensor shape: {input_signal.shape}")
            return [imgs[i].permute(1, 2, 0).cpu().numpy() for i in range(imgs.shape[0])]
        elif isinstance(input_signal, np.ndarray):
            if input_signal.ndim == 3:
                return [input_signal]
            elif input_signal.ndim == 4:
                return [input_signal[i] for i in range(input_signal.shape[0])]
            else:
                raise ValueError(f"Unsupported array shape: {input_signal.shape}")
        else:
            raise ValueError(f"Unsupported input type: {type(input_signal)}")

    def check_interaction(self, interaction):
        """Check if interaction is in the interaction template."""
        if interaction not in self.interaction_template:
            raise ValueError(
                f"Interaction '{interaction}' not in interaction_template. "
                f"Available interactions: {self.interaction_template}"
            )
        return True

    def get_interaction(self, interaction):
        """Add interaction to current_interaction list after validation."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, num_frames: Optional[int] = None) -> Dict[str, Any]:
        """Process current interactions and return feature flags for representation."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")

        latest_interaction = self.current_interaction[-1]
        self.interaction_history.append(latest_interaction)

        result = {
            "predict_points": True,
            "predict_cameras": True,
            "predict_depth": True,
            "predict_conf": True,
            "use_conditions": False,
        }

        if latest_interaction in ("3d_reconstruction", "point_cloud_generation", "multi_view_reconstruction"):
            pass
        elif latest_interaction == "depth_estimation":
            result["predict_points"] = False
            result["predict_cameras"] = False
        elif latest_interaction == "camera_pose_estimation":
            result["predict_points"] = False
            result["predict_depth"] = False
        elif latest_interaction == "conditional_reconstruction":
            result["use_conditions"] = True

        if num_frames is not None:
            result["num_frames"] = num_frames

        return result

    def delete_last_interaction(self):
        """Delete the last interaction from current_interaction list."""
        if len(self.current_interaction) > 0:
            self.current_interaction = self.current_interaction[:-1]
        else:
            raise ValueError("No interaction to delete.")
