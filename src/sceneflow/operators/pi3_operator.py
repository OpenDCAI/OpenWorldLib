import os
import math
import cv2
import numpy as np
import torch
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from .base_operator import BaseOperator


NAVIGATION_TEMPLATE = [
    "forward", "backward", "left", "right",
    "forward_left", "forward_right", "backward_left", "backward_right",
    "camera_up", "camera_down", "camera_l", "camera_r",
    "camera_ul", "camera_ur", "camera_dl", "camera_dr",
    "camera_zoom_in", "camera_zoom_out",
]

NAVIGATION_DELTAS = {
    "forward":        [0.0,  0.0, -0.5, 0.0,  0.0],
    "backward":       [0.0,  0.0,  0.5, 0.0,  0.0],
    "left":           [-0.5, 0.0,  0.0, 0.0,  0.0],
    "right":          [0.5,  0.0,  0.0, 0.0,  0.0],
    "forward_left":   [-0.35, 0.0, -0.35, 0.0, 0.0],
    "forward_right":  [0.35,  0.0, -0.35, 0.0, 0.0],
    "backward_left":  [-0.35, 0.0,  0.35, 0.0, 0.0],
    "backward_right": [0.35,  0.0,  0.35, 0.0, 0.0],
    "camera_up":      [0.0,  0.0,  0.0, -0.15, 0.0],
    "camera_down":    [0.0,  0.0,  0.0,  0.15, 0.0],
    "camera_l":       [0.0,  0.0,  0.0,  0.0, -0.15],
    "camera_r":       [0.0,  0.0,  0.0,  0.0,  0.15],
    "camera_ul":      [0.0,  0.0,  0.0, -0.1, -0.1],
    "camera_ur":      [0.0,  0.0,  0.0, -0.1,  0.1],
    "camera_dl":      [0.0,  0.0,  0.0,  0.1, -0.1],
    "camera_dr":      [0.0,  0.0,  0.0,  0.1,  0.1],
    "camera_zoom_in": [0.0,  0.0, -0.3,  0.0,  0.0],
    "camera_zoom_out":[0.0,  0.0,  0.3,  0.0,  0.0],
}


class Pi3Operator(BaseOperator):

    PATCH_SIZE = 14
    PIXEL_LIMIT = 255000

    def __init__(
        self,
        operation_types=["visual_instruction", "action_instruction"],
        interaction_template=None,
    ):
        if interaction_template is None:
            interaction_template = list(NAVIGATION_TEMPLATE)
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
        VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        return os.path.splitext(path)[1].lower() in VIDEO_EXTS

    def _load_video_frames(self, video_path: str, interval: int = 10) -> List[np.ndarray]:
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
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise ValueError(f"Could not read image from {image_path}")
        return cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _compute_target_size(
        W_orig: int, H_orig: int, patch_size: int = 14, pixel_limit: int = 255000,
    ) -> tuple:
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
        if interaction not in self.interaction_template:
            raise ValueError(
                f"Interaction '{interaction}' not in interaction_template. "
                f"Available interactions: {self.interaction_template}"
            )
        return True

    def get_interaction(self, interaction):
        if isinstance(interaction, list):
            for i in interaction:
                self.check_interaction(i)
                self.current_interaction.append(i)
        else:
            self.check_interaction(interaction)
            self.current_interaction.append(interaction)

    def process_interaction(self, num_frames: Optional[int] = None) -> List[float]:
        """Process navigation interactions and return accumulated camera delta [dx,dy,dz,theta_x,theta_z]."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")

        delta = [0.0, 0.0, 0.0, 0.0, 0.0]
        for interaction in self.current_interaction:
            self.interaction_history.append(interaction)
            d = NAVIGATION_DELTAS.get(interaction, [0.0, 0.0, 0.0, 0.0, 0.0])
            for i in range(5):
                delta[i] += d[i]

        self.current_interaction = []
        return delta

    def delete_last_interaction(self):
        if len(self.current_interaction) > 0:
            self.current_interaction = self.current_interaction[:-1]
        else:
            raise ValueError("No interaction to delete.")
