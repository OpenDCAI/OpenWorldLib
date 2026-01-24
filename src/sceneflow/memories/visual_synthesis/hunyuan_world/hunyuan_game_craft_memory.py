# sceneflow/memories/visual_synthesis/hunyuan_world/hunyuan_game_craft_memory.py

from ...base_memory import BaseMemory
from typing import Optional, List, Dict, Any
from PIL import Image
import numpy as np
import torch


def np_to_pil(frame: np.ndarray) -> Image.Image:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame must be np.ndarray, got {type(frame)}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame)


class HunyuanGameCraftMemory(BaseMemory):
    """
    Streaming memory for Hunyuan-GameCraft:
    - Keep ref_images / last_latents / ref_latents to continue generation across turns
    - Accumulate all frames for exporting (rank0 only to save memory)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage: List[Dict[str, Any]] = []
        self.all_frames: List[np.ndarray] = []

        self.ref_images: Optional[List[Image.Image]] = None
        self.last_latents: Optional[torch.Tensor] = None
        self.ref_latents: Optional[torch.Tensor] = None

        self.n_generated_segments: int = 0  # how many stream() video segments generated so far

    def has_context(self) -> bool:
        return (self.last_latents is not None) and (self.ref_latents is not None)

    def record(self, data, record_frames: bool = True, **kwargs):
        """
        data:
          - PIL.Image: initial image
          - list[np.ndarray]: video frames from convert_videos_to_grid
        kwargs:
          - visual_context: dict with ref_images/last_latents/ref_latents (for initial image)
          - last_latents/ref_latents: updated latents after generation
        """
        current_image: Optional[Image.Image] = None

        if isinstance(data, Image.Image):
            current_image = data
            visual_context = kwargs.get("visual_context", None)
            if visual_context is not None:
                self.ref_images = visual_context.get("ref_images", self.ref_images)
                self.last_latents = visual_context.get("last_latents", self.last_latents)
                self.ref_latents = visual_context.get("ref_latents", self.ref_latents)

        elif isinstance(data, list):
            frames = data
            if len(frames) > 0:
                # rank0 才真正存全量帧，其他 rank 只维护 latents/context
                if record_frames:
                    self.all_frames.extend(frames)
                try:
                    current_image = np_to_pil(frames[-1])
                except Exception:
                    current_image = None

            if "last_latents" in kwargs and kwargs["last_latents"] is not None:
                self.last_latents = kwargs["last_latents"]
            if "ref_latents" in kwargs and kwargs["ref_latents"] is not None:
                self.ref_latents = kwargs["ref_latents"]

            self.n_generated_segments += 1

        else:
            raise TypeError(f"Unsupported data type for record(): {type(data)}")

        self.storage.append({
            "content": current_image,
            "type": "image",
            "timestamp": len(self.all_frames),
            "metadata": {
                "n_generated_segments": self.n_generated_segments
            }
        })

    def select(self, **kwargs) -> Optional[Image.Image]:
        if len(self.storage) == 0:
            return None
        return self.storage[-1]["content"]

    def select_context(self) -> Optional[Dict[str, Any]]:
        if not self.has_context():
            return None
        return {
            "ref_images": self.ref_images,
            "last_latents": self.last_latents,
            "ref_latents": self.ref_latents
        }

    def manage(self, action: str = "reset", **kwargs):
        if action == "reset":
            self.storage = []
            self.all_frames = []
            self.ref_images = None
            self.last_latents = None
            self.ref_latents = None
            self.n_generated_segments = 0
