import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Union

from ...base_memory import BaseMemory


def tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """
    Transfer single tensor to PIL.Image.
    """
    if frame.ndim != 3:
        raise ValueError(f"Expected frame shape (C, H, W), got {frame.shape}")

    if frame.min() < 0:
        frame = (frame.clamp(-1, 1) + 1.0) / 2.0
    else:
        frame = frame.clamp(0, 1)

    frame = (frame * 255.0).byte()  # -> uint8, [0, 255]
    # (C, H, W) -> (H, W, C)
    np_img = frame.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np_img)


class CosmosPredict2p5Memory(BaseMemory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage = [] 
        self.all_frames = []

    def record(self, data: torch.Tensor, **kwargs):

        if isinstance(data, torch.Tensor):
            video = data
            if video.ndim != 4:
                raise ValueError(f"Unexpected video tensor shape: {video.shape}")

            if video.shape[0] <= 4:
                # (C, T, H, W) -> (T, C, H, W)
                video = video.permute(1, 0, 2, 3)

            frames = []
            for t in range(video.shape[0]):
                frame = video[t]        # (C, H, W)
                pil_img = tensor_frame_to_pil(frame)
                frames.append(pil_img)
                self.all_frames.append(pil_img)

            current_image = frames[-1]

        else:
            raise TypeError(
                f"CosmosPredict2p5Memory.record expects torch.Tensor or List[PIL.Image], got {type(data)}"
            )

        self.storage.append(
            {
                "content": current_image,
                "type": "image",
                "timestamp": len(self.all_frames),
                "metadata": {},
            }
        )

    def select(self, **kwargs) -> Optional[Image.Image]:
        if len(self.storage) == 0:
            return None
        return self.storage[-1]["content"]

    def manage(self, action: str = "reset", **kwargs):
        if action == "reset":
            self.storage = []
            self.all_frames = []
