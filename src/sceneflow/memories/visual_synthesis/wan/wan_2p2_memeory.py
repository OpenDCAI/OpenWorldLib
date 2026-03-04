from ...base_memory import BaseMemory
import numpy as np
from PIL import Image
from typing import Optional, List, Union

import torch


def tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """
    将单帧张量转换为 PIL.Image。
    假定输入 frame 形状为 (C, H, W)，数值范围在 [-1, 1] 或 [0, 1]。
    """
    if frame.ndim != 3:
        raise ValueError(f"Expected frame shape (C, H, W), got {frame.shape}")

    # 如果是 [-1, 1]，先映射到 [0, 1]
    if frame.min() < 0:
        frame = (frame.clamp(-1, 1) + 1.0) / 2.0
    else:
        frame = frame.clamp(0, 1)

    frame = (frame * 255.0).byte()  # -> uint8, [0, 255]
    # (C, H, W) -> (H, W, C)
    np_img = frame.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np_img)


class Wan2p2Memory(BaseMemory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage = [] 
        self.all_frames = []

    def record(self, data: Union[torch.Tensor, List[Image.Image]], **kwargs):

        if isinstance(data, torch.Tensor):
            video = data
            if video.ndim != 4:
                raise ValueError(f"Unexpected video tensor shape: {video.shape}")

            # 简单判断时间维：如果第 0 维 <= 4，通常是 (C, T, H, W)，否则是 (T, C, H, W)
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
                f"Wan2p2Memory.record expects torch.Tensor or List[PIL.Image], got {type(data)}"
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