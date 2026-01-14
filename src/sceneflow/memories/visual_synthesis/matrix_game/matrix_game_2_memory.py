from ...base_memory import BaseMemory
import numpy as np
from PIL import Image
from typing import Optional


def tensor_to_pil(tensor: np.ndarray) -> Image.Image:
    """将numpy数组转换为PIL Image"""
    last_frame = (tensor * 255).astype(np.uint8)
    return Image.fromarray(last_frame)


class MatrixGame2Memory(BaseMemory):
    """
    MatrixGame2 专用的 Memory 模块
    storage 为列表，current image 取最后一个
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage = []

    def record(self, data, **kwargs):
        """记录帧到 storage"""
        if isinstance(data, Image.Image):
            self.storage.append(data)
        elif isinstance(data, list):
            last_frame = data[-1]
            self.storage.append(tensor_to_pil(last_frame))

    def select(self, **kwargs) -> Optional[Image.Image]:
        """获取最后一帧作为 current image"""
        if len(self.storage) == 0:
            return None
        return self.storage[-1]

    def manage(self, action: str = "reset", **kwargs):
        """管理 storage"""
        if action == "reset":
            self.storage = []

## matrix game 2 memory可以利用first image改成非常简单的记忆处理机制，可以处理折返运动有问题的时候
