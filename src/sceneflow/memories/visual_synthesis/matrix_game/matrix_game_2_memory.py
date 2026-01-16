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
    MatrixGame2 Memory module
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage = []
        self.all_frames = []

    def record(self, data, **kwargs):
        """record to storage"""
        if isinstance(data, Image.Image):
            current_image = data
        elif isinstance(data, list):
            last_frame = data[-1]
            current_image = tensor_to_pil(last_frame)
            self.all_frames.extend(data)
        self.storage.append({
            'content': current_image,
            'type': 'image',
            'timestamp': len(self.all_frames),
            'metadata': {}
        })

    def select(self, **kwargs) -> Optional[Image.Image]:
        """select the last frame as current image"""
        if len(self.storage) == 0:
            return None
        return self.storage[-1]['content']

    def manage(self, action: str = "reset", **kwargs):
        """manage storage"""
        if action == "reset":
            self.storage = []
            self.all_frames = []
