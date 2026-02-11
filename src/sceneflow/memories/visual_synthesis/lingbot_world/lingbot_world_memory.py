from ...base_memory import BaseMemory
import numpy as np
from PIL import Image
from typing import Optional, List, Union

def numpy_to_pil(image_array: np.ndarray) -> Image.Image:
    """
    Convert (H, W, C) numpy array in range [0, 1] or [0, 255] to PIL Image.
    """
    if image_array.dtype == np.float32 or image_array.dtype == np.float16:
        image_array = (image_array * 255).astype(np.uint8)
    return Image.fromarray(image_array)

class LingBotMemory(BaseMemory):
    """
    LingBot Memory module implementing BaseMemory.
    """
    def __init__(self, capacity=10, **kwargs):
        super().__init__(capacity=capacity, **kwargs)
        self.storage = []     # Stores context for each round
        self.all_frames = []  # Stores all generated frames for final export

    def record(self, data: Union[Image.Image, np.ndarray], type="image", metadata=None, **kwargs):
        """
        Record data.
        If data is PIL Image: It's usually the initial start frame.
        If data is numpy array (Video): It's the output of a generation.
        """
        current_image = None
        
        # Case 1: Initial Image (PIL)
        if isinstance(data, Image.Image):
            current_image = data
            # Stored only in storage as context
            
        # Case 2: Generated Video (Numpy Array [T, H, W, C])
        elif isinstance(data, np.ndarray):
            # Expected shape: [T, H, W, C]
            self.all_frames.append(data)
            
            # Use the last frame as the context for the next round
            last_frame = data[-1]
            current_image = numpy_to_pil(last_frame)
        
        if current_image is not None:
            entry = {
                'content': current_image,
                'type': 'image',
                'metadata': metadata or {}
            }
            self.storage.append(entry)
            
            # Simple capacity management
            if self.capacity and len(self.storage) > self.capacity:
                self.storage.pop(0)

    def select(self, context_query=None, **kwargs) -> Optional[Image.Image]:
        """
        Select the latest image to serve as the starting frame for the next generation.
        """
        if len(self.storage) == 0:
            return None
        return self.storage[-1]['content']

    def manage(self, action: str = "reset", **kwargs):
        """Manage storage."""
        if action == "reset":
            self.storage = []
            self.all_frames = []