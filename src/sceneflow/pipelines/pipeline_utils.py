import torch
from typing import Optional, Generator, List


class PipelineABC:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls):
        return cls()
    
    def process(self, *args, **kwds):
        pass
    
    def __call__(self, *args, **kwds):
        pass

    def stream(self, *args, **kwds)-> Generator[torch.Tensor, List[str], None]:
        pass

    def save_pretrained(self, save_directory: str):
        """
        finish this part after the training pipeline is prepared.
        """
        pass
