# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from . import configs

__all__ = ["configs", "WanI2V", "WanT2V", "Yume"]


def __getattr__(name):
    if name == "WanI2V":
        from .image2video import WanI2V

        return WanI2V
    if name == "WanT2V":
        from .text2video import WanT2V

        return WanT2V
    if name == "Yume":
        from .textimage2video import Yume

        return Yume
    raise AttributeError(name)
