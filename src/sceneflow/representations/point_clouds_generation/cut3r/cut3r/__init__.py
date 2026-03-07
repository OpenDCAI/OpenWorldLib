"""
CUT3R - Unified inference package
整合了 dust3r 和 croco 的推理功能
"""

# 导出主要类和函数 - 使用相对导入
from .model import ARCroco3DStereo, ARCroco3DStereoConfig
from .inference import inference, inference_recurrent, inference_step
from .utils.image import load_images
from .utils.camera import pose_encoding_to_camera
from .utils.geometry import geotrf, depthmap_to_pts3d
from .post_process import estimate_focal_knowing_depth

__all__ = [
    "ARCroco3DStereo",
    "ARCroco3DStereoConfig",
    "inference",
    "inference_recurrent",
    "inference_step",
    "load_images",
    "pose_encoding_to_camera",
    "geotrf",
    "depthmap_to_pts3d",
    "estimate_focal_knowing_depth",
]
