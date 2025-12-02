import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from typing import Optional, Dict
import torchvision.transforms as tvf

ToTensor = tvf.Compose([tvf.ToTensor()])

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def ImageAugmentation(
    apply_aug: bool = False,
    color_jitter: Optional[Dict[str, float]] = None,
    gray_scale: bool = True,
    gau_blur: bool = False
) -> Optional[tvf.Compose]:
    """Create a composition of image augmentations.

    Args:
        color_jitter: Dictionary containing color jitter parameters:
            - brightness: float (default: 0.5)
            - contrast: float (default: 0.5)
            - saturation: float (default: 0.5)
            - hue: float (default: 0.1)
            - p: probability of applying (default: 0.9)
            If None, uses default values
        gray_scale: Whether to apply random grayscale (default: True)
        gau_blur: Whether to apply gaussian blur (default: False)

    Returns:
        A Compose object of transforms or None if no transforms are added
    """
    if not apply_aug:
        # Return an identity transformation
        return tvf.Compose([tvf.Lambda(lambda x: x)])

    transform_list = []
    default_jitter = {
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "hue": 0.1,
        "p": 0.9
    }

    # Handle color jitter
    if color_jitter is not None:
        # Merge with defaults for missing keys
        effective_jitter = {**default_jitter, **color_jitter}
    else:
        effective_jitter = default_jitter

    transform_list.append(
        tvf.RandomApply(
            [
                tvf.ColorJitter(
                    brightness=effective_jitter["brightness"],
                    contrast=effective_jitter["contrast"],
                    saturation=effective_jitter["saturation"],
                    hue=effective_jitter["hue"],
                )
            ],
            p=effective_jitter["p"],
        )
    )

    if gray_scale:
        transform_list.append(tvf.RandomGrayscale(p=0.05))

    if gau_blur:
        transform_list.append(
            tvf.RandomApply(
                [tvf.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05
            )
        )

    return tvf.Compose(transform_list)