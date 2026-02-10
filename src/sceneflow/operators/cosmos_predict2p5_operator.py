from PIL import Image
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from .base_operator import BaseOperator


def get_image_size(src_size: tuple[int, int], dst_size: Any, mode: str, max_size: int | None = None, multiple: int | None = None):
    """Compute destination size given mode and constraints.

    Args:
        src_size: (W, H) source size.
        dst_size: Target size or scalar depending on mode.
        mode: 'long','short','height','width','fixed','outer_fit','inner_fit','area'.
        max_size: Max length constraint on longer side.
        multiple: Round both dims to nearest multiple.

    Returns:
        (W, H) destination size.
    """
    width, height = src_size
    if mode in ('long', 'short', 'height', 'width'):
        if isinstance(dst_size, (list, tuple)):
            assert dst_size[0] == dst_size[1]
            dst_size = dst_size[0]
        if mode == 'long':
            scale = float(dst_size) / max(height, width)
        elif mode == 'short':
            scale = float(dst_size) / min(height, width)
        elif mode == 'height':
            scale = float(dst_size) / height
        elif mode == 'width':
            scale = float(dst_size) / width
        dst_height = int(round(height * scale))
        dst_width = int(round(width * scale))
    elif mode in ('fixed', 'outer_fit', 'inner_fit'):
        if isinstance(dst_size, (list, tuple)):
            dst_width, dst_height = dst_size
        else:
            dst_width, dst_height = dst_size, dst_size
        if mode == 'outer_fit':
            if float(dst_height) / height > float(dst_width) / width:
                dst_height = int(round(float(dst_width) / width * height))
            else:
                dst_width = int(round(float(dst_height) / height * width))
        elif mode == 'inner_fit':
            if float(dst_height) / height < float(dst_width) / width:
                dst_height = int(round(float(dst_width) / width * height))
            else:
                dst_width = int(round(float(dst_height) / height * width))
    elif mode == 'area':
        if isinstance(dst_size, (list, tuple)):
            dst_width, dst_height = dst_size
        else:
            dst_width, dst_height = dst_size, dst_size
        aspect_ratio = float(height) / float(width)
        dst_area = dst_height * dst_width
        dst_height = int(round(np.sqrt(dst_area * aspect_ratio)))
        dst_width = int(round(np.sqrt(dst_area / aspect_ratio)))
    else:
        assert False
    if max_size is not None and max(dst_height, dst_width) > max_size:
        if dst_height > dst_width:
            dst_width = int(round(float(max_size) / dst_height * dst_width))
            dst_height = max_size
        else:
            dst_height = int(round(float(max_size) / dst_width * dst_height))
            dst_width = max_size
    if multiple is not None:
        dst_height = int(round(dst_height / multiple)) * multiple
        dst_width = int(round(dst_width / multiple)) * multiple
    return dst_width, dst_height


def _load_input_image(input_path: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(input_path, Image.Image):
        return input_path.convert("RGB")

    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found: {p}")
    return Image.open(p).convert("RGB")


class CosmosPredict2p5Operator(BaseOperator):
    """
    Cosmos-Predict2.5 data processing Operator

    - process_interaction: process input text prompt
    - process_perception: process input image
    """

    def __init__(self, operation_types=None) -> None:
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super().__init__(operation_types=operation_types)

        self.interaction_template = ["prompt", "neg_prompt"]
        self.interaction_template_init()

    def get_interaction(self, interaction: str):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction: str) -> bool:
        if not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string, got {type(interaction)}")
        return True

    def process_interaction(self, **kwargs) -> Dict[str, Any]:
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        prompt = self.current_interaction[-1]
        self.interaction_history.append(prompt)
        return {"input_prompt": prompt}

    def process_perception(
        self,
        input_path: Optional[Union[str, Path, Image.Image]] = None,
        height: int = 704,
        width: int = 1280,
        **kwargs,
    ) -> Dict[str, Any]:
        """Path to the conditioning image (for `img2world` task), Optional."""
        input_image = None
        orig_width = orig_height = None

        dst_width, dst_height = width, height

        if input_path is not None:
            input_image = _load_input_image(input_path)
            orig_width, orig_height = input_image.width, input_image.height

            dst_width, dst_height = get_image_size(
                (orig_width, orig_height),
                (width, height),
                mode="area",
                multiple=16,
            )

            # scale then center-crop to (dst_width, dst_height)
            if float(dst_height) / orig_height < float(dst_width) / orig_width:
                new_width = dst_width
                new_height = int(round(float(dst_width) / orig_width * orig_height))
            else:
                new_height = dst_height
                new_width = int(round(float(dst_height) / orig_height * orig_width))

            x1 = (new_width - dst_width) // 2
            y1 = (new_height - dst_height) // 2

            input_image = F.resize(input_image, (new_height, new_width), InterpolationMode.BILINEAR)
            input_image = F.crop(input_image, y1, x1, dst_height, dst_width)

        return {
            "input_image": input_image,
            "height": dst_height,
            "width": dst_width,
            "orig_height": orig_height,
            "orig_width": orig_width,
        }
