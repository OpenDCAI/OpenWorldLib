import logging
from PIL import Image
from pathlib import Path
from giga_datasets import image_utils
from typing import Any, Dict, Optional, Union

from .base_operator import BaseOperator


def _load_input_image(input_path: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(input_path, Image.Image):
        return input_path.convert("RGB")

    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found: {p}")
    img = Image.open(p).convert("RGB")
    return img


class CosmosPrecict2p5Operator(BaseOperator):
    """
    Cosmos-Predict2.5 data processing Operator

    - process_interaction: process input text prompt
    - process_perception: process input image
    """

    def __init__(self, operation_types=None) -> None:
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(Wan2p2Operator, self).__init__(operation_types=operation_types)

        self.interaction_template = ["text_prompt", "image_prompt"]
        self.interaction_template_init()

    def get_interaction(self, interaction):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        if not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string, got {type(interaction)}")
        return True

    def process_interaction(
        self,
        **kwargs,
    ) -> Dict[str, Any]:
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        prompt = self.current_interaction[-1]
        self.interaction_history.append(prompt)
        return {
            "input_prompt": prompt,
        }

    def process_perception(
        self,
        input_path: Optional[Union[str, Path, Image.Image]] = None,
        height: int = 704,
        width: int = 1280,
        **kwargs,
    ) -> Dict[str, Any]:
        """Path to the conditioning image (for `img2world` task), Optional."""
        if input_path is not None:
            input_image = _load_input_image(input_path)
            # Resize & crop to fit the input size
            image_width, image_height = input_image.width, input_image.height
            dst_width, dst_height = image_utils.get_image_size((image_width, image_height), (width, height), mode='area', multiple=16)
            if float(dst_height) / image_height < float(dst_width) / image_width:
                new_height = int(round(float(dst_width) / image_width * image_height))
                new_width = dst_width
            else:
                new_height = dst_height
                new_width = int(round(float(dst_height) / image_height * image_width))
            assert dst_width <= new_width and dst_height <= new_height
            x1 = (new_width - dst_width) // 2
            y1 = (new_height - dst_height) // 2
            input_image = F.resize(image, (new_height, new_width), InterpolationMode.BILINEAR)
            input_image = F.crop(input_image, y1, x1, dst_height, dst_width)
        else:
            input_image = None
            dst_width, dst_height = width, height
        return {
            "input_image": input_image,
            "height": image_height,
            "width": image_width,
        }
