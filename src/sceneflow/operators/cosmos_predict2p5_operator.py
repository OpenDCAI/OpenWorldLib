import logging
from PIL import Image
from pathlib import Path
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
        task: str,
        image: Optional[Image.Image] = None,
        base_seed: int = -1,
        **kwargs,
    ) -> Dict[str, Any]:
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        prompt = self.current_interaction[-1]
        self.interaction_history.append(prompt)
        return {
            "processed_prompt": prompt,
        }

    def process_perception(
        self,
        input_path: Optional[Union[str, Path, Image.Image]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Path to the conditioning image (for `img2world` task), Optional."""
        if input_path is None:
            input_image = None
        else:
            input_image = _load_input_image(input_path)
        return {
            "input_image": input_image,
        }
