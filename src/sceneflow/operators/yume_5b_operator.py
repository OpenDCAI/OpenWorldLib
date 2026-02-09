from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image

from .base_operator import BaseOperator


def _load_input_image(input_path: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(input_path, Image.Image):
        return input_path.convert("RGB")

    image_path = Path(input_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


class Yume5bOperator(BaseOperator):
    """
    Lightweight YUME input processing:
    - process_perception: optional reference image loading
    - process_interaction: pass-through prompt
    """

    def __init__(self, operation_types=None) -> None:
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(Yume5bOperator, self).__init__(operation_types=operation_types)
        self.interaction_template = ["text_prompt", "image_prompt"]
        self.interaction_template_init()

    def get_interaction(self, interaction: str):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction: Any):
        if not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string, got {type(interaction)}")
        return True

    def process_interaction(self, **kwargs) -> Dict[str, Any]:
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        prompt = self.current_interaction[-1]
        self.interaction_history.append(prompt)
        return {"processed_prompt": prompt}

    def process_perception(
        self,
        *,
        input_path: Optional[Union[str, Path, Image.Image]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if input_path is None or input_path == "":
            input_image = None
        else:
            input_image = _load_input_image(input_path)
        return {"input_image": input_image}
