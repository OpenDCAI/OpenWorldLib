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


INTERACTION_CAPTIONS = {
    "forward": (
        "First-person perspective. The camera pushes forward (W). "
        "The rotation direction of the camera remains stationary (.). "
        "Actual distance moved:4 at 100 meters per second. "
        "Angular change rate (turn speed):0. View rotation speed:0."
    ),
    "camera_l": (
        "First-person perspective. The camera pushes forward (W)."
        "The camera pans to the left (←)."
        "Actual distance moved:0 at 100 meters per second."
        "Angular change rate (turn speed):4 View rotation speed:4."
    ),
    "camera_r": (
        "First-person perspective. The camera pushes forward (W)."
        "The camera pans to the right (→)."
        "Actual distance moved:0 at 100 meters per second."
        "Angular change rate (turn speed):4 View rotation speed:4."
    )
}


class YumeOperator(BaseOperator):
    """Lightweight operator for YUME prompt/image preprocessing."""

    def __init__(self, operation_types=None) -> None:
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(YumeOperator, self).__init__(operation_types=operation_types)
        self.interaction_template = ["forward", "camera_l", "camera_r"]
        self.interaction_template_init()

    def get_interaction(self, interaction: str):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        if not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string, got {type(interaction)}")
        # # interaction should be in the template, otherwise raise an error
        # if interaction not in self.interaction_template:
        #     raise ValueError(f"{interaction} not in template")
        return True

    def process_interaction(self, **kwargs) -> Dict[str, Any]:
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        prompt = self.current_interaction[-1]
        ## 输入应该是 "forward", "camera_l", "camera_r"这样的信号 然后 prompt = INTERACTION_CAPTIONS[self.current_interaction[-1]] 进行转换
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
