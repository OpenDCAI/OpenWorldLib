from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image

from .base_operator import BaseOperator


SUPPORTED_LINGBOT_VIDEO_MODES = {"t2i", "t2v", "i2v", "ti2v"}


def _load_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


class LingBotVideoOperator(BaseOperator):
    """Input processing for LingBot-Video T2I/T2V/TI2V calls."""

    def __init__(self, operation_types=None) -> None:
        if operation_types is None:
            operation_types = ["text_prompt", "image_prompt"]
        super().__init__(operation_types=operation_types)
        self.interaction_template = ["prompt", "image"]
        self.interaction_template_init()

    @staticmethod
    def normalize_mode(mode: str) -> str:
        normalized = mode.lower()
        if normalized not in SUPPORTED_LINGBOT_VIDEO_MODES:
            raise ValueError(
                f"Unsupported LingBot-Video mode: {mode}. "
                f"Supported modes: {sorted(SUPPORTED_LINGBOT_VIDEO_MODES)}"
            )
        return "ti2v" if normalized == "i2v" else normalized

    def get_interaction(self, interaction):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        if not isinstance(interaction, str) or not interaction.strip():
            raise ValueError("LingBot-Video prompt must be a non-empty string.")
        return True

    def process_interaction(self, *, mode: str, prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        normalized_mode = self.normalize_mode(mode)
        if prompt is not None:
            self.get_interaction(prompt)
        if not self.current_interaction:
            raise ValueError("No prompt to process.")
        processed_prompt = self.current_interaction[-1].strip()
        self.interaction_history.append(processed_prompt)
        self.current_interaction = []
        return {
            "mode": normalized_mode,
            "prompt": processed_prompt,
        }

    def process_perception(
        self,
        *,
        mode: str,
        images: Optional[Union[str, Path, Image.Image]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        normalized_mode = self.normalize_mode(mode)
        image = _load_image(images) if images is not None else None
        if normalized_mode == "ti2v" and image is None:
            raise ValueError("LingBot-Video i2v/ti2v mode requires an input image.")
        return {
            "image": image,
        }
