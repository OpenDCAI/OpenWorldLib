from __future__ import annotations

from typing import List, Optional, Sequence

from PIL import Image

from .base_operator import BaseOperator


class RollingForcingOperator(BaseOperator):
    """Text prompt and optional image-context operator for Rolling Forcing."""

    def __init__(self, operation_types: Optional[List[str]] = None):
        super().__init__(operation_types=operation_types or ["textual_instruction", "visual_instruction"])
        self.interaction_template = ["prompt", "image"]
        self.interaction_template_init()

    def check_interaction(self, interaction):
        if isinstance(interaction, str):
            if interaction.strip() == "":
                raise ValueError("RollingForcing prompt cannot be empty.")
            return True
        if isinstance(interaction, (list, tuple)):
            if len(interaction) == 0:
                raise ValueError("RollingForcing prompt list cannot be empty.")
            for prompt in interaction:
                if not isinstance(prompt, str) or prompt.strip() == "":
                    raise ValueError("Each RollingForcing prompt must be a non-empty string.")
            return True
        raise TypeError(f"Unsupported RollingForcing interaction type: {type(interaction)}")

    def get_interaction(self, interaction):
        if isinstance(interaction, str):
            prompts = [interaction]
        else:
            prompts = list(interaction)
        self.check_interaction(prompts)
        self.current_interaction.append(prompts)

    def process_interaction(self, num_samples: int = 1):
        if len(self.current_interaction) == 0:
            raise ValueError("No RollingForcing prompt has been registered.")

        prompts = list(self.current_interaction[-1])
        self.interaction_history.append(prompts)

        if len(prompts) == 1 and num_samples > 1:
            prompts = prompts * int(num_samples)
        elif len(prompts) != int(num_samples):
            if num_samples != 1:
                raise ValueError(
                    "When multiple prompts are provided, their count must match num_samples."
                )
            num_samples = len(prompts)

        return {
            "prompts": prompts,
            "num_samples": int(num_samples),
        }

    def process_perception(self, images=None, size=None):
        if images is None:
            return {"image": None, "size": size}
        if not isinstance(images, Image.Image):
            raise ValueError("RollingForcing currently expects `images` to be a PIL.Image.")
        return {"image": images.convert("RGB"), "size": size}

    def process_prompts(self, prompt: Optional[str], prompts: Optional[Sequence[str]]) -> List[str]:
        if prompts is not None:
            resolved = list(prompts)
        elif prompt is not None:
            resolved = [prompt]
        else:
            raise ValueError("Provide either `prompt` or `prompts`.")
        self.check_interaction(resolved)
        return resolved
