from __future__ import annotations

from typing import List, Optional

from .base_operator import BaseOperator


class LongLiveOperator(BaseOperator):
    """Prompt-switch operator for LongLive text-to-video generation."""

    def __init__(self, operation_types: Optional[List[str]] = None):
        super().__init__(operation_types=operation_types or ["textual_instruction"])
        self.interaction_template = ["prompt"]
        self.interaction_template_init()

    def check_interaction(self, interaction):
        if isinstance(interaction, str):
            if interaction.strip() == "":
                raise ValueError("LongLive prompt cannot be empty.")
            return True
        if isinstance(interaction, (list, tuple)):
            if len(interaction) == 0:
                raise ValueError("LongLive prompt list cannot be empty.")
            for prompt in interaction:
                if not isinstance(prompt, str) or prompt.strip() == "":
                    raise ValueError("Each LongLive prompt must be a non-empty string.")
            return True
        raise TypeError(f"Unsupported LongLive interaction type: {type(interaction)}")

    def get_interaction(self, interaction):
        if isinstance(interaction, str):
            prompts = [interaction]
        else:
            prompts = list(interaction)
        self.check_interaction(prompts)
        self.current_interaction.append(prompts)

    def process_interaction(
        self,
        num_frames: int,
        switch_frame_indices: Optional[List[int]] = None,
    ):
        if len(self.current_interaction) == 0:
            raise ValueError("No LongLive prompts have been registered.")

        prompts = list(self.current_interaction[-1])
        self.interaction_history.append(prompts)
        if len(prompts) == 1:
            return {
                "mode": "single",
                "prompts": prompts,
                "switch_frame_indices": [],
            }

        if switch_frame_indices is None:
            switch_frame_indices = self._even_switch_indices(
                num_frames=num_frames,
                num_segments=len(prompts),
            )
        switch_frame_indices = [int(index) for index in switch_frame_indices]
        if len(switch_frame_indices) != len(prompts) - 1:
            raise ValueError(
                "switch_frame_indices length must be one less than the number of prompts."
            )
        previous = 0
        for index in switch_frame_indices:
            if index <= previous or index >= num_frames:
                raise ValueError(
                    "switch_frame_indices must be strictly increasing and within "
                    f"(0, num_frames). Got {switch_frame_indices} for num_frames={num_frames}."
                )
            previous = index

        return {
            "mode": "interactive",
            "prompts": prompts,
            "switch_frame_indices": switch_frame_indices,
        }

    def process_perception(self, text_prompt=None):
        if text_prompt is None:
            return {}
        self.check_interaction(text_prompt)
        return {"prompt": text_prompt}

    @staticmethod
    def _even_switch_indices(num_frames: int, num_segments: int) -> List[int]:
        if num_segments <= 1:
            return []
        segment_length = max(num_frames // num_segments, 1)
        return [segment_length * idx for idx in range(1, num_segments)]
