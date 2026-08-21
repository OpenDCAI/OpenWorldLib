from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch

from ...memories.visual_synthesis.longlive.longlive_memory import LongLiveMemory
from ...operators.longlive_operator import LongLiveOperator
from ...synthesis.visual_generation.longlive.longlive_synthesis import (
    DEFAULT_LONGLIVE_MODEL_ROOT,
    LongLiveSynthesis,
)


class LongLivePipeline:
    def __init__(
        self,
        operators: Optional[LongLiveOperator] = None,
        synthesis_model: Optional[LongLiveSynthesis] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
    ):
        self.synthesis_model = synthesis_model
        self.operators = operators
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        required_components: Optional[dict] = None,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
        **kwargs,
    ) -> "LongLivePipeline":
        synthesis_model = LongLiveSynthesis.from_pretrained(
            pretrained_model_path=model_path or DEFAULT_LONGLIVE_MODEL_ROOT,
            required_components=required_components,
            device=device,
            weight_dtype=weight_dtype,
            **kwargs,
        )
        return cls(
            operators=LongLiveOperator(),
            synthesis_model=synthesis_model,
            memory_module=LongLiveMemory(),
            device=device,
            weight_dtype=weight_dtype,
        )

    def process(
        self,
        prompts: Sequence[str] | str,
        num_output_frames: int,
        switch_frame_indices: Optional[Sequence[int]] = None,
    ):
        self.operators.get_interaction(prompts)
        operator_condition = self.operators.process_interaction(
            num_frames=num_output_frames,
            switch_frame_indices=switch_frame_indices,
        )
        self.operators.delete_last_interaction()
        return operator_condition

    def __call__(
        self,
        prompt: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
        num_frames: int = 120,
        switch_frame_indices: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
        return_latents: bool = False,
        **kwargs,
    ):
        prompt_sequence = self._resolve_prompts(prompt=prompt, prompts=prompts)
        operator_condition = self.process(
            prompts=prompt_sequence,
            num_output_frames=num_frames,
            switch_frame_indices=switch_frame_indices,
        )

        output = self.synthesis_model.predict(
            prompts=operator_condition["prompts"],
            num_output_frames=num_frames,
            switch_frame_indices=operator_condition["switch_frame_indices"],
            seed=seed,
            return_latents=return_latents,
            **kwargs,
        )
        return output

    def stream(
        self,
        prompt: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
        num_frames: int = 120,
        switch_frame_indices: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
        return_latents: bool = False,
        reset: bool = False,
        **kwargs,
    ):
        if reset:
            self.memory_module.manage(action="reset")

        prompt_sequence = self._resolve_prompts(prompt=prompt, prompts=prompts)
        effective_switch_indices = switch_frame_indices
        if effective_switch_indices is None and len(prompt_sequence) > 1:
            effective_switch_indices = self.operators._even_switch_indices(
                num_frames=num_frames,
                num_segments=len(prompt_sequence),
            )

        output = self.__call__(
            prompts=prompt_sequence,
            num_frames=num_frames,
            switch_frame_indices=effective_switch_indices,
            seed=seed,
            return_latents=return_latents,
            **kwargs,
        )

        metadata = {
            "type": "video",
            "prompts": prompt_sequence,
            "num_frames": num_frames,
            "switch_frame_indices": list(effective_switch_indices or []),
        }
        if return_latents:
            video, latents = output
            self.memory_module.record(video, metadata=metadata)
            latents_metadata = dict(metadata)
            latents_metadata["type"] = "latents"
            self.memory_module.record(latents, metadata=latents_metadata)
        else:
            self.memory_module.record(output, metadata=metadata)
        return output

    @staticmethod
    def _resolve_prompts(prompt: Optional[str], prompts: Optional[Sequence[str]]) -> List[str]:
        if prompts is not None:
            resolved = list(prompts)
        elif prompt is not None:
            resolved = [prompt]
        else:
            raise ValueError("Provide either `prompt` or `prompts`.")
        if len(resolved) == 0:
            raise ValueError("Prompt list cannot be empty.")
        return resolved
