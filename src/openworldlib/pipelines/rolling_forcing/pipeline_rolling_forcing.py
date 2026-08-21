from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from PIL import Image

from ...memories.visual_synthesis.rolling_forcing.rolling_forcing_memory import (
    RollingForcingMemory,
)
from ...operators.rolling_forcing_operator import RollingForcingOperator
from ...synthesis.visual_generation.rolling_forcing.rolling_forcing_synthesis import (
    DEFAULT_ROLLING_FORCING_MODEL_ROOT,
    RollingForcingSynthesis,
)


class RollingForcingPipeline:
    def __init__(
        self,
        operators: Optional[RollingForcingOperator] = None,
        synthesis_model: Optional[RollingForcingSynthesis] = None,
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
        max_latent_context_frames: int = 24,
        **kwargs,
    ) -> "RollingForcingPipeline":
        synthesis_model = RollingForcingSynthesis.from_pretrained(
            pretrained_model_path=model_path or DEFAULT_ROLLING_FORCING_MODEL_ROOT,
            required_components=required_components,
            device=device,
            weight_dtype=weight_dtype,
            **kwargs,
        )
        return cls(
            operators=RollingForcingOperator(),
            synthesis_model=synthesis_model,
            memory_module=RollingForcingMemory(max_latent_context_frames=max_latent_context_frames),
            device=device,
            weight_dtype=weight_dtype,
        )

    def process(
        self,
        prompt: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
        images: Optional[Image.Image] = None,
        num_samples: int = 1,
        size: tuple[int, int] = (480, 832),
    ):
        prompt_sequence = self.operators.process_prompts(prompt=prompt, prompts=prompts)
        self.operators.get_interaction(prompt_sequence)
        interaction = self.operators.process_interaction(num_samples=num_samples)
        self.operators.delete_last_interaction()

        perception = self.operators.process_perception(images=images, size=size)
        return {
            "prompts": interaction["prompts"],
            "num_samples": interaction["num_samples"],
            "image": perception["image"],
            "size": perception["size"],
        }

    def __call__(
        self,
        prompt: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
        images: Optional[Image.Image] = None,
        num_frames: int = 126,
        num_samples: int = 1,
        seed: Optional[int] = None,
        return_latents: bool = False,
        include_context: bool = True,
        size: tuple[int, int] = (480, 832),
        **kwargs,
    ):
        processed = self.process(
            prompt=prompt,
            prompts=prompts,
            images=images,
            num_samples=num_samples,
            size=size,
        )
        return self.synthesis_model.predict(
            prompts=processed["prompts"],
            num_output_frames=num_frames,
            num_samples=processed["num_samples"],
            seed=seed,
            images=processed["image"],
            return_latents=return_latents,
            include_context=include_context,
            size=processed["size"],
            **kwargs,
        )

    def stream(
        self,
        prompt: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
        images: Optional[Image.Image] = None,
        num_frames: int = 126,
        num_samples: int = 1,
        seed: Optional[int] = None,
        reset: bool = False,
        use_memory: bool = True,
        return_latents: bool = False,
        return_full_video: bool = False,
        size: tuple[int, int] = (480, 832),
        **kwargs,
    ):
        if reset:
            self.memory_module.manage(action="reset")

        processed = self.process(
            prompt=prompt,
            prompts=prompts,
            images=images,
            num_samples=num_samples,
            size=size,
        )

        initial_latents = None
        if processed["image"] is None and use_memory:
            initial_latents = self.memory_module.select(type="latents")
        context_frames = int(initial_latents.shape[1]) if initial_latents is not None else 0
        if processed["image"] is not None:
            context_frames = 1

        output_video, output_latents = self.synthesis_model.predict(
            prompts=processed["prompts"],
            num_output_frames=num_frames + context_frames,
            num_samples=processed["num_samples"],
            seed=seed,
            images=processed["image"],
            initial_latents=initial_latents,
            return_latents=True,
            include_context=True,
            size=processed["size"],
            **kwargs,
        )

        visible_video = output_video
        if not return_full_video and context_frames > 0:
            visible_video = output_video[:, context_frames:]

        metadata = {
            "type": "video",
            "prompts": processed["prompts"],
            "num_frames": num_frames,
            "context_frames": context_frames,
        }
        self.memory_module.record(visible_video, metadata=metadata)
        latents_metadata = dict(metadata)
        latents_metadata["type"] = "latents"
        self.memory_module.record(output_latents, metadata=latents_metadata)

        if return_latents:
            return visible_video, output_latents
        return visible_video
