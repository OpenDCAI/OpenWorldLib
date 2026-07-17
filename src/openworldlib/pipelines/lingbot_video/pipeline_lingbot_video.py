import json
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from ...memories.visual_synthesis.lingbot_video import LingBotVideoMemory
from ...operators.lingbot_video_operator import LingBotVideoOperator
from ...reasoning.general_reasoning.lingbot_video import LingBotVideoReasoning
from ...synthesis.visual_generation.lingbot_video import LingBotVideoSynthesis


class LingBotVideoPipeline:
    """OpenWorldLib pipeline interface for LingBot-Video T2I/T2V/TI2V generation."""

    def __init__(
        self,
        *,
        operator: LingBotVideoOperator,
        synthesis_model: LingBotVideoSynthesis,
        memory_module: Optional[LingBotVideoMemory] = None,
        reasoning_model: Optional[LingBotVideoReasoning] = None,
        mode: str = "t2v",
    ) -> None:
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module or LingBotVideoMemory()
        self.reasoning_model = reasoning_model
        self.mode = self.operator.normalize_mode(mode)
        self.backend = self.synthesis_model.backend
        self.requested_backend = self.synthesis_model.requested_backend
        self.engine_name = self.synthesis_model.engine_name

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        mode: str = "t2v",
        backend: str = "diffusers",
        reasoning_model_path: Optional[str] = None,
        reasoning_adapter_path: Optional[str] = None,
        **kwargs,
    ) -> "LingBotVideoPipeline":
        if bool(reasoning_model_path) != bool(reasoning_adapter_path):
            raise ValueError("Both reasoning_model_path and reasoning_adapter_path are required.")
        operator = LingBotVideoOperator()
        normalized_mode = operator.normalize_mode(mode)
        synthesis_model = LingBotVideoSynthesis.from_pretrained(
            pretrained_model_path=model_path,
            mode=normalized_mode,
            backend=backend,
            **kwargs,
        )
        reasoning_model = None
        if reasoning_model_path:
            reasoning_model = LingBotVideoReasoning.from_pretrained(
                reasoning_model_path,
                reasoning_adapter_path,
            )
        return cls(
            operator=operator,
            synthesis_model=synthesis_model,
            memory_module=LingBotVideoMemory(),
            reasoning_model=reasoning_model,
            mode=normalized_mode,
        )

    def process(
        self,
        *,
        prompt: str,
        images: Any = None,
        mode: Optional[str] = None,
        rewrite_prompt: bool = False,
        duration: float = 5.0,
    ) -> Dict[str, Any]:
        effective_mode = self.operator.normalize_mode(mode or self.mode)
        if effective_mode != self.mode:
            raise ValueError(
                f"This pipeline was loaded for mode {self.mode!r}; "
                f"load a separate pipeline for mode {effective_mode!r}."
            )
        interaction = self.operator.process_interaction(mode=effective_mode, prompt=prompt)
        perception = self.operator.process_perception(mode=effective_mode, images=images)
        processed_prompt = interaction["prompt"]
        if rewrite_prompt:
            if self.reasoning_model is None:
                raise ValueError("rewrite_prompt=True requires a configured reasoning model.")
            reasoning = self.reasoning_model.inference(
                processed_prompt,
                mode=effective_mode,
                image=perception["image"],
                duration=duration,
            )
            if reasoning["json"] is None:
                raise ValueError("LingBot-Video prompt rewriter did not return a JSON object.")
            processed_prompt = json.dumps(reasoning["json"], ensure_ascii=False, separators=(",", ":"))
        processed = {
            "mode": interaction["mode"],
            "prompt": processed_prompt,
            "image": perception["image"],
        }
        self.memory_module.record(processed, type="text", metadata={"mode": effective_mode})
        if perception["image"] is not None:
            self.memory_module.record(perception["image"], type="image", metadata={"mode": effective_mode})
        return processed

    def __call__(
        self,
        *,
        prompt: str,
        images: Any = None,
        mode: Optional[str] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        steps: int = 40,
        guidance_scale: float = 3.0,
        shift: float = 3.0,
        seed: int = 42,
        rewrite_prompt: bool = False,
        duration: float = 5.0,
        **kwargs,
    ) -> Any:
        processed_inputs = self.process(
            prompt=prompt,
            images=images,
            mode=mode,
            rewrite_prompt=rewrite_prompt,
            duration=duration,
        )
        output = self.synthesis_model.predict(
            processed_inputs=processed_inputs,
            height=height,
            width=width,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=guidance_scale,
            shift=shift,
            seed=seed,
            **kwargs,
        )
        output_type = "video" if processed_inputs["mode"] != "t2i" else "image_output"
        self.memory_module.record(output, type=output_type)
        last_frame = self._last_frame(output)
        if last_frame is not None:
            self.memory_module.record(last_frame, type="image")
        return output

    @staticmethod
    def _last_frame(output: Any) -> Optional[Image.Image]:
        frames = output.frames if hasattr(output, "frames") else output
        if isinstance(frames, (list, tuple)):
            if not frames:
                return None
            frames = frames[0]
        if isinstance(frames, np.ndarray):
            while frames.ndim > 3:
                frames = frames[-1]
            if frames.ndim != 3:
                return None
            if np.issubdtype(frames.dtype, np.floating):
                frames = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                frames = np.clip(frames, 0, 255).astype(np.uint8)
            return Image.fromarray(frames).convert("RGB")
        return frames.convert("RGB") if isinstance(frames, Image.Image) else None

    def stream(self, **kwargs) -> Any:
        if self.mode == "ti2v" and kwargs.get("images") is None:
            last_image = self.memory_module.select(type="image")
            if last_image is not None:
                kwargs["images"] = last_image
        return self(**kwargs)
