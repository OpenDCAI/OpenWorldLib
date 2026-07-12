import contextlib
import json
from typing import Any, Dict, Optional

from ...base_reasoning import BaseReasoning
from .prompt_templates import (
    IMAGE_STEP1_EXPAND,
    IMAGE_STEP2_MAP,
    VIDEO_STEP1_EXPAND,
    VIDEO_STEP2_MAP,
)


_EXPAND_PROMPTS = {
    "t2i": IMAGE_STEP1_EXPAND,
    "t2v": VIDEO_STEP1_EXPAND,
    "ti2v": VIDEO_STEP1_EXPAND,
}

_MAP_PROMPTS = {
    "t2i": IMAGE_STEP2_MAP,
    "t2v": VIDEO_STEP2_MAP,
    "ti2v": VIDEO_STEP2_MAP,
}


class LingBotVideoReasoning(BaseReasoning):
    """Two-stage prompt rewriter used by LingBot-Video."""

    def __init__(self, backend: Any) -> None:
        super().__init__()
        self.backend = backend

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        adapter_path: str,
        device: str = "auto",
        max_new_tokens: int = 6144,
        **kwargs,
    ) -> "LingBotVideoReasoning":
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(pretrained_model_path, trust_remote_code=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path).eval()
        return cls(_TransformersRewriterBackend(model, processor, max_new_tokens))

    def inference(
        self,
        prompt: str,
        mode: str = "t2v",
        image: Any = None,
        duration: float = 5.0,
        **kwargs,
    ) -> Dict[str, Any]:
        if mode not in _EXPAND_PROMPTS:
            raise ValueError(f"Unsupported LingBot-Video reasoning mode: {mode}")
        if mode == "ti2v" and image is None:
            raise ValueError("LingBot-Video ti2v prompt rewriting requires an image.")

        duration_line = "" if mode == "t2i" else f"\n\nVideo Duration: {int(round(duration))} seconds"
        expand_input = f"{_EXPAND_PROMPTS[mode]}\n\n{prompt}{duration_line}"
        detailed = self.backend.generate(expand_input, image=image, use_lora=False).strip()
        map_duration = "" if mode == "t2i" else f"Video Duration: {int(round(duration))} seconds\n\n"
        raw = self.backend.generate(
            f"{_MAP_PROMPTS[mode]}\n\n{map_duration}DETAILED CAPTION:\n{detailed}",
            image=image,
            use_lora=True,
        ).strip()
        try:
            structured = json.loads(raw)
        except json.JSONDecodeError:
            from json_repair import repair_json

            structured = repair_json(raw, return_objects=True)
        if not isinstance(structured, dict):
            raise ValueError("LingBot-Video prompt rewriter did not return a JSON object.")
        return {
            "mode": mode,
            "detailed_prompt": detailed,
            "structured_prompt": structured,
            "prompt": json.dumps(structured, ensure_ascii=False, separators=(",", ":")),
        }


class _TransformersRewriterBackend:
    def __init__(self, model: Any, processor: Any, max_new_tokens: int) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def generate(self, text: str, image: Optional[Any], use_lora: bool) -> str:
        import torch

        content = ([{"type": "image", "image": image}] if image is not None else [])
        content.append({"type": "text", "text": text})
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[chat],
            images=[image] if image is not None else None,
            return_tensors="pt",
        ).to(self.model.device)
        adapter_context = contextlib.nullcontext() if use_lora else self.model.disable_adapter()
        with torch.no_grad(), adapter_context:
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]
