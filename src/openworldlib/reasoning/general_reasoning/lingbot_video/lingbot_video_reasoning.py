import contextlib
import io
import re
from typing import Any, Dict, Optional

import requests
from PIL import Image

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

from ...base_reasoning import BaseReasoning
from .lingbot_video.prompt_templates import (
    IMAGE_STEP1_EXPAND,
    IMAGE_STEP2_MAP,
    VIDEO_STEP1_EXPAND,
    VIDEO_STEP2_MAP,
)


MODES = {
    "t2v": dict(s1=VIDEO_STEP1_EXPAND, s2=VIDEO_STEP2_MAP, image=False, duration=True),
    "ti2v": dict(s1=VIDEO_STEP1_EXPAND, s2=VIDEO_STEP2_MAP, image=True, duration=True),
    "t2i": dict(s1=IMAGE_STEP1_EXPAND, s2=IMAGE_STEP2_MAP, image=False, duration=False),
}


def _has_cjk(text: str) -> bool:
    return any("一" <= char <= "鿿" for char in text)


def load_image(source: Any) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, str) and re.match(r"^https?://", source):
        return Image.open(io.BytesIO(requests.get(source, timeout=30).content)).convert("RGB")
    return Image.open(source).convert("RGB")


def _step1_text(mode: str, prompt: str, duration: int) -> str:
    system_prompt = MODES[mode]["s1"]
    if mode == "t2i":
        return system_prompt + "\n\nUser image prompt:\n" + prompt
    duration_line = (
        f"\n\n视频时长：{duration} 秒"
        if _has_cjk(prompt)
        else f"\n\nVideo Duration: {duration} seconds"
    )
    return system_prompt + "\n\n" + prompt + duration_line


def _step2_text(mode: str, detailed: str, duration: int) -> str:
    system_prompt = MODES[mode]["s2"]
    if mode == "t2i":
        return system_prompt + "\n\nDETAILED CAPTION:\n" + detailed
    return (
        system_prompt
        + f"\n\nVideo Duration: {duration} seconds\n\nDETAILED CAPTION:\n"
        + detailed
        + "\n\nOutput the JSON now."
    )


def parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if repair_json is None:
        raise ImportError("rewriter parsing requires the json_repair package.")
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        parsed = repair_json(text, return_objects=True)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


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
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        config = MODES[mode]
        rounded_duration = int(round(duration))
        resolved_image = None
        if config["image"]:
            if image is None:
                raise ValueError(f"{mode} requires first_frame (path / URL / PIL.Image)")
            resolved_image = load_image(image)

        detailed = self.backend.generate(
            _step1_text(mode, prompt, rounded_duration),
            image=resolved_image,
            use_lora=False,
        ).strip()
        raw = self.backend.generate(
            _step2_text(mode, detailed, rounded_duration),
            image=resolved_image,
            use_lora=True,
        ).strip()
        return {
            "mode": mode,
            "detailed": detailed,
            "json": parse_json(raw),
            "json_raw": raw,
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
