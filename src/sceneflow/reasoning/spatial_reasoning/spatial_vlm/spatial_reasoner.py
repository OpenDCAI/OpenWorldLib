"""
SpatialReasoner wrapper aligned with SpatialLadder.

Provides a BaseReasoning-compatible interface with optional image/video inputs
and batched chat templates.
"""

from typing import List, Optional, Sequence, Union

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from ...base_reasoning import BaseReasoning


ImageLike = Union[str, bytes]
VideoLike = Union[str, bytes]


class SpatialReasoner(BaseReasoning):
    def __init__(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        processor: AutoProcessor,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = torch.device(device) if device is not None else self._get_default_device()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = "ccvl/SpatialReasoner",
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = None,
        device_map: Union[str, dict] = "auto",
        **kwargs,
    ) -> "SpatialReasoner":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            **kwargs,
        )
        processor = AutoProcessor.from_pretrained(pretrained_model_path)
        return cls(model=model, processor=processor, device=device)

    def api_init(self, api_key, endpoint):
        raise NotImplementedError("API init is not supported for SpatialReasoner.")

    def _get_default_device(self) -> torch.device:
        if hasattr(self.model, "device"):
            return self.model.device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_messages(
        self,
        image_paths: Optional[Union[ImageLike, Sequence[ImageLike]]],
        video_paths: Optional[Union[VideoLike, Sequence[VideoLike]]],
        instruction: str,
    ):
        if image_paths is None:
            image_paths = []
        if video_paths is None:
            video_paths = []
        if isinstance(image_paths, (str, bytes)):
            image_paths = [image_paths]
        if isinstance(video_paths, (str, bytes)):
            video_paths = [video_paths]

        content = [{"type": "image", "image": path} for path in image_paths]
        content += [{"type": "video", "video": path} for path in video_paths]
        content.append({"type": "text", "text": instruction})
        return [{"role": "user", "content": content}]

    @torch.no_grad()
    def inference(
        self,
        instruction: str = "",
        image_paths: Optional[Union[ImageLike, Sequence[ImageLike]]] = None,
        video_paths: Optional[Union[VideoLike, Sequence[VideoLike]]] = None,
        max_new_tokens: int = 2048,
        messages: Optional[list] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> List[str]:
        """
        Run SpatialReasoner generation. Supports batched messages when `messages`
        is provided as list[list[dict]]; otherwise builds a single-sample batch
        from image_paths/video_paths + instruction. Either images or videos can
        be empty.
        """
        if messages is None:
            batched_messages = [
                self._build_messages(
                    image_paths=image_paths,
                    video_paths=video_paths,
                    instruction=instruction,
                )
            ]
        else:
            if not messages:
                raise ValueError("messages must be non-empty.")
            batched_messages = [messages] if isinstance(messages[0], dict) else messages

        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batched_messages
        ]

        vision_info = [process_vision_info(m) for m in batched_messages]
        image_inputs, video_inputs = [], []
        for imgs, vids in vision_info:
            image_inputs.append(imgs if imgs else None)
            video_inputs.append(vids if vids else None)
        if all(v is None for v in image_inputs):
            image_inputs = None
        if all(v is None for v in video_inputs):
            video_inputs = None

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
