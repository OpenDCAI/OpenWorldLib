"""
Qwen2.5_Omni reasoning wrapper.

This module follows the BaseReasoning interface and exposes a simple helper
around the HuggingFace Qwen2.5-Omni checkpoint.
"""

from typing import List, Optional, Sequence, Union

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from ...base_reasoning import BaseReasoning


ImageLike = Union[str, bytes]
AudioLike = Union[str, bytes]


class Qwen2p5OmniReasoning(BaseReasoning):
    """
    Thin wrapper for Qwen2.5-Omni multimodal model.
    Supports text, image, audio, and video inputs with optional audio output.
    """

    def __init__(
        self,
        model: Qwen2_5OmniForConditionalGeneration,
        processor: Qwen2_5OmniProcessor,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = torch.device(device) if device is not None else self._get_default_device()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = "Qwen/Qwen2.5-Omni-7B",
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = None,
        device_map: Union[str, dict] = "auto",
        
        **kwargs,
    ) -> "Qwen2p5Omni":
        """
        Load Qwen2.5-Omni model and processor.

        Extra kwargs are forwarded to transformers.from_pretrained.
        """
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            **kwargs,
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(pretrained_model_path)
        return cls(model=model, processor=processor, device=device)

    def api_init(self, api_key, endpoint):
        # API-based inference is not implemented for Qwen2.5-Omni yet.
        raise NotImplementedError("API init is not supported for Qwen2.5-Omni.")

    def _get_default_device(self) -> torch.device:
        # Prefer model's device when device_map is set, otherwise fall back to CUDA/CPU.
        if hasattr(self.model, "device"):
            return self.model.device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_messages(
        self,
        image_paths: Optional[Union[ImageLike, Sequence[ImageLike]]] = None,
        audio_paths: Optional[Union[AudioLike, Sequence[AudioLike]]] = None,
        instruction: str = "",
    ):
        """
        Build messages for Qwen2.5-Omni supporting images, audio, and text.
        
        Args:
            image_paths: Single path/bytes or a sequence for multi-image inputs.
            audio_paths: Single path/bytes or a sequence for multi-audio inputs.
            instruction: Text instruction.
            
        Returns:
            List of message dictionaries.
        """
        content = []
        
        # Process image inputs
        if image_paths is not None:
            if isinstance(image_paths, (str, bytes)):
                image_paths = [image_paths]
            content.extend([{"type": "image", "image": path} for path in image_paths])
        
        # Process audio inputs
        if audio_paths is not None:
            if isinstance(audio_paths, (str, bytes)):
                audio_paths = [audio_paths]
            content.extend([{"type": "audio", "audio": path} for path in audio_paths])
        
        # Add text instruction
        if instruction:
            content.append({"type": "text", "text": instruction})
        
        return [{"role": "user", "content": content}]

    @torch.no_grad()
    def inference(
        self,
        image_paths: Optional[Union[ImageLike, Sequence[ImageLike]]] = None,
        audio_paths: Optional[Union[AudioLike, Sequence[AudioLike]]] = None,
        instruction: str = "",
        max_new_tokens: int = 128,
        messages: Optional[list] = None,
        generation_kwargs: Optional[dict] = None,
        use_audio_in_video: bool = True,
        return_audio: bool = False,
    ) -> Union[List[str], tuple]:
        """
        Run Qwen2.5-Omni generation with support for images, audio, video, and text.
        
        Supports batched messages when `messages` is provided as list[list[dict]]; 
        otherwise builds a single-sample batch from image_paths/audio_paths + instruction.

        Args:
            image_paths: Single path/bytes or a sequence for multi-image inputs.
            audio_paths: Single path/bytes or a sequence for multi-audio inputs.
            instruction: Text instruction appended after the images/audio.
            max_new_tokens: Default generation length for convenience.
            messages: Optional raw chat template; when provided, image_paths/audio_paths/instruction
                      are ignored and messages is used directly. Supports a single
                      conversation (list[dict]) or a batch (list[list[dict]]).
            generation_kwargs: Extra kwargs forwarded to model.generate.
            use_audio_in_video: Whether to use audio track in video inputs.
            return_audio: Whether to return generated audio output.

        Returns:
            If return_audio is False: List of decoded text strings (one per batch element).
            If return_audio is True: Tuple of (List of text strings, audio tensor).
        """
        if messages is None:
            batched_messages = [
                self._build_messages(
                    image_paths=image_paths, 
                    audio_paths=audio_paths, 
                    instruction=instruction
                )
            ]
        else:
            if not messages:
                raise ValueError("messages must be non-empty.")
            batched_messages = [messages] if isinstance(messages[0], dict) else messages

        # Apply chat template
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batched_messages
        ]

        # Process multimodal inputs (images, videos, audio)
        audios, images, videos = process_mm_info(batched_messages, use_audio_in_video=use_audio_in_video)

        # Prepare inputs for the model
        inputs = self.processor(
            text=texts,
            audio=audios,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(self.device).to(self.model.dtype)

        # Prepare generation kwargs
        gen_kwargs = {"max_new_tokens": max_new_tokens, "use_audio_in_video": use_audio_in_video}
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        # Generate
        if return_audio:
            text_ids, audio = self.model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, text_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return output_text, audio
        else:
            gen_kwargs["return_audio"] = False
            text_ids = self.model.generate(**inputs, **gen_kwargs)
            # Trim the input tokens from the output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, text_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return output_text