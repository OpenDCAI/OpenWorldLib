"""
OmniVinci reasoning wrapper.

This module follows the BaseReasoning interface and exposes a simple helper
around the HuggingFace OmniVinci checkpoint.
"""

from typing import Optional, Union, List, Dict, Any
import os
import torch
import copy
import torch.nn as nn
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModel, AutoConfig, PreTrainedModel, PretrainedConfig, Qwen2AudioEncoder
from transformers.generation import GenerationMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.siglip import SiglipVisionModel
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel, SiglipImageProcessor
from transformers.image_processing_utils import BaseImageProcessor
from ...base_reasoning import BaseReasoning
from .omnivinci.modules import from_pretrained_new


ImageLike = Union[str, bytes]
AudioLike = Union[str, bytes]
VideoLike = Union[str, bytes]

class OmniVinciReasoning(BaseReasoning):
    """
    Thin wrapper for OmniVinci multimodal model.
    Supports text, image, audio, and video inputs.
    """

    def __init__(
        self,
        model: AutoModel,
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
        pretrained_model_path: str = "./",
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: str = "torch.float16",
        device_map: Union[str, Dict] = "auto",
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "OmniVinciReasoning":
        """
        Load OmniVinci model and processor.

        Extra kwargs are forwarded to transformers.from_pretrained.
        """
        # Load config
        config = AutoConfig.from_pretrained(
            pretrained_model_path, 
            trust_remote_code=trust_remote_code
        )
        
        # Load model
        config._attn_implementation = "eager"  # 禁用 FlashAttention2
        config.use_flash_attn = False
        config.dynamic_s2 = True
        AutoModel.from_pretrained = from_pretrained_new
        model = AutoModel.from_pretrained(
            pretrained_model_path,
            trust_remote_code=trust_remote_code,
            config = config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager",
            **kwargs,
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            pretrained_model_path, 
            trust_remote_code=trust_remote_code
        )
        
        return cls(model=model, processor=processor, device=device)

    def api_init(self, api_key, endpoint):
        # API-based inference is not implemented for OmniVinci yet.
        raise NotImplementedError("API init is not supported for OmniVinci.")

    def _get_default_device(self) -> torch.device:
        # Prefer model's device when device_map is set, otherwise fall back to CUDA/CPU.
        if hasattr(self.model, "device"):
            return self.model.device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_messages(
        self,
        image_paths: Optional[Union[ImageLike, List[ImageLike]]] = None,
        audio_paths: Optional[Union[AudioLike, List[AudioLike]]] = None,
        video_paths: Optional[Union[VideoLike, List[VideoLike]]] = None,
        instruction: str = "",
    ):
        """
        Build messages for OmniVinci supporting images, audio, video, and text.
        
        Args:
            image_paths: Single path/bytes or a sequence for multi-image inputs.
            audio_paths: Single path/bytes or a sequence for multi-audio inputs.
            video_paths: Single path/bytes or a sequence for multi-video inputs.
            instruction: Text instruction.
            
        Returns:
            List of message dictionaries.
        """
        content = []
        
        # Process video inputs
        if video_paths is not None:
            if isinstance(video_paths, (str, bytes)):
                video_paths = [video_paths]
            content.extend([{"type": "video", "video": path} for path in video_paths])
        
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
        image_paths: Optional[Union[ImageLike, List[ImageLike]]] = None,
        audio_paths: Optional[Union[AudioLike, List[AudioLike]]] = None,
        video_paths: Optional[Union[VideoLike, List[VideoLike]]] = None,
        instruction: str = "",
        max_new_tokens: int = 1024,
        messages: Optional[List[Dict]] = None,
        generation_kwargs: Optional[Dict] = None,
        load_audio_in_video: bool = True,
        num_video_frames: int = 128,
        audio_length: str = "max_3600",
    ) -> str:
        """
        Run OmniVinci generation with support for images, audio, video, and text.
        
        Supports batched messages when `messages` is provided as list[dict]; 
        otherwise builds a single-sample batch from image_paths/audio_paths/video_paths + instruction.

        Args:
            image_paths: Single path/bytes or a sequence for multi-image inputs.
            audio_paths: Single path/bytes or a sequence for multi-audio inputs.
            video_paths: Single path/bytes or a sequence for multi-video inputs.
            instruction: Text instruction appended after the images/audio/video.
            max_new_tokens: Default generation length for convenience.
            messages: Optional raw chat template; when provided, image_paths/audio_paths/video_paths/instruction
                      are ignored and messages is used directly.
            generation_kwargs: Extra kwargs forwarded to model.generate.
            load_audio_in_video: Whether to use audio track in video inputs.
            num_video_frames: Number of frames to extract from video.
            audio_length: Maximum audio length to process.

        Returns:
            Decoded text string.
        """
        # Build messages if not provided
        if messages is None:
            messages = self._build_messages(
                image_paths=image_paths,
                audio_paths=audio_paths,
                video_paths=video_paths,
                instruction=instruction
            )
        
        # Configure model and processor for video/audio processing
        if hasattr(self.model, 'config'):
            self.model.config.load_audio_in_video = load_audio_in_video
            if num_video_frames > 0:
                self.model.config.num_video_frames = num_video_frames
            if audio_length != -1:
                self.model.config.audio_chunk_length = audio_length
        
        if hasattr(self.processor, 'config'):
            self.processor.config.load_audio_in_video = load_audio_in_video
            if num_video_frames > 0:
                self.processor.config.num_video_frames = num_video_frames
            if audio_length != -1:
                self.processor.config.audio_chunk_length = audio_length
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor([text])
        
        # Move inputs to device
        if hasattr(inputs, 'input_ids'):
            inputs.input_ids = inputs.input_ids.to(self.device)
        
        # Prepare generation config
        generation_config = self.model.default_generation_config if hasattr(self.model, 'default_generation_config') else {}
        gen_kwargs = {"max_new_tokens": max_new_tokens, "max_length": 99999999}
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)
        generation_config.update(**gen_kwargs)
        # Generate
        output_ids = self.model.generate(
            input_ids=inputs.input_ids,
            media=getattr(inputs, 'media', None),
            media_config=getattr(inputs, 'media_config', None),
            generation_config=generation_config
        )
        
        # Decode output
        output_text = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        return output_text
