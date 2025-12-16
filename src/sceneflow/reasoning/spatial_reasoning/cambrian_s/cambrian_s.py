"""
Lightweight Cambrian-S loader & inference helper.

Wraps the Cambrian builder utilities into a BaseReasoning-style interface that
can run single-turn image or video prompts.
"""

from typing import List, Optional, Sequence, Union
from io import BytesIO

import torch
from PIL import Image

from .cambrian.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from .cambrian.conversation import conv_templates
from .cambrian.mm_utils import (
    get_model_name_from_path,
    process_images,
    process_videos,
    tokenizer_image_token,
)
from .cambrian.model.builder import load_pretrained_model
from ...base_reasoning import BaseReasoning


ImageLike = Union[str, bytes, Image.Image]
VideoLike = Union[str, bytes]


class CambrianS(BaseReasoning):
    """
    Single-turn Cambrian-S inference with optional image or video input.
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        device: Optional[Union[str, torch.device]] = None,
        conv_template: str = "qwen_2",
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_template = conv_template
        self.device = torch.device(device) if device is not None else self._get_default_device()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[Union[str, torch.device]] = None,
        device_map: Union[str, dict] = "auto",
        torch_dtype: torch.dtype = torch.float16,
        use_flash_attn: bool = False,
        video_max_frames: int = 32,
        video_fps: int = 1,
        video_force_sample: bool = False,
        add_time_instruction: bool = False,
        miv_token_len: int = 196,
        si_token_len: int = 729,
        image_aspect_ratio: str = "anyres",
        anyres_max_subimages: int = 9,
        conv_template: str = "qwen_2",
        **kwargs,
    ) -> "CambrianS":
        model_name = get_model_name_from_path(pretrained_model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            pretrained_model_path,
            None,
            model_name,
            device_map=device_map,
            use_flash_attn=use_flash_attn,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        model.config.video_max_frames = video_max_frames
        model.config.video_fps = video_fps
        model.config.video_force_sample = video_force_sample
        model.config.add_time_instruction = add_time_instruction
        model.config.miv_token_len = miv_token_len
        model.config.si_token_len = si_token_len
        model.config.image_aspect_ratio = image_aspect_ratio
        model.config.anyres_max_subimages = anyres_max_subimages

        return cls(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device,
            conv_template=conv_template,
        )

    def api_init(self, api_key, endpoint):
        raise NotImplementedError("API init is not supported for CambrianS.")

    def _get_default_device(self) -> torch.device:
        if hasattr(self.model, "device"):
            return self.model.device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_image(self, img: ImageLike) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, bytes):
            return Image.open(BytesIO(img)).convert("RGB")
        return Image.open(img).convert("RGB")

    def _prepare_vision(
        self,
        images: Optional[Sequence[ImageLike]],
        videos: Optional[Sequence[VideoLike]],
    ):
        has_images = images is not None and len(images) > 0
        has_videos = videos is not None and len(videos) > 0
        if has_images and has_videos:
            raise ValueError("CambrianS currently supports either images or videos per call, not both.")

        if has_images:
            pil_images = [self._load_image(img) for img in images]
            visual_tensors, visual_sizes = process_images(
                pil_images,
                self.image_processor,
                self.model.config,
            )
            return visual_tensors, visual_sizes

        if has_videos:
            visual_tensors, visual_sizes, _ = process_videos(
                videos,
                self.image_processor,
                self.model.config,
            )
            return visual_tensors, visual_sizes

        return None, None

    def _build_prompt(self, instruction: str, num_visuals: int) -> str:
        conv = conv_templates[self.conv_template].copy()

        if num_visuals > 0:
            if getattr(self.model.config, "mm_use_im_start_end", False):
                visual_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN) * num_visuals
            else:
                visual_token = DEFAULT_IMAGE_TOKEN * num_visuals
            user_msg = visual_token + "\n" + instruction
        else:
            user_msg = instruction

        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    @torch.no_grad()
    def inference(
        self,
        instruction: str,
        image_paths: Optional[Union[ImageLike, Sequence[ImageLike]]] = None,
        video_paths: Optional[Union[VideoLike, Sequence[VideoLike]]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
        generation_kwargs: Optional[dict] = None,
    ) -> List[str]:
        if image_paths is None:
            image_paths = []
        if video_paths is None:
            video_paths = []
        if isinstance(image_paths, (str, bytes, Image.Image)):
            image_paths = [image_paths]
        if isinstance(video_paths, (str, bytes)):
            video_paths = [video_paths]

        visual_tensors, visual_sizes = self._prepare_vision(image_paths, video_paths)
        num_visuals = 0
        if visual_tensors is not None:
            # visual_tensors is a list (per vision tower)
            num_visuals = len(image_paths) if image_paths else len(video_paths)
            visual_tensors = [vt.to(self.device, dtype=torch.float16) for vt in visual_tensors]

        prompt = self._build_prompt(instruction, num_visuals)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "num_beams": num_beams,
        }
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        outputs = self.model.generate(
            inputs=input_ids,
            images=visual_tensors,
            image_sizes=visual_sizes,
            use_cache=True,
            **gen_kwargs,
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [decoded[0].strip()]
