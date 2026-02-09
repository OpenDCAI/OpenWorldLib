from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import sys

from PIL import Image
import torch

from ...operators.yume_5b_operator import Yume5bOperator
from ...synthesis.visual_generation.yume.yume5b_synthesis import Yume5bSynthesis
from ...memories.visual_synthesis.wan.wan_2p2_memeory import Wan2p2Memory

DEFAULT_MODEL_VARIANT = "ti2v-5B"
SUPPORTED_SIZES = ("704*1280", "1280*704")
SizeLike = Union[str, Tuple[int, int]]

EXAMPLE_PROMPT = (
    "First-person perspective, walking down a busy neon-lit street at night, "
    "smooth camera motion and cinematic lighting."
)


class Yume5bPipeline:

    @staticmethod
    def _normalize_size_key(size: SizeLike) -> str:
        if isinstance(size, str):
            return size
        if (
            isinstance(size, tuple)
            and len(size) == 2
            and all(isinstance(v, int) and v > 0 for v in size)
        ):
            return f"{size[0]}*{size[1]}"
        raise TypeError(
            "size must be either a string like '1280*704' or a tuple like (1280, 704)."
        )

    def __init__(
        self,
        *,
        operator: Yume5bOperator,
        synthesis_model: Yume5bSynthesis,
        memory_module: Optional[Wan2p2Memory] = None,
        model_variant: str = DEFAULT_MODEL_VARIANT,
        size: SizeLike = "1280*704",
        prompt: Optional[str] = None,
        image: Optional[str] = None,
        save_file: Optional[str] = None,
        num_euler_timesteps: int = 4,
        sigma_shift: float = 7.0,
        latent_frame_zero: int = 8,
        frame_zero: int = 32,
        rollout_steps: int = 1,
        base_seed: int = -1,
        show_progress: bool = True,
    ) -> None:
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module if memory_module else Wan2p2Memory()

        self.model_variant = model_variant
        self.size = self._normalize_size_key(size)
        self.prompt = prompt
        self.image = image
        self.save_file = save_file
        self.num_euler_timesteps = num_euler_timesteps
        self.sigma_shift = sigma_shift
        self.latent_frame_zero = latent_frame_zero
        self.frame_zero = frame_zero
        self.rollout_steps = rollout_steps
        self.base_seed = base_seed
        self.show_progress = show_progress

    @classmethod
    def from_pretrained(
        cls,
        synthesis_model_path: str,
        *,
        size: SizeLike = "1280*704",
        prompt: Optional[str] = None,
        image: Optional[str] = None,
        save_file: Optional[str] = None,
        num_euler_timesteps: int = 4,
        sigma_shift: float = 7.0,
        latent_frame_zero: int = 8,
        frame_zero: int = 32,
        rollout_steps: int = 1,
        base_seed: int = -1,
        show_progress: bool = True,
        t5_fsdp: bool = False,
        t5_cpu: bool = False,
        dit_fsdp: bool = False,
        ulysses_size: int = 1,
        convert_model_dtype: bool = False,
        device_id: int = 0,
        rank: int = 0,
        **kwargs,
    ) -> "Yume5bPipeline":
        model_variant = kwargs.pop("model_variant", kwargs.pop("task", DEFAULT_MODEL_VARIANT))
        if model_variant != DEFAULT_MODEL_VARIANT:
            raise ValueError(
                f"Unsupported YUME model variant: {model_variant}. "
                f"Yume5bPipeline currently supports {DEFAULT_MODEL_VARIANT}."
            )

        if prompt is None:
            prompt = EXAMPLE_PROMPT

        size_key = cls._normalize_size_key(size)
        if size_key not in SUPPORTED_SIZES:
            raise ValueError(
                f"Unsupported size {size}, supported sizes are: {', '.join(SUPPORTED_SIZES)}"
            )

        if base_seed < 0:
            base_seed = random.randint(0, sys.maxsize)
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed)

        operator = Yume5bOperator()
        memory_module = Wan2p2Memory()
        synthesis_model = Yume5bSynthesis.from_pretrained(
            task=model_variant,
            ckpt_dir=synthesis_model_path,
            device_id=device_id,
            rank=rank,
            t5_fsdp=t5_fsdp,
            dit_fsdp=dit_fsdp,
            ulysses_size=ulysses_size,
            t5_cpu=t5_cpu,
            convert_model_dtype=convert_model_dtype,
        )

        return cls(
            operator=operator,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            model_variant=model_variant,
            size=size_key,
            prompt=prompt,
            image=image,
            save_file=save_file,
            num_euler_timesteps=num_euler_timesteps,
            sigma_shift=sigma_shift,
            latent_frame_zero=latent_frame_zero,
            frame_zero=frame_zero,
            rollout_steps=rollout_steps,
            base_seed=base_seed,
            show_progress=show_progress,
        )

    @staticmethod
    def _load_seed_video_from_path(
        video_path: str,
        *,
        max_frames: int = 33,
    ) -> torch.Tensor:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")

        decord_error: Optional[Exception] = None
        try:
            from decord import VideoReader

            reader = VideoReader(str(path))
            if len(reader) == 0:
                raise ValueError(f"Video has no frames: {path}")

            frame_count = min(len(reader), max_frames)
            frame_indices = list(range(frame_count))
            frames = reader.get_batch(frame_indices).asnumpy()  # [T, H, W, C]
            if frames.ndim != 4:
                raise ValueError(f"Unexpected video frame shape from decord: {frames.shape}")

            frames = frames[..., :3]
            video_tchw = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
            return video_tchw.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
        except Exception as exc:  # noqa: BLE001
            decord_error = exc

        try:
            import numpy as np
            import imageio.v3 as iio
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to import video readers for {path}. "
                f"decord error: {decord_error!r}"
            ) from exc

        try:
            frames_np = []
            for idx, frame in enumerate(iio.imiter(str(path))):
                if frame.ndim == 2:
                    frame = np.stack([frame, frame, frame], axis=-1)
                elif frame.ndim != 3:
                    raise ValueError(f"Unexpected frame shape from imageio: {frame.shape}")
                frames_np.append(frame[..., :3])
                if idx + 1 >= max_frames:
                    break

            if not frames_np:
                raise ValueError(f"Video has no frames: {path}")

            frames = np.stack(frames_np, axis=0)  # [T, H, W, C]
            video_tchw = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
            return video_tchw.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to read video from path: {path}. "
                f"Tried decord ({decord_error!r}) and imageio ({exc!r})."
            ) from exc

    @staticmethod
    def _infer_generation_mode(
        *,
        image_path: Optional[str],
        image: Optional[Image.Image],
        video_path: Optional[str] = None,
        seed_video: Optional[torch.Tensor] = None,
    ) -> str:
        if seed_video is not None or (video_path is not None and video_path != ""):
            return "v2v"
        if image is not None:
            return "i2v"
        if image_path is None or image_path == "":
            return "t2v"
        return "i2v"

    def process(
        self,
        *,
        prompt: str,
        image_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
        generation_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        input_for_perception = image if image is not None else image_path
        perception = self.operator.process_perception(input_path=input_for_perception)
        img = perception["input_image"]

        self.operator.get_interaction(prompt)
        interaction = self.operator.process_interaction()
        if generation_mode is None:
            generation_mode = self._infer_generation_mode(
                image_path=image_path,
                image=img,
            )

        return {
            "prompt": interaction["processed_prompt"],
            "image": img,
            "paths": {"image_path": image_path},
            "meta": {
                "model_variant": self.model_variant,
                "mode": generation_mode,
            },
        }

    @staticmethod
    def _load_prompt_schedule_from_caption(
        *,
        caption_path: str,
        prompt: str,
    ) -> List[str]:
        caption_file = Path(caption_path)
        if not caption_file.exists():
            raise FileNotFoundError(f"Caption file not found: {caption_file}")

        with caption_file.open("r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]
        if not lines:
            return [prompt]
        return [f"{line}{prompt}" for line in lines]

    def __call__(
        self,
        *,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
        video_path: Optional[str] = None,
        seed_video: Optional[torch.Tensor] = None,
        caption_path: Optional[str] = None,
        prompt_schedule: Optional[List[str]] = None,
        rollout_steps: Optional[int] = None,
        num_euler_timesteps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        latent_frame_zero: Optional[int] = None,
        frame_zero: Optional[int] = None,
        show_progress: Optional[bool] = None,
    ) -> torch.Tensor:
        if prompt is None:
            if self.prompt is None:
                raise ValueError("prompt must be provided either in initialization or call().")
            prompt = self.prompt

        if image is None and image_path is None:
            image_path = self.image

        if seed_video is not None and video_path not in (None, ""):
            raise ValueError("Only one of seed_video and video_path can be provided.")

        if seed_video is None and video_path not in (None, ""):
            seed_video = self._load_seed_video_from_path(video_path)

        generation_mode = self._infer_generation_mode(
            image_path=image_path,
            image=image,
            video_path=video_path,
            seed_video=seed_video,
        )

        if prompt_schedule is None and caption_path is not None:
            prompt_schedule = self._load_prompt_schedule_from_caption(
                caption_path=caption_path,
                prompt=prompt,
            )

        if seed_video is not None:
            self.operator.get_interaction(prompt)
            interaction = self.operator.process_interaction()
            processed = {
                "prompt": interaction["processed_prompt"],
                "image": None,
                "seed_video": seed_video,
                "paths": {
                    "image_path": image_path,
                    "video_path": video_path,
                },
                "meta": {
                    "model_variant": self.model_variant,
                    "mode": generation_mode,
                },
            }
        else:
            processed = self.process(
                prompt=prompt,
                image_path=image_path,
                image=image,
                generation_mode=generation_mode,
            )

        synthesis_params = {
            "size": self.size,
            "num_euler_timesteps": (
                num_euler_timesteps
                if num_euler_timesteps is not None
                else self.num_euler_timesteps
            ),
            "sigma_shift": sigma_shift if sigma_shift is not None else self.sigma_shift,
            "latent_frame_zero": (
                latent_frame_zero
                if latent_frame_zero is not None
                else self.latent_frame_zero
            ),
            "frame_zero": frame_zero if frame_zero is not None else self.frame_zero,
            "rollout_steps": rollout_steps if rollout_steps is not None else self.rollout_steps,
            "prompt_schedule": prompt_schedule,
            "base_seed": self.base_seed,
            "show_progress": self.show_progress if show_progress is None else show_progress,
        }

        return self.synthesis_model.predict(
            processed_inputs=processed,
            **synthesis_params,
        )

    def stream(
        self,
        *,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
        video_path: Optional[str] = None,
        caption_path: Optional[str] = None,
        prompt_schedule: Optional[List[str]] = None,
        rollout_steps: Optional[int] = None,
        show_progress: Optional[bool] = None,
    ) -> torch.Tensor:
        video = self.__call__(
            prompt=prompt,
            image_path=image_path,
            image=image,
            video_path=video_path,
            caption_path=caption_path,
            prompt_schedule=prompt_schedule,
            rollout_steps=rollout_steps,
            show_progress=show_progress,
        )

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"[Yume5bPipeline.stream] Expected torch.Tensor from predict, got {type(video)}"
            )

        self.memory_module.record(video)
        print(
            f"[Yume5bPipeline.stream] Recorded segment. "
            f"Total frames in memory: {len(getattr(self.memory_module, 'all_frames', []))}"
        )

        return video
