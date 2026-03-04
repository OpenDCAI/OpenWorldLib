from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image
import torch

from ...memories.visual_synthesis.wan.wan_2p2_memeory import Wan2p2Memory
from ...operators.yume_operator import YumeOperator
from ...synthesis.visual_generation.yume.yume_synthesis import (
    MODEL_DEFAULTS,
    SizeLike,
    YumeSynthesis,
)


EXAMPLE_PROMPT = (
    "First-person perspective, walking down a busy neon-lit street at night, "
    "smooth camera motion and cinematic lighting."
)


class YumePipeline:

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
        synthesis_model: YumeSynthesis,
        operator: Optional[YumeOperator] = None,
        memory_module: Optional[Wan2p2Memory] = None,
    ) -> None:
        self.operator = operator if operator is not None else YumeOperator()
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module if memory_module else Wan2p2Memory()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[Union[int, str, torch.device]] = None,
        *,
        model_variant: Optional[str] = None,
        **kwargs,
    ) -> "YumePipeline":
        if model_path is None:
            synthesis_model_path = "stdstu123/Yume-5B-720P"
        else:
            synthesis_model_path = model_path

        print(f"Loading Yume synthesis model from {synthesis_model_path}...")
        synthesis_model = YumeSynthesis.from_pretrained(
            pretrained_model_path=synthesis_model_path,
            model_variant=model_variant,
            device=device,
            **kwargs,
        )
        return cls(synthesis_model=synthesis_model)

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

    @staticmethod
    def _build_prompt_schedule(
        prompt: str,
        caption: Optional[Union[str, Sequence[str]]],
        *,
        prompt_schedule: Optional[Sequence[str]],
    ) -> Optional[List[str]]:
        if prompt_schedule is not None:
            return [line for line in prompt_schedule if line]

        if caption is None:
            return None

        if isinstance(caption, str):
            lines = [line.strip() for line in caption.splitlines() if line.strip()]
            if not lines:
                lines = [caption.strip()] if caption.strip() else []
        else:
            lines = [str(line).strip() for line in caption if str(line).strip()]

        if not lines:
            return None

        return [f"{line}{prompt}" for line in lines]

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
                "model_variant": self.synthesis_model.task,
                "mode": generation_mode,
            },
        }

    def __call__(
        self,
        prompt: Optional[str] = None,
        interactions: Optional[Union[str, Sequence[str]]] = None,
        image_path: Optional[str] = None,
        images: Optional[Image.Image] = None,
        size: Optional[SizeLike] = None,
        seed: Optional[int] = None,
        *,
        video_path: Optional[str] = None,
        seed_video: Optional[torch.Tensor] = None,
        prompt_schedule: Optional[Sequence[str]] = None,
        rollout_steps: Optional[int] = None,
        num_euler_timesteps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        latent_frame_zero: Optional[int] = None,
        frame_zero: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        sampling_method: Optional[str] = None,
        sde_eta: Optional[float] = None,
        rand_num_img: Optional[float] = None,
        show_progress: Optional[bool] = None,
    ) -> torch.Tensor:
        if prompt is None:
            prompt = EXAMPLE_PROMPT

        if seed_video is not None and video_path not in (None, ""):
            raise ValueError("Only one of seed_video and video_path can be provided.")

        if seed_video is None and video_path not in (None, ""):
            seed_video = self._load_seed_video_from_path(video_path)

        generation_mode = self._infer_generation_mode(
            image_path=image_path,
            image=images,
            video_path=video_path,
            seed_video=seed_video,
        )

        resolved_prompt_schedule = self._build_prompt_schedule(
            prompt=prompt,
            caption=interactions,
            prompt_schedule=prompt_schedule,
        )

        ## get_interaction 应该是使用 interactions 作为输入
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
                    "model_variant": self.synthesis_model.task,
                    "mode": generation_mode,
                },
            }
        else:
            processed = self.process(
                prompt=prompt,
                image_path=image_path,
                image=images,
                generation_mode=generation_mode,
            )

        defaults = MODEL_DEFAULTS[self.synthesis_model.task]

        synthesis_params = {
            "size": self._normalize_size_key(size if size is not None else defaults["size"]),
            "num_euler_timesteps": (
                num_euler_timesteps
                if num_euler_timesteps is not None
                else defaults["num_euler_timesteps"]
            ),
            "sigma_shift": sigma_shift if sigma_shift is not None else defaults["sigma_shift"],
            "latent_frame_zero": (
                latent_frame_zero
                if latent_frame_zero is not None
                else defaults["latent_frame_zero"]
            ),
            "frame_zero": frame_zero if frame_zero is not None else defaults["frame_zero"],
            "rollout_steps": rollout_steps if rollout_steps is not None else 1,
            "prompt_schedule": resolved_prompt_schedule,
            "base_seed": -1 if seed is None else int(seed),
            "guidance_scale": (
                guidance_scale if guidance_scale is not None else defaults["guidance_scale"]
            ),
            "sampling_method": (
                sampling_method if sampling_method is not None else "ode"
            ),
            "sde_eta": sde_eta if sde_eta is not None else 0.3,
            "rand_num_img": rand_num_img if rand_num_img is not None else 0.6,
            "show_progress": True if show_progress is None else show_progress,
        }

        return self.synthesis_model.predict(
            processed_inputs=processed,
            **synthesis_params,
        )

    def stream(
        self,
        prompt: Optional[str] = None,
        interactions: Optional[Union[str, Sequence[str]]] = None,
        image_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
        *,
        video_path: Optional[str] = None,
        prompt_schedule: Optional[Sequence[str]] = None,
        rollout_steps: Optional[int] = None,
        show_progress: Optional[bool] = None,
    ) -> torch.Tensor:
        video = self.__call__(
            prompt=prompt,
            interactions=interactions,
            image_path=image_path,
            image=image,
            video_path=video_path,
            prompt_schedule=prompt_schedule,
            rollout_steps=rollout_steps,
            show_progress=show_progress,
        )

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"[YumePipeline.stream] Expected torch.Tensor from predict, got {type(video)}"
            )

        self.memory_module.record(video)
        print(
            f"[YumePipeline.stream] Recorded segment. "
            f"Total frames in memory: {len(getattr(self.memory_module, 'all_frames', []))}"
        )

        return video
