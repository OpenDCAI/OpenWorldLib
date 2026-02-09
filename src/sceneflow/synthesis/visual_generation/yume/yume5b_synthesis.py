from __future__ import annotations

from contextlib import nullcontext
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import os
import random
import sys
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "704*1280": (704, 1280),
    "1280*704": (1280, 704),
}

MAX_AREA_CONFIGS = {
    key: value[0] * value[1] for key, value in SIZE_CONFIGS.items()
}
SizeLike = Union[str, Tuple[int, int]]


class _YumeCompatAdapter:
    """
    Compatibility adapter that exposes YUME-style `generate(...)` setup outputs
    on top of SceneFlow's internal `wan_2p2.WanTI2V`.
    """

    def __init__(self, wan_model: Any) -> None:
        self._wan = wan_model
        self.device = wan_model.device
        self.t5_cpu = wan_model.t5_cpu
        self.text_encoder = wan_model.text_encoder
        self.vae = wan_model.vae
        self.model = wan_model.model
        self.vae_stride = wan_model.vae_stride
        self.patch_size = wan_model.patch_size
        self.sp_size = wan_model.sp_size
        self.sample_neg_prompt = wan_model.sample_neg_prompt

    @staticmethod
    def _masks_like_yume(
        tensor: List[torch.Tensor],
        *,
        zero: bool = False,
        generator: Optional[torch.Generator] = None,
        p: float = 0.2,
        latent_frame_zero: int = 8,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        out1 = [torch.ones_like(u) for u in tensor]
        out2 = [torch.ones_like(u) for u in tensor]
        if not zero:
            return out1, out2

        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[:, :-latent_frame_zero] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator,
                    ).expand_as(u[:, :-latent_frame_zero]).exp()
                    v[:, :-latent_frame_zero] = torch.zeros_like(v[:, :-latent_frame_zero])
        else:
            for u, v in zip(out1, out2):
                u[:, :-latent_frame_zero] = torch.zeros_like(u[:, :-latent_frame_zero])
                v[:, :-latent_frame_zero] = torch.zeros_like(v[:, :-latent_frame_zero])
        return out1, out2

    def _encode_prompts(
        self,
        input_prompt: str,
        n_prompt: str,
        *,
        offload_model: bool,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        return context, context_null

    def _t2v_setup(
        self,
        input_prompt: str,
        *,
        size: tuple[int, int],
        frame_num: int,
        n_prompt: str,
        seed: int,
        offload_model: bool,
    ) -> tuple[Dict[str, Any], Dict[str, Any], torch.Tensor]:
        target_shape = (
            self.vae.model.z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.patch_size[1] * self.patch_size[2])
            * target_shape[1]
            / self.sp_size
        ) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        context, context_null = self._encode_prompts(
            input_prompt, n_prompt, offload_model=offload_model
        )

        noise = torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g,
        )
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        return arg_c, arg_null, noise

    def _i2v_setup(
        self,
        input_prompt: str,
        *,
        img: torch.Tensor,
        max_area: int,
        frame_num: int,
        n_prompt: str,
        seed: int,
        offload_model: bool,
        latent_frame_zero: int,
    ) -> tuple[Dict[str, Any], Dict[str, Any], torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        try:
            from ....base_models.diffusion_model.video.wan_2p2.utils.utils import best_output_size
        except Exception as exc:
            raise ImportError(
                "Failed to import SceneFlow WAN utilities (best_output_size)."
            ) from exc

        z = img.to(self.device)
        if z.ndim == 3:
            z = z.unsqueeze(1)
        if z.ndim != 4:
            raise ValueError(f"img must be [C,F,H,W] tensor for i2v setup, got shape={z.shape}")

        ih, iw = z.shape[2:]
        dh = self.patch_size[1] * self.vae_stride[1]
        dw = self.patch_size[2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]
        ) * (ow // self.vae_stride[2]) // (self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        noise = torch.randn(
            self.vae.model.z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )

        context, context_null = self._encode_prompts(
            input_prompt, n_prompt, offload_model=offload_model
        )

        f_target = noise.shape[1]
        f_z = z.shape[1]
        if f_target > f_z:
            padding = f_target - f_z
            z = torch.cat(
                [z, torch.zeros_like(z[:, -1:, :, :]).repeat(1, padding, 1, 1)], dim=1
            )
        elif f_target < f_z:
            z = z[:, :f_target, :, :]
        z_list = [z]

        _, mask2 = self._masks_like_yume(
            [noise], zero=True, latent_frame_zero=latent_frame_zero
        )
        latent = (1.0 - mask2[0]) * z_list[0] + mask2[0] * noise
        noise = latent

        arg_c = {"context": [context[0]], "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        return arg_c, arg_null, noise, mask2, z_list

    def generate(
        self,
        input_prompt: str,
        *,
        img: Optional[torch.Tensor] = None,
        size: tuple[int, int] = (1280, 704),
        max_area: int = 704 * 1280,
        frame_num: int = 81,
        latent_frame_zero: int = 8,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
        **kwargs,
    ):
        if img is not None:
            return self._i2v_setup(
                input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
                latent_frame_zero=latent_frame_zero,
            )
        return self._t2v_setup(
            input_prompt,
            size=size,
            frame_num=frame_num,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model,
        )


class Yume5bSynthesis:
    """
    YUME-5B inference wrapper:
    - Prefers YUME `wan_2p3` implementation when available for original sampling behavior.
    - Falls back to SceneFlow internal WAN 2.2 implementation if `wan_2p3` is unavailable.
    - Runs custom Euler latent iterations following sample_5b.py.
    """

    def __init__(
        self,
        *,
        task: str,
        cfg: Any,
        model: Any,
        device_id: int,
        rank: int = 0,
    ) -> None:
        self.task = task
        self.cfg = cfg
        self.model = model
        self.device_id = device_id
        self.rank = rank
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        try:
            self._supports_flag = "flag" in inspect.signature(self.model.model.forward).parameters
        except (TypeError, ValueError):
            self._supports_flag = False

    @staticmethod
    def _resolve_checkpoint_dir(ckpt_dir: str) -> str:
        if os.path.isdir(ckpt_dir):
            return ckpt_dir

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise FileNotFoundError(
                f"Checkpoint directory not found: {ckpt_dir}. "
                "Install huggingface_hub or pass a local checkpoint directory."
            ) from exc

        repo_name = ckpt_dir.split("/")[-1]
        local_dir = Path.cwd() / repo_name
        local_dir.mkdir(parents=True, exist_ok=True)
        return str(
            snapshot_download(
                repo_id=ckpt_dir,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        task: str,
        ckpt_dir: str,
        device_id: int = 0,
        rank: int = 0,
        t5_fsdp: bool = False,
        dit_fsdp: bool = False,
        ulysses_size: int = 1,
        t5_cpu: bool = False,
        convert_model_dtype: bool = False,
    ) -> "Yume5bSynthesis":
        ckpt_dir = cls._resolve_checkpoint_dir(ckpt_dir)

        # Preferred path: SceneFlow vendored YUME wan_2p3 implementation.
        try:
            from ....base_models.diffusion_model.video import wan_2p3
            from ....base_models.diffusion_model.video.wan_2p3.configs import (
                WAN_CONFIGS as WAN2P3_CONFIGS,
            )
        except Exception:
            wan_2p3 = None
            WAN2P3_CONFIGS = None

        if wan_2p3 is not None and WAN2P3_CONFIGS is not None and task in WAN2P3_CONFIGS:
            try:
                cfg = WAN2P3_CONFIGS[task]
                yume_model = wan_2p3.Yume(
                    config=cfg,
                    checkpoint_dir=ckpt_dir,
                    device_id=device_id,
                    rank=rank,
                    t5_fsdp=t5_fsdp,
                    dit_fsdp=dit_fsdp,
                    use_sp=(ulysses_size > 1),
                    t5_cpu=t5_cpu,
                    convert_model_dtype=convert_model_dtype,
                )
                if isinstance(yume_model.model, torch.nn.Module):
                    yume_model.model.to(yume_model.device)
                    # Keep runtime behavior aligned with YUME sample_5b.py
                    # (transformer converted to bf16 before sampling).
                    if yume_model.device.type == "cuda":
                        yume_model.model.to(torch.bfloat16)
                yume_model.model.eval().requires_grad_(False)
                return cls(
                    task=task,
                    cfg=cfg,
                    model=yume_model,
                    device_id=device_id,
                    rank=rank,
                )
            except Exception as exc:
                warnings.warn(
                    f"SceneFlow wan_2p3 initialization failed ({exc!r}), "
                    "falling back to SceneFlow internal WAN2.2 compatibility mode.",
                    RuntimeWarning,
                )

        warnings.warn(
            "SceneFlow wan_2p3 is unavailable, falling back to SceneFlow internal WAN2.2 compatibility mode. "
            "This path is self-contained but may not fully match original YUME long-context behavior.",
            RuntimeWarning,
        )

        # Fallback path: internal WAN2.2 + YUME-compatible generate adapter.
        try:
            from ....base_models.diffusion_model.video import wan_2p2
            from ....base_models.diffusion_model.video.wan_2p2.configs import WAN_CONFIGS
        except Exception as exc:
            raise ImportError(
                "Failed to import SceneFlow internal WAN 2.2 modules. "
                "Please check SceneFlow dependencies."
            ) from exc

        if task not in WAN_CONFIGS:
            raise ValueError(f"Unsupported YUME task: {task}")

        cfg = WAN_CONFIGS[task]

        wan_model = wan_2p2.WanTI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=t5_fsdp,
            dit_fsdp=dit_fsdp,
            use_sp=(ulysses_size > 1),
            t5_cpu=t5_cpu,
            convert_model_dtype=convert_model_dtype,
        )
        wan_model.model.eval().requires_grad_(False)
        model = _YumeCompatAdapter(wan_model)

        return cls(
            task=task,
            cfg=cfg,
            model=model,
            device_id=device_id,
            rank=rank,
        )

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

    @staticmethod
    def _get_sampling_sigmas(sampling_steps: int, shift: float) -> np.ndarray:
        sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
        sigma = shift * sigma / (1 + (shift - 1) * sigma)
        return sigma

    @staticmethod
    def _normalize_prompt_schedule(
        *,
        prompt: str,
        prompt_schedule: Optional[List[str]],
        rollout_steps: Optional[int],
    ) -> List[str]:
        prompts = [p for p in (prompt_schedule or [prompt]) if p]
        if not prompts:
            raise ValueError("Prompt schedule is empty.")

        steps = rollout_steps if rollout_steps is not None else len(prompts)
        steps = max(1, int(steps))

        if len(prompts) < steps:
            prompts.extend([prompts[-1]] * (steps - len(prompts)))
        elif len(prompts) > steps:
            prompts = prompts[:steps]
        return prompts

    def _build_seed_video_from_image(
        self,
        image: Image.Image,
        *,
        size: Tuple[int, int],
        total_frames: int = 33,
    ) -> torch.Tensor:
        target_w, target_h = size

        image_np = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(self.device)
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        image_tensor = image_tensor.sub(0.5).div(0.5).clamp(-1.0, 1.0)

        seed_video = torch.zeros(
            (3, total_frames, target_h, target_w),
            dtype=image_tensor.dtype,
            device=self.device,
        )
        seed_video[:, 0] = image_tensor
        return seed_video

    def _prepare_seed_video_tensor(
        self,
        seed_video: torch.Tensor,
        *,
        size: Tuple[int, int],
        total_frames: int = 33,
    ) -> torch.Tensor:
        target_w, target_h = size
        video = seed_video

        if video.ndim != 4:
            raise ValueError(f"seed_video must be 4D tensor, got shape={video.shape}")

        # Accept both [C, T, H, W] and [T, C, H, W].
        if video.shape[0] in (1, 3) and video.shape[1] > 4:
            video_cthw = video
        elif video.shape[1] in (1, 3):
            video_cthw = video.permute(1, 0, 2, 3)
        else:
            raise ValueError(
                "Cannot infer seed_video format. Expected [C,T,H,W] or [T,C,H,W]."
            )

        video_cthw = video_cthw.to(self.device).float()
        if video_cthw.max() > 1.5:
            video_cthw = video_cthw / 255.0
        if video_cthw.min() >= 0.0:
            video_cthw = video_cthw.sub(0.5).div(0.5)
        video_cthw = video_cthw.clamp(-1.0, 1.0)

        video_tchw = video_cthw.permute(1, 0, 2, 3)
        video_tchw = F.interpolate(
            video_tchw,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        video_cthw = video_tchw.permute(1, 0, 2, 3).contiguous()

        if video_cthw.shape[1] < total_frames:
            pad_count = total_frames - video_cthw.shape[1]
            pad = video_cthw[:, -1:, :, :].repeat(1, pad_count, 1, 1)
            video_cthw = torch.cat([video_cthw, pad], dim=1)
        elif video_cthw.shape[1] > total_frames:
            video_cthw = video_cthw[:, :total_frames, :, :]

        return video_cthw

    def _build_masked_timestep(
        self,
        *,
        sigma_value: float,
        mask2: List[torch.Tensor],
        arg_c: Dict[str, Any],
        latent_frame_zero: int,
    ) -> torch.Tensor:
        timestep = torch.tensor([sigma_value * 1000.0], device=self.device, dtype=torch.float32)
        temp_ts = mask2[0][0][:-latent_frame_zero, ::2, ::2].flatten()
        temp_ts = torch.cat(
            [
                temp_ts,
                temp_ts.new_ones(arg_c["seq_len"] - temp_ts.size(0)) * timestep,
            ]
        )
        return temp_ts.unsqueeze(0)

    def _predict_noise(
        self,
        *,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        arg_c: Dict[str, Any],
        flag: Optional[bool] = None,
    ) -> torch.Tensor:
        forward_kwargs = dict(arg_c)
        if flag is not None and self._supports_flag:
            forward_kwargs["flag"] = flag
        return self.model.model([latent], t=timestep, **forward_kwargs)[0]

    def _run_conditioned_euler_segment(
        self,
        *,
        context_latents: torch.Tensor,
        arg_c: Dict[str, Any],
        noise: torch.Tensor,
        mask2: List[torch.Tensor],
        img_latents: List[torch.Tensor],
        latent_frame_zero: int,
        num_euler_timesteps: int,
        sigma_shift: float,
        show_progress: bool = True,
        progress_desc: str = "YUME denoising",
    ) -> torch.Tensor:
        if img_latents[0].shape[1] <= latent_frame_zero:
            raise ValueError(
                "Invalid temporal latent length for conditioned Euler step: "
                f"img_latents[0].shape[1]={img_latents[0].shape[1]} <= "
                f"latent_frame_zero={latent_frame_zero}. "
                "This usually indicates frame_num was passed in latent-frame units "
                "instead of pixel-frame units."
            )

        sampling_sigmas = self._get_sampling_sigmas(num_euler_timesteps, sigma_shift)
        latent = torch.cat(
            [img_latents[0][:, :-latent_frame_zero, :, :], noise[:, -latent_frame_zero:, :, :]],
            dim=1,
        )

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            euler_iter = tqdm(
                range(num_euler_timesteps),
                desc=progress_desc,
                leave=False,
                disable=not show_progress,
            )
            for i in euler_iter:
                timestep = self._build_masked_timestep(
                    sigma_value=float(sampling_sigmas[i]),
                    mask2=mask2,
                    arg_c=arg_c,
                    latent_frame_zero=latent_frame_zero,
                )
                noise_pred_cond = self._predict_noise(
                    latent=latent,
                    timestep=timestep,
                    arg_c=arg_c,
                )

                if i + 1 == num_euler_timesteps:
                    delta = 0.0 - float(sampling_sigmas[i])
                else:
                    delta = float(sampling_sigmas[i + 1]) - float(sampling_sigmas[i])

                temp_x0 = latent[:, -latent_frame_zero:, :, :] + delta * noise_pred_cond[
                    :, -latent_frame_zero:, :, :
                ]
                latent = torch.cat(
                    [context_latents[:, :-latent_frame_zero, :, :], temp_x0], dim=1
                )

        return latent

    def _run_t2v_first_segment(
        self,
        *,
        prompt: str,
        max_area: int,
        frame_zero: int,
        latent_frame_zero: int,
        num_euler_timesteps: int,
        sigma_shift: float,
        show_progress: bool = True,
        progress_desc: str = "YUME denoising",
    ) -> torch.Tensor:
        arg_c, _, noise = self.model.generate(
            prompt,
            frame_num=frame_zero,
            max_area=max_area,
            latent_frame_zero=latent_frame_zero,
        )
        latent = noise.clone()
        sampling_sigmas = self._get_sampling_sigmas(num_euler_timesteps, sigma_shift)

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            euler_iter = tqdm(
                range(num_euler_timesteps),
                desc=progress_desc,
                leave=False,
                disable=not show_progress,
            )
            for i in euler_iter:
                timestep = torch.tensor(
                    [float(sampling_sigmas[i]) * 1000.0],
                    device=self.device,
                    dtype=torch.float32,
                )
                noise_pred_cond = self._predict_noise(
                    latent=latent,
                    timestep=timestep,
                    arg_c=arg_c,
                    flag=False,
                )

                if i + 1 == num_euler_timesteps:
                    delta = 0.0 - float(sampling_sigmas[i])
                else:
                    delta = float(sampling_sigmas[i + 1]) - float(sampling_sigmas[i])
                latent = latent + delta * noise_pred_cond

        return latent

    def _decode_tail_segment(
        self,
        *,
        model_input: torch.Tensor,
        latent_frame_zero: int,
        frame_zero: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            video = self.model.vae.decode([model_input[:, -latent_frame_zero:, :, :].to(torch.float32)])[0]
        return video[:, -frame_zero:, :, :]

    @torch.no_grad()
    def predict(
        self,
        *,
        processed_inputs: Dict[str, Any],
        size: SizeLike = "1280*704",
        max_area: Optional[int] = None,
        num_euler_timesteps: int = 4,
        sigma_shift: float = 7.0,
        latent_frame_zero: int = 8,
        frame_zero: int = 32,
        rollout_steps: Optional[int] = None,
        prompt_schedule: Optional[List[str]] = None,
        base_seed: int = -1,
        show_progress: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        size_key = self._normalize_size_key(size)
        if size_key not in SIZE_CONFIGS:
            raise ValueError(f"Unsupported size: {size}. Supported: {list(SIZE_CONFIGS.keys())}")

        if base_seed >= 0:
            random.seed(base_seed)
            np.random.seed(base_seed % (2**32 - 1))
            torch.manual_seed(base_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(base_seed)

        prompt = processed_inputs["prompt"]
        image: Optional[Image.Image] = processed_inputs.get("image")
        seed_video: Optional[torch.Tensor] = processed_inputs.get("seed_video")
        prompts = self._normalize_prompt_schedule(
            prompt=prompt,
            prompt_schedule=prompt_schedule,
            rollout_steps=rollout_steps,
        )

        max_area = max_area if max_area is not None else MAX_AREA_CONFIGS[size_key]
        model_input: Optional[torch.Tensor]
        decoded_segments: List[torch.Tensor] = []

        if seed_video is not None:
            seed_video = self._prepare_seed_video_tensor(
                seed_video, size=SIZE_CONFIGS[size_key], total_frames=33
            )
            model_input = torch.cat(
                [seed_video[:, 0].unsqueeze(1).repeat(1, 16, 1, 1), seed_video[:, :33]],
                dim=1,
            )
            model_input = torch.cat(
                [
                    self.model.vae.encode([model_input[:, :-32, :, :]])[0],
                    self.model.vae.encode([model_input[:, -32:, :, :]])[0],
                ],
                dim=1,
            )
        elif image is not None:
            seed_video = self._build_seed_video_from_image(
                image, size=SIZE_CONFIGS[size_key], total_frames=33
            )
            # sample_5b seed layout: [first frame x16] + [33-frame seed video]
            model_input = torch.cat(
                [seed_video[:, 0].unsqueeze(1).repeat(1, 16, 1, 1), seed_video[:, :33]],
                dim=1,
            )
            model_input = torch.cat(
                [
                    self.model.vae.encode([model_input[:, :-32, :, :]])[0],
                    self.model.vae.encode([model_input[:, -32:, :, :]])[0],
                ],
                dim=1,
            )
        else:
            model_input = None

        rollout_iter = enumerate(
            tqdm(
                prompts,
                desc="YUME rollout",
                leave=True,
                disable=not show_progress,
            )
        )
        for step_idx, step_prompt in rollout_iter:
            if step_idx == 0 and model_input is None:
                # t2v first segment: sample_5b flag=False branch
                model_input = self._run_t2v_first_segment(
                    prompt=step_prompt,
                    max_area=max_area,
                    frame_zero=frame_zero,
                    latent_frame_zero=latent_frame_zero,
                    num_euler_timesteps=num_euler_timesteps,
                    sigma_shift=sigma_shift,
                    show_progress=show_progress,
                    progress_desc=f"YUME denoising {step_idx + 1}/{len(prompts)}",
                )
            else:
                assert model_input is not None
                temporal_stride = (
                    int(self.model.vae_stride[0])
                    if hasattr(self.model, "vae_stride")
                    else 4
                )
                if step_idx == 0:
                    # sample_5b uses pixel-frame count for generate(frame_num=...),
                    # while model_input is latent frames here.
                    frame_num = (model_input.shape[1] - 1) * temporal_stride + 1
                    context_latents = model_input
                    generate_img = model_input[:, :-latent_frame_zero, :, :]
                    arg_c, _, noise, mask2, img_latents = self.model.generate(
                        step_prompt,
                        frame_num=frame_num,
                        max_area=max_area,
                        latent_frame_zero=latent_frame_zero,
                        img=generate_img,
                    )
                else:
                    frame_num = (model_input.shape[1] - 1) * temporal_stride + 1 + frame_zero
                    context_latents = torch.cat(
                        [
                            model_input,
                            torch.zeros(
                                (
                                    model_input.shape[0],
                                    latent_frame_zero,
                                    model_input.shape[2],
                                    model_input.shape[3],
                                ),
                                device=self.device,
                                dtype=model_input.dtype,
                            ),
                        ],
                        dim=1,
                    )
                    arg_c, _, _, mask2, img_latents = self.model.generate(
                        step_prompt,
                        frame_num=frame_num,
                        max_area=max_area,
                        latent_frame_zero=latent_frame_zero,
                        img=model_input,
                    )
                    noise = torch.randn_like(context_latents)

                latent = self._run_conditioned_euler_segment(
                    context_latents=context_latents,
                    arg_c=arg_c,
                    noise=noise,
                    mask2=mask2,
                    img_latents=img_latents,
                    latent_frame_zero=latent_frame_zero,
                    num_euler_timesteps=num_euler_timesteps,
                    sigma_shift=sigma_shift,
                    show_progress=show_progress,
                    progress_desc=f"YUME denoising {step_idx + 1}/{len(prompts)}",
                )

                if step_idx == 0:
                    model_input = torch.cat(
                        [model_input[:, :-latent_frame_zero, :, :], latent[:, -latent_frame_zero:, :, :]],
                        dim=1,
                    )
                else:
                    model_input = torch.cat([model_input, latent[:, -latent_frame_zero:, :, :]], dim=1)

            decoded_segments.append(
                self._decode_tail_segment(
                    model_input=model_input,
                    latent_frame_zero=latent_frame_zero,
                    frame_zero=frame_zero,
                )
            )

        return torch.cat(decoded_segments, dim=1)
