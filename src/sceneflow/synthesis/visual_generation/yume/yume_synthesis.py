from __future__ import annotations

from contextlib import nullcontext
import inspect
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

from ...base_synthesis import BaseSynthesis


SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "544*960": (544, 960),
    "960*544": (960, 544),
    "704*1280": (704, 1280),
    "1280*704": (1280, 704),
    "1024*704": (1024, 704),
    "704*1024": (704, 1024),
}
MAX_AREA_CONFIGS = {key: value[0] * value[1] for key, value in SIZE_CONFIGS.items()}
SizeLike = Union[str, Tuple[int, int]]

MODEL_VARIANT_ALIASES = {
    "5b": "ti2v-5B",
    "ti2v": "ti2v-5B",
    "ti2v-5b": "ti2v-5B",
    "yume-5b": "ti2v-5B",
    "yume-5b-720p": "ti2v-5B",
    "i2v-a14b": "i2v-14B",
    "i2v-14b": "i2v-14B",
    "yume-i2v-540p": "i2v-14B",
    "14b": "i2v-14B",
    "t2v-a14b": "t2v-14B",
    "t2v-14b": "t2v-14B",
}

MODEL_DEFAULTS = {
    "ti2v-5B": {
        "size": "1280*704",
        "num_euler_timesteps": 4,
        "sigma_shift": 7.0,
        "latent_frame_zero": 8,
        "frame_zero": 32,
        "context_blend": False,
        "use_masked_timestep": True,
        "guidance_scale": 1.0,
    },
    "i2v-14B": {
        "size": "960*544",
        "num_euler_timesteps": 50,
        "sigma_shift": 3.0,
        "latent_frame_zero": 8,
        "frame_zero": 32,
        "context_blend": True,
        "use_masked_timestep": False,
        "guidance_scale": 5.0,
    },
    "t2v-14B": {
        "size": "832*480",
        "num_euler_timesteps": 50,
        "sigma_shift": 5.0,
        "latent_frame_zero": 8,
        "frame_zero": 32,
        "context_blend": False,
        "use_masked_timestep": False,
        "guidance_scale": 5.0,
    },
}


class _YumeCompatAdapter:
    """Expose YUME-like setup outputs on top of SceneFlow wan_2p2.WanTI2V."""

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
            input_prompt,
            n_prompt,
            offload_model=offload_model,
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
        from ....base_models.diffusion_model.video.wan_2p2.utils.utils import best_output_size

        z = img.to(self.device)
        if z.ndim == 3:
            z = z.unsqueeze(1)
        if z.ndim != 4:
            raise ValueError(f"img must be [C,F,H,W], got shape={z.shape}")

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
            input_prompt,
            n_prompt,
            offload_model=offload_model,
        )

        f_target = noise.shape[1]
        f_z = z.shape[1]
        if f_target > f_z:
            padding = f_target - f_z
            z = torch.cat([z, torch.zeros_like(z[:, -1:, :, :]).repeat(1, padding, 1, 1)], dim=1)
        elif f_target < f_z:
            z = z[:, :f_target, :, :]
        z_list = [z]

        _, mask2 = self._masks_like_yume(
            [noise],
            zero=True,
            latent_frame_zero=latent_frame_zero,
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


class _YumeI2VAdapter:
    """SceneFlow-local loader for Yume-I2V-540P checkpoints."""

    def __init__(
        self,
        *,
        device: torch.device,
        t5_cpu: bool,
        text_encoder: Any,
        vae: Any,
        clip: Any,
        model: Any,
        vae_stride: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        sample_neg_prompt: str,
        param_dtype: torch.dtype,
        rank: int = 0,
    ) -> None:
        self.device = device
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.text_encoder = text_encoder
        self.vae = vae
        self.clip = clip
        self.model = model
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.sp_size = 1
        self.sample_neg_prompt = sample_neg_prompt
        self.param_dtype = param_dtype

    @staticmethod
    def _upsample_conv3d_weights(conv_small: torch.nn.Conv3d, size: Tuple[int, int, int]) -> torch.nn.Conv3d:
        old_weight = conv_small.weight.data
        new_weight = F.interpolate(
            old_weight,
            size=size,
            mode="trilinear",
            align_corners=False,
        )
        conv_large = torch.nn.Conv3d(
            in_channels=16,
            out_channels=5120,
            kernel_size=size,
            stride=size,
            padding=0,
        )
        conv_large.weight.data = new_weight
        if conv_small.bias is not None:
            conv_large.bias.data = conv_small.bias.data.clone()
        return conv_large

    @staticmethod
    def _load_safetensor_state_dict(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
        from safetensors import safe_open
        from safetensors.torch import load_file

        index_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)

            model_weights: Dict[str, torch.Tensor] = {}
            for shard_name in sorted(set(index_data["weight_map"].values())):
                shard_path = os.path.join(checkpoint_dir, shard_name)
                if not os.path.exists(shard_path):
                    raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")

                with safe_open(shard_path, framework="pt", device="cpu") as shard_file:
                    for key in shard_file.keys():
                        model_weights[key] = shard_file.get_tensor(key)
            return model_weights

        full_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing model weight file: {full_path}")
        return load_file(full_path, device="cpu")

    @classmethod
    def from_pretrained(
        cls,
        *,
        checkpoint_root: str,
        yume_dit_dir: str,
        config: Any,
        device_id: int,
        rank: int,
        t5_cpu: bool,
        convert_model_dtype: bool,
    ) -> "_YumeI2VAdapter":
        from ....base_models.diffusion_model.video.wan_2p1.modules.clip import CLIPModel
        from ....base_models.diffusion_model.video.wan_2p1.modules.model import WanModel
        from ....base_models.diffusion_model.video.wan_2p1.modules.t5 import T5EncoderModel
        from ....base_models.diffusion_model.video.wan_2p1.modules.vae import WanVAE

        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_root, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_root, config.t5_tokenizer),
        )
        text_encoder.model.eval().requires_grad_(False)
        if not t5_cpu and device.type == "cuda":
            text_encoder.model.to(device)

        vae = WanVAE(
            vae_pth=os.path.join(checkpoint_root, config.vae_checkpoint),
            device=device,
        )
        vae.model.eval().requires_grad_(False)

        clip = CLIPModel(
            dtype=config.clip_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_root, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_root, config.clip_tokenizer),
        )
        clip.model.eval().requires_grad_(False)
        if device.type == "cuda":
            clip.model.to(torch.bfloat16).to(device)

        model_config = {
            "model_type": "i2v",
            "text_len": 512,
            "in_dim": 36,
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "out_dim": 16,
            "num_heads": 40,
            "num_layers": 40,
            "eps": 1e-6,
            "_class_name": "WanModel",
            "_diffusers_version": "0.30.0",
        }
        model = WanModel.from_config(model_config)
        model.patch_embedding_2x = cls._upsample_conv3d_weights(model.patch_embedding, (1, 4, 4))
        model.patch_embedding_2x_f = torch.nn.Conv3d(
            36,
            36,
            kernel_size=(1, 4, 4),
            stride=(1, 4, 4),
        )
        model.patch_embedding_4x = cls._upsample_conv3d_weights(model.patch_embedding, (1, 8, 8))
        model.patch_embedding_8x = cls._upsample_conv3d_weights(model.patch_embedding, (1, 16, 16))
        model.patch_embedding_16x = cls._upsample_conv3d_weights(model.patch_embedding, (1, 32, 32))

        model_weights = cls._load_safetensor_state_dict(yume_dit_dir)
        model.load_state_dict(model_weights, strict=False)
        del model_weights

        model.eval().requires_grad_(False)
        if convert_model_dtype:
            model.to(config.param_dtype)
        if device.type == "cuda":
            model.to(device)

        return cls(
            device=device,
            t5_cpu=t5_cpu,
            text_encoder=text_encoder,
            vae=vae,
            clip=clip,
            model=model,
            vae_stride=tuple(config.vae_stride),
            patch_size=tuple(config.patch_size),
            sample_neg_prompt=config.sample_neg_prompt,
            param_dtype=config.param_dtype,
            rank=rank,
        )


class YumeSynthesis(BaseSynthesis):
    """
    Unified YUME synthesis implementation.

    - Single class supports 5B and 14B model variants.
    - Unified predict API supports t2v / i2v / v2v auto-inference.
    - Supports deterministic ODE sampling and stochastic SDE sampling.
    """

    def __init__(
        self,
        *,
        model: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self._task: Optional[str] = None
        self._backend: Optional[str] = None
        self.device = self._resolve_model_device(model)
        self._supports_flag = self._detect_flag_support()
        self._supports_rand_num_img = self._detect_forward_arg_support("rand_num_img")
        self._supports_latent_frame_zero = self._detect_forward_arg_support("latent_frame_zero")
        self._transformer_dtype = self._infer_transformer_dtype()

    def _resolve_runtime_route(self) -> Tuple[str, str]:
        if self._task is not None and self._backend is not None:
            return self._task, self._backend

        task = self._task
        backend = self._backend
        model_cls_name = self.model.__class__.__name__.lower()

        if task is None:
            if isinstance(self.model, _YumeCompatAdapter):
                task = "ti2v-5B"
            elif isinstance(self.model, _YumeI2VAdapter):
                task = "i2v-14B"
            elif "ti2v" in model_cls_name:
                task = "ti2v-5B"
            elif "t2v" in model_cls_name and "ti2v" not in model_cls_name:
                task = "t2v-14B"
            elif "i2v" in model_cls_name:
                task = "i2v-14B"
            elif (
                getattr(self.model, "low_noise_model", None) is not None
                and getattr(self.model, "high_noise_model", None) is not None
            ):
                task = "i2v-14B"
            else:
                raise RuntimeError(
                    f"Unable to infer yume task from model type: {self.model.__class__.__name__}"
                )

        task = self._normalize_model_variant(task)

        if backend is None:
            if isinstance(self.model, _YumeCompatAdapter):
                backend = "sceneflow_wan22_compat"
            elif isinstance(self.model, _YumeI2VAdapter):
                backend = "sceneflow_yume_i2v_native"
            elif task == "ti2v-5B":
                backend = "sceneflow_wan22_compat"
            elif task == "t2v-14B":
                backend = "sceneflow_wan22_native_t2v"
            elif task == "i2v-14B":
                if (
                    getattr(self.model, "low_noise_model", None) is not None
                    and getattr(self.model, "high_noise_model", None) is not None
                ):
                    backend = "sceneflow_wan22_native_i2v"
                else:
                    backend = "sceneflow_yume_i2v_native"
            else:
                raise RuntimeError(f"Unable to infer yume backend for task={task}")

        self._task = task
        self._backend = backend
        return task, backend

    @property
    def task(self) -> str:
        return self._resolve_runtime_route()[0]

    @property
    def backend(self) -> str:
        return self._resolve_runtime_route()[1]

    @staticmethod
    def _normalize_model_variant(model_variant: str) -> str:
        key = model_variant.strip()
        key_l = key.lower()
        if key_l in MODEL_VARIANT_ALIASES:
            return MODEL_VARIANT_ALIASES[key_l]
        if key in MODEL_DEFAULTS:
            return key
        raise ValueError(
            f"Unsupported YUME model variant: {model_variant}. "
            f"Supported: {sorted(MODEL_DEFAULTS.keys())}"
        )

    @classmethod
    def infer_model_variant_from_path(
        cls,
        pretrained_model_path: str,
        *,
        prefer_i2v_14b: bool = True,
    ) -> str:
        lower_path = pretrained_model_path.lower()
        if "yume-i2v-540p" in lower_path or ("i2v" in lower_path and "540p" in lower_path):
            return "i2v-14B"
        if "yume-5b-720p" in lower_path:
            return "ti2v-5B"
        if "5b" in lower_path or "ti2v" in lower_path:
            return "ti2v-5B"
        if "t2v" in lower_path:
            return "t2v-14B"
        if "i2v" in lower_path or "14b" in lower_path:
            return "i2v-14B"
        return "i2v-14B" if prefer_i2v_14b else "ti2v-5B"

    @staticmethod
    def _resolve_device_id(device: Optional[Union[int, str, torch.device]]) -> int:
        if isinstance(device, int):
            return device
        if isinstance(device, torch.device):
            if device.type == "cuda":
                return int(device.index or 0)
            return 0
        if isinstance(device, str):
            if device.startswith("cuda"):
                if ":" in device:
                    return int(device.split(":", 1)[1])
                return 0
            return 0
        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
        return 0

    @classmethod
    def _resolve_model_device(cls, model: Any) -> torch.device:
        model_device = getattr(model, "device", None)
        if isinstance(model_device, torch.device):
            return model_device
        if isinstance(model_device, str):
            return torch.device(model_device)
        device_id = cls._resolve_device_id(model_device)
        return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _resolve_yume_i2v_layout(model_root: str) -> Optional[Tuple[str, str]]:
        yume_dit_subdir = os.path.join(model_root, "Yume-Dit")
        if os.path.isdir(yume_dit_subdir):
            return model_root, yume_dit_subdir

        if os.path.basename(os.path.normpath(model_root)) == "Yume-Dit":
            parent = os.path.dirname(os.path.normpath(model_root))
            return parent, model_root

        return None

    def _detect_flag_support(self) -> bool:
        model_core = getattr(self.model, "model", None)
        if model_core is None:
            return False
        try:
            return "flag" in inspect.signature(model_core.forward).parameters
        except (TypeError, ValueError, AttributeError):
            return False

    def _detect_forward_arg_support(self, arg_name: str) -> bool:
        model_core = getattr(self.model, "model", None)
        if model_core is None:
            return False
        try:
            return arg_name in inspect.signature(model_core.forward).parameters
        except (TypeError, ValueError, AttributeError):
            return False

    def _infer_transformer_dtype(self) -> torch.dtype:
        model_candidates: List[Any] = []
        model_core = getattr(self.model, "model", None)
        if model_core is not None:
            model_candidates.append(model_core)
        for attr in ("low_noise_model", "high_noise_model"):
            model_part = getattr(self.model, attr, None)
            if model_part is not None:
                model_candidates.append(model_part)

        for candidate in model_candidates:
            try:
                first_param = next(candidate.parameters())
                return first_param.dtype
            except (StopIteration, AttributeError, TypeError):
                continue

        param_dtype = getattr(self.model, "param_dtype", None)
        if isinstance(param_dtype, torch.dtype):
            return param_dtype
        return torch.float32

    @staticmethod
    def _cast_floating_tree_dtype(data: Any, dtype: torch.dtype) -> Any:
        if isinstance(data, torch.Tensor):
            if data.is_floating_point() and data.dtype != dtype:
                return data.to(dtype=dtype)
            return data
        if isinstance(data, list):
            return [YumeSynthesis._cast_floating_tree_dtype(item, dtype) for item in data]
        if isinstance(data, tuple):
            return tuple(YumeSynthesis._cast_floating_tree_dtype(item, dtype) for item in data)
        if isinstance(data, dict):
            return {k: YumeSynthesis._cast_floating_tree_dtype(v, dtype) for k, v in data.items()}
        return data

    def _build_transformer_autocast_ctx(self):
        if self.device.type != "cuda":
            return nullcontext()
        if self._transformer_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", dtype=self._transformer_dtype)
        return nullcontext()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        *,
        model_variant: Optional[str] = None,
        device: Optional[Union[int, str, torch.device]] = None,
        **kwargs,
    ) -> "YumeSynthesis":
        distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        distributed_mode = distributed_world_size > 1
        env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        env_rank = int(os.environ.get("RANK", "0"))

        if distributed_mode:
            import torch.distributed as dist

            if torch.cuda.is_available():
                torch.cuda.set_device(env_local_rank)
            if not dist.is_initialized():
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend)
            if device is None:
                device = env_local_rank

        ckpt_dir = kwargs.pop("ckpt_dir", pretrained_model_path)
        if os.path.isdir(ckpt_dir):
            model_root = ckpt_dir
        else:
            from huggingface_hub import snapshot_download

            print(f"Downloading weights from HuggingFace repo: {ckpt_dir}")
            model_root = snapshot_download(ckpt_dir)
            print(f"Model downloaded to: {model_root}")

        requested_variant = model_variant or kwargs.pop("task", None)
        if requested_variant is None:
            requested_variant = cls.infer_model_variant_from_path(ckpt_dir)
        task = cls._normalize_model_variant(requested_variant)

        device_id = cls._resolve_device_id(device)

        rank = int(kwargs.pop("rank", env_rank if distributed_mode else 0))
        t5_fsdp = bool(kwargs.pop("t5_fsdp", False))
        dit_fsdp = bool(kwargs.pop("dit_fsdp", False))
        ulysses_size = int(
            kwargs.pop("ulysses_size", distributed_world_size if distributed_mode else 1)
        )
        t5_cpu = bool(kwargs.pop("t5_cpu", False))
        init_on_cpu = bool(kwargs.pop("init_on_cpu", False))
        convert_model_dtype = bool(kwargs.pop("convert_model_dtype", True))

        yume_i2v_layout = cls._resolve_yume_i2v_layout(model_root) if task == "i2v-14B" else None
        if yume_i2v_layout is not None:
            from ....base_models.diffusion_model.video.wan_2p1.configs import WAN_CONFIGS as WAN21_CONFIGS

            checkpoint_root, yume_dit_dir = yume_i2v_layout
            model = _YumeI2VAdapter.from_pretrained(
                checkpoint_root=checkpoint_root,
                yume_dit_dir=yume_dit_dir,
                config=WAN21_CONFIGS["i2v-14B"],
                device_id=device_id,
                rank=rank,
                t5_cpu=t5_cpu,
                convert_model_dtype=convert_model_dtype,
            )
        else:
            from ....base_models.diffusion_model.video import wan_2p2
            from ....base_models.diffusion_model.video.wan_2p2.configs import WAN_CONFIGS

            model_spec = {
                "ti2v-5B": ("ti2v-5B", wan_2p2.WanTI2V),
                "i2v-14B": ("i2v-A14B", wan_2p2.WanI2V),
                "t2v-14B": ("t2v-A14B", wan_2p2.WanT2V),
            }
            cfg_key, model_cls = model_spec[task]
            cfg = WAN_CONFIGS[cfg_key]
            model = model_cls(
                config=cfg,
                checkpoint_dir=model_root,
                device_id=device_id,
                rank=rank,
                t5_fsdp=t5_fsdp,
                dit_fsdp=dit_fsdp,
                use_sp=(ulysses_size > 1),
                t5_cpu=t5_cpu,
                init_on_cpu=init_on_cpu,
                convert_model_dtype=convert_model_dtype,
            )

        if task == "ti2v-5B":
            model.model.eval().requires_grad_(False)
            model = _YumeCompatAdapter(model)

        return cls(model=model)

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
        prompt_schedule: Optional[Sequence[str]],
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

    def _call_model(
        self,
        *,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        arg_c: Dict[str, Any],
        flag: Optional[bool] = None,
        forward_extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        model_core = self._resolve_forward_model(
            timestep=timestep,
            target_device=latent.device,
        )

        forward_kwargs = dict(arg_c)
        if forward_extra_kwargs:
            forward_kwargs.update(forward_extra_kwargs)
        if flag is not None and self._supports_flag:
            forward_kwargs["flag"] = flag

        if latent.is_floating_point() and latent.dtype != self._transformer_dtype:
            latent = latent.to(dtype=self._transformer_dtype)
        forward_kwargs = self._cast_floating_tree_dtype(forward_kwargs, self._transformer_dtype)

        try:
            outputs = model_core([latent], t=timestep, **forward_kwargs)
        except TypeError:
            # Some model variants (14B) do not accept certain extra kwargs.
            filtered_kwargs = dict(arg_c)
            if flag is not None and self._supports_flag:
                filtered_kwargs["flag"] = flag
            filtered_kwargs = self._cast_floating_tree_dtype(filtered_kwargs, self._transformer_dtype)
            outputs = model_core([latent], t=timestep, **filtered_kwargs)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if not isinstance(outputs, torch.Tensor):
            raise TypeError(f"Unexpected model output type: {type(outputs)}")
        return outputs

    def _resolve_forward_model(
        self,
        *,
        timestep: torch.Tensor,
        target_device: torch.device,
    ) -> Any:
        model_core = getattr(self.model, "model", None)
        if model_core is not None:
            try:
                first_param = next(model_core.parameters())
                if first_param.device != target_device:
                    model_core.to(target_device)
            except (StopIteration, AttributeError, TypeError):
                pass
            return model_core

        low_noise_model = getattr(self.model, "low_noise_model", None)
        high_noise_model = getattr(self.model, "high_noise_model", None)
        if low_noise_model is None or high_noise_model is None:
            raise RuntimeError(
                f"Backend {self.backend} does not expose a compatible transformer for custom sampling"
            )

        t_value = float(timestep.flatten()[0].item())
        boundary = float(getattr(self.model, "boundary", 0.5)) * float(
            getattr(self.model, "num_train_timesteps", 1000)
        )
        if t_value >= boundary:
            model_core = high_noise_model
            offload_model = low_noise_model
        else:
            model_core = low_noise_model
            offload_model = high_noise_model

        try:
            first_param = next(model_core.parameters())
            if first_param.device != target_device:
                model_core.to(target_device)
        except (StopIteration, AttributeError, TypeError):
            pass

        if bool(getattr(self.model, "init_on_cpu", False)):
            try:
                offload_param = next(offload_model.parameters())
                if offload_param.device.type == "cuda":
                    offload_model.to("cpu")
            except (StopIteration, AttributeError, TypeError):
                pass

        return model_core

    @staticmethod
    def _integrate_tail(
        *,
        latent_tail: torch.Tensor,
        noise_pred_tail: torch.Tensor,
        sigma_now: float,
        sigma_next: float,
        sampling_method: str,
        sde_eta: float,
    ) -> torch.Tensor:
        temp_x0 = latent_tail + (sigma_next - sigma_now) * noise_pred_tail

        if sampling_method == "sde":
            sigma_safe = max(abs(sigma_now), 1e-6)
            pred_original_sample = latent_tail + (0.0 - sigma_now) * noise_pred_tail
            delta_t = max(sigma_now - sigma_next, 0.0)
            dsigma = sigma_next - sigma_now

            std_dev_t = sde_eta * math.sqrt(delta_t)
            score_estimate = -(
                latent_tail - pred_original_sample * (1.0 - sigma_now)
            ) / (sigma_safe * sigma_safe)
            log_term = -0.5 * (sde_eta ** 2) * score_estimate
            prev_sample_mean = temp_x0 + log_term * dsigma

            if std_dev_t > 0:
                temp_x0 = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t
            else:
                temp_x0 = prev_sample_mean

        return temp_x0

    def _run_conditioned_segment(
        self,
        *,
        context_latents: torch.Tensor,
        arg_c: Dict[str, Any],
        noise: torch.Tensor,
        latent_frame_zero: int,
        num_euler_timesteps: int,
        sigma_shift: float,
        sampling_method: str,
        sde_eta: float,
        show_progress: bool,
        progress_desc: str,
        mask2: Optional[List[torch.Tensor]] = None,
        use_masked_timestep: bool = False,
        context_blend: bool = False,
        guidance_scale: float = 1.0,
        arg_null: Optional[Dict[str, Any]] = None,
        forward_extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if context_latents.shape[1] <= latent_frame_zero:
            raise ValueError(
                "Invalid latent temporal length: "
                f"context_latents.shape[1]={context_latents.shape[1]} <= "
                f"latent_frame_zero={latent_frame_zero}"
            )

        sampling_sigmas = self._get_sampling_sigmas(num_euler_timesteps, sigma_shift)

        if context_blend:
            sigma0 = float(sampling_sigmas[0])
            head_init = noise[:, :-latent_frame_zero, :, :] * sigma0 + (
                1.0 - sigma0
            ) * context_latents[:, :-latent_frame_zero, :, :]
        else:
            head_init = context_latents[:, :-latent_frame_zero, :, :]
        latent = torch.cat([head_init, noise[:, -latent_frame_zero:, :, :]], dim=1)

        autocast_ctx = self._build_transformer_autocast_ctx()

        with torch.no_grad(), autocast_ctx:
            euler_iter = tqdm(
                range(num_euler_timesteps),
                desc=progress_desc,
                leave=False,
                disable=not show_progress,
            )
            for i in euler_iter:
                sigma_now = float(sampling_sigmas[i])
                sigma_next = (
                    0.0
                    if i + 1 == num_euler_timesteps
                    else float(sampling_sigmas[i + 1])
                )

                if use_masked_timestep:
                    if mask2 is None:
                        raise ValueError("mask2 is required when use_masked_timestep=True")
                    timestep = self._build_masked_timestep(
                        sigma_value=sigma_now,
                        mask2=mask2,
                        arg_c=arg_c,
                        latent_frame_zero=latent_frame_zero,
                    )
                else:
                    timestep = torch.tensor([sigma_now * 1000.0], device=self.device, dtype=torch.float32)

                noise_pred_cond = self._call_model(
                    latent=latent,
                    timestep=timestep,
                    arg_c=arg_c,
                    forward_extra_kwargs=forward_extra_kwargs,
                )

                if arg_null is not None and guidance_scale != 1.0:
                    noise_pred_uncond = self._call_model(
                        latent=latent,
                        timestep=timestep,
                        arg_c=arg_null,
                        forward_extra_kwargs=forward_extra_kwargs,
                    )
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = noise_pred_cond

                tail = self._integrate_tail(
                    latent_tail=latent[:, -latent_frame_zero:, :, :],
                    noise_pred_tail=noise_pred[:, -latent_frame_zero:, :, :],
                    sigma_now=sigma_now,
                    sigma_next=sigma_next,
                    sampling_method=sampling_method,
                    sde_eta=sde_eta,
                )

                if context_blend:
                    head = noise[:, :-latent_frame_zero, :, :] * sigma_next + (
                        1.0 - sigma_next
                    ) * context_latents[:, :-latent_frame_zero, :, :]
                else:
                    head = context_latents[:, :-latent_frame_zero, :, :]
                latent = torch.cat([head, tail], dim=1)

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
        sampling_method: str,
        sde_eta: float,
        show_progress: bool,
        progress_desc: str,
    ) -> torch.Tensor:
        arg_c, _, noise = self.model.generate(
            prompt,
            frame_num=frame_zero,
            max_area=max_area,
            latent_frame_zero=latent_frame_zero,
        )
        latent = noise.clone()
        sampling_sigmas = self._get_sampling_sigmas(num_euler_timesteps, sigma_shift)

        autocast_ctx = self._build_transformer_autocast_ctx()
        with torch.no_grad(), autocast_ctx:
            euler_iter = tqdm(
                range(num_euler_timesteps),
                desc=progress_desc,
                leave=False,
                disable=not show_progress,
            )
            for i in euler_iter:
                sigma_now = float(sampling_sigmas[i])
                sigma_next = (
                    0.0
                    if i + 1 == num_euler_timesteps
                    else float(sampling_sigmas[i + 1])
                )
                timestep = torch.tensor(
                    [sigma_now * 1000.0],
                    device=self.device,
                    dtype=torch.float32,
                )
                noise_pred_cond = self._call_model(
                    latent=latent,
                    timestep=timestep,
                    arg_c=arg_c,
                    flag=False,
                )

                latent = self._integrate_tail(
                    latent_tail=latent,
                    noise_pred_tail=noise_pred_cond,
                    sigma_now=sigma_now,
                    sigma_next=sigma_next,
                    sampling_method=sampling_method,
                    sde_eta=sde_eta,
                )

        return latent

    def _decode_tail_segment(
        self,
        *,
        model_input: torch.Tensor,
        latent_frame_zero: int,
        frame_zero: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            video = self.model.vae.decode(
                [model_input[:, -latent_frame_zero:, :, :].to(torch.float32)]
            )[0]
        return video[:, -frame_zero:, :, :]

    @staticmethod
    def _first_frame_to_pil(seed_video: torch.Tensor) -> Image.Image:
        frame = seed_video[:, 0].detach().cpu().float().clamp(-1.0, 1.0)
        frame = (frame + 1.0) * 0.5
        frame = (frame * 255.0).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        return Image.fromarray(frame)

    def _predict_ti2v(
        self,
        *,
        processed_inputs: Dict[str, Any],
        size_key: str,
        max_area: int,
        num_euler_timesteps: int,
        sigma_shift: float,
        latent_frame_zero: int,
        frame_zero: int,
        rollout_steps: Optional[int],
        prompt_schedule: Optional[List[str]],
        sampling_method: str,
        sde_eta: float,
        show_progress: bool,
    ) -> torch.Tensor:
        prompt = processed_inputs["prompt"]
        image: Optional[Image.Image] = processed_inputs.get("image")
        seed_video: Optional[torch.Tensor] = processed_inputs.get("seed_video")
        prompts = self._normalize_prompt_schedule(
            prompt=prompt,
            prompt_schedule=prompt_schedule,
            rollout_steps=rollout_steps,
        )

        model_input: Optional[torch.Tensor]
        decoded_segments: List[torch.Tensor] = []

        if seed_video is not None:
            seed_video = self._prepare_seed_video_tensor(
                seed_video,
                size=SIZE_CONFIGS[size_key],
                total_frames=33,
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
                image,
                size=SIZE_CONFIGS[size_key],
                total_frames=33,
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
                model_input = self._run_t2v_first_segment(
                    prompt=step_prompt,
                    max_area=max_area,
                    frame_zero=frame_zero,
                    latent_frame_zero=latent_frame_zero,
                    num_euler_timesteps=num_euler_timesteps,
                    sigma_shift=sigma_shift,
                    sampling_method=sampling_method,
                    sde_eta=sde_eta,
                    show_progress=show_progress,
                    progress_desc=f"YUME denoising {step_idx + 1}/{len(prompts)}",
                )
            else:
                assert model_input is not None
                temporal_stride = int(getattr(self.model, "vae_stride", [4])[0])

                if step_idx == 0:
                    frame_num = (model_input.shape[1] - 1) * temporal_stride + 1
                    context_latents = model_input
                    generate_img = model_input[:, :-latent_frame_zero, :, :]
                    arg_c, arg_null, noise, mask2, _ = self.model.generate(
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
                    arg_c, arg_null, _, mask2, _ = self.model.generate(
                        step_prompt,
                        frame_num=frame_num,
                        max_area=max_area,
                        latent_frame_zero=latent_frame_zero,
                        img=model_input,
                    )
                    noise = torch.randn_like(context_latents)

                latent = self._run_conditioned_segment(
                    context_latents=context_latents,
                    arg_c=arg_c,
                    noise=noise,
                    latent_frame_zero=latent_frame_zero,
                    num_euler_timesteps=num_euler_timesteps,
                    sigma_shift=sigma_shift,
                    sampling_method=sampling_method,
                    sde_eta=sde_eta,
                    show_progress=show_progress,
                    progress_desc=f"YUME denoising {step_idx + 1}/{len(prompts)}",
                    mask2=mask2,
                    use_masked_timestep=True,
                    context_blend=MODEL_DEFAULTS[self.task]["context_blend"],
                    guidance_scale=1.0,
                    arg_null=None,
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

    def _encode_native_i2v_prompts(
        self,
        *,
        input_prompt: str,
        n_prompt: str = "",
        offload_model: bool = False,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        if n_prompt == "":
            n_prompt = self.model.sample_neg_prompt

        if not self.model.t5_cpu:
            self.model.text_encoder.model.to(self.device)
            context = self.model.text_encoder([input_prompt], self.device)
            context_null = self.model.text_encoder([n_prompt], self.device)
            if offload_model:
                self.model.text_encoder.model.cpu()
        else:
            context = self.model.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.model.text_encoder([n_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        return context, context_null

    def _build_native_i2v_mask(
        self,
        *,
        known_frames: int,
        frame_zero: int,
        lat_h: int,
        lat_w: int,
    ) -> torch.Tensor:
        total_frames = known_frames + frame_zero
        msk = torch.ones(1, total_frames, lat_h, lat_w, device=self.device)
        msk[:, -frame_zero:] = 0
        msk = torch.cat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
            dim=1,
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        return msk.transpose(1, 2)[0]

    def _encode_native_i2v_clip(
        self,
        *,
        known_video: torch.Tensor,
        rand_num_img: float,
    ) -> Optional[torch.Tensor]:
        clip_model = getattr(self.model, "clip", None)
        if clip_model is None:
            return None

        if rand_num_img >= 0.4:
            clip_input = known_video[:, -1:, :, :]
        else:
            clip_input = known_video[:, :1, :, :]

        clip_model.model.to(self.device)
        with torch.no_grad():
            return clip_model.visual([clip_input])

    def _build_native_i2v_args(
        self,
        *,
        step_prompt: str,
        known_video: torch.Tensor,
        frame_zero: int,
        rand_num_img: float,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if known_video.ndim != 4:
            raise ValueError(f"known_video must be [C,T,H,W], got shape={known_video.shape}")

        known_video = known_video.to(self.device)
        _, known_frames, height, width = known_video.shape

        lat_h = height // self.model.vae_stride[1]
        lat_w = width // self.model.vae_stride[2]
        total_frames = known_frames + frame_zero
        seq_len = ((total_frames - 1) // self.model.vae_stride[0] + 1) * lat_h * lat_w // (
            self.model.patch_size[1] * self.model.patch_size[2]
        )
        seq_len = int(math.ceil(seq_len / self.model.sp_size)) * self.model.sp_size

        condition_video = torch.cat(
            [
                known_video,
                torch.zeros(
                    (3, frame_zero, height, width),
                    device=self.device,
                    dtype=known_video.dtype,
                ),
            ],
            dim=1,
        )
        y = self.model.vae.encode([condition_video])[0]
        mask = self._build_native_i2v_mask(
            known_frames=known_frames,
            frame_zero=frame_zero,
            lat_h=lat_h,
            lat_w=lat_w,
        )
        y = torch.cat([mask, y], dim=0)

        context, context_null = self._encode_native_i2v_prompts(input_prompt=step_prompt)
        clip_context = self._encode_native_i2v_clip(
            known_video=known_video,
            rand_num_img=rand_num_img,
        )
        arg_c = {
            "context": [context[0]],
            "seq_len": seq_len,
            "y": [y],
        }
        arg_null = {
            "context": context_null,
            "seq_len": seq_len,
            "y": [y],
        }
        if clip_context is not None:
            arg_c["clip_fea"] = clip_context
            arg_null["clip_fea"] = clip_context
        return arg_c, arg_null

    def _encode_native_i2v_seed_latents(
        self,
        *,
        model_input_de: torch.Tensor,
        frame_zero: int,
    ) -> torch.Tensor:
        model_input_de = model_input_de.to(self.device)
        if model_input_de.shape[1] <= frame_zero:
            return self.model.vae.encode([model_input_de])[0]

        return torch.cat(
            [
                self.model.vae.encode([model_input_de[:, :-frame_zero, :, :]])[0],
                self.model.vae.encode([model_input_de[:, -frame_zero:, :, :]])[0],
            ],
            dim=1,
        )

    def _predict_native_wan22_i2v(
        self,
        *,
        processed_inputs: Dict[str, Any],
        size_key: str,
        num_euler_timesteps: int,
        sigma_shift: float,
        latent_frame_zero: int,
        frame_zero: int,
        rollout_steps: Optional[int],
        prompt_schedule: Optional[List[str]],
        guidance_scale: float,
        show_progress: bool,
        sampling_method: str,
        sde_eta: float,
        rand_num_img: float,
    ) -> torch.Tensor:
        prompt = processed_inputs["prompt"]
        image: Optional[Image.Image] = processed_inputs.get("image")
        seed_video: Optional[torch.Tensor] = processed_inputs.get("seed_video")

        if seed_video is None and image is None:
            raise ValueError("i2v-14B requires image_path or video_path")

        prompts = self._normalize_prompt_schedule(
            prompt=prompt,
            prompt_schedule=prompt_schedule,
            rollout_steps=rollout_steps,
        )

        if seed_video is not None:
            seed_video = self._prepare_seed_video_tensor(
                seed_video,
                size=SIZE_CONFIGS[size_key],
                total_frames=33,
            )
        else:
            assert image is not None
            seed_video = self._build_seed_video_from_image(
                image,
                size=SIZE_CONFIGS[size_key],
                total_frames=33,
            )

        model_input_de = torch.cat(
            [seed_video[:, 0].unsqueeze(1).repeat(1, 16, 1, 1), seed_video[:, :33]],
            dim=1,
        )
        model_input = self._encode_native_i2v_seed_latents(
            model_input_de=model_input_de,
            frame_zero=frame_zero,
        )

        decoded_segments: List[torch.Tensor] = []
        rollout_iter = enumerate(
            tqdm(
                prompts,
                desc="YUME rollout",
                leave=True,
                disable=not show_progress,
            )
        )
        for step_idx, step_prompt in rollout_iter:
            if step_idx == 0:
                context_latents = model_input
                known_video = model_input_de[:, :-frame_zero, :, :]
            else:
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
                known_video = model_input_de

            arg_c, arg_null = self._build_native_i2v_args(
                step_prompt=step_prompt,
                known_video=known_video,
                frame_zero=frame_zero,
                rand_num_img=rand_num_img,
            )

            forward_extra_kwargs: Optional[Dict[str, Any]] = None
            if self._supports_rand_num_img or self._supports_latent_frame_zero:
                forward_extra_kwargs = {}
                if self._supports_rand_num_img:
                    forward_extra_kwargs["rand_num_img"] = rand_num_img
                if self._supports_latent_frame_zero:
                    forward_extra_kwargs["latent_frame_zero"] = latent_frame_zero
                if not forward_extra_kwargs:
                    forward_extra_kwargs = None

            latent = self._run_conditioned_segment(
                context_latents=context_latents,
                arg_c=arg_c,
                noise=torch.randn_like(context_latents),
                latent_frame_zero=latent_frame_zero,
                num_euler_timesteps=num_euler_timesteps,
                sigma_shift=sigma_shift,
                sampling_method=sampling_method,
                sde_eta=sde_eta,
                show_progress=show_progress,
                progress_desc=f"YUME denoising {step_idx + 1}/{len(prompts)}",
                mask2=None,
                use_masked_timestep=False,
                context_blend=True,
                guidance_scale=guidance_scale,
                arg_null=arg_null,
                forward_extra_kwargs=forward_extra_kwargs,
            )

            if step_idx == 0:
                model_input = torch.cat(
                    [model_input[:, :-latent_frame_zero, :, :], latent[:, -latent_frame_zero:, :, :]],
                    dim=1,
                )
            else:
                model_input = torch.cat([model_input, latent[:, -latent_frame_zero:, :, :]], dim=1)

            segment = self._decode_tail_segment(
                model_input=model_input,
                latent_frame_zero=latent_frame_zero,
                frame_zero=frame_zero,
            )
            decoded_segments.append(segment)

            if step_idx == 0:
                model_input_de = torch.cat([model_input_de[:, :-frame_zero, :, :], segment], dim=1)
            else:
                model_input_de = torch.cat([model_input_de, segment], dim=1)

        return torch.cat(decoded_segments, dim=1)

    def _predict_native_wan22_t2v(
        self,
        *,
        processed_inputs: Dict[str, Any],
        size_key: str,
        num_euler_timesteps: int,
        sigma_shift: float,
        frame_zero: int,
        rollout_steps: Optional[int],
        prompt_schedule: Optional[List[str]],
        guidance_scale: float,
        show_progress: bool,
        base_seed: int,
    ) -> torch.Tensor:
        prompt = processed_inputs["prompt"]

        prompts = self._normalize_prompt_schedule(
            prompt=prompt,
            prompt_schedule=prompt_schedule,
            rollout_steps=rollout_steps,
        )

        decoded_segments: List[torch.Tensor] = []
        for idx, step_prompt in enumerate(
            tqdm(prompts, desc="YUME rollout", leave=True, disable=not show_progress)
        ):
            seed = base_seed + idx if base_seed >= 0 else -1

            video = self.model.generate(
                input_prompt=step_prompt,
                size=SIZE_CONFIGS[size_key],
                frame_num=(frame_zero - 1) * 2 + 1,
                shift=sigma_shift,
                sample_solver="unipc",
                sampling_steps=num_euler_timesteps,
                guide_scale=guidance_scale,
                seed=seed,
                offload_model=False,
            )

            segment = video[:, -frame_zero:, :, :]
            decoded_segments.append(segment)

        return torch.cat(decoded_segments, dim=1)

    @torch.no_grad()
    def predict(
        self,
        *,
        processed_inputs: Dict[str, Any],
        size: Optional[SizeLike] = None,
        max_area: Optional[int] = None,
        num_euler_timesteps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        latent_frame_zero: Optional[int] = None,
        frame_zero: Optional[int] = None,
        rollout_steps: Optional[int] = None,
        prompt_schedule: Optional[List[str]] = None,
        base_seed: int = -1,
        guidance_scale: Optional[float] = None,
        sampling_method: str = "ode",
        sde_eta: float = 0.3,
        rand_num_img: float = 0.6,
        show_progress: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        defaults = MODEL_DEFAULTS[self.task]

        size_key = self._normalize_size_key(size or defaults["size"])
        if size_key not in SIZE_CONFIGS:
            raise ValueError(f"Unsupported size: {size}. Supported: {list(SIZE_CONFIGS.keys())}")

        num_euler_timesteps = int(
            num_euler_timesteps
            if num_euler_timesteps is not None
            else defaults["num_euler_timesteps"]
        )
        sigma_shift = float(sigma_shift if sigma_shift is not None else defaults["sigma_shift"])
        latent_frame_zero = int(
            latent_frame_zero
            if latent_frame_zero is not None
            else defaults["latent_frame_zero"]
        )
        frame_zero = int(frame_zero if frame_zero is not None else defaults["frame_zero"])
        guidance_scale = float(
            guidance_scale if guidance_scale is not None else defaults["guidance_scale"]
        )
        max_area = int(max_area if max_area is not None else MAX_AREA_CONFIGS[size_key])

        sampling_method = sampling_method.lower().strip()
        if sampling_method not in {"ode", "sde"}:
            raise ValueError("sampling_method must be one of {'ode', 'sde'}")

        if base_seed >= 0:
            random.seed(base_seed)
            np.random.seed(base_seed % (2**32 - 1))
            torch.manual_seed(base_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(base_seed)

        if self.backend in {
            "sceneflow_wan22_compat",
        }:
            return self._predict_ti2v(
                processed_inputs=processed_inputs,
                size_key=size_key,
                max_area=max_area,
                num_euler_timesteps=num_euler_timesteps,
                sigma_shift=sigma_shift,
                latent_frame_zero=latent_frame_zero,
                frame_zero=frame_zero,
                rollout_steps=rollout_steps,
                prompt_schedule=prompt_schedule,
                sampling_method=sampling_method,
                sde_eta=sde_eta,
                show_progress=show_progress,
            )

        if self.backend == "sceneflow_wan22_native_i2v":
            return self._predict_native_wan22_i2v(
                processed_inputs=processed_inputs,
                size_key=size_key,
                num_euler_timesteps=num_euler_timesteps,
                sigma_shift=sigma_shift,
                latent_frame_zero=latent_frame_zero,
                frame_zero=frame_zero,
                rollout_steps=rollout_steps,
                prompt_schedule=prompt_schedule,
                guidance_scale=guidance_scale,
                show_progress=show_progress,
                sampling_method=sampling_method,
                sde_eta=sde_eta,
                rand_num_img=rand_num_img,
            )

        if self.backend == "sceneflow_yume_i2v_native":
            return self._predict_native_wan22_i2v(
                processed_inputs=processed_inputs,
                size_key=size_key,
                num_euler_timesteps=num_euler_timesteps,
                sigma_shift=sigma_shift,
                latent_frame_zero=latent_frame_zero,
                frame_zero=frame_zero,
                rollout_steps=rollout_steps,
                prompt_schedule=prompt_schedule,
                guidance_scale=guidance_scale,
                show_progress=show_progress,
                sampling_method=sampling_method,
                sde_eta=sde_eta,
                rand_num_img=rand_num_img,
            )

        if self.backend == "sceneflow_wan22_native_t2v":
            return self._predict_native_wan22_t2v(
                processed_inputs=processed_inputs,
                size_key=size_key,
                num_euler_timesteps=num_euler_timesteps,
                sigma_shift=sigma_shift,
                frame_zero=frame_zero,
                rollout_steps=rollout_steps,
                prompt_schedule=prompt_schedule,
                guidance_scale=guidance_scale,
                show_progress=show_progress,
                base_seed=base_seed,
            )

        raise RuntimeError(f"Unsupported backend: {self.backend}")
