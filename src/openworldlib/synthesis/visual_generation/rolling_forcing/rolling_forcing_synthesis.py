from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

from ...base_synthesis import BaseSynthesis
from .runtime.pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from .runtime.pipeline.rolling_forcing_inference import CausalInferencePipeline
from .runtime.utils.misc import set_seed


DEFAULT_ROLLING_FORCING_MODEL_ROOT = "checkpoints/RollingForcing"
DEFAULT_WAN_MODEL_ROOT = "checkpoints/Wan2.1-T2V-1.3B"
DEFAULT_BASE_CONFIG_NAME = "default_config.yaml"
DEFAULT_CONFIG_NAME = "rolling_forcing_dmd.yaml"


def _resolve_device(device: str | torch.device | None) -> torch.device:
    resolved = torch.device(device or "cuda")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("RollingForcing inference requires CUDA.")
        torch.cuda.set_device(resolved.index if resolved.index is not None else 0)
    return resolved


def _first_existing_or_default(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _strip_fsdp_prefix(state_dict):
    return OrderedDict(
        (key.replace("_fsdp_wrapped_module.", ""), value)
        for key, value in state_dict.items()
    )


def _load_generator_state_dict(checkpoint_path: Path, use_ema: bool):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and ("generator" in state_dict or "generator_ema" in state_dict):
        key = "generator_ema" if use_ema and "generator_ema" in state_dict else "generator"
        state_dict = state_dict[key]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    if isinstance(state_dict, dict):
        state_dict = _strip_fsdp_prefix(state_dict)
    return state_dict


class RollingForcingSynthesis(BaseSynthesis):
    def __init__(
        self,
        pipeline: CausalInferencePipeline | CausalDiffusionInferencePipeline,
        config,
        device: torch.device,
        weight_dtype,
        latent_shape: Sequence[int],
        component_paths: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.config = config
        self.device = device
        self.weight_dtype = weight_dtype
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.component_paths = component_paths or {}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        required_components: Optional[Dict[str, str]] = None,
        device: str | torch.device | None = "cuda",
        weight_dtype=torch.bfloat16,
        config_path: Optional[str] = None,
        base_config_path: Optional[str] = None,
        use_ema: bool = True,
        latent_height: int = 60,
        latent_width: int = 104,
        **kwargs,
    ) -> "RollingForcingSynthesis":
        resolved_device = _resolve_device(device)
        model_root = Path(pretrained_model_path or DEFAULT_ROLLING_FORCING_MODEL_ROOT)
        required_components = required_components or {}

        config_dir = Path(__file__).resolve().parent / "configs"
        base_config = OmegaConf.load(base_config_path or config_dir / DEFAULT_BASE_CONFIG_NAME)
        config = OmegaConf.load(config_path or config_dir / DEFAULT_CONFIG_NAME)
        config = OmegaConf.merge(base_config, config)

        wan_model_path = Path(required_components.get("wan_model_path") or DEFAULT_WAN_MODEL_ROOT)
        generator_ckpt = Path(
            required_components.get("generator_ckpt")
            or required_components.get("generator_ckpt_path")
            or _first_existing_or_default(
                [
                    model_root / "checkpoints/rolling_forcing_dmd.pt",
                    model_root / "rolling_forcing_dmd.pt",
                    Path(str(getattr(config, "generator_ckpt", "checkpoints/rolling_forcing_dmd.pt"))),
                    Path("checkpoints/rolling_forcing_dmd.pt"),
                ]
            )
        )

        missing = [
            str(path)
            for path in [
                wan_model_path / "config.json",
                wan_model_path / "models_t5_umt5-xxl-enc-bf16.pth",
                wan_model_path / "Wan2.1_VAE.pth",
                wan_model_path / "google/umt5-xxl",
                generator_ckpt,
            ]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "RollingForcing required weights are missing. Download them under "
                f"`checkpoints/`. Missing: {missing}"
            )

        config.wan_model_path = str(wan_model_path)
        config.generator_ckpt = str(generator_ckpt)
        config.verbose = bool(kwargs.pop("verbose", False))

        seed = int(getattr(config, "seed", 0))
        set_seed(seed)

        if hasattr(config, "denoising_step_list"):
            pipeline = CausalInferencePipeline(config, device=resolved_device)
        else:
            pipeline = CausalDiffusionInferencePipeline(config, device=resolved_device)

        generator_state_dict = _load_generator_state_dict(generator_ckpt, use_ema=use_ema)
        missing_keys, unexpected_keys = pipeline.generator.load_state_dict(
            generator_state_dict,
            strict=False,
        )
        if missing_keys:
            print(f"[RollingForcing] Missing generator keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[RollingForcing] Unexpected generator keys: {len(unexpected_keys)}")

        pipeline = pipeline.to(device=resolved_device, dtype=weight_dtype)
        pipeline.eval()

        return cls(
            pipeline=pipeline,
            config=config,
            device=resolved_device,
            weight_dtype=weight_dtype,
            latent_shape=(16, latent_height, latent_width),
            component_paths={
                "model_root": str(model_root),
                "wan_model_path": str(wan_model_path),
                "generator_ckpt": str(generator_ckpt),
            },
        )

    @torch.no_grad()
    def predict(
        self,
        prompts: Sequence[str] | str,
        num_output_frames: Optional[int] = None,
        num_samples: int = 1,
        seed: Optional[int] = None,
        images: Optional[Image.Image] = None,
        initial_latents: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        include_context: bool = True,
        size: tuple[int, int] = (480, 832),
        profile: bool = False,
        **kwargs,
    ):
        prompt_list = self._resolve_prompt_batch(prompts=prompts, num_samples=num_samples)
        batch_size = len(prompt_list)

        if seed is not None:
            set_seed(int(seed))

        if initial_latents is None and images is not None:
            if not self._supports_single_image_context():
                block = int(getattr(self.config, "num_frame_per_block", 1))
                raise ValueError(
                    "The default RollingForcing DMD config is text-to-video and cannot "
                    "use a single image latent as context because num_frame_per_block="
                    f"{block}. Use text prompts or provide memory latents with a compatible "
                    "frame count."
                )
            initial_latents = self.encode_image_to_latents(images=images, size=size)

        initial_latents = self._prepare_initial_latents(initial_latents, batch_size)
        num_context_frames = int(initial_latents.shape[1]) if initial_latents is not None else 0
        num_total_frames = int(num_output_frames or getattr(self.config, "num_output_frames", 126))
        if num_total_frames <= num_context_frames:
            raise ValueError(
                "num_output_frames must be greater than the number of context latent frames "
                f"({num_context_frames})."
            )

        noise_frame_count = num_total_frames - num_context_frames
        noise_frame_count = self._align_noise_frames(noise_frame_count, has_context=initial_latents is not None)
        num_total_frames = noise_frame_count + num_context_frames

        noise = torch.randn(
            [batch_size, noise_frame_count, *self.latent_shape],
            device=self.device,
            dtype=self.weight_dtype,
        )

        if hasattr(self.pipeline, "inference_rolling_forcing"):
            output = self.pipeline.inference_rolling_forcing(
                noise=noise,
                text_prompts=prompt_list,
                initial_latent=initial_latents,
                return_latents=return_latents,
                profile=profile,
            )
        else:
            output = self.pipeline.inference(
                noise=noise,
                text_prompts=prompt_list,
                initial_latent=initial_latents,
                return_latents=return_latents,
            )

        if return_latents:
            video, latents = output
            video = self._format_video(video)
            if not include_context and num_context_frames > 0:
                video = video[:, num_context_frames:]
                latents = latents[:, num_context_frames:]
            return video, latents

        video = self._format_video(output)
        if not include_context and num_context_frames > 0:
            video = video[:, num_context_frames:]
        return video

    @torch.no_grad()
    def encode_image_to_latents(
        self,
        images: Image.Image,
        size: tuple[int, int] = (480, 832),
    ) -> torch.Tensor:
        from torchvision import transforms

        height, width = size
        transform = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        image_tensor = transform(images.convert("RGB")).unsqueeze(0).unsqueeze(2)
        image_tensor = image_tensor.to(device=self.device, dtype=self.weight_dtype)
        return self.pipeline.vae.encode_to_latent(image_tensor).to(
            device=self.device,
            dtype=self.weight_dtype,
        )

    def _prepare_initial_latents(self, initial_latents: Optional[torch.Tensor], batch_size: int):
        if initial_latents is None:
            return None
        latents = initial_latents.to(device=self.device, dtype=self.weight_dtype)
        if latents.ndim != 5:
            raise ValueError("initial_latents must have shape [B, T, C, H, W].")
        if latents.shape[0] == 1 and batch_size > 1:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)
        if latents.shape[0] != batch_size:
            raise ValueError(
                f"initial_latents batch size ({latents.shape[0]}) must match prompt batch ({batch_size})."
            )
        if not bool(getattr(self.config, "independent_first_frame", False)):
            block = int(getattr(self.config, "num_frame_per_block", 1))
            if latents.shape[1] % block != 0:
                raise ValueError(
                    "This RollingForcing config requires context latents to be a multiple of "
                    f"num_frame_per_block={block}. Use stream memory latents or a compatible config."
                )
        return latents

    def _align_noise_frames(self, noise_frame_count: int, has_context: bool) -> int:
        block = int(getattr(self.config, "num_frame_per_block", 1))
        independent_first_frame = bool(getattr(self.config, "independent_first_frame", False))
        if block <= 1:
            return noise_frame_count

        if independent_first_frame and not has_context:
            if (noise_frame_count - 1) % block == 0:
                return noise_frame_count
            return ((noise_frame_count - 1 + block - 1) // block) * block + 1

        if noise_frame_count % block == 0:
            return noise_frame_count
        return ((noise_frame_count + block - 1) // block) * block

    def _supports_single_image_context(self) -> bool:
        if bool(getattr(self.config, "independent_first_frame", False)):
            return True
        return int(getattr(self.config, "num_frame_per_block", 1)) == 1

    @staticmethod
    def _resolve_prompt_batch(prompts: Sequence[str] | str, num_samples: int) -> list[str]:
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)
        if len(prompt_list) == 0:
            raise ValueError("prompts cannot be empty.")
        if len(prompt_list) == 1 and num_samples > 1:
            return prompt_list * int(num_samples)
        if num_samples != 1 and len(prompt_list) != int(num_samples):
            raise ValueError("Prompt count must be 1 or match num_samples.")
        return prompt_list

    @staticmethod
    def _format_video(video: torch.Tensor) -> torch.Tensor:
        return (
            rearrange(video, "b t c h w -> b t h w c")
            .detach()
            .cpu()
            .mul(255.0)
            .clamp(0, 255)
            .to(torch.uint8)
        )
