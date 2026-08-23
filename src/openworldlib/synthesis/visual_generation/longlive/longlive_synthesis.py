from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from einops import rearrange
from omegaconf import OmegaConf

from ...base_synthesis import BaseSynthesis
from .runtime.pipeline.causal_inference import CausalInferencePipeline
from .runtime.pipeline.interactive_causal_inference import (
    InteractiveCausalInferencePipeline,
)
from .runtime.utils.lora_utils import configure_lora_for_model
from .runtime.utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
from .runtime.utils.misc import set_seed


DEFAULT_LONGLIVE_MODEL_ROOT = "checkpoints/LongLive"
DEFAULT_CONFIG_NAME = "longlive_inference.yaml"
DEFAULT_INTERACTIVE_CONFIG_NAME = "longlive_interactive_inference.yaml"


def _resolve_device(device: str | torch.device | None) -> torch.device:
    resolved = torch.device(device or "cuda")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("LongLive inference requires CUDA.")
        torch.cuda.set_device(resolved.index if resolved.index is not None else 0)
    return resolved


def _load_generator_state_dict(checkpoint_path: Path, use_ema: bool):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and ("generator" in state_dict or "generator_ema" in state_dict):
        raw_state_dict = state_dict["generator_ema" if use_ema else "generator"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        raw_state_dict = state_dict["model"]
    else:
        raw_state_dict = state_dict

    if use_ema:
        return {
            key.replace("_fsdp_wrapped_module.", ""): value
            for key, value in raw_state_dict.items()
        }
    return raw_state_dict


def _first_existing_or_default(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _apply_lora(pipeline, config, lora_ckpt_path: Optional[Path]):
    adapter = getattr(config, "adapter", None)
    if adapter is None:
        return False

    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=adapter,
        is_main_process=True,
    )
    if lora_ckpt_path is not None and lora_ckpt_path.exists():
        import peft

        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            lora_state = lora_checkpoint["generator_lora"]
        else:
            lora_state = lora_checkpoint
        peft.set_peft_model_state_dict(pipeline.generator.model, lora_state)
    return True


class LongLiveSynthesis(BaseSynthesis):
    def __init__(
        self,
        pipeline: CausalInferencePipeline,
        interactive_pipeline: InteractiveCausalInferencePipeline,
        config,
        device: torch.device,
        weight_dtype,
        latent_shape: Sequence[int],
        low_memory: bool = False,
        component_paths: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.interactive_pipeline = interactive_pipeline
        self.config = config
        self.device = device
        self.weight_dtype = weight_dtype
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.low_memory = low_memory
        self.component_paths = component_paths or {}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        required_components: Optional[Dict[str, str]] = None,
        device: str | torch.device | None = "cuda",
        weight_dtype=torch.bfloat16,
        config_path: Optional[str] = None,
        interactive_config_path: Optional[str] = None,
        use_lora: bool = True,
        low_memory: Optional[bool] = None,
        latent_height: int = 60,
        latent_width: int = 104,
        **kwargs,
    ) -> "LongLiveSynthesis":
        resolved_device = _resolve_device(device)
        model_root = Path(pretrained_model_path or DEFAULT_LONGLIVE_MODEL_ROOT)
        required_components = required_components or {}

        config_dir = Path(__file__).resolve().parent / "configs"
        config = OmegaConf.load(config_path or config_dir / DEFAULT_CONFIG_NAME)
        interactive_config = OmegaConf.load(
            interactive_config_path or config_dir / DEFAULT_INTERACTIVE_CONFIG_NAME
        )

        if required_components.get("wan_model_path"):
            wan_model_path = Path(required_components["wan_model_path"])
        else:
            wan_model_path = _first_existing_or_default(
                [
                    Path("checkpoints/Wan2.1-T2V-1.3B"),
                ]
            )
        generator_ckpt = Path(
            required_components.get("generator_ckpt")
            or required_components.get("generator_ckpt_path")
            or _first_existing_or_default(
                [
                    model_root / "models/longlive_base.pt",
                    model_root / "longlive_models/models/longlive_base.pt",
                ]
            )
        )
        lora_ckpt_value = (
            required_components.get("lora_ckpt")
            or required_components.get("lora_ckpt_path")
            or _first_existing_or_default(
                [
                    model_root / "models/lora.pt",
                    model_root / "longlive_models/models/lora.pt",
                ]
            )
        )
        lora_ckpt = Path(lora_ckpt_value) if lora_ckpt_value else None

        config.wan_model_path = str(wan_model_path)
        interactive_config.wan_model_path = str(wan_model_path)
        config.generator_ckpt = str(generator_ckpt)
        interactive_config.generator_ckpt = str(generator_ckpt)
        config.lora_ckpt = str(lora_ckpt) if lora_ckpt is not None else None
        interactive_config.lora_ckpt = str(lora_ckpt) if lora_ckpt is not None else None

        missing = [
            str(path)
            for path in [
                wan_model_path / "models_t5_umt5-xxl-enc-bf16.pth",
                wan_model_path / "Wan2.1_VAE.pth",
                wan_model_path / "config.json",
                generator_ckpt,
            ]
            if not path.exists()
        ]
        if use_lora and (lora_ckpt is None or not lora_ckpt.exists()):
            missing.append(str(lora_ckpt or "lora_ckpt"))
        if missing:
            raise FileNotFoundError(
                "LongLive required weights are missing. Download them under "
                f"{model_root}. Missing: {missing}"
            )

        seed = int(getattr(config, "seed", 0))
        set_seed(seed)

        pipeline = CausalInferencePipeline(config, device=resolved_device)
        generator_state_dict = _load_generator_state_dict(
            generator_ckpt,
            use_ema=bool(getattr(config, "use_ema", False)),
        )
        missing_keys, unexpected_keys = pipeline.generator.load_state_dict(
            generator_state_dict, strict=False
        )
        if missing_keys:
            print(f"[LongLive] Missing generator keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[LongLive] Unexpected generator keys: {len(unexpected_keys)}")

        if use_lora:
            pipeline.is_lora_enabled = _apply_lora(pipeline, config, lora_ckpt)
        else:
            pipeline.is_lora_enabled = False

        pipeline = pipeline.to(dtype=weight_dtype)
        if low_memory is None:
            low_memory = (
                resolved_device.type == "cuda"
                and get_cuda_free_memory_gb(resolved_device) < 40
            )
        if low_memory:
            DynamicSwapInstaller.install_model(pipeline.text_encoder, device=resolved_device)
        pipeline.generator.to(device=resolved_device)
        pipeline.vae.to(device=resolved_device)

        interactive_pipeline = InteractiveCausalInferencePipeline(
            interactive_config,
            device=resolved_device,
            generator=pipeline.generator,
            text_encoder=pipeline.text_encoder,
            vae=pipeline.vae,
        )
        interactive_pipeline.is_lora_enabled = pipeline.is_lora_enabled

        return cls(
            pipeline=pipeline,
            interactive_pipeline=interactive_pipeline,
            config=config,
            device=resolved_device,
            weight_dtype=weight_dtype,
            latent_shape=(16, latent_height, latent_width),
            low_memory=bool(low_memory),
            component_paths={
                "model_root": str(model_root),
                "wan_model_path": str(wan_model_path),
                "generator_ckpt": str(generator_ckpt),
                "lora_ckpt": str(lora_ckpt) if lora_ckpt is not None else "",
            },
        )

    @torch.no_grad()
    def predict(
        self,
        prompts: Sequence[str] | str,
        num_output_frames: Optional[int] = None,
        switch_frame_indices: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
        return_latents: bool = False,
        low_memory: Optional[bool] = None,
        **kwargs,
    ):
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)
        if len(prompt_list) == 0:
            raise ValueError("prompts cannot be empty")

        if seed is not None:
            set_seed(int(seed))

        num_output_frames = int(num_output_frames or getattr(self.config, "num_output_frames", 120))
        if num_output_frames % int(getattr(self.config, "num_frame_per_block", 1)) != 0:
            block = int(getattr(self.config, "num_frame_per_block", 1))
            num_output_frames = ((num_output_frames + block - 1) // block) * block

        noise = torch.randn(
            [1, num_output_frames, *self.latent_shape],
            device=self.device,
            dtype=self.weight_dtype,
        )
        use_low_memory = self.low_memory if low_memory is None else bool(low_memory)

        if len(prompt_list) == 1:
            output = self.pipeline.inference(
                noise=noise,
                text_prompts=prompt_list,
                return_latents=return_latents,
                low_memory=use_low_memory,
                profile=False,
            )
        else:
            if switch_frame_indices is None:
                segment_length = max(num_output_frames // len(prompt_list), 1)
                switch_frame_indices = [
                    segment_length * idx for idx in range(1, len(prompt_list))
                ]
            text_prompts_list = [[prompt] for prompt in prompt_list]
            output = self.interactive_pipeline.inference(
                noise=noise,
                text_prompts_list=text_prompts_list,
                switch_frame_indices=[int(idx) for idx in switch_frame_indices],
                return_latents=return_latents,
                low_memory=use_low_memory,
            )

        if return_latents:
            video, latents = output
            return self._format_video(video), latents
        return self._format_video(output)

    @staticmethod
    def _format_video(video: torch.Tensor) -> torch.Tensor:
        return (rearrange(video, "b t c h w -> b t h w c").detach().cpu() * 255.0).clamp(0, 255).to(torch.uint8)
