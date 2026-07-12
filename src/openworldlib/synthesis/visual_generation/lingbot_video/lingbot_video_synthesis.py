from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ...base_synthesis import BaseSynthesis


class LingBotVideoSynthesis(BaseSynthesis):
    """LingBot-Video synthesis wrapper around the external inference package."""

    def __init__(
        self,
        *,
        model: Any,
        mode: str,
        model_path: str,
        backend: str = "diffusers",
        device: Optional[torch.device] = None,
        batch_cfg: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.mode = mode
        self.model_path = model_path
        self.backend = backend
        self.device = device if device is not None else self._default_device()
        self.batch_cfg = batch_cfg

    @staticmethod
    def _default_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        mode: str = "t2v",
        backend: str = "diffusers",
        default_dtype: str = "bf16",
        transformer_dtype: str = "bf16",
        text_encoder_dtype: str = "bf16",
        vae_dtype: str = "fp32",
        transformer_subfolder: str = "transformer",
        diffusers_attn_backend: str = "",
        allow_tf32: bool = True,
        batch_cfg: bool = False,
        **kwargs,
    ) -> "LingBotVideoSynthesis":
        from lingbot_video.inference_backend import resolve_backend_engine
        from lingbot_video.runner import (
            _component_dtypes,
            _configure_pipeline_logs,
            _load_pipe,
            _make_dtype_map,
        )
        import argparse
        import os

        normalized_mode = "ti2v" if mode == "i2v" else mode
        engine = resolve_backend_engine(engine=None, backend=backend)
        if diffusers_attn_backend:
            os.environ["DIFFUSERS_ATTN_BACKEND"] = diffusers_attn_backend
        if allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        unsupported = {
            name: kwargs[name]
            for name in (
                "cfg_parallel_degree",
                "context_parallel_degree",
                "enable_fsdp_inference",
                "run_refiner",
                "refiner_model_dir",
                "reuse_condition_features",
            )
            if kwargs.get(name)
        }
        if unsupported:
            raise NotImplementedError(
                "The OpenWorldLib LingBot-Video wrapper does not support: "
                + ", ".join(sorted(unsupported))
            )
        args = argparse.Namespace(
            model_dir=str(Path(pretrained_model_path).expanduser()),
            mode=normalized_mode,
            backend=backend,
            engine=engine,
            default_dtype=default_dtype,
            transformer_dtype=transformer_dtype,
            text_encoder_dtype=text_encoder_dtype,
            vae_dtype=vae_dtype,
            transformer_subfolder=transformer_subfolder,
            diffusers_attn_backend=diffusers_attn_backend,
            allow_tf32=allow_tf32,
        )
        dtype_map = _make_dtype_map(args)
        model, _engine_name = _load_pipe(args, dtype_map)
        _configure_pipeline_logs(model)
        component_dtypes = _component_dtypes(model)
        expected_vae_dtype = str(dtype_map["vae"]).replace("torch.", "")
        if component_dtypes.get("vae") != expected_vae_dtype:
            raise AssertionError(
                f"VAE dtype mismatch: requested {expected_vae_dtype}, got {component_dtypes}"
            )

        return cls(
            model=model,
            mode=normalized_mode,
            model_path=pretrained_model_path,
            backend=backend,
            device=cls._default_device(),
            batch_cfg=batch_cfg,
        )

    @torch.no_grad()
    def predict(
        self,
        *,
        processed_inputs: Dict[str, Any],
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        steps: int = 40,
        guidance_scale: float = 3.0,
        shift: float = 3.0,
        seed: int = 42,
        negative_prompt: Optional[str] = None,
        batch_cfg: Optional[bool] = None,
        output_type: str = "np",
        **kwargs,
    ) -> Any:
        from lingbot_video.pipeline_lingbot_video import (
            DEFAULT_NEGATIVE_PROMPT,
            DEFAULT_NEGATIVE_PROMPT_IMAGE,
        )

        mode = processed_inputs["mode"]
        if mode == "t2i":
            num_frames = 1
        if negative_prompt is None:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT_IMAGE if mode == "t2i" else DEFAULT_NEGATIVE_PROMPT

        if kwargs:
            raise TypeError(f"Unsupported LingBot-Video inference arguments: {sorted(kwargs)}")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        call_kwargs = {
            "prompt": processed_inputs["prompt"],
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "shift": shift,
            "generator": generator,
            "output_type": output_type,
            "batch_cfg": self.batch_cfg if batch_cfg is None else batch_cfg,
        }
        if mode == "ti2v":
            call_kwargs["image"] = processed_inputs["image"]

        return self.model(**call_kwargs)
