from __future__ import annotations

from pathlib import Path
from typing import Dict

from ...base_synthesis import BaseSynthesis
from ...runtime_utils import require_outputs


class Cosmos3Synthesis(BaseSynthesis):
    def __init__(self, model_path: str = "nvidia/Cosmos3-Nano", torch_dtype: str = "bfloat16"):
        super().__init__()
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self._pipe = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = "nvidia/Cosmos3-Nano",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ) -> "Cosmos3Synthesis":
        return cls(model_path=pretrained_model_path, torch_dtype=torch_dtype)

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        dtype = getattr(torch, self.torch_dtype)
        pipe = Cosmos3OmniPipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="cuda",
            safety_checker=None,
            enable_safety_checker=False,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)
        self._pipe = pipe
        return pipe

    def predict(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str = "",
        image=None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: int = 24,
        num_inference_steps: int = 35,
        guidance_scale: float = 6.0,
        seed: int = 1234,
        enable_sound: bool = False,
        **kwargs,
    ) -> Dict[str, str]:
        import torch
        from diffusers.utils import export_to_video

        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        pipe = self._load_pipe()
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_frames=int(num_frames),
            height=int(height),
            width=int(width),
            fps=int(fps),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            enable_sound=bool(enable_sound),
            add_resolution_template=False,
            add_duration_template=False,
            generator=generator,
        )
        export_to_video(result.video, str(output), fps=int(fps), macro_block_size=1)
        require_outputs([output])
        return {"video_path": str(output)}
