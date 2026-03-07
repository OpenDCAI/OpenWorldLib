from typing import Any, Dict, Optional

import torch
import os
from pathlib import Path
from huggingface_hub import snapshot_download

from ....base_models.diffusion_model.video import wan_2p2
from ....base_models.diffusion_model.video.wan_2p2.configs import (
    WAN_CONFIGS,
    SIZE_CONFIGS,
    MAX_AREA_CONFIGS,
)


class Wan2p2Synthesis:
    """
    Wan 推理层：只负责
    - 根据 mode 创建对应的 Wan* 管线
    - 根据 processed_inputs + args 调用 .generate(...)
    """

    def __init__(
        self,
        *,
        mode: str,
        cfg: Any,
        model: Any,
        device: int,
        rank: int = 0,
    ) -> None:
        self.mode = mode
        self.cfg = cfg
        self.model = model
        self.device = device
        self.rank = rank


    @classmethod
    def from_pretrained(
        cls,
        *,
        mode: str,
        ckpt_dir: str,
        device: int = 0,
        rank: int = 0,
        t5_fsdp: bool = False,
        dit_fsdp: bool = False,
        ulysses_size: int = 1,
        t5_cpu: bool = False,
        convert_model_dtype: bool = False,
    ) -> "Wan2p2Synthesis":
        """
        目前只关注 ti2v 任务，这里仅支持构建 WanTI2V。
        其他 task 如需支持，可以在后续按需补充。
        """
        if mode not in WAN_CONFIGS:
            raise ValueError(f"Unsupported mode: {mode}")

        if "ti2v" not in mode:
            raise ValueError(
                f"Wan2p2Synthesis.from_pretrained only support ti2v mode, got mode={mode!r}"
            )
        cfg = WAN_CONFIGS[mode]


        if os.path.isdir(ckpt_dir):
            model_root = ckpt_dir
        else:
            repo_name = ckpt_dir.split("/")[-1]
            local_dir = Path.cwd() / repo_name
            local_dir.mkdir(parents=True, exist_ok=True)
            model_root = Path(snapshot_download(
                repo_id=ckpt_dir,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False
            ))

        common_kwargs = dict(
            config=cfg,
            checkpoint_dir=model_root,
            device_id=device,
            rank=rank,
            t5_fsdp=t5_fsdp,
            dit_fsdp=dit_fsdp,
            use_sp=(ulysses_size > 1),
            t5_cpu=t5_cpu,
            convert_model_dtype=convert_model_dtype,
        )

        model = wan_2p2.WanTI2V(**common_kwargs)

        return cls(
            mode=mode,
            cfg=cfg,
            model=model,
            device=device,
            rank=rank,
        )


    @torch.no_grad()
    def predict(
        self,
        *,
        processed_inputs: Dict[str, Any],
        size: str = "1280*720",
        frame_num: Optional[int] = None,
        sample_shift: Optional[float] = None,
        sample_solver: str = "unipc",
        sample_steps: Optional[int] = None,
        sample_guide_scale: Optional[float] = None,
        base_seed: int = -1,
        offload_model: Optional[bool] = None,
        **kwargs
    ) -> Any:

        prompt: str = processed_inputs["prompt"]
        img = processed_inputs.get("image")

        if "ti2v" not in self.mode:
            raise ValueError(
                f"Wan2p2Synthesis.predict only support ti2v mode, got mode={self.mode!r}"
            )

        video = self.model.generate(
            prompt,
            img=img,
            size=SIZE_CONFIGS[size],
            max_area=MAX_AREA_CONFIGS[size],
            frame_num=frame_num,
            shift=sample_shift,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale=sample_guide_scale,
            seed=base_seed,
            offload_model=offload_model,
        )

        return video
        