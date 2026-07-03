from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

from ...base_synthesis import BaseSynthesis
from ...runtime_utils import ensure_path, package_dir, require_outputs, run_method_command


class FantasyWorldSynthesis(BaseSynthesis):
    def __init__(self, wan_ckpt_path: str, model_ckpt: str, python_bin: str = sys.executable):
        super().__init__()
        self.runtime_dir = package_dir(__file__)
        self.wan_ckpt_path = str(ensure_path(wan_ckpt_path, "FantasyWorld Wan checkpoint", expect_file=False))
        self.model_ckpt = str(ensure_path(model_ckpt, "FantasyWorld checkpoint", expect_file=True))
        self.python_bin = python_bin

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        wan_ckpt_path: str,
        python_bin: str = sys.executable,
        **kwargs,
    ) -> "FantasyWorldSynthesis":
        return cls(
            wan_ckpt_path=wan_ckpt_path,
            model_ckpt=pretrained_model_path,
            python_bin=python_bin,
        )

    def api_init(self, api_key, endpoint):
        raise NotImplementedError("FantasyWorld is a local checkpoint pipeline and does not expose an API backend.")

    def predict(
        self,
        image_path: str,
        camera_json_path: str,
        prompt: str,
        output_dir: str,
        sample_steps: int = 50,
        frames: int = 17,
        using_scale: bool = True,
        cuda_visible_devices: str = "0",
        moge_ckpt: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, str]:
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_bin,
            "inference_wan21.py",
            "--wan_ckpt_path",
            self.wan_ckpt_path,
            "--model_ckpt",
            self.model_ckpt,
            "--image_path",
            str(ensure_path(image_path, "FantasyWorld input image", expect_file=True)),
            "--camera_json_path",
            str(ensure_path(camera_json_path, "FantasyWorld camera json", expect_file=True)),
            "--prompt",
            prompt,
            "--output_dir",
            str(output_root),
            "--sample_steps",
            str(int(sample_steps)),
            "--frames",
            str(int(frames)),
            "--using_scale",
            "True" if using_scale else "False",
        ]
        repo_src = self.runtime_dir.parents[3]
        utils3d_parent = (
            repo_src
            / "openworldlib"
            / "base_models"
            / "three_dimensions"
            / "general_3d"
            / "eastern_journalist"
        )
        completed = run_method_command(
            command,
            cwd=self.runtime_dir,
            env={
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
                "FANTASY_WORLD_MOGE_CKPT": moge_ckpt or kwargs.get("moge_ckpt") or "Ruicheng/moge-2-vitl-normal",
            },
            python_paths=[repo_src, self.runtime_dir, utils3d_parent],
            timeout=timeout,
        )
        video_path = output_root / "video.mp4"
        ply_candidates = sorted(output_root.glob("recon_confthresh*.ply"))
        if not ply_candidates:
            require_outputs([video_path, output_root / "recon_confthresh1.0.ply"])
        require_outputs([video_path, ply_candidates[-1] if ply_candidates else output_root / "recon_confthresh1.0.ply"])
        return {"video_path": str(video_path), "point_cloud_path": str(ply_candidates[-1]), "stdout": completed.stdout}
