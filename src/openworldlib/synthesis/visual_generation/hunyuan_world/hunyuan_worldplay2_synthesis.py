from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

from ...base_synthesis import BaseSynthesis
from ...runtime_utils import ensure_path, package_dir, require_outputs, run_method_command


class HunyuanWorldPlay2Synthesis(BaseSynthesis):
    def __init__(
        self,
        model_path: str = "tencent/HY-World-2.0",
        subfolder: str = "HY-WorldMirror-2.0",
        python_bin: str = sys.executable,
    ):
        super().__init__()
        self.runtime_dir = package_dir(__file__, "hunyuan_worldplay2")
        self.model_path = model_path
        self.subfolder = subfolder
        self.python_bin = python_bin

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = "tencent/HY-World-2.0",
        subfolder: str = "HY-WorldMirror-2.0",
        python_bin: str = sys.executable,
        **kwargs,
    ) -> "HunyuanWorldPlay2Synthesis":
        return cls(model_path=pretrained_model_path, subfolder=subfolder, python_bin=python_bin)

    def api_init(self, api_key, endpoint):
        raise NotImplementedError("HY-World-2.0 is a local checkpoint pipeline and does not expose an API backend.")

    def predict(
        self,
        input_path: str,
        output_dir: str,
        target_size: int = 952,
        video_max_frames: int = 32,
        fps: int = 1,
        save_rendered: bool = False,
        render_depth: bool = False,
        cuda_visible_devices: str = "0",
        timeout: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, object]:
        source = ensure_path(input_path, "HY-World-2.0 input")
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        command = [
            self.python_bin,
            "-m",
            "hyworld2.worldrecon.pipeline",
            "--input_path",
            str(source),
            "--output_path",
            str(output_root),
            "--strict_output_path",
            str(output_root),
            "--pretrained_model_name_or_path",
            self.model_path,
            "--subfolder",
            self.subfolder,
            "--target_size",
            str(int(target_size)),
            "--video_max_frames",
            str(int(video_max_frames)),
            "--fps",
            str(int(fps)),
            "--no_interactive",
        ]
        if save_rendered:
            command.append("--save_rendered")
        if render_depth:
            command.append("--render_depth")

        completed = run_method_command(
            command,
            cwd=self.runtime_dir,
            env={"CUDA_VISIBLE_DEVICES": cuda_visible_devices},
            python_paths=[self.runtime_dir],
            timeout=timeout,
        )
        ply_outputs = [output_root / "gaussians.ply", output_root / "points.ply"]
        require_outputs(ply_outputs)
        rendered = sorted((output_root / "rendered").glob("*.mp4")) if (output_root / "rendered").exists() else []
        result = {
            "output_dir": str(output_root),
            "gaussians_path": str(output_root / "gaussians.ply"),
            "points_path": str(output_root / "points.ply"),
            "stdout": completed.stdout,
        }
        if rendered:
            result["rendered_video_paths"] = [str(path) for path in rendered]
        return result
