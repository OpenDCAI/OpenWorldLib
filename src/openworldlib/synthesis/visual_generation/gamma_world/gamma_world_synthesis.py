from __future__ import annotations

import sys
import glob
import os
from pathlib import Path
from typing import Dict, Optional

from ...base_synthesis import BaseSynthesis
from ...runtime_utils import ensure_path, package_dir, require_outputs, run_method_command


def _local_or_remote_path(path: Optional[str], name: str) -> Optional[str]:
    if path is None:
        return None
    if path.startswith("hf://"):
        return path
    return str(ensure_path(path, name))


class GammaWorldSynthesis(BaseSynthesis):
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        vae: Optional[str] = None,
        text_encoder: Optional[str] = None,
        python_bin: str = sys.executable,
    ):
        super().__init__()
        self.runtime_dir = package_dir(__file__)
        self.checkpoint = _local_or_remote_path(checkpoint, "Gamma-World checkpoint")
        self.vae = _local_or_remote_path(vae, "Gamma-World VAE")
        self.text_encoder = _local_or_remote_path(text_encoder, "Gamma-World text encoder")
        self.python_bin = python_bin

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str],
        vae: Optional[str] = None,
        text_encoder: Optional[str] = None,
        python_bin: str = sys.executable,
        **kwargs,
    ) -> "GammaWorldSynthesis":
        return cls(
            checkpoint=pretrained_model_path,
            vae=vae,
            text_encoder=text_encoder,
            python_bin=python_bin,
        )

    def api_init(self, api_key, endpoint):
        raise NotImplementedError("Gamma-World is a local checkpoint pipeline and does not expose an API backend.")

    def predict(
        self,
        output_dir: str,
        eval_dir: Optional[str] = None,
        mode: str = "causal_few_step",
        n_players: int = 2,
        max_eval_samples: Optional[int] = 1,
        prompt: Optional[str] = None,
        num_frames: int = 189,
        height: int = 320,
        width: int = 480,
        fps: int = 16,
        nproc_per_node: int = 1,
        cuda_visible_devices: str = "0",
        timeout: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, object]:
        default_eval_dir = self.runtime_dir.parents[4] / "data" / "test_case" / "gamma_world"
        eval_root = ensure_path(eval_dir or str(default_eval_dir), "Gamma-World eval data", expect_file=False)
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        command = [
            self.python_bin,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={int(nproc_per_node)}",
            "scripts/inference.py",
            "--mode",
            mode,
            "--eval-dir",
            str(eval_root),
            "--n-players",
            str(int(n_players)),
            "--output",
            str(output_root),
            "--num-frames",
            str(int(num_frames)),
            "--height",
            str(int(height)),
            "--width",
            str(int(width)),
            "--fps",
            str(int(fps)),
        ]
        if self.checkpoint:
            command.extend(["--checkpoint", self.checkpoint])
        if self.vae:
            command.extend(["--vae", self.vae])
        if self.text_encoder:
            command.extend(["--text-encoder", self.text_encoder])
        if max_eval_samples is not None:
            command.extend(["--max-eval-samples", str(int(max_eval_samples))])
        if prompt:
            command.extend(["--prompt", prompt])

        completed = run_method_command(
            command,
            cwd=self.runtime_dir,
            env={
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
                "LD_LIBRARY_PATH": self._library_path(),
                "CUDA_HOME": str(Path(self.python_bin).expanduser().resolve().parent.parent),
            },
            python_paths=[self.runtime_dir, self.runtime_dir / "packages" / "cosmos-oss"],
            timeout=timeout,
        )
        outputs = sorted(output_root.glob("**/generated.mp4"))
        require_outputs(outputs or [output_root / "generated.mp4"])
        return {"video_paths": [str(path) for path in outputs], "stdout": completed.stdout}

    def _library_path(self) -> str:
        env_root = Path(self.python_bin).expanduser().resolve().parent.parent
        nvidia_libs = glob.glob(str(env_root / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib"))
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        parts = nvidia_libs + ([existing] if existing else [])
        return ":".join(parts)
