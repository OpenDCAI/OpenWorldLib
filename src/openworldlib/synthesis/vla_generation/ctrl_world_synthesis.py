from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

from ..base_synthesis import BaseSynthesis
from ..runtime_utils import ensure_path, package_dir, require_outputs, run_method_command


class CtrlWorldSynthesis(BaseSynthesis):
    def __init__(
        self,
        ckpt_path: str,
        svd_model_path: str,
        clip_model_path: str,
        dataset_root_path: Optional[str] = None,
        dataset_meta_info_path: Optional[str] = None,
        python_bin: str = sys.executable,
    ):
        super().__init__()
        self.runtime_dir = package_dir(__file__, "ctrl_world")
        self.ckpt_path = str(ensure_path(ckpt_path, "Ctrl-World checkpoint", expect_file=True))
        self.svd_model_path = str(ensure_path(svd_model_path, "SVD model", expect_file=False))
        self.clip_model_path = str(ensure_path(clip_model_path, "CLIP model", expect_file=False))
        resolved_dataset_root = dataset_root_path or os.environ.get("CTRL_WORLD_DATASET_ROOT")
        resolved_dataset_meta = dataset_meta_info_path or os.environ.get("CTRL_WORLD_DATASET_META")
        if not resolved_dataset_root:
            raise ValueError("Ctrl-World dataset root must be provided via dataset_root_path or CTRL_WORLD_DATASET_ROOT.")
        if not resolved_dataset_meta:
            raise ValueError("Ctrl-World dataset meta must be provided via dataset_meta_info_path or CTRL_WORLD_DATASET_META.")
        self.dataset_root_path = str(ensure_path(resolved_dataset_root, "Ctrl-World dataset root", expect_file=False))
        self.dataset_meta_info_path = str(ensure_path(resolved_dataset_meta, "Ctrl-World dataset meta", expect_file=False))
        self.python_bin = python_bin

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        svd_model_path: str,
        clip_model_path: str,
        dataset_root_path: Optional[str] = None,
        dataset_meta_info_path: Optional[str] = None,
        python_bin: str = sys.executable,
        **kwargs,
    ) -> "CtrlWorldSynthesis":
        return cls(
            ckpt_path=pretrained_model_path,
            svd_model_path=svd_model_path,
            clip_model_path=clip_model_path,
            dataset_root_path=dataset_root_path,
            dataset_meta_info_path=dataset_meta_info_path,
            python_bin=python_bin,
        )

    def api_init(self, api_key, endpoint):
        raise NotImplementedError("Ctrl-World is a local checkpoint pipeline and does not expose an API backend.")

    def predict(
        self,
        keyboard: str = "ddcu",
        task_type: str = "keyboard",
        output_dir: str = "./output/ctrl_world",
        cuda_visible_devices: str = "0",
        timeout: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, str]:
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        script = "scripts/rollout_key_board.py" if task_type == "keyboard" else "scripts/rollout_replay_traj.py"
        command = [
            self.python_bin,
            script,
            "--dataset_root_path",
            self.dataset_root_path,
            "--dataset_meta_info_path",
            self.dataset_meta_info_path,
            "--dataset_names",
            "droid_subset",
            "--svd_model_path",
            self.svd_model_path,
            "--clip_model_path",
            self.clip_model_path,
            "--ckpt_path",
            self.ckpt_path,
            "--task_type",
            task_type,
            "--save_dir",
            str(output_root),
        ]
        if task_type == "keyboard":
            command.extend(["--keyboard", keyboard])

        completed = run_method_command(
            command,
            cwd=self.runtime_dir,
            env={
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
                "PATH": os.pathsep.join([str(Path(self.python_bin).resolve().parent), os.environ.get("PATH", "")]),
            },
            python_paths=[self.runtime_dir],
            timeout=timeout,
        )
        match = re.findall(r"Saving video to (.+?\.mp4)", completed.stdout)
        if not match:
            raise RuntimeError(f"Ctrl-World completed but did not report a video path.\n{completed.stdout}")
        output_path = str((self.runtime_dir / match[-1]).resolve())
        require_outputs([output_path])
        return {"video_path": output_path, "stdout": completed.stdout}
