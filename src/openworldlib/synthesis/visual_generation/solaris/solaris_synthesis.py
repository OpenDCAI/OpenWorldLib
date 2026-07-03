from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

from ...base_synthesis import BaseSynthesis
from ...runtime_utils import ensure_path, package_dir, require_outputs, run_method_command


class SolarisSynthesis(BaseSynthesis):
    def __init__(
        self,
        pretrained_model_dir: str,
        dataset_dir: str,
        python_bin: str = sys.executable,
    ):
        super().__init__()
        self.runtime_dir = package_dir(__file__)
        self.pretrained_model_dir = str(ensure_path(pretrained_model_dir, "Solaris pretrained model directory", expect_file=False))
        self.dataset_dir = str(ensure_path(dataset_dir, "Solaris dataset directory", expect_file=False))
        self.python_bin = python_bin

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        dataset_dir: str,
        python_bin: str = sys.executable,
        **kwargs,
    ) -> "SolarisSynthesis":
        return cls(pretrained_model_dir=pretrained_model_path, dataset_dir=dataset_dir, python_bin=python_bin)

    def api_init(self, api_key, endpoint):
        raise NotImplementedError("Solaris is a local checkpoint pipeline and does not expose an API backend.")

    def predict(
        self,
        output_dir: str,
        eval_num_samples: int = 1,
        eval_datasets=None,
        num_frames_eval=None,
        experiment_name: str = "solaris",
        cuda_visible_devices: str = "0",
        eval_metrics: str = "fid",
        timeout: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, object]:
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_root / "checkpoint"
        jax_cache_dir = output_root / "jax_cache"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        jax_cache_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_bin,
            "src/inference.py",
            f"experiment_name={experiment_name}",
            f"device.eval_num_samples={int(eval_num_samples)}",
            f"device.data_dir={self.dataset_dir}",
            f"device.pretrained_model_dir={self.pretrained_model_dir}",
            f"device.output_dir={output_root}",
            f"device.checkpoint_dir={checkpoint_dir}",
            f"device.jax_cache_dir={jax_cache_dir}",
        ]
        if num_frames_eval is not None:
            command.append(f"runner.num_frames_eval={int(num_frames_eval)}")
        if eval_datasets:
            selected = {eval_datasets} if isinstance(eval_datasets, str) else set(eval_datasets)
            default_datasets = {
                "eval_structure",
                "eval_translation",
                "eval_rotation",
                "eval_turn_to_look",
                "eval_turn_to_look_opposite",
                "eval_both_look_away",
                "eval_one_looks_away",
            }
            unknown = selected - default_datasets
            if unknown:
                raise ValueError(f"Unknown Solaris eval datasets: {sorted(unknown)}")
            for dataset_name in sorted(default_datasets - selected):
                command.append(f"~eval_datasets.{dataset_name}")
        completed = run_method_command(
            command,
            cwd=self.runtime_dir,
            env={
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
                "SOLARIS_EVAL_METRICS": eval_metrics,
            },
            python_paths=[self.runtime_dir],
            timeout=timeout,
        )
        outputs = sorted(output_root.glob(f"{experiment_name}/**/*.mp4"))
        if not outputs:
            require_outputs([output_root / experiment_name / "eval_structure" / "video_0_side_by_side.mp4"])
        require_outputs(outputs)
        return {"video_paths": [str(path) for path in outputs], "stdout": completed.stdout}
