from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

from ...base_synthesis import BaseSynthesis
from ...runtime_utils import ensure_path, package_dir, require_outputs, run_method_command


class MemFlowSynthesis(BaseSynthesis):
    def __init__(
        self,
        checkpoint_dir: str,
        wan_model_path: str,
        python_bin: str = sys.executable,
    ):
        super().__init__()
        self.runtime_dir = package_dir(__file__)
        self.checkpoint_dir = str(ensure_path(checkpoint_dir, "MemFlow checkpoint directory", expect_file=False))
        self.wan_model_path = str(ensure_path(wan_model_path, "MemFlow Wan model", expect_file=False))
        self.python_bin = python_bin

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        wan_model_path: str,
        python_bin: str = sys.executable,
        **kwargs,
    ) -> "MemFlowSynthesis":
        return cls(checkpoint_dir=pretrained_model_path, wan_model_path=wan_model_path, python_bin=python_bin)

    def api_init(self, api_key, endpoint):
        raise NotImplementedError("MemFlow is a local checkpoint pipeline and does not expose an API backend.")

    def predict(
        self,
        prompt: str,
        output_dir: str,
        num_output_frames: int = 120,
        num_samples: int = 1,
        cuda_visible_devices: str = "0",
        timeout: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, str]:
        import yaml

        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="memflow_openworldlib_") as tmp:
            tmp_root = Path(tmp)
            wan_link = tmp_root / "wan_models" / "Wan2.1-T2V-1.3B"
            wan_link.parent.mkdir(parents=True, exist_ok=True)
            wan_link.symlink_to(Path(self.wan_model_path).expanduser().resolve(), target_is_directory=True)
            prompt_path = Path(tmp) / "prompt.txt"
            prompt_path.write_text(prompt.strip() + "\n", encoding="utf-8")
            config_path = Path(tmp) / "inference.yaml"
            with open(self.runtime_dir / "configs" / "inference.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            config["data_path"] = str(prompt_path)
            config["output_folder"] = str(output_root)
            config["inference_iter"] = 0
            config["num_output_frames"] = int(num_output_frames)
            config["num_samples"] = int(num_samples)
            config["generator_ckpt"] = str(Path(self.checkpoint_dir) / "base.pt")
            config["lora_ckpt"] = str(Path(self.checkpoint_dir) / "lora.pt")
            config["model_name"] = "Wan2.1-T2V-1.3B"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            completed = run_method_command(
                [self.python_bin, str(self.runtime_dir / "inference.py"), "--config_path", str(config_path)],
                cwd=tmp_root,
                env={"CUDA_VISIBLE_DEVICES": cuda_visible_devices},
                python_paths=[self.runtime_dir.parents[3], self.runtime_dir],
                timeout=timeout,
            )

        outputs = sorted(output_root.glob("rank*-0_regular.mp4")) or sorted(output_root.glob("*.mp4"))
        if not outputs:
            outputs = [output_root / "rank0-0-0_regular.mp4"]
        verified = require_outputs(outputs)
        return {"video_path": verified[-1], "stdout": completed.stdout}
