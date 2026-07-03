from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from ...operators.hunyuan_worldplay2_operator import HunyuanWorldPlay2Operator
from ...synthesis.visual_generation.hunyuan_world.hunyuan_worldplay2_synthesis import HunyuanWorldPlay2Synthesis


class HunyuanWorldPlay2Pipeline:
    def __init__(
        self,
        operators: Optional[HunyuanWorldPlay2Operator] = None,
        synthesis_model: Optional[HunyuanWorldPlay2Synthesis] = None,
        device: str = "cuda",
    ) -> None:
        self.operators = operators
        self.synthesis_model = synthesis_model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "tencent/HY-World-2.0",
        subfolder: str = "HY-WorldMirror-2.0",
        python_bin: str = sys.executable,
        device: str = "cuda",
        **kwargs,
    ) -> "HunyuanWorldPlay2Pipeline":
        return cls(
            operators=HunyuanWorldPlay2Operator(),
            synthesis_model=HunyuanWorldPlay2Synthesis.from_pretrained(
                pretrained_model_path=model_path,
                subfolder=subfolder,
                python_bin=python_bin,
            ),
            device=device,
        )

    def process(self, target_size: int = 952, video_max_frames: int = 32) -> Dict[str, Any]:
        if self.operators is None:
            raise ValueError("operators must be provided")
        interaction = self.operators.process_interaction(
            target_size=target_size,
            video_max_frames=video_max_frames,
        )
        self.operators.delete_last_interaction()
        return interaction

    def __call__(self, input_path: str, output_dir: str, target_size: int = 952, video_max_frames: int = 32, **kwargs):
        if self.synthesis_model is None:
            raise ValueError("synthesis_model must be provided")
        processed = self.process(
            target_size=target_size,
            video_max_frames=video_max_frames,
        )
        return self.synthesis_model.predict(
            input_path=input_path,
            output_dir=output_dir,
            target_size=processed["target_size"],
            video_max_frames=processed["video_max_frames"],
            **kwargs,
        )
