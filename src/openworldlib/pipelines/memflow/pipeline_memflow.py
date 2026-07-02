from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from ...operators.memflow_operator import MemFlowOperator
from ...synthesis.visual_generation.memflow.memflow_synthesis import MemFlowSynthesis


class MemFlowPipeline:
    def __init__(
        self,
        operators: Optional[MemFlowOperator] = None,
        synthesis_model: Optional[MemFlowSynthesis] = None,
        device: str = "cuda",
    ) -> None:
        self.operators = operators
        self.synthesis_model = synthesis_model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        wan_model_path: Optional[str] = None,
        required_components: Optional[Dict[str, str]] = None,
        python_bin: str = sys.executable,
        device: str = "cuda",
        **kwargs,
    ) -> "MemFlowPipeline":
        if required_components is not None and wan_model_path is None:
            wan_model_path = required_components.get("wan_model_path")
        if wan_model_path is None:
            raise ValueError("wan_model_path must be provided directly or through required_components.")
        return cls(
            operators=MemFlowOperator(),
            synthesis_model=MemFlowSynthesis.from_pretrained(
                pretrained_model_path=model_path,
                wan_model_path=wan_model_path,
                python_bin=python_bin,
            ),
            device=device,
        )

    def process(self, prompt: str) -> Dict[str, Any]:
        if self.operators is None:
            raise ValueError("operators must be provided")
        self.operators.get_interaction(prompt)
        interaction = self.operators.process_interaction()
        self.operators.delete_last_interaction()
        return interaction

    def __call__(
        self,
        prompt: str,
        output_dir: str,
        num_output_frames: int = 120,
        num_samples: int = 1,
        cuda_visible_devices: str = "0",
        timeout: Optional[int] = None,
        **kwargs,
    ):
        if self.synthesis_model is None:
            raise ValueError("synthesis_model must be provided")
        processed = self.process(prompt=prompt)
        return self.synthesis_model.predict(
            prompt=processed["prompt"],
            output_dir=output_dir,
            num_output_frames=num_output_frames,
            num_samples=num_samples,
            cuda_visible_devices=cuda_visible_devices,
            timeout=timeout,
            **kwargs,
        )
