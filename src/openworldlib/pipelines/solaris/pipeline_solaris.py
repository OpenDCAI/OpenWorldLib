from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from ...operators.solaris_operator import SolarisOperator
from ...synthesis.visual_generation.solaris.solaris_synthesis import SolarisSynthesis


class SolarisPipeline:
    def __init__(
        self,
        operators: Optional[SolarisOperator] = None,
        synthesis_model: Optional[SolarisSynthesis] = None,
        device: str = "cuda",
    ) -> None:
        self.operators = operators
        self.synthesis_model = synthesis_model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dataset_dir: Optional[str] = None,
        python_bin: str = sys.executable,
        device: str = "cuda",
        required_components: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> "SolarisPipeline":
        if required_components is not None and dataset_dir is None:
            dataset_dir = required_components.get("dataset_dir")
        if dataset_dir is None:
            raise ValueError("dataset_dir must be provided directly or through required_components.")
        return cls(
            operators=SolarisOperator(),
            synthesis_model=SolarisSynthesis.from_pretrained(
                pretrained_model_path=model_path,
                dataset_dir=dataset_dir,
                python_bin=python_bin,
            ),
            device=device,
        )

    def process(self, eval_num_samples: int = 1) -> Dict[str, Any]:
        if self.operators is None:
            raise ValueError("operators must be provided")
        interaction = self.operators.process_interaction(eval_num_samples=eval_num_samples)
        self.operators.delete_last_interaction()
        return interaction

    def __call__(
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
    ):
        if self.synthesis_model is None:
            raise ValueError("synthesis_model must be provided")
        processed = self.process(eval_num_samples=eval_num_samples)
        return self.synthesis_model.predict(
            output_dir=output_dir,
            eval_num_samples=processed["eval_num_samples"],
            eval_datasets=eval_datasets,
            num_frames_eval=num_frames_eval,
            experiment_name=experiment_name,
            cuda_visible_devices=cuda_visible_devices,
            eval_metrics=eval_metrics,
            timeout=timeout,
            **kwargs,
        )
