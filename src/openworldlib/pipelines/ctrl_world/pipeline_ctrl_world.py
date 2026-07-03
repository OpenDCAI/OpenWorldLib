from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, Optional, Union

from ...operators.ctrl_world_operator import CtrlWorldOperator
from ...synthesis.vla_generation.ctrl_world_synthesis import CtrlWorldSynthesis


class CtrlWorldPipeline:
    def __init__(
        self,
        operators: Optional[CtrlWorldOperator] = None,
        synthesis_model: Optional[CtrlWorldSynthesis] = None,
        device: str = "cuda",
    ) -> None:
        self.operators = operators
        self.synthesis_model = synthesis_model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        svd_model_path: Optional[str] = None,
        clip_model_path: Optional[str] = None,
        dataset_root_path: Optional[str] = None,
        dataset_meta_info_path: Optional[str] = None,
        python_bin: str = sys.executable,
        device: str = "cuda",
        required_components: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> "CtrlWorldPipeline":
        if required_components is not None:
            svd_model_path = required_components.get("svd_model_path", svd_model_path)
            clip_model_path = required_components.get("clip_model_path", clip_model_path)
            dataset_root_path = required_components.get("dataset_root_path", dataset_root_path)
            dataset_meta_info_path = required_components.get("dataset_meta_info_path", dataset_meta_info_path)
        if svd_model_path is None or clip_model_path is None:
            raise ValueError("svd_model_path and clip_model_path must be provided directly or through required_components.")
        return cls(
            operators=CtrlWorldOperator(),
            synthesis_model=CtrlWorldSynthesis.from_pretrained(
                pretrained_model_path=model_path,
                svd_model_path=svd_model_path,
                clip_model_path=clip_model_path,
                dataset_root_path=dataset_root_path,
                dataset_meta_info_path=dataset_meta_info_path,
                python_bin=python_bin,
            ),
            device=device,
        )

    def process(self, interactions: Union[Iterable[str], str]) -> Dict[str, Any]:
        if self.operators is None:
            raise ValueError("operators must be provided")
        self.operators.get_interaction(interactions)
        interaction = self.operators.process_interaction()
        self.operators.delete_last_interaction()
        return interaction

    def __call__(
        self,
        interactions: Union[Iterable[str], str] = "ddcu",
        task_type: str = "keyboard",
        output_dir: str = "./output/ctrl_world",
        cuda_visible_devices: str = "0",
        timeout: Optional[int] = None,
        **kwargs,
    ):
        if self.synthesis_model is None:
            raise ValueError("synthesis_model must be provided")
        processed = self.process(interactions=interactions)
        return self.synthesis_model.predict(
            keyboard=processed["keyboard"],
            task_type=task_type,
            output_dir=output_dir,
            cuda_visible_devices=cuda_visible_devices,
            timeout=timeout,
            **kwargs,
        )
