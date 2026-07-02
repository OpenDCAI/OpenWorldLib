from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from ...operators.fantasy_world_operator import FantasyWorldOperator
from ...synthesis.visual_generation.fantasy_world.fantasy_world_synthesis import FantasyWorldSynthesis


class FantasyWorldPipeline:
    def __init__(
        self,
        operators: Optional[FantasyWorldOperator] = None,
        synthesis_model: Optional[FantasyWorldSynthesis] = None,
        device: str = "cuda",
    ) -> None:
        self.operators = operators
        self.synthesis_model = synthesis_model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        wan_ckpt_path: Optional[str] = None,
        required_components: Optional[Dict[str, str]] = None,
        python_bin: str = sys.executable,
        device: str = "cuda",
        **kwargs,
    ) -> "FantasyWorldPipeline":
        if required_components is not None and wan_ckpt_path is None:
            wan_ckpt_path = required_components.get("wan_ckpt_path")
        if wan_ckpt_path is None:
            raise ValueError("wan_ckpt_path must be provided directly or through required_components.")
        return cls(
            operators=FantasyWorldOperator(),
            synthesis_model=FantasyWorldSynthesis.from_pretrained(
                pretrained_model_path=model_path,
                wan_ckpt_path=wan_ckpt_path,
                python_bin=python_bin,
            ),
            device=device,
        )

    def process(self, image_path: str, camera_json_path: str, prompt: str) -> Dict[str, Any]:
        if self.operators is None:
            raise ValueError("operators must be provided")
        perception = self.operators.process_perception(image_path=image_path, camera_json_path=camera_json_path)
        self.operators.get_interaction(prompt)
        interaction = self.operators.process_interaction()
        self.operators.delete_last_interaction()
        return {
            **perception,
            **interaction,
        }

    def __call__(
        self,
        image_path: str,
        camera_json_path: str,
        prompt: str,
        output_dir: str,
        sample_steps: int = 50,
        frames: int = 17,
        using_scale: bool = True,
        cuda_visible_devices: str = "0",
        timeout: Optional[int] = None,
        **kwargs,
    ):
        if self.synthesis_model is None:
            raise ValueError("synthesis_model must be provided")
        processed = self.process(image_path=image_path, camera_json_path=camera_json_path, prompt=prompt)
        return self.synthesis_model.predict(
            image_path=processed["image_path"],
            camera_json_path=processed["camera_json_path"],
            prompt=processed["prompt"],
            output_dir=output_dir,
            sample_steps=sample_steps,
            frames=frames,
            using_scale=using_scale,
            cuda_visible_devices=cuda_visible_devices,
            timeout=timeout,
            **kwargs,
        )
