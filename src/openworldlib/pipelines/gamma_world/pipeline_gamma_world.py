from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from ...operators.gamma_world_operator import GammaWorldOperator
from ...synthesis.visual_generation.gamma_world.gamma_world_synthesis import GammaWorldSynthesis


class GammaWorldPipeline:
    def __init__(
        self,
        operators: Optional[GammaWorldOperator] = None,
        synthesis_model: Optional[GammaWorldSynthesis] = None,
        device: str = "cuda",
    ) -> None:
        self.operators = operators
        self.synthesis_model = synthesis_model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        required_components: Optional[Dict[str, str]] = None,
        vae: Optional[str] = None,
        text_encoder: Optional[str] = None,
        python_bin: str = sys.executable,
        device: str = "cuda",
        **kwargs,
    ) -> "GammaWorldPipeline":
        if required_components is not None:
            vae = required_components.get("vae", vae)
            text_encoder = required_components.get("text_encoder", text_encoder)
        return cls(
            operators=GammaWorldOperator(),
            synthesis_model=GammaWorldSynthesis.from_pretrained(
                pretrained_model_path=model_path,
                vae=vae,
                text_encoder=text_encoder,
                python_bin=python_bin,
            ),
            device=device,
        )

    def process(
        self,
        mode: str = "causal_few_step",
        n_players: int = 2,
        max_eval_samples: Optional[int] = 1,
    ) -> Dict[str, Any]:
        if self.operators is None:
            raise ValueError("operators must be provided")
        interaction = self.operators.process_interaction(
            mode=mode,
            n_players=n_players,
            max_eval_samples=max_eval_samples,
        )
        self.operators.delete_last_interaction()
        return interaction

    def __call__(
        self,
        output_dir: str,
        mode: str = "causal_few_step",
        n_players: int = 2,
        max_eval_samples: Optional[int] = 1,
        **kwargs,
    ):
        if self.synthesis_model is None:
            raise ValueError("synthesis_model must be provided")
        processed = self.process(
            mode=mode,
            n_players=n_players,
            max_eval_samples=max_eval_samples,
        )
        return self.synthesis_model.predict(
            output_dir=output_dir,
            mode=processed["mode"],
            n_players=processed["n_players"],
            max_eval_samples=processed["max_eval_samples"],
            **kwargs,
        )
