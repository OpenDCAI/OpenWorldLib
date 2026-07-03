from __future__ import annotations

from typing import Any, Dict, Optional

from ...operators.cosmos3_operator import Cosmos3Operator
from ...synthesis.visual_generation.cosmos.cosmos3_synthesis import Cosmos3Synthesis


class Cosmos3Pipeline:
    def __init__(
        self,
        operators: Optional[Cosmos3Operator] = None,
        synthesis_model: Optional[Cosmos3Synthesis] = None,
        device: str = "cuda",
    ) -> None:
        self.operators = operators
        self.synthesis_model = synthesis_model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "nvidia/Cosmos3-Nano",
        torch_dtype: str = "bfloat16",
        device: str = "cuda",
        **kwargs,
    ) -> "Cosmos3Pipeline":
        return cls(
            operators=Cosmos3Operator(),
            synthesis_model=Cosmos3Synthesis.from_pretrained(
                pretrained_model_path=model_path,
                torch_dtype=torch_dtype,
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

    def __call__(self, prompt: str, output_path: str, **kwargs):
        if self.synthesis_model is None:
            raise ValueError("synthesis_model must be provided")
        processed = self.process(prompt=prompt)
        return self.synthesis_model.predict(prompt=processed["prompt"], output_path=output_path, **kwargs)
