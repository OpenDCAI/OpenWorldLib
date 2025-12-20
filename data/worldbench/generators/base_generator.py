# generators/base_generator.py
from abc import ABC, abstractmethod
import os
import json
from typing import Any
from core import MODEL_REGISTRY, GenerationResult

class BaseGenerator(ABC):
    """
    Base class for all generation tasks (t2v / i2v / audio / edit ...)
    """

    def __init__(self, model_cfg, data_cfg, output_dir):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # init model
        model_cls = MODEL_REGISTRY.get(model_cfg["model_name"])
        self.model = model_cls(model_cfg)

    # ---------- dataset ----------
    def load_dataset(self):
        with open(self.data_cfg["annotation_file"], "r") as f:
            return [json.loads(l) for l in f]

    # ---------- hooks ----------
    @abstractmethod
    def build_inputs(self, sample: dict) -> dict:
        """sample -> model inputs"""
        pass

    def run_step(self, inputs: dict):
        """Single-step generation"""
        return self.model.generate(inputs)

    def postprocess(self, output: Any, sample: Dict[str, Any]) -> GenerationResult:
            return {
                "sample_id": str(sample.get("id")),
                "task_type": self.data_cfg.get("task_type"),

                "prediction": output,  
                "ground_truth": sample.get("ground_truth"), 
                "source_input": sample.get("source_input"),
                
                "trajectory": None,
                "action_history": None,
                
                "meta": sample
            }

    # ---------- main loop ----------
    def run(self):
        results = []

        for sample in self.load_dataset():
            inputs = self.build_inputs(sample)
            output = self.run_step(inputs)
            record = self.postprocess(output, sample)
            results.append(record)

        save_path = os.path.join(self.output_dir, "predictions.jsonl")
        with open(save_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        return save_path
