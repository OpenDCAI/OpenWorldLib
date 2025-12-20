# metrics/base_metric.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseMetric(ABC):
    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    @abstractmethod
    def compute_one(self, record: Dict[str, Any]) -> float:
        """
        Compute metric for a single sample.
        """
        pass

    def compute(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Default dataset-level evaluation.
        """
        values = []
        for r in results:
            try:
                values.append(self.compute_one(r))
            except Exception:
                continue

        return {
            self.__class__.__name__: sum(values) / max(len(values), 1)
        }
