from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .segment import SegmentId


@dataclass
class Batch:
    obs: np.ndarray
    act: np.ndarray
    real_lengths: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obs": self.obs,
            "act": self.act,
            "real_lengths": self.real_lengths,
        }
