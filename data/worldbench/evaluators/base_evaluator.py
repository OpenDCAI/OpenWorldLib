# evaluator/base_evaluator.py
import json
from core import METRIC_REGISTRY
from metrics.metric_groups import TASK_METRICS

class BaseEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.task_type = cfg.task_type

        self.metric_names = self._resolve_metrics()
        self.metrics = [
            METRIC_REGISTRY.get(name)(cfg)
            for name in self.metric_names
        ]

    def _resolve_metrics(self):
        """
        Resolve final metric list:
        task default + optional override
        """
        default_metrics = TASK_METRICS.get(self.task_type, [])

        # 完全 override（debug / ablation）
        if self.cfg.metrics is not None:
            return self.cfg.metrics

        return default_metrics

    def load_predictions(self, pred_file):
        with open(pred_file) as f:
            return [json.loads(l) for l in f]

    def evaluate(self, pred_file):
        results = self.load_predictions(pred_file)
        final_scores = {}

        for metric in self.metrics:
            score_dict = metric.compute(results)
            final_scores.update(score_dict)

        return final_scores
