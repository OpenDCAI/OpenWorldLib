# configs/base_config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class BaseConfig:
    # 任务大类，确定默认的metrics
    task_name: str
    # 任务子类
    task_type: str
    model_name: str

    input_data_path: str
    annotation_file: str

    # 可选, 我希望的是根据task_type直接定测评指标，也就是在task里面封装
    metrics: Optional[List[str]] = None

    result_dir: str = "results/output"
    device: str = "cuda"

    # 允许任务的参数传进来
    task_args: Dict[str, Any] = field(default_factory=dict)
    model_args: Dict[str, Any] = field(default_factory=dict)