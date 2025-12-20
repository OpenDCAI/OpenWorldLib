# worldbench/core/protocol.py
from typing import TypedDict, Any, Dict, List, Optional

# 定义标准的输出结构
class GenerationResult(TypedDict):
    """
    所有 Generator 的输出必须符合此结构，直接写入 jsonl。
    """
    
    # --- 1. 基础索引 ---
    sample_id: str          # 唯一 ID，必须与 Annotation 一致
    task_type: str          # e.g. "navigation", "video_edit"
    
    # --- 2. 核心结果 ---
    prediction: Any         # 模型输出 (路径/文本/动作)
    ground_truth: Any       # 标签 (用于直接对比)
    
    # --- 3. 上下文 (World Model 刚需) ---
    source_input: Optional[Any]   # 输入的原图/原视频 (用于一致性计算)
    
    # --- 4. 过程数据 (Agent/Reasoning 刚需) ---
    trajectory: Optional[List[Any]] # 轨迹/思维链
    action_history: Optional[List[str]] 

    meta: Dict[str, Any]    # 原始数据集的完整 meta 信息