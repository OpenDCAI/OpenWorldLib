"""
VLA LIBERO Benchmark Configuration

This benchmark evaluates VLA models on LIBERO robotic manipulation tasks.
"""

info = {
    "input_keys": [
        "main_view",      # 主视角图片路径
        "wrist_view",     # 手腕视角图片路径
        "raw_state",      # 机器人初始状态
        "task",           # 任务描述
        "robot_type"      # 机器人类型
    ],
    "output_keys": ["generated_actions"],
    "perception_data_path": "test_images/",
    "metadata_path": "metadata.jsonl",
}

benchmarks = {
    "vla_libero_test": info,
}
