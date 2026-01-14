import os
import numpy as np
from PIL import Image

from src.sceneflow.pipelines.kling.pipeline_astra import AstraPipeline

def test():
    # 模型路径
    DIT_PATH = "../models/Astra/checkpoints/diffusion_pytorch_model.ckpt"
    WAN_MODEL_PATH = "../models/Wan-AI/Wan2.1-T2V-1.3B"
    
    # 输入素材
    CONDITION_IMAGE = "data.test_case1.ref_image.png"
    
    # 交互控制
    PROMPT = "A first-person view walking through a beautiful garden."
    DIRECTION = "left"  # 可选: forward, backward, left, right, forward_left, s_curve
    
    # 输出路径
    OUTPUT_PATH = "./results/astra_test_clean.mp4"

    print("Initializing Astra Pipeline...")
    
    pipeline = AstraPipeline.from_pretrained(
        dit_path=DIT_PATH,
        wan_model_path=WAN_MODEL_PATH,
        device="cuda"
    )

    print("Running Inference...")
    
    # 执行生成
    # 所有的复杂参数（步数、CFG、滑动窗口）都在 Pipeline 的 DefaultConfig 里设置
    pipeline.process(
        condition_image=CONDITION_IMAGE,
        prompt=PROMPT,
        direction=DIRECTION,
        output_path=OUTPUT_PATH
    )

    print(f"Test finished! Video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    test()