import os
import sys
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import numpy as np
import imageio
from src.sceneflow.pipelines.kling.pipeline_astra import AstraPipeline
def export_to_video(video_frames, output_video_path, fps=20):
    """
    将 PIL.Image 列表保存为视频文件
    """
    print(f"Exporting video to {output_video_path}")
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # 转换为 numpy array 供 imageio 使用
    frames_np = [np.array(frame) for frame in video_frames]
    
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in frames_np:
            writer.append_data(frame)
    print("Export complete.")

def test():
    DIT_PATH = "EvanEternal/Astra" 
    
    # Wan2.1 的官方 ID 
    WAN_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B" 
    
    CONDITION_IMAGE = os.path.join(project_root, "data/test_case1/ref_image.png")
    
    # 交互控制
    interaction = {
        "prompt": "A sunlit European street lined with historic buildings and vibrant greenery creates a warm, charming, and inviting atmosphere. The scene shows a picturesque open square paved with red bricks, surrounded by classic narrow townhouses featuring tall windows, gabled roofs, and dark-painted facades. On the right side, a lush arrangement of potted plants and blooming flowers adds rich color and texture to the foreground. A vintage-style streetlamp stands prominently near the center-right, contributing to the timeless character of the street. Mature trees frame the background, their leaves glowing in the warm afternoon sunlight. Bicycles are visible along the edges of the buildings, reinforcing the urban yet leisurely feel. The sky is bright blue with scattered clouds, and soft sun flares enter the frame from the left, enhancing the scene’s inviting, peaceful mood.",
        "direction": "left" # 可选: forward, backward, left, right, forward_left, s_curve
    } 
    
    # 输出路径
    OUTPUT_PATH = "./results/astra_test_clean.mp4"

    print("Initializing Astra Pipeline...")
    
    pipeline = AstraPipeline.from_pretrained(
        dit_path=DIT_PATH,
        wan_model_path=WAN_MODEL_PATH,
        device="cuda",
        moe_num_experts=3 # 确保与模型权重匹配
    )

    print("Running Inference...")
    
    # 执行 Pipeline -> 获取 PIL Image 列表
    video_frames = pipeline(
        input_=CONDITION_IMAGE,
        interaction=interaction
    )

    # 用户自行保存
    export_to_video(video_frames, OUTPUT_PATH, fps=20)

    print(f"Test finished! Video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    test()