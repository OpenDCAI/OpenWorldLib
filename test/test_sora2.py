import sys 
sys.path.append("..") 
import os
import imageio
from src.sceneflow.pipelines.sora.pipeline_sora2 import Sora2Pipeline
import os
import time

def save_video(client, video, task_type, output_dir="./output/sora2", filename_prefix="sora2"):
    os.makedirs(output_dir, exist_ok=True)
    content = client.videos.download_content(video.id, variant="video")
    video_path = os.path.join(output_dir, f"{filename_prefix}_{task_type}.mp4")
    content.write_to_file(video_path)
    print(f"Saved video to: {video_path}")
    return video_path


# 配置参数
image_path = "./data/test_case1/ref_image.png"
test_prompt = "An old-fashioned European village with thatched roofs on the houses."
output_dir = "./output/sora2"

pipeline = Sora2Pipeline.api_init(
    endpoint="https://api.openai.com/v1", 
    api_key="your api key"
)

# 自动判断任务类型，pipeline 内部轮询等待完成
result = pipeline(
    prompt=test_prompt,
    # reference_image=image_path,
    wait=True  # 轮询在 pipeline 内完成
)

save_video(
    pipeline.get_synthesis_model().client,
    result["response"],
    result["task_type"],
    output_dir=output_dir,
    filename_prefix="sora2"
)
