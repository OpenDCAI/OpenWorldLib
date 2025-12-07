import sys 
sys.path.append("..") 
from src.sceneflow.pipelines.wan.pipeline_wan_2p5 import Wan2p5Pipeline
from http import HTTPStatus
import os
import requests


def download_file(url: str, save_path: str, chunk_size: int = 1024 * 1024) -> None:
    """下载远程文件到本地"""
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)


def process_and_save_result(result, output_dir="./output/wan25", save_content=True):
    """
    处理并保存 API 调用结果
    
    Args:
        result: pipeline 返回的结果字典，包含 task_type, prompt, response
        output_dir: 输出目录
        save_content: 是否保存内容
        
    Returns:
        处理后的结果字典，包含 output_path（如果保存）
    """
    response = result["response"]
    task_type = result["task_type"]
    
    # 检查响应状态
    if response.status_code != HTTPStatus.OK:
        print(f"API调用失败，状态码: {response.status_code}")
        if hasattr(response, 'message'):
            print(f"错误信息: {response.message}")
        return result
    
    print(f"API调用成功，状态码: {response.status_code}")
    
    video_file_path = None

    # 保存结果
    if save_content:
        os.makedirs(output_dir, exist_ok=True)
        
        video_url = None
        task_id = None
        if hasattr(response, 'output') and response.output:
            output_data = response.output
            video_url = output_data.get('video_url')
            task_id = output_data.get('task_id')
        
        if video_url:
            video_filename = f"wan25_{task_type}_{task_id or 'result'}.mp4"
            video_file_path = os.path.join(output_dir, video_filename)
            download_file(video_url, video_file_path)
            print(f"视频已保存到: {video_file_path}")
        
    
    return video_file_path


image_path = "./data/test_case1/ref_image.png"
test_prompt = "An old-fashioned European village with thatched roofs on the houses."
output_dir = "./output/wan25"

wan25_pipeline = Wan2p5Pipeline.api_init(
    endpoint="your endpoint", 
    api_key="your apikey"
)

result = wan25_pipeline(
    prompt=test_prompt,
    reference_image=image_path,  # 提供图像则自动使用 i2av
)

# 处理并保存结果
video_file_path = process_and_save_result(result, output_dir=output_dir, save_content=True)

