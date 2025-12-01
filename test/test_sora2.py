import sys 
sys.path.append("..") 
from src.sceneflow.pipelines.sora2.pipeline_sora2 import Sora2Pipeline
import os
import time


def wait_and_download_video(openai_client, video, output_dir="./output/sora2", filename_prefix="sora2"):
    """
    轮询视频生成状态，并在完成后下载视频
    
    Args:
        openai_client: OpenAI 客户端实例
        video: create() 返回的视频对象
        output_dir: 保存目录
        filename_prefix: 文件名前缀
        
    Returns:
        本地视频文件路径，或 None（失败时）
    """
    print("Video generation started")
    
    bar_length = 30
    progress_raw = getattr(video, "progress", 0)
    progress = progress_raw if progress_raw is not None else 0
    
    
    # 定义完成状态
    completed_statuses = ("completed", "succeeded", "success", "done")
    
    # 轮询直到完成或失败
    max_retries = 600  # 最多轮询 20 分钟（300 * 2秒）
    retry_count = 0
    
    while video.status.lower() not in [s.lower() for s in completed_statuses] and video.status.lower() not in ["failed", "error"]:
        if retry_count >= max_retries:
            print(f"\n Failed to poll, reached the maximum number of retries ({max_retries})")
            return None
            
        # 获取最新状态
        try:
            video = openai_client.videos.retrieve(video.id)
        except Exception as e:
            print(f"\n Failed to get video status: {e}")
            time.sleep(2)
            retry_count += 1
            continue
            
        # 获取进度，确保是数字类型
        progress_raw = getattr(video, "progress", 0)
        progress = progress_raw if progress_raw is not None else 0
        status_lower = video.status.lower()

        # 显示进度
        filled_length = int((progress / 100) * bar_length) if progress is not None else 0
        bar = "=" * filled_length + "-" * (bar_length - filled_length)
        
        # 根据状态显示不同的文本
        if status_lower == "queued":
            status_text = "Queued"
        elif status_lower in ["submitted", "not_start"]:
            status_text = "Submitted"
        elif status_lower in ["in_progress", "processing", "running"]:
            status_text = "Processing"
        else:
            status_text = video.status

        progress_display = f"{progress:.1f}%" if progress is not None else "N/A"
        sys.stdout.write(f"\r{status_text}: [{bar}] {progress_display} (Status: {video.status})")
        sys.stdout.flush()
        
        # 如果已完成，退出循环
        if status_lower in [s.lower() for s in completed_statuses]:
            break
            
        time.sleep(2)
        retry_count += 1

    sys.stdout.write("\n")  # 换行

    # 检查是否失败
    if video.status.lower() in ["failed", "error"]:
        error_msg = getattr(getattr(video, "error", None), "message", "Unknown error")
        print(f"Failed to generate video: {error_msg}")
        return None

    # 检查是否真正完成
    status_lower = video.status.lower()
    if status_lower not in [s.lower() for s in completed_statuses]:
        print(f"Video status is abnormal: {video.status}, trying to download...")


    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 下载视频内容（重试机制）
    max_download_retries = 5
    for attempt in range(max_download_retries):
        try:
            content = openai_client.videos.download_content(video.id, variant="video")
            video_path = os.path.join(output_dir, f"{filename_prefix}_{video.id}.mp4")
            content.write_to_file(video_path)
            print(f"Wrote video to: {video_path}")
            return video_path
        except Exception as e:
            error_str = str(e)
            # 如果任务还未完成，继续等待
            if "not completed" in error_str.lower() or "not_start" in error_str.lower():
                if attempt < max_download_retries - 1:
                    print(f"\n Task is not completed, waiting 5 seconds and retrying... (attempt {attempt + 1}/{max_download_retries})")
                    time.sleep(5)
                    # 重新获取状态
                    try:
                        video = openai_client.videos.retrieve(video.id)
                        print(f"Current status: {video.status}, progress: {getattr(video, 'progress', 0)}%")
                    except:
                        pass
                    continue
            print(f"Failed to download video (attempt {attempt + 1}/{max_download_retries}): {e}")
            if attempt == max_download_retries - 1:
                return None
            time.sleep(3)
    
    return None


# 配置参数
image_path = "./data/test_case1/ref_image.png"
test_prompt = "An old-fashioned European village with thatched roofs on the houses."
output_dir = "./output/sora2"

# 初始化 pipeline
sora2_pipeline = Sora2Pipeline.from_pretrained(
    base_url="https://api.openai.com/v1", 
    api_key="your api key"
)

# 使用 __call__ 方法（推荐，统一接口）
# 自动判断任务类型
result = sora2_pipeline(
    prompt=test_prompt,
    reference_image=image_path,  # 提供图像则自动使用 i2av
)

wait_and_download_video(
    sora2_pipeline.get_synthesis_model().client, 
    result["response"], 
    output_dir=output_dir, 
    filename_prefix="sora2"
)
