import sys
import time
from multiprocessing import process
from PIL import Image
from typing import Optional, Union, Dict, Any

from ...operators.sora2_operator import Sora2Operator
from ...synthesis.visual_generation.sora.sora2.sora2_synthesis import Sora2Synthesis



class Sora2Pipeline:
    def __init__(
        self, 
        operator: Optional[Sora2Operator] = None,
        synthesis_model: Optional[Sora2Synthesis] = None,
        endpoint: str = "https://api.openai.com/v1", 
        api_key: str = "your_api_key"):
        """
        初始化 Sora2Pipeline
        Args:
            operator: Sora2 operator 实例（如果为None则自动创建）
            synthesis_model: Sora2 synthesis 模型实例（如果为None则自动创建）
            endpoint: API基础URL
            api_key: API密钥
        """
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.endpoint = endpoint
        self.api_key = api_key

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://api.openai.com/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ) -> 'Sora2Pipeline':
        """
        从配置加载完整的 pipeline
        
        Args:
            endpoint: API基础URL
            api_key: API密钥
            logger: 日志记录器
            **kwargs: 额外参数
        """
        if logger:
            logger.info(f"Loading Sora2 pipeline with endpoint: {endpoint}")
        
        # 加载 synthesis 模型
        if logger:
            logger.info("Loading Sora2 synthesis model...")
        synthesis_model = Sora2Synthesis.api_init(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
            **kwargs
        )
        
        if logger:
            logger.info("Initializing Sora2 operator...")
        operator = Sora2Operator()
        
        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            endpoint=endpoint,
            api_key=api_key
        )
        
        if logger:
            logger.info("Sora2 pipeline loaded successfully")
        
        return pipeline

    def process(
        self,
        prompt: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理输入，通过 operator 预处理后传给 synthesis 模型
        """
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        processed_data: Dict[str, Any] = {}

        # 视文本为交互输入通过process_interaction处理
        self.operator.get_interaction(prompt)
        processed_interaction = self.operator.process_interaction()
        processed_data['prompt'] = processed_interaction['processed_prompt']

        # 视图片为感知输入通过process_perception处理
        processed_perception = self.operator.process_perception(
            reference_image=reference_image,
            **kwargs
        )
        processed_data['encoded_image'] = processed_perception['encoded_image']
        processed_data['reference_image'] = processed_perception['reference_image']
        
        return processed_data

    def __call__(
        self,
        prompt: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
        size: str = "1280x720",
        duration: int = 4,
        task_type: str = "auto",
        wait: bool = True,
        poll_interval: int = 2,
        max_retries: int = 600,
        **kwargs
    ) -> Dict[str, Any]:
        """
        自动根据是否提供 reference_image 选择 T2V 或 I2V
        """

        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        
        # 使用 operator 预处理输入
        processed_data = self.process(
            prompt=prompt,
            reference_image=reference_image,
            **kwargs
        )
        
        # 使用 synthesis 模型的 predict 方法进行推理
        result = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            size=size,
            duration=duration,
            **kwargs
        )

        if wait:
            final_video = self._poll_video_status(
                video=result["response"],
                poll_interval=poll_interval,
                max_retries=max_retries
            )
            result["response"] = final_video

        return result

    def _poll_video_status(self, video: Any, poll_interval: int = 2, max_retries: int = 600) -> Any:
        """
        轮询视频生成状态，直到完成或失败，返回最终视频对象
        """
        if self.synthesis_model is None or not hasattr(self.synthesis_model, "client"):
            raise ValueError("Synthesis model client is not initialized")

        openai_client = self.synthesis_model.client

        completed_statuses = ("completed", "succeeded", "success", "done")
        retry_count = 0

        # 进度打印相关
        bar_length = 30
        progress_raw = getattr(video, "progress", 0)
        progress = progress_raw if progress_raw is not None else 0

        while (
            video.status.lower() not in [s.lower() for s in completed_statuses]
            and video.status.lower() not in ["failed", "error"]
        ):
            if retry_count >= max_retries:
                print(f"\nFailed to poll, reached the maximum number of retries ({max_retries})")
                return video

            try:
                video = openai_client.videos.retrieve(video.id)
            except Exception as e:
                print(f"\nFailed to get video status: {e}")
                time.sleep(poll_interval)
                retry_count += 1
                continue

            progress_raw = getattr(video, "progress", 0)
            progress = progress_raw if progress_raw is not None else 0
            status_lower = video.status.lower()

            filled_length = int((progress / 100) * bar_length) if progress is not None else 0
            bar = "=" * filled_length + "-" * (bar_length - filled_length)

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

            if status_lower in [s.lower() for s in completed_statuses]:
                break

            time.sleep(poll_interval)
            retry_count += 1

        sys.stdout.write("\n")

        if video.status.lower() in ["failed", "error"]:
            error_msg = getattr(getattr(video, "error", None), "message", "Unknown error")
            print(f"Failed to generate video: {error_msg}")

        return video

    def get_operator(self) -> Optional[Sora2Operator]:
        """获取 operator 实例"""
        return self.operator
    
    def get_synthesis_model(self) -> Optional[Sora2Synthesis]:
        """获取 synthesis 模型实例"""
        return self.synthesis_model