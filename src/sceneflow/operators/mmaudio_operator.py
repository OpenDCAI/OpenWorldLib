from .base_operator import BaseOperator
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from fractions import Fraction

import numpy as np
import torch
import av
from PIL import Image
from torchvision.transforms import v2

log = logging.getLogger()


_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


@dataclass
class VideoInfo:
    duration_sec: float
    fps: Fraction
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    all_frames: Optional[list[np.ndarray]]

    @property
    def height(self):
        return self.all_frames[0].shape[0]

    @property
    def width(self):
        return self.all_frames[0].shape[1]

    @classmethod
    def from_image_info(cls, image_info: 'ImageInfo', duration_sec: float,
                        fps: Fraction) -> 'VideoInfo':
        num_frames = int(duration_sec * fps)
        all_frames = [image_info.original_frame] * num_frames
        return cls(duration_sec=duration_sec,
                   fps=fps,
                   clip_frames=image_info.clip_frames,
                   sync_frames=image_info.sync_frames,
                   all_frames=all_frames)


@dataclass
class ImageInfo:
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    original_frame: Optional[np.ndarray]

    @property
    def height(self):
        return self.original_frame.shape[0]

    @property
    def width(self):
        return self.original_frame.shape[1]



class MMAudioOperator(BaseOperator):
    """
    MMAudio 数据处理 Operator
    
    负责视频加载、文本处理、数据验证等数据预处理工作
    不涉及模型推理
    """
    
    def __init__(
        self, 
        operation_types: list = None
    ):
        """
        初始化 MMAudioOperator
        
        Args:
            operation_types: 操作类型列表
        """
        if operation_types is None:
            operation_types = ["video_processing", "text_processing"]
        super().__init__(operation_types=operation_types)
        
    @staticmethod
    def _read_frames(
        video_path: Path, 
        list_of_fps: list[float], 
        start_sec: float, 
        end_sec: float,
        need_all_frames: bool
    ) -> tuple[list[np.ndarray], list[np.ndarray], Fraction]:
        output_frames = [[] for _ in list_of_fps]
        next_frame_time_for_each_fps = [0.0 for _ in list_of_fps]
        time_delta_for_each_fps = [1 / fps for fps in list_of_fps]
        all_frames = []

        # container = av.open(video_path)
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps = stream.guessed_rate
            stream.thread_type = 'AUTO'
            for packet in container.demux(stream):
                for frame in packet.decode():
                    frame_time = frame.time
                    if frame_time < start_sec:
                        continue
                    if frame_time > end_sec:
                        break

                    frame_np = None
                    if need_all_frames:
                        frame_np = frame.to_ndarray(format='rgb24')
                        all_frames.append(frame_np)

                    for i, _ in enumerate(list_of_fps):
                        this_time = frame_time
                        while this_time >= next_frame_time_for_each_fps[i]:
                            if frame_np is None:
                                frame_np = frame.to_ndarray(format='rgb24')

                            output_frames[i].append(frame_np)
                            next_frame_time_for_each_fps[i] += time_delta_for_each_fps[i]

        output_frames = [np.stack(frames) for frames in output_frames]
        return output_frames, all_frames, fps

    def load_video(self, video_path: Union[str, Path], duration_sec: float, load_all_frames: bool = True) -> VideoInfo:
        """
        加载视频文件并处理为模型所需格式
        
        Args:
            video_path: 视频文件路径
            duration_sec: 音频持续时间（秒）
            load_all_frames: 是否加载所有帧（用于后续视频合成）
            
        Returns:
            VideoInfo: 包含 clip_frames, sync_frames, all_frames 等信息
        """
        video_path = Path(video_path) if isinstance(video_path, str) else video_path
        clip_transform = v2.Compose([
            v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        output_frames, all_frames, orig_fps = self._read_frames(
            video_path,
            list_of_fps=[_CLIP_FPS, _SYNC_FPS],
            start_sec=0,
            end_sec=duration_sec,
            need_all_frames=load_all_frames
        )

        clip_chunk, sync_chunk = output_frames
        clip_chunk = torch.from_numpy(clip_chunk).permute(0, 3, 1, 2)
        sync_chunk = torch.from_numpy(sync_chunk).permute(0, 3, 1, 2)

        clip_frames = clip_transform(clip_chunk)
        sync_frames = sync_transform(sync_chunk)

        clip_length_sec = clip_frames.shape[0] / _CLIP_FPS
        sync_length_sec = sync_frames.shape[0] / _SYNC_FPS

        if clip_length_sec < duration_sec:
            log.warning(f'Clip video is too short: {clip_length_sec:.2f} < {duration_sec:.2f}')
            log.warning(f'Truncating to {clip_length_sec:.2f} sec')
            duration_sec = clip_length_sec

        if sync_length_sec < duration_sec:
            log.warning(f'Sync video is too short: {sync_length_sec:.2f} < {duration_sec:.2f}')
            log.warning(f'Truncating to {sync_length_sec:.2f} sec')
            duration_sec = sync_length_sec

        clip_frames = clip_frames[:int(_CLIP_FPS * duration_sec)]
        sync_frames = sync_frames[:int(_SYNC_FPS * duration_sec)]

        video_info = VideoInfo(
            duration_sec=duration_sec,
            fps=orig_fps,
            clip_frames=clip_frames,
            sync_frames=sync_frames,
            all_frames=all_frames if load_all_frames else None,
        )
        return video_info


    def check_interaction(
        self, 
        video_path: Optional[Union[str, Path]], 
        prompt: str, 
        negative_prompt: str, 
        duration: float,
        **kwargs
    ):
        """
        验证输入参数的合法性
        
        Args:
            video_path: 视频路径
            prompt: 文本提示
            negative_prompt: 负面提示
            duration: 持续时间
        """
        if video_path is not None:
            video_path = Path(video_path) if isinstance(video_path, str) else video_path
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be string, got {type(prompt)}")
            
        if not isinstance(negative_prompt, str):
            raise TypeError(f"Negative prompt must be string, got {type(negative_prompt)}")
    
    def process_text(
        self, 
        prompt: str, 
        negative_prompt: str
    ) -> tuple[list[str], list[str]]:
        """
        处理文本输入
        
        Args:
            prompt: 文本提示
            negative_prompt: 负面提示
            
        Returns:
            (prompts, negative_prompts): 转换为列表格式
        """
        prompts = [prompt] if isinstance(prompt, str) else prompt
        neg_prompts = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        return prompts, neg_prompts

    def get_interaction(
        self,
        video: Optional[Union[str, Path]] = None,
        prompt: str = "",
        negative_prompt: str = "",
        duration: float = 8.0,
        mask_away_clip: bool = False,
        load_all_frames: bool = True,
        **kwargs
    ):
        """
        将一次交互（视频 + 文本 + 其他控制参数）写入 current_interaction。

        与其它 Operator 保持一致的交互获取逻辑：
        - 先通过 check_interaction 做合法性检查；
        - 再把本次交互保存到 self.current_interaction 中，供 process_interaction 使用。
        """
        # 先做合法性检查
        self.check_interaction(video, prompt, negative_prompt, duration, **kwargs)

        # 规范化 video 路径，避免后续重复判断
        if video is not None and isinstance(video, str):
            video = Path(video)

        interaction: Dict[str, Any] = {
            "video": video,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "duration": duration,
            "mask_away_clip": mask_away_clip,
            "load_all_frames": load_all_frames,
            "extra_kwargs": kwargs,
        }

        self.current_interaction.append(interaction)
    
    def process_perception(
        self, 
        video: Optional[Union[str, Path]] = None,
        prompt: str = "",
        negative_prompt: str = "",
        duration: float = 8.0,
        mask_away_clip: bool = False,
        load_all_frames: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理感知输入，生成模型所需的输入格式。

        支持两种使用方式：
        1）外部直接传入 video / prompt 等参数，本函数内部会调用 get_interaction 记录本次交互；
        2）外部先多次调用 get_interaction 手动累积交互，这里不再显式传参，直接处理 current_interaction 中最后一条。

        Args:
            video: 视频文件路径（可选）
            prompt: 文本提示
            negative_prompt: 负面提示
            duration: 音频持续时间（秒）
            mask_away_clip: 是否屏蔽 CLIP 特征
            load_all_frames: 是否加载所有帧（用于后续视频合成）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含处理后的输入数据：
                - clip_frames: CLIP 视频帧 [1, T, C, H, W] 或 None
                - sync_frames: Sync 视频帧 [1, T, C, H, W] 或 None
                - prompt: 文本提示列表
                - negative_prompt: 负面提示列表
                - duration: 实际音频时长
                - video_info: VideoInfo 对象（如果有视频）
        """
        # 如果显式传入了参数，则先登记本次交互
        if any([video is not None, bool(prompt), bool(negative_prompt), bool(kwargs)]):
            self.get_interaction(
                video=video,
                prompt=prompt,
                negative_prompt=negative_prompt,
                duration=duration,
                mask_away_clip=mask_away_clip,
                load_all_frames=load_all_frames,
                **kwargs,
            )

        # 保证当前至少有一条待处理交互
        if not self.current_interaction:
            raise ValueError("No interaction to process, please call get_interaction first.")

        # 取出最近一次交互并记录到历史
        interaction = self.current_interaction[-1]
        self.interaction_history.append(interaction)

        # 从交互中解包参数
        video = interaction["video"]
        prompt = interaction["prompt"]
        negative_prompt = interaction["negative_prompt"]
        duration = interaction["duration"]
        mask_away_clip = interaction["mask_away_clip"]
        load_all_frames = interaction["load_all_frames"]
        extra_kwargs = interaction.get("extra_kwargs", {}) or {}

        # 处理视频（如果有）
        video_info = None
        if video is not None:
            log.info(f"Loading video from {video}")
            video_info = self.load_video(video, duration, load_all_frames=load_all_frames)
            clip_frames = video_info.clip_frames
            sync_frames = video_info.sync_frames
            duration = video_info.duration_sec
            
            # 处理 mask_away_clip 选项
            if mask_away_clip:
                clip_frames = None
                log.info("Masking away CLIP features")
            else:
                clip_frames = clip_frames.unsqueeze(0)
            sync_frames = sync_frames.unsqueeze(0)
        else:
            log.info("No video provided -- text-to-audio mode")
            clip_frames = None
            sync_frames = None
        
        # 处理文本
        prompts, neg_prompts = self.process_text(prompt, negative_prompt)
        
        # 构建返回数据
        processed_data = {
            "clip_frames": clip_frames,
            "sync_frames": sync_frames,
            "prompt": prompts,
            "negative_prompt": neg_prompts,
            "duration": duration,
            "video_info": video_info,  # 保存以便后续视频合成
        }

        processed_data.update(extra_kwargs)

        return processed_data