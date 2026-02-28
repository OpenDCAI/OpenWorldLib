from typing import Optional, Generator, List
import torch
from ..pipeline_utils import PipelineABC
from ...operators.hunyuan_worldplay_operator import HunyuanWorldPlayOperator
from ...synthesis.visual_generation.hunyuan_world.hunyuan_worldplay_synthesis import HunyuanWorldPlaySynthesis


class HunyuanWorldPlayPipeline(PipelineABC):
    def __init__(
        self,
        *,
        representation_model=None,
        reasoning_model=None,
        synthesis_model: Optional[HunyuanWorldPlaySynthesis] = None,
        operators=None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.operators = operators
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        *,
        representation_path: Optional[str] = None,
        reasoning_path: Optional[str] = None,
        synthesis_path: Optional[str] = None,
        transformer_version: str = '480p_i2v',
        device: str = 'cuda',
        create_sr_pipeline: bool = False,
        force_sparse_attn: bool = False,
        transformer_dtype=torch.bfloat16,
        enable_offloading: bool = None,
        enable_group_offloading: bool = None,
        overlap_group_offloading: bool = True,
        action_ckpt: Optional[str] = None,
        init_infer_state: bool = True,
        infer_state_kwargs: Optional[dict] = None,
        **kwargs
    ) -> 'HunyuanWorldPlayPipeline':
        """
        从预训练模型加载 Pipeline
        
        Args:
            representation_path: 表示模型路径（可选）
            reasoning_path: 推理模型路径（可选）
            synthesis_path: 合成模型路径
            transformer_version: transformer 版本（如 "480p_i2v"）
            device: 设备
            create_sr_pipeline: 是否创建超分辨率 pipeline
            force_sparse_attn: 是否强制使用稀疏注意力
            transformer_dtype: transformer 数据类型
            enable_offloading: 是否启用 offloading
            enable_group_offloading: 是否启用 group offloading
            overlap_group_offloading: 是否重叠 group offloading
            action_ckpt: action 模型检查点路径
            **kwargs: 其他参数
            
        Returns:
            HunyuanWorldPlayPipeline: Pipeline 实例
        """
        if init_infer_state:
            default_infer_state_kwargs = {
                "sage_blocks_range": "0-53",
                "use_sageattn": False,
                "enable_torch_compile": False,
                "use_fp8_gemm": False,
                "quant_type": "fp8-per-block",
                "include_patterns": "double_blocks",
                "use_vae_parallel": False,
            }
            if infer_state_kwargs is not None:
                default_infer_state_kwargs.update(infer_state_kwargs)
            cls.initialize_infer_state(**default_infer_state_kwargs)
        synthesis_model = HunyuanWorldPlaySynthesis.from_pretrained(
            synthesis_path,
            transformer_version,
            device=device,
            create_sr_pipeline=create_sr_pipeline,
            force_sparse_attn=force_sparse_attn,
            transformer_dtype=transformer_dtype,
            enable_offloading=enable_offloading,
            enable_group_offloading=enable_group_offloading,
            overlap_group_offloading=overlap_group_offloading,
            action_ckpt=action_ckpt,
            **kwargs
        )
        operators = HunyuanWorldPlayOperator()
        
        return cls(
            synthesis_model=synthesis_model,
            operators=operators,
            device=device
        )

    def process(self, *, input_, interaction, video_length: int):
        """
        处理输入
        
        Args:
            input_: 输入数据（图片、视频等）
            interaction: 交互信号
            
        Returns:
            处理后的数据
        """
        if self.operators is None:
            raise ValueError("operators must be provided")
        input_ = self.operators.process_perception(input_)
        self.operators.get_interaction(interaction)
        latent_frames = (video_length - 1) // 4 + 1
        operator_condition = self.operators.process_interaction(latent_frames=latent_frames)
        self.operators.delete_last_interaction()
        return {
            "reference_image": input_,
            "operator_condition": operator_condition,
        }

    @staticmethod
    def initialize_infer_state(**kwargs):
        from ...synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.commons.infer_state import (
            initialize_infer_state,
        )

        return initialize_infer_state(**kwargs)

    @staticmethod
    def save_video(video, path: str, fps: int = 24):
        import torch
        import einops
        import imageio

        if video.ndim == 5:
            if video.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got shape: {tuple(video.shape)}")
            video = video[0]
        vid = (video * 255).clamp(0, 255).to(torch.uint8)
        vid = einops.rearrange(vid, "c f h w -> f h w c")
        imageio.mimwrite(path, vid, fps=fps)

    def __call__(
        self,
        *,
        prompt: str,
        reference_image: Optional[str] = None,
        pose: Optional[str] = None,
        interaction_signal: Optional[str] = None,
        aspect_ratio: str = '16:9',
        video_length: int = 125,
        num_inference_steps: int = 4,
        negative_prompt: str = "",
        seed: int = 1,
        output_type: str = "pt",
        prompt_rewrite: bool = False,
        enable_sr: bool = False,
        sr_num_inference_steps: Optional[int] = None,
        return_pre_sr_video: bool = False,
        few_step: bool = True,
        chunk_latent_frames: int = 4,
        model_type: str = "ar",
        user_height: Optional[int] = None,
        user_width: Optional[int] = None,
        **kwargs
    ):
        """
        Pipeline 调用入口
        
        Args:
            prompt: 文本提示
            reference_image: 参考图像路径
            pose: 相机轨迹（如 "w-10, right-10, d-11"）
            aspect_ratio: 宽高比
            video_length: 视频长度
            num_inference_steps: 推理步数
            negative_prompt: 负面提示
            seed: 随机种子
            output_type: 输出类型
            prompt_rewrite: 是否重写提示
            enable_sr: 是否启用超分辨率
            sr_num_inference_steps: 超分辨率推理步数
            return_pre_sr_video: 是否返回超分辨率前的视频
            few_step: 是否使用少步推理
            chunk_latent_frames: chunk latent frames
            model_type: 模型类型（"ar" 或 "bi"）
            user_height: 用户指定高度
            user_width: 用户指定宽度
            **kwargs: 其他参数
            
        Returns:
            HunyuanVideoPipelineOutput: 包含生成的视频帧
        """
        pose_value = interaction_signal if interaction_signal is not None else pose
        if pose_value is None:
            raise ValueError("pose or interaction_signal must be provided")
        inferred_video_length = self.operators.infer_video_length(pose_value)
        if video_length != inferred_video_length:
            print(f"video_length {video_length} != inferred_video_length {inferred_video_length}, auto setting")
            video_length = inferred_video_length
        processed = self.process(
            input_=reference_image,
            interaction=pose_value,
            video_length=video_length,
        )
        operator_condition = processed["operator_condition"]
        
        output = self.synthesis_model(
            enable_sr=enable_sr,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            sr_num_inference_steps=sr_num_inference_steps,
            video_length=video_length,
            negative_prompt=negative_prompt,
            seed=seed,
            output_type=output_type,
            prompt_rewrite=prompt_rewrite,
            return_pre_sr_video=return_pre_sr_video,
            viewmats=operator_condition["viewmats"].unsqueeze(0),
            Ks=operator_condition["Ks"].unsqueeze(0),
            action=operator_condition["action"].unsqueeze(0),
            few_step=few_step,
            chunk_latent_frames=chunk_latent_frames,
            model_type=model_type,
            user_height=user_height,
            user_width=user_width,
            reference_image=processed["reference_image"],
            **kwargs
        )
        
        return output

    def stream(self, *args, **kwds) -> Generator[torch.Tensor, List[str], None]:
        """
        流式输出
        """
        pass

    def save_pretrained(self, save_directory: str):
        """
        保存模型（训练 pipeline 准备好后完成）
        """
        pass
