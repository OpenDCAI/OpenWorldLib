from typing import Optional
from PIL import Image
import logging
from pathlib import Path
from ....base_models.diffusion_model.diffsynth import ModelManager, WanVideoPipeline
from ...base_synthesis import BaseSynthesis
from huggingface_hub import snapshot_download
import os
import torch


def ensure_in_channels_36_for_image_input(model=None):
    """确保模型支持36个输入通道（x:16 + y:20）用于图像输入模式。"""
    if hasattr(model, 'patch_embedding'):
        if hasattr(model.patch_embedding, 'module'):
            patch_emb = model.patch_embedding.module
        else:
            patch_emb = model.patch_embedding
        current_in_dim = patch_emb.weight.shape[1]
        if current_in_dim != 36:
            print(f"[INFO] 扩展 patch_embedding 输入通道数从 {current_in_dim} 到 36（用于图像输入模式）")
            original_weight = patch_emb.weight.data
            out_dims = list(original_weight.shape)
            out_channels = out_dims[0]
            in_channels = out_dims[1]
            remaining_shape = out_dims[2:]
            new_weight = torch.zeros((out_channels, 36, *remaining_shape), dtype=original_weight.dtype, device=original_weight.device)
            new_weight[:, :in_channels] = original_weight
            torch.nn.init.normal_(new_weight[:, in_channels:], mean=0.0, std=0.02)
            patch_emb.weight.data = new_weight
        else:
            print(f"[INFO] patch_embedding 输入通道数已经是 36")
    else:
        print("[WARN] 去噪模型没有 patch_embedding；请验证架构")


class WoWSynthesis(BaseSynthesis):
    
    def __init__(
        self, 
        pipeline,
        device='cuda'):

        super(WoWSynthesis, self).__init__()
        self.pipeline = pipeline
        self.device = device

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_path="WoW-world-model/WoW-1-Wan-1.3B-2M",
                        synthesis_args=None,
                        device="cuda",
                        **kwargs):
        """
        从预训练模型路径加载 WoWSynthesis
        
        Args:
            pretrained_model_path: 模型路径（本地路径或HuggingFace repo_id）
            synthesis_args: 合成参数对象
            device: 设备（如 'cuda' 或 'cpu'）
            **kwargs: 其他参数
            
        Returns:
            WoWSynthesis 实例
        """
        if os.path.isdir(pretrained_model_path):
            model_root = Path(pretrained_model_path)
        else:
            repo_name = pretrained_model_path.split("/")[-1]
            local_dir = Path.cwd() / repo_name
            local_dir.mkdir(parents=True, exist_ok=True)
            model_root = Path(snapshot_download(
                repo_id=pretrained_model_path,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False
            ))
            print(f"Model downloaded to: {model_root}")
        
        clip_model_path = model_root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        t5_model_path = model_root / "models_t5_umt5-xxl-enc-bf16.pth"
        vae_model_path = model_root / "Wan2.1_VAE.pth"
        dit_paths = [
            str(model_root / f"diffusion_pytorch_model.safetensors")
        ]

        # 加载模型 - 与原始demo保持一致：明确指定torch_dtype=torch.bfloat16
        mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        mm.load_models([str(clip_model_path)], torch_dtype=torch.float32)
        mm.load_models(
            [dit_paths, str(t5_model_path), str(vae_model_path)],
            torch_dtype=torch.bfloat16,
        )

        dit_model = mm.fetch_model("wan_video_dit")
        
        default_checkpoint = model_root / "WoW_video_dit.pt"
        state_dict = torch.load(str(default_checkpoint), map_location="cpu")
        if dit_model is not None:
            ensure_in_channels_36_for_image_input(dit_model)
            dit_model.load_state_dict(state_dict, strict=False)
            dit_model.has_image_input = True
            print("checkpoint (WoW_video_dit.pt) 加载成功")
        
        # 创建pipeline（在checkpoint加载之后）
        pipeline = WanVideoPipeline.from_model_manager(mm, torch_dtype=torch.bfloat16, device=device)

        # 启用VRAM管理（通过 enable_vram_management / no_vram_management 控制）
        enable_vram = True
        if synthesis_args:
            enable_vram = (
                getattr(synthesis_args, 'enable_vram_management', True)
                and not getattr(synthesis_args, 'no_vram_management', False)
            )

        if enable_vram:
            persistent_param_gb = getattr(synthesis_args, 'persistent_param_gb', 70)
            num_persistent_params = int(persistent_param_gb * 10**9)
            pipeline.enable_vram_management(num_persistent_param_in_dit=num_persistent_params)

        pipeline.denoising_model().has_image_input = True

        return cls(pipeline=pipeline, device=device)

    @torch.no_grad()
    def predict(self,
                input_image: Image.Image,
                text_prompt: str,
                synthesis_args=None,
                **kwargs):

        if synthesis_args is None:
            steps = 50
            seed = 42
            tiled = True
            num_frames = 81
        else:
            steps = getattr(synthesis_args, 'steps', 50)
            seed = getattr(synthesis_args, 'seed', 42)
            tiled = not getattr(synthesis_args, 'no_tiled', False)
            num_frames = getattr(synthesis_args, 'num_frames', 81)
        
        print(f"[WoWSynthesis] 准备生成视频，输入图片尺寸: {input_image.size if input_image else 'None'}")
        output_video = self.pipeline(
            prompt=text_prompt,
            negative_prompt="low quality, distorted, ugly, bad anatomy",
            input_image=input_image,
            num_inference_steps=steps,
            seed=seed,
            tiled=tiled,
            num_frames=num_frames,
        )
        print(f"[WoWSynthesis] 视频生成完成，输出帧数: {len(output_video) if output_video else 0}")
        return output_video