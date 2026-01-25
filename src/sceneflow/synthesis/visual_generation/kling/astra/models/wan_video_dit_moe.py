import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional
from einops import rearrange
from ......base_models.diffusion_model.diffsynth.models.utils import hash_state_dict_keys
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
from .wan_video_dit import SelfAttention
from ......base_models.diffusion_model.diffsynth.models.wan_video_dit import (
    flash_attention,
    modulate,
    sinusoidal_embedding_1d,
    precompute_freqs_cis_3d,
    precompute_freqs_cis,
    rope_apply,
    RMSNorm,
    AttentionModule,
    CrossAttention,
    MLP,
    Head,
    WanModelStateDictConverter
)

    

class ModalityProcessor(nn.Module):
    """模态处理器 - 将不同模态投影到统一维度"""
    
    def __init__(self, modality_name: str, input_dim: int, unified_dim: int = 30):
        super().__init__()
        self.modality_name = modality_name
        self.input_dim = input_dim
        self.unified_dim = unified_dim
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, unified_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] 或 [batch_size, input_dim]
        Returns:
            projected: [batch_size, seq_len, unified_dim]
        """
        # 🔧 修正：确保输入数据类型匹配
        original_dtype = x.dtype
        
        # 确保有seq_len维度
        if x.dim() == 2:  # [batch, input_dim]
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        # 🔧 关键修复：确保数据类型匹配projector的权重类型
        x = x.to(self.projector[0].weight.dtype)
        
        output = self.projector(x)
        
        # 🔧 可选：保持原始数据类型
        output = output.to(original_dtype)
        
        return output


class MultiModalMoE(nn.Module):
    """简化的多模态MoE - 只保留专家，不包含router"""
    
    def __init__(self, unified_dim: int = 30, hidden_dim: int = 60, output_dim: int = None, 
                 num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.unified_dim = unified_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim or unified_dim
        
        # 🔧 定义模态到专家的映射
        self.modality_to_expert = {
            "sekai": 0,      # sekai数据使用专家0
            "nuscenes": 1,   # nuscenes数据使用专家1
            "openx": 2,      # openx数据使用专家2
            "unknown": 0     # 默认使用专家0
        }
        
        # 🔧 移除router，只保留专家网络
        # Experts - 输入unified_dim，输出output_dim (每层独立)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(unified_dim, self.output_dim)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor, top_k_indices: torch.Tensor, 
                modality_type: str = "unknown") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, unified_dim]
            expert_weights: [batch_size, seq_len, top_k] - 从全局router得到的权重
            top_k_indices: [batch_size, seq_len, top_k] - 从全局router得到的专家索引
            modality_type: 模态类型标识（用于专家分配和统计）
        Returns:
            output: [batch_size, seq_len, output_dim]
            expert_stats: 专家选择统计信息
        """
        batch_size, seq_len, input_dim = x.shape
        assert input_dim == self.unified_dim, f"Expected input dim {self.unified_dim}, got {input_dim}"
        
        # 🔧 修正：确保数据类型匹配
        original_dtype = x.dtype
        x = x.to(self.experts[0][0].weight.dtype)
        
        # 🔧 获取该模态应该使用的目标专家
        target_expert_id = self.modality_to_expert.get(modality_type, 0)
        
        # 🔧 收集专家选择统计信息
        expert_stats = self.collect_expert_statistics(expert_weights, top_k_indices, modality_type, target_expert_id)
        
        # Expert processing (使用当前层的独立experts)
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # [batch, seq, output_dim]
            expert_outputs.append(expert_output)
        
        expert_outputs = torch.stack(expert_outputs, dim=-2)  # [batch, seq, num_experts, output_dim]
        
        # Weighted combination using provided weights and indices
        output = torch.zeros(batch_size, seq_len, self.output_dim, 
                           device=x.device, dtype=x.dtype)
        
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # [batch, seq]
            weight = expert_weights[:, :, k:k+1]  # [batch, seq, 1]
            
            expert_output = torch.gather(
                expert_outputs, 
                dim=2, 
                index=expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, expert_outputs.shape[-1])
            ).squeeze(2)  # [batch, seq, output_dim]
            
            output += weight * expert_output
        
        # 🔧 恢复原始数据类型
        output = output.to(original_dtype)
        
        return output, expert_stats
    
    def collect_expert_statistics(self, expert_weights, top_k_indices, modality_type, target_expert_id):
        """🔧 收集专家选择统计信息"""
        with torch.no_grad():
            # 计算每个专家被选中的频率
            expert_selection_count = torch.zeros(self.num_experts, device=expert_weights.device)
            for expert_id in range(self.num_experts):
                expert_selection_count[expert_id] = (top_k_indices == expert_id).float().sum()
            
            total_selections = expert_selection_count.sum()
            expert_selection_ratio = expert_selection_count / (total_selections + 1e-8)
            
            # 计算平均权重
            avg_expert_weights = torch.zeros(self.num_experts, device=expert_weights.device)
            for expert_id in range(self.num_experts):
                mask = (top_k_indices == expert_id)
                if mask.sum() > 0:
                    avg_expert_weights[expert_id] = expert_weights[mask].mean()
            
            # 计算Top-K权重统计
            avg_top_k_weights = expert_weights.mean(dim=(0, 1))
            
            # 🔧 计算目标专家的使用率
            target_expert_usage = expert_selection_ratio[target_expert_id].item()
            
            # 返回统计信息字典
            return {
                'modality_type': modality_type,
                'target_expert_id': target_expert_id,
                'target_expert_usage': target_expert_usage,
                'expert_selection_ratio': expert_selection_ratio.float().cpu().numpy(),
                'avg_expert_weights': avg_expert_weights.float().cpu().numpy(),
                'avg_top_k_weights': avg_top_k_weights.float().cpu().numpy(),
                'num_experts': self.num_experts,
                'top_k': self.top_k
            }
                                    
class DiTBlockWithMoE(nn.Module):
    """集成MoE的DiT Block"""
    
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, 
                 eps: float = 1e-6, use_moe: bool = True, moe_config: Optional[dict] = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_moe = use_moe
        
        # 原有的DiT组件
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), 
            nn.GELU(approximate='tanh'), 
            nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # 🔧 只在启用MoE时初始化MoE组件（无router版本）
        if self.use_moe and moe_config:
            unified_dim = moe_config.get("unified_dim", 30)  
            # MoE模块 - 输入unified_dim，输出dim用于残差连接，无router
            self.moe = MultiModalMoE(
                unified_dim=unified_dim,
                output_dim=dim,  # 输出维度与transformer block的dim匹配
                num_experts=moe_config.get("num_experts", 4),
                top_k=moe_config.get("top_k", 2)
            )

    def forward(self, x, context, cam_emb, t_mod, freqs, 
                modality_inputs: Optional[dict] = None,
                router_weights: Optional[torch.Tensor] = None,
                router_indices: Optional[torch.Tensor] = None):
        # 原有的modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        # 🔧 MoE处理 - 使用全局router的结果
        if self.use_moe and modality_inputs and hasattr(self, 'moe') and router_weights is not None:
            # 合并所有模态的输入（已经通过全局processor处理过）
            combined_modality_input = None
            active_modality = "unknown"
            for modality_type, processed_input in modality_inputs.items():
                active_modality = modality_type  # 记录当前活跃的模态
                if combined_modality_input is None:
                    combined_modality_input = processed_input
                else:
                    combined_modality_input = combined_modality_input + processed_input
            
            if combined_modality_input is not None:
                # 🔧 使用全局router的权重和索引
                moe_output, expert_stats = self.moe(
                    combined_modality_input, 
                    router_weights, 
                    router_indices, 
                    active_modality
                )
                input_x = input_x + moe_output
                
                # 🔧 存储专家统计信息供后续收集
                if not hasattr(self, 'expert_stats_buffer'):
                    self.expert_stats_buffer = []
                    
                self.expert_stats_buffer.append(expert_stats)
        elif cam_emb is not None and hasattr(self, 'cam_encoder'):
            # 传统camera编码器作为fallback
            cam_emb = cam_emb.to(self.cam_encoder.weight.dtype)
            cam_emb = self.cam_encoder(cam_emb)
            input_x = input_x + cam_emb

        input_x = input_x.to(self.projector.weight.dtype)

        # Ensure self.self_attn output dtype matches self.projector.weight dtype
        attn_output = self.self_attn(input_x, freqs)
        attn_output = attn_output.to(self.projector.weight.dtype)

        x = x + gate_msa * self.projector(attn_output)
        x = x.to(self.norm3.weight.dtype)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x
                




class WanModelMoe(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        # 🔧 新增MoE参数
        use_moe: bool = True,
        moe_config: Optional[dict] = None
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.use_moe = use_moe  # 🔧 保存MoE配置
        self.moe_config = moe_config or {}

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # # 🔧 新增：创建全局router - 放在WanModel级别
        # if use_moe and moe_config:
        #     unified_dim = moe_config.get("unified_dim", 30)
        #     num_experts = moe_config.get("num_experts", 4)
        #     self.top_k = moe_config.get("top_k", 2)
            
            # 🔧 定义模态到专家的映射
        self.modality_to_expert = {
            "sekai": 0,      # sekai数据使用专家0
            "nuscenes": 1,   # nuscenes数据使用专家1
            "openx": 2,      # openx数据使用专家2
            "unknown": 0     # 默认使用专家0
        }
        self.top_k = 1
            
        #     self.global_router = nn.Linear(unified_dim, num_experts)
        #     print(f"✅ 创建了全局router: input_dim={unified_dim}, num_experts={num_experts}")
        # else:
        #     self.global_router = None
        #     self.modality_to_expert = {}

        # 🔧 根据是否使用MoE创建不同的blocks
        self.blocks = nn.ModuleList([
            DiTBlockWithMoE(has_image_input, dim, num_heads, ffn_dim, eps, use_moe, moe_config)
            for _ in range(num_layers)
        ])
        
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

    def compute_router_decisions(self, combined_modality_input: torch.Tensor, modality_type: str):
        """
        不用router，直接根据modality_to_expert写死专家选择和权重
        """
        batch_size, seq_len, _ = combined_modality_input.shape
        num_experts = len(self.modality_to_expert)
        top_k = self.top_k if hasattr(self, "top_k") else 1

        # 获取目标专家id
        target_expert_id = self.modality_to_expert.get(modality_type, 0)

        # router_indices: 全部填目标专家
        router_indices = torch.full((batch_size, seq_len, top_k), target_expert_id, dtype=torch.long, device=combined_modality_input.device)
        # router_weights: 全部为1
        router_weights = torch.ones((batch_size, seq_len, top_k), dtype=combined_modality_input.dtype, device=combined_modality_input.device)

        # 专业化损失直接为0
        specialization_loss = torch.tensor(0.0, device=combined_modality_input.device)

        return router_weights, router_indices, specialization_loss

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def create_clean_x_embedder(self):
        """创建类似FramePack的clean_x_embedder"""        
        class CleanXEmbedder(nn.Module):
            def __init__(self, inner_dim):
                super().__init__()
                # 参考hunyuan_video_packed.py的设计
                self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
                self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
            
            def forward(self, x, scale="1x"):
                if scale == "1x":
                    return self.proj(x)
                elif scale == "2x":
                    return self.proj_2x(x)
                elif scale == "4x":
                    return self.proj_4x(x)
                else:
                    raise ValueError(f"Unsupported scale: {scale}")
        
        return CleanXEmbedder(self.dim)

    def rope(self, frame_indices, height, width, device):
        """🔧 模仿HunyuanVideo的rope方法"""
        batch_size = frame_indices.shape[0]
        seq_len = frame_indices.shape[1]
        
        # 使用frame_indices生成时间维度的频率
        f_freqs = self.freqs[0][frame_indices.to("cpu")]  # [batch, seq_len, freq_dim]
        
        # 为每个spatial位置生成频率
        h_positions = torch.arange(height, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        w_positions = torch.arange(width, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # 获取h和w的频率
        h_freqs = self.freqs[1][h_positions.to("cpu")]  # [batch, seq_len, height, h_freq_dim]
        w_freqs = self.freqs[2][w_positions.to("cpu")]  # [batch, seq_len, width, w_freq_dim]
        
        # 扩展到完整的spatial grid
        f_freqs_expanded = f_freqs.unsqueeze(2).unsqueeze(3).expand(-1, -1, height, width, -1)
        h_freqs_expanded = h_freqs.unsqueeze(3).expand(-1, -1, -1, width, -1)
        w_freqs_expanded = w_freqs.unsqueeze(2).expand(-1, -1, height, -1, -1)
        
        # 合并所有频率
        rope_freqs = torch.cat([f_freqs_expanded, h_freqs_expanded, w_freqs_expanded], dim=-1)
        
        return rope_freqs  # [batch, seq_len, height, width, total_freq_dim]

    def pad_for_3d_conv(self, x, kernel_size):
        """3D卷积的padding - 参考hunyuan实现"""
        if len(x.shape) == 5:  # [B, C, T, H, W]
            b, c, t, h, w = x.shape
            pt, ph, pw = kernel_size
            pad_t = (pt - (t % pt)) % pt
            pad_h = (ph - (h % ph)) % ph
            pad_w = (pw - (w % pw)) % pw
            return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode='replicate')
        elif len(x.shape) == 6:  # [B, T, H, W, C] (RoPE频率)
            b, t, h, w, c = x.shape
            pt, ph, pw = kernel_size
            pad_t = (pt - (t % pt)) % pt
            pad_h = (ph - (h % ph)) % ph
            pad_w = (pw - (w % pw)) % pw
            return torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t), mode='replicate')
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

    def center_down_sample_3d(self, x, scale_factor):
        """🔧 模仿HunyuanVideo的center_down_sample_3d"""
        if len(x.shape) == 6:  # [B, T, H, W, C] (RoPE频率)
            st, sh, sw = scale_factor
            return x[:, ::st, ::sh, ::sw, :]
        elif len(x.shape) == 5:  # [B, C, T, H, W]
            st, sh, sw = scale_factor
            return x[:, :, ::st, ::sh, ::sw]
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

    def process_modality_inputs(self, modality_inputs):
        """🔧 全局处理模态输入，返回统一格式的embeddings"""
        if not modality_inputs or not self.use_moe:
            return None
        
        processed_modality_inputs = {}
        
        for modality_type, input_data in modality_inputs.items():
            if modality_type == "sekai" and hasattr(self, 'sekai_processor'):
                processed = self.sekai_processor(input_data)
                processed_modality_inputs[modality_type] = processed
            elif modality_type == "nuscenes" and hasattr(self, 'nuscenes_processor'):
                processed = self.nuscenes_processor(input_data)
                processed_modality_inputs[modality_type] = processed
            elif modality_type == "openx" and hasattr(self, 'openx_processor'):
                processed = self.openx_processor(input_data)
                processed_modality_inputs[modality_type] = processed
            else:
                print(f"⚠️ 未知的模态类型: {modality_type}")
                continue
        
        return processed_modality_inputs, processed
    
    def process_input_hidden_states(self, 
                                latents, latent_indices=None,
                                clean_latents=None, clean_latent_indices=None,
                                clean_latents_2x=None, clean_latent_2x_indices=None,
                                clean_latents_4x=None, clean_latent_4x_indices=None,
                                cam_emb=None,
                                modality_inputs: Optional[dict] = None):  # 🔧 新增modality_inputs参数
        """🔧 处理FramePack风格的多尺度输入 + MoE模态输入处理 - 完全照wan_video_dit_recam_future实现"""
        
        # 主要latents处理
        hidden_states, grid_size = self.patchify(latents)
        B, T_patches, C = hidden_states.shape
        f, h, w = grid_size
        
        # 🔧 修正：使用latent_indices指定的时间位置计算RoPE频率
        if latent_indices is None:
            latent_indices = torch.arange(0, f, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        
        # 为主要latents计算RoPE频率
        main_rope_freqs_list = []
        for b in range(B):
            batch_rope_freqs = []
            for t_idx in latent_indices[b]:
                f_freq = self.freqs[0][t_idx:t_idx+1]  # [1, freq_dim]
                h_freq = self.freqs[1][:h]  # [h, freq_dim] 
                w_freq = self.freqs[2][:w]  # [w, freq_dim]
                
                spatial_freqs = torch.cat([
                    f_freq.view(1, 1, 1, -1).expand(1, h, w, -1),
                    h_freq.view(1, h, 1, -1).expand(1, h, w, -1), 
                    w_freq.view(1, 1, w, -1).expand(1, h, w, -1)
                ], dim=-1).reshape(h * w, -1)  # [h*w, total_freq_dim]
                
                batch_rope_freqs.append(spatial_freqs)
            
            batch_rope_freqs = torch.cat(batch_rope_freqs, dim=0)  # [f*h*w, total_freq_dim]
            main_rope_freqs_list.append(batch_rope_freqs)
        
        rope_freqs = torch.stack(main_rope_freqs_list, dim=0)  # [B, f*h*w, total_freq_dim]
        
        # 🔧 准备主要scale (1x) 的modality embeddings - 空间维度为 h*w
        start_indice = clean_latent_indices[0][0].item() if clean_latent_indices is not None else 0
        combined_modality_embeddings = None
        
        # 🔧 兼容原有的cam_emb处理（完全照抄wan_video_dit_recam_future的逻辑）
        if cam_emb is not None:
            # 提取target部分的camera（基于latent_indices）
            target_start = latent_indices[0].min().item() - start_indice
            target_end = latent_indices[0].max().item() + 1 - start_indice
            target_camera = cam_emb[:, target_start:target_end, :]  # [B, target_frames, cam_dim]
            
            # 🔧 为主要latents处理camera空间扩展
            target_camera_spatial = target_camera.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
            target_camera_spatial = rearrange(target_camera_spatial, 'b f h w d -> b (f h w) d')
            combined_modality_embeddings = target_camera_spatial
        
        # 🔧 处理clean_latents (1x scale) - 完全参考wan_video_dit_recam_future
        if clean_latents is not None and clean_latent_indices is not None:
            clean_latents = clean_latents.to(hidden_states)
            clean_hidden_states = self.clean_x_embedder(clean_latents, scale="1x")
            clean_hidden_states = rearrange(clean_hidden_states, 'b c f h w -> b (f h w) c')
            
            # 🔧 为clean_latents计算RoPE频率
            clean_rope_freqs_list = []
            for b in range(B):
                clean_batch_rope_freqs = []
                for t_idx in clean_latent_indices[b]:
                    f_freq = self.freqs[0][t_idx:t_idx+1]
                    h_freq = self.freqs[1][:h]
                    w_freq = self.freqs[2][:w]
                    
                    spatial_freqs = torch.cat([
                        f_freq.view(1, 1, 1, -1).expand(1, h, w, -1),
                        h_freq.view(1, h, 1, -1).expand(1, h, w, -1),
                        w_freq.view(1, 1, w, -1).expand(1, h, w, -1)
                    ], dim=-1).reshape(h * w, -1)
                    
                    clean_batch_rope_freqs.append(spatial_freqs)
                
                clean_batch_rope_freqs = torch.cat(clean_batch_rope_freqs, dim=0)
                clean_rope_freqs_list.append(clean_batch_rope_freqs)
            
            clean_rope_freqs = torch.stack(clean_rope_freqs_list, dim=0)
            
            # 🔧 处理clean modality embeddings - 1x空间维度
            if cam_emb is not None:
                clean_start = clean_latent_indices[0].min().item() - start_indice
                clean_end = clean_latent_indices[0].max().item() + 1 - start_indice

                if clean_start == clean_end:
                    clean_camera = cam_emb[:, clean_start:clean_start+1, :]   # [B, 1, cam_dim]
                else:
                    clean_camera = cam_emb[:, [clean_start, clean_end], :]   # [B, 2, cam_dim]  
                                
                # 扩展到1x空间维度 h*w
                clean_camera_spatial = clean_camera.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
                clean_camera_spatial = rearrange(clean_camera_spatial, 'b f h w d -> b (f h w) d')
                combined_modality_embeddings = torch.cat([clean_camera_spatial, combined_modality_embeddings], dim=1)
            
            # cat clean latents和frequencies到前面
            hidden_states = torch.cat([clean_hidden_states, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_rope_freqs, rope_freqs], dim=1)
        
        # 🔧 处理clean_latents_2x (2x scale) - 完全参考wan_video_dit_recam_future
        if clean_latents_2x is not None and clean_latent_2x_indices is not None and clean_latent_2x_indices.numel() > 0:
            # 过滤有效索引（非-1）
            valid_2x_indices = clean_latent_2x_indices[clean_latent_2x_indices >= 0]
            
            if len(valid_2x_indices) > 0:
                clean_latents_2x = clean_latents_2x.to(hidden_states)
                clean_latents_2x = self.pad_for_3d_conv(clean_latents_2x, (2, 4, 4))
                clean_hidden_states_2x = self.clean_x_embedder(clean_latents_2x, scale="2x")
                
                _, _, clean_2x_f, clean_2x_h, clean_2x_w = clean_hidden_states_2x.shape
                clean_hidden_states_2x = rearrange(clean_hidden_states_2x, 'b c f h w -> b (f h w) c')
                
                # 🔧 为2x latents计算RoPE频率 - 基于实际的下采样结果
                clean_2x_rope_freqs_list = []
                for b in range(B):
                    clean_2x_batch_rope_freqs = []
                    
                    # 🔧 使用clean_2x_f作为实际的时间帧数
                    for frame_idx in range(clean_2x_f):
                        # 计算对应的原始时间索引
                        if frame_idx < len(valid_2x_indices):
                            t_idx = valid_2x_indices[frame_idx]
                        else:
                            # 如果超出有效索引，使用0频率
                            t_idx = valid_2x_indices[-1] if len(valid_2x_indices) > 0 else 0
                        
                        f_freq = self.freqs[0][t_idx:t_idx+1]
                        h_freq = self.freqs[1][:clean_2x_h]
                        w_freq = self.freqs[2][:clean_2x_w]
                        
                        spatial_freqs = torch.cat([
                            f_freq.view(1, 1, 1, -1).expand(1, clean_2x_h, clean_2x_w, -1),
                            h_freq.view(1, clean_2x_h, 1, -1).expand(1, clean_2x_h, clean_2x_w, -1),
                            w_freq.view(1, 1, clean_2x_w, -1).expand(1, clean_2x_h, clean_2x_w, -1)
                        ], dim=-1).reshape(clean_2x_h * clean_2x_w, -1)
                        
                        clean_2x_batch_rope_freqs.append(spatial_freqs)
                    
                    clean_2x_batch_rope_freqs = torch.cat(clean_2x_batch_rope_freqs, dim=0)
                    clean_2x_rope_freqs_list.append(clean_2x_batch_rope_freqs)
                
                clean_2x_rope_freqs = torch.stack(clean_2x_rope_freqs_list, dim=0)
                
                # 🔧 处理2x modality embeddings
                if cam_emb is not None:
                    # 创建2x camera，0填充无效部分
                    clean_2x_camera = torch.zeros(B, clean_2x_f, cam_emb.shape[-1], dtype=cam_emb.dtype, device=cam_emb.device)
                    
                    for frame_idx in range(min(clean_2x_f, len(valid_2x_indices))):
                        cam_idx = valid_2x_indices[frame_idx].item() - start_indice
                        if 0 <= cam_idx < cam_emb.shape[1]:
                            clean_2x_camera[:, frame_idx, :] = cam_emb[:, cam_idx, :]
                    
                    clean_2x_camera_spatial = clean_2x_camera.unsqueeze(2).unsqueeze(3).repeat(1, 1, clean_2x_h, clean_2x_w, 1)
                    clean_2x_camera_spatial = rearrange(clean_2x_camera_spatial, 'b f h w d -> b (f h w) d')
                    combined_modality_embeddings = torch.cat([clean_2x_camera_spatial, combined_modality_embeddings], dim=1)
                
                hidden_states = torch.cat([clean_hidden_states_2x, hidden_states], dim=1)
                rope_freqs = torch.cat([clean_2x_rope_freqs, rope_freqs], dim=1)
        
        # 🔧 处理clean_latents_4x (4x scale) - 完全参考wan_video_dit_recam_future
        if clean_latents_4x is not None and clean_latent_4x_indices is not None and clean_latent_4x_indices.numel() > 0:
            # 过滤有效索引（非-1）
            valid_4x_indices = clean_latent_4x_indices[clean_latent_4x_indices >= 0]
            
            if len(valid_4x_indices) > 0:
                clean_latents_4x = clean_latents_4x.to(hidden_states)
                clean_latents_4x = self.pad_for_3d_conv(clean_latents_4x, (4, 8, 8))
                clean_hidden_states_4x = self.clean_x_embedder(clean_latents_4x, scale="4x")
                
                _, _, clean_4x_f, clean_4x_h, clean_4x_w = clean_hidden_states_4x.shape
                clean_hidden_states_4x = rearrange(clean_hidden_states_4x, 'b c f h w -> b (f h w) c')
                
                # 🔧 为4x latents计算RoPE频率 - 基于实际的下采样结果
                clean_4x_rope_freqs_list = []
                for b in range(B):
                    clean_4x_batch_rope_freqs = []
                    
                    # 🔧 使用clean_4x_f作为实际的时间帧数
                    for frame_idx in range(clean_4x_f):
                        # 计算对应的原始时间索引
                        if frame_idx < len(valid_4x_indices):
                            t_idx = valid_4x_indices[frame_idx]
                        else:
                            # 如果超出有效索引，使用0频率
                            t_idx = valid_4x_indices[-1] if len(valid_4x_indices) > 0 else 0
                        
                        f_freq = self.freqs[0][t_idx:t_idx+1]
                        h_freq = self.freqs[1][:clean_4x_h]
                        w_freq = self.freqs[2][:clean_4x_w]
                        
                        spatial_freqs = torch.cat([
                            f_freq.view(1, 1, 1, -1).expand(1, clean_4x_h, clean_4x_w, -1),
                            h_freq.view(1, clean_4x_h, 1, -1).expand(1, clean_4x_h, clean_4x_w, -1),
                            w_freq.view(1, 1, clean_4x_w, -1).expand(1, clean_4x_h, clean_4x_w, -1)
                        ], dim=-1).reshape(clean_4x_h * clean_4x_w, -1)
                        
                        clean_4x_batch_rope_freqs.append(spatial_freqs)
                    
                    clean_4x_batch_rope_freqs = torch.cat(clean_4x_batch_rope_freqs, dim=0)
                    clean_4x_rope_freqs_list.append(clean_4x_batch_rope_freqs)
                
                clean_4x_rope_freqs = torch.stack(clean_4x_rope_freqs_list, dim=0)
                
                # 🔧 处理4x modality embeddings
                if cam_emb is not None:
                    # 创建4x camera，0填充无效部分
                    clean_4x_camera = torch.zeros(B, clean_4x_f, cam_emb.shape[-1], dtype=cam_emb.dtype, device=cam_emb.device)
                    
                    for frame_idx in range(min(clean_4x_f, len(valid_4x_indices))):
                        cam_idx = valid_4x_indices[frame_idx].item() - start_indice
                        if 0 <= cam_idx < cam_emb.shape[1]:
                            clean_4x_camera[:, frame_idx, :] = cam_emb[:, cam_idx, :]
                    
                    clean_4x_camera_spatial = clean_4x_camera.unsqueeze(2).unsqueeze(3).repeat(1, 1, clean_4x_h, clean_4x_w, 1)
                    clean_4x_camera_spatial = rearrange(clean_4x_camera_spatial, 'b f h w d -> b (f h w) d')
                    combined_modality_embeddings = torch.cat([clean_4x_camera_spatial, combined_modality_embeddings], dim=1)
                
                hidden_states = torch.cat([clean_hidden_states_4x, hidden_states], dim=1)
                rope_freqs = torch.cat([clean_4x_rope_freqs, rope_freqs], dim=1)
        
        rope_freqs = rope_freqs.unsqueeze(2).to(device=hidden_states.device)
        
        # 🔧 关键修正：在return前处理modality_inputs
        processed_modality_inputs = None
        if modality_inputs and self.use_moe:
            # 确定模态类型并将处理好的combined_modality_embeddings赋值给对应的模态
            processed_modality_inputs = {}
            for modality_type in modality_inputs.keys():
                # 将处理好的camera embeddings赋给对应的模态
                processed_modality_inputs[modality_type] = combined_modality_embeddings
        
        return hidden_states, rope_freqs, grid_size, combined_modality_embeddings, processed_modality_inputs

    def forward(self, 
                latents, timestep, cam_emb,
                # 🔧 FramePack参数
                latent_indices=None,
                clean_latents=None, clean_latent_indices=None,
                clean_latents_2x=None, clean_latent_2x_indices=None,
                clean_latents_4x=None, clean_latent_4x_indices=None,
                # 🔧 MoE参数
                modality_inputs: Optional[dict] = None,
                **kwargs):
        
        modality_inputs, cam_emb = self.process_modality_inputs(modality_inputs)
        
        # 🔧 清空之前的专家统计信息
        for block in self.blocks:
            if hasattr(block, 'expert_stats_buffer'):
                block.expert_stats_buffer = []
        
        # 🔧 使用新的处理方法来处理多尺度输入和RoPE频率 + MoE模态输入
        hidden_states, rope_freqs, grid_size, processed_cam_emb, processed_modality_inputs = self.process_input_hidden_states(
            latents, latent_indices,
            clean_latents, clean_latent_indices,
            clean_latents_2x, clean_latent_2x_indices,
            clean_latents_4x, clean_latent_4x_indices,
            cam_emb, modality_inputs
        )
        
        # 计算原始latent序列长度（用于最后提取）
        batch_size, num_channels, num_frames, height, width = latents.shape
        p, p_t = self.patch_size[2], self.patch_size[0]  # [t, h, w]
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        original_context_length = post_patch_num_frames * post_patch_height * post_patch_width
        
        # 处理其他embeddings
        context = kwargs.get("context", None)
        if context is not None:
            context = self.text_embedding(context)
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        #t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        with torch.amp.autocast("cuda", enabled=False):
            # Force time projection (and parameters) to run in fp32 to bypass bf16 autocast
            t_fp32 = t.float()
            t_activated = self.time_projection[0](t_fp32)
            linear = self.time_projection[1]
            weight_fp32 = linear.weight.float()
            bias_fp32 = linear.bias.float() if linear.bias is not None else None
            t_proj = F.linear(t_activated, weight_fp32, bias_fp32)
            t_proj = t_proj.to(t.dtype)
        t_mod = t_proj.unflatten(1, (6, self.dim))

        # 确保rope_freqs与hidden_states的序列长度匹配
        assert rope_freqs.shape[1] == hidden_states.shape[1], \
            f"RoPE频率序列长度 {rope_freqs.shape[1]} 与 hidden_states序列长度 {hidden_states.shape[1]} 不匹配"
        
        # 🔧 全局router决策计算（一次性为所有层计算）
        router_weights, router_indices, total_specialization_loss = None, None, torch.tensor(0.0, device=hidden_states.device)
        active_modality = "unknown"
        
        if self.use_moe and processed_modality_inputs:
            # 合并所有模态的输入
            combined_modality_input = None
            for modality_type, processed_input in processed_modality_inputs.items():
                active_modality = modality_type
                if combined_modality_input is None:
                    combined_modality_input = processed_input
                else:
                    combined_modality_input = combined_modality_input + processed_input
            
            #router_input = torch.cat([hidden_states, combined_modality_input], dim=-1)
            if combined_modality_input is not None:
                router_weights, router_indices, total_specialization_loss = self.compute_router_decisions(
                    combined_modality_input, active_modality
                )
        
        # 🔧 Transformer blocks - 传递全局router的结果
        for block in self.blocks:
            hidden_states = block(
                hidden_states, 
                context, 
                processed_cam_emb, 
                t_mod, 
                rope_freqs, 
                processed_modality_inputs,
                router_weights,  # 🔧 传递全局router权重
                router_indices   # 🔧 传递全局router索引
            )
        
        # 🔧 收集并打印整体专家统计信息
        #self.print_overall_expert_statistics()
        
        # 🔧 只对原始预测目标部分进行输出投影
        hidden_states = hidden_states[:, -original_context_length:, :]
        hidden_states = self.head(hidden_states, t)
        hidden_states = self.unpatchify(hidden_states, grid_size)
        
        return hidden_states, total_specialization_loss

    def print_overall_expert_statistics(self):
        """🔧 新增：打印整体专家统计信息 - 更新版本，显示全局router信息"""
        all_expert_stats = []
        
        # 收集所有block的专家统计信息
        for i, block in enumerate(self.blocks):
            if hasattr(block, 'expert_stats_buffer') and len(block.expert_stats_buffer) > 0:
                all_expert_stats.extend(block.expert_stats_buffer)
        
        if not all_expert_stats:
            return
        
        # 按模态类型分组统计
        modality_stats = {}
        for stats in all_expert_stats:
            modality = stats['modality_type']
            if modality not in modality_stats:
                modality_stats[modality] = {
                    'selection_ratios': [],
                    'expert_weights': [],
                    'top_k_weights': [],
                    'target_expert_usages': [],
                    'target_expert_id': stats['target_expert_id'],
                    'count': 0
                }
            
            modality_stats[modality]['selection_ratios'].append(stats['expert_selection_ratio'])
            modality_stats[modality]['expert_weights'].append(stats['avg_expert_weights'])
            modality_stats[modality]['top_k_weights'].append(stats['avg_top_k_weights'])
            modality_stats[modality]['target_expert_usages'].append(stats['target_expert_usage'])
            modality_stats[modality]['count'] += 1
        
        # 打印整体统计信息
        print("\n" + "="*60)
        print("📊 【样本整体专家专业化统计】(全局Router + 分层Experts)")
        print("="*60)
        
        for modality, stats in modality_stats.items():
            if stats['count'] == 0:
                continue
                
            # 计算该模态的平均统计
            avg_selection_ratio = np.mean(stats['selection_ratios'], axis=0)
            avg_expert_weights = np.mean(stats['expert_weights'], axis=0)
            avg_top_k_weights = np.mean(stats['top_k_weights'], axis=0)
            avg_target_expert_usage = np.mean(stats['target_expert_usages'])
            target_expert_id = stats['target_expert_id']
            
            print(f"\n {modality.upper()} modality (Source {stats['count']} MoE blocks)")
            print(f"   expected expert: Expert-{target_expert_id}")
            print(f"   expected expert usage: {avg_target_expert_usage:.3f} ({avg_target_expert_usage*100:.1f}%)")
            
            print(f"   Expert chosen weight (global Router decision):")
            for i, ratio in enumerate(avg_selection_ratio):
                status = "🔥" if i == target_expert_id else "  "
                print(f"    {status} Expert-{i}: {ratio:.3f} ({ratio*100:.1f}%)")
            
            print(f"    Expert avg weight:")
            for i, weight in enumerate(avg_expert_weights):
                status = "🔥" if i == target_expert_id else "  "
                print(f"    {status} Expert-{i}: {weight:.3f}")
            
            # 专业化程度评估
            if avg_target_expert_usage > 0.8:
                specialization_status = " 高度专业化"
            elif avg_target_expert_usage > 0.5:
                specialization_status = " 良好专业化"
            else:
                specialization_status = "  专业化不足"
            
            print(f"   专业化程度: {specialization_status}")
            
            # 找出最常用的专家
            most_used_expert = np.argmax(avg_selection_ratio)
            most_used_ratio = avg_selection_ratio[most_used_expert]
            if most_used_expert == target_expert_id:
                print(f"   Actual most expert: Expert-{most_used_expert} ({most_used_ratio:.3f}) - OK!")
            else:
                print(f"   Actual most expert: Expert-{most_used_expert} ({most_used_ratio:.3f}) - No")
        
        print("="*60)

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
