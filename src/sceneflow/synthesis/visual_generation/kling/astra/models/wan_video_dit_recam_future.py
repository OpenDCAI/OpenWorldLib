import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
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
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, causal=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads, causal=False):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, causal: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        x = x.to(self.q.weight.dtype)
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        #self.self_attn = SelfAttention(dim, num_heads, eps, causal=True)  # Enable causal masking
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, context, cam_emb, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        if cam_emb is not None:
            # 🔧 简化：cam_emb已经在process_input_hidden_states中处理过空间扩展和重排列
            cam_emb = cam_emb.to(self.cam_encoder.weight.dtype)
            cam_emb = self.cam_encoder(cam_emb)  # [batch, seq_len, dim]
            input_x = input_x + cam_emb

        # Ensure input_x dtype matches self.projector.weight dtype
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

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModelFuture(torch.nn.Module):
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
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

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
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

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

    def process_input_hidden_states(self, 
                                latents, latent_indices=None,
                                clean_latents=None, clean_latent_indices=None,
                                clean_latents_2x=None, clean_latent_2x_indices=None,
                                clean_latents_4x=None, clean_latent_4x_indices=None,
                                cam_emb=None):
        """🔧 处理FramePack风格的多尺度输入 - 修正RoPE频率维度匹配问题"""
        
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
        
        # 🔧 准备主要scale (1x) 的camera embeddings - 空间维度为 h*w
        start_indice = clean_latent_indices[0][0].item()
        combined_camera_embeddings = None
        
        if cam_emb is not None:
            # 提取target部分的camera（基于latent_indices）
            target_start = latent_indices[0].min().item() - start_indice
            target_end = latent_indices[0].max().item() + 1 - start_indice
            target_camera = cam_emb[:, target_start:target_end, :]  # [B, target_frames, cam_dim]
            
            # 🔧 为主要latents处理camera空间扩展
            target_camera_spatial = target_camera.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
            target_camera_spatial = rearrange(target_camera_spatial, 'b f h w d -> b (f h w) d')
            combined_camera_embeddings = target_camera_spatial
        
        # 🔧 处理clean_latents (1x scale)
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
            
            # 🔧 处理clean camera embeddings - 1x空间维度
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
                combined_camera_embeddings = torch.cat([clean_camera_spatial, combined_camera_embeddings], dim=1)
            
            # cat clean latents和frequencies到前面
            hidden_states = torch.cat([clean_hidden_states, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_rope_freqs, rope_freqs], dim=1)
        
        # 🔧 处理clean_latents_2x (2x scale) - 修正RoPE频率计算
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
                    
                    # 🔧 修正：使用clean_2x_f作为实际的时间帧数
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
                
                # 🔧 处理2x camera embeddings
                if cam_emb is not None:
                    # 创建2x camera，0填充无效部分
                    clean_2x_camera = torch.zeros(B, clean_2x_f, cam_emb.shape[-1], dtype=cam_emb.dtype, device=cam_emb.device)
                    
                    for frame_idx in range(min(clean_2x_f, len(valid_2x_indices))):
                        cam_idx = valid_2x_indices[frame_idx].item() - start_indice
                        if 0 <= cam_idx < cam_emb.shape[1]:
                            clean_2x_camera[:, frame_idx, :] = cam_emb[:, cam_idx, :]
                    
                    clean_2x_camera_spatial = clean_2x_camera.unsqueeze(2).unsqueeze(3).repeat(1, 1, clean_2x_h, clean_2x_w, 1)
                    clean_2x_camera_spatial = rearrange(clean_2x_camera_spatial, 'b f h w d -> b (f h w) d')
                    combined_camera_embeddings = torch.cat([clean_2x_camera_spatial, combined_camera_embeddings], dim=1)
                
                hidden_states = torch.cat([clean_hidden_states_2x, hidden_states], dim=1)
                rope_freqs = torch.cat([clean_2x_rope_freqs, rope_freqs], dim=1)
        
        # 🔧 处理clean_latents_4x (4x scale) - 修正RoPE频率计算
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
                    
                    # 🔧 修正：使用clean_4x_f作为实际的时间帧数
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
                
                # 🔧 处理4x camera embeddings
                if cam_emb is not None:
                    # 创建4x camera，0填充无效部分
                    clean_4x_camera = torch.zeros(B, clean_4x_f, cam_emb.shape[-1], dtype=cam_emb.dtype, device=cam_emb.device)
                    
                    for frame_idx in range(min(clean_4x_f, len(valid_4x_indices))):
                        cam_idx = valid_4x_indices[frame_idx].item() - start_indice
                        if 0 <= cam_idx < cam_emb.shape[1]:
                            clean_4x_camera[:, frame_idx, :] = cam_emb[:, cam_idx, :]
                    
                    clean_4x_camera_spatial = clean_4x_camera.unsqueeze(2).unsqueeze(3).repeat(1, 1, clean_4x_h, clean_4x_w, 1)
                    clean_4x_camera_spatial = rearrange(clean_4x_camera_spatial, 'b f h w d -> b (f h w) d')
                    combined_camera_embeddings = torch.cat([clean_4x_camera_spatial, combined_camera_embeddings], dim=1)
                
                hidden_states = torch.cat([clean_hidden_states_4x, hidden_states], dim=1)
                rope_freqs = torch.cat([clean_4x_rope_freqs, rope_freqs], dim=1)
        
        rope_freqs = rope_freqs.unsqueeze(2).to(device=hidden_states.device)
        return hidden_states, rope_freqs, grid_size, combined_camera_embeddings

    def forward(self, 
                latents, timestep, cam_emb,
                # 🔧 FramePack参数
                latent_indices=None,
                clean_latents=None, clean_latent_indices=None,
                clean_latents_2x=None, clean_latent_2x_indices=None,
                clean_latents_4x=None, clean_latent_4x_indices=None,
                **kwargs):
        
        # 🔧 使用新的处理方法来处理多尺度输入和RoPE频率
        hidden_states, rope_freqs, grid_size, processed_cam_emb = self.process_input_hidden_states(
            latents, latent_indices,
            clean_latents, clean_latent_indices,
            clean_latents_2x, clean_latent_2x_indices,
            clean_latents_4x, clean_latent_4x_indices,
            cam_emb
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
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # 确保rope_freqs与hidden_states的序列长度匹配
        assert rope_freqs.shape[1] == hidden_states.shape[1], \
            f"RoPE频率序列长度 {rope_freqs.shape[1]} 与 hidden_states序列长度 {hidden_states.shape[1]} 不匹配"
        
        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, context, processed_cam_emb, t_mod, rope_freqs)
        
        # 🔧 只对原始预测目标部分进行输出投影
        # 提取最后original_context_length个tokens（对应原始latents）
        hidden_states = hidden_states[:, -original_context_length:, :]
        hidden_states = self.head(hidden_states, t)
        hidden_states = self.unpatchify(hidden_states, grid_size)
        
        return hidden_states

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        else:
            config = {}
        return state_dict, config
