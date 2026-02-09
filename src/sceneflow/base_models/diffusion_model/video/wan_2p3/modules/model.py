# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']
from torch.cuda import amp  # PyTorch的自动混合精度

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs, b_, s_, n_, d_, mask_token=None, ids_restore=None, ids_keep=None, rand_num_img=None, flag=None):
    if not flag: #False:
        if ids_restore!=None:
            x = x.view(b_,s_,n_*d_)

            mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

            x = x.view(b_,ids_restore.shape[1],n_,d_)

        n, c = x.size(2), x.size(3) // 2
        
        # split freqs
        freqs = freqs.squeeze().split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # loop over samples
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w

            # precompute multipliers
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
                seq_len, n, -1, 2))
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
                                dim=-1).reshape(seq_len, 1, -1)

            # apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])

            # append to collection
            output.append(x_i)
        
        if ids_keep!=None:
            output = torch.stack(output).float()
            output = output.view(b_,ids_restore.shape[1],n_*d_)
            output = torch.gather(
                output, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, n_*d_))
            output = output.view(b_,s_,n_,d_)
            return output

        return torch.stack(output).float()
    else:
        if ids_restore!=None:
            x = x.view(b_,s_,n_*d_)

            mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

            x = x.view(b_,ids_restore.shape[1],n_,d_)

        n, c = x.size(2), x.size(3) // 2

        # loop over samples
        output = []
        seq_len = x.shape[1]
        x_i = torch.view_as_complex(x[0, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = freqs
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[0, seq_len:]])
        output.append(x_i)


        if ids_keep!=None:
            output = torch.stack(output).float()
            output = output.view(b_,ids_restore.shape[1],n_*d_)
            output = torch.gather(
                output, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, n_*d_))
            output = output.view(b_,s_,n_,d_)
            return output

        return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, mask_token, ids_restore, ids_keep, flag):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs, b, s, n, d, mask_token, ids_restore, ids_keep, flag = flag),
            k=rope_apply(k, grid_sizes, freqs, b, s, n, d, mask_token, ids_restore, ids_keep, flag = flag),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        ids_keep=None,
        ids_restore=None,
        mask_token=None,
        flag=True,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs, ids_restore=ids_restore, ids_keep=ids_keep, mask_token=mask_token,flag=flag)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x

from typing import Optional, Tuple, List, Dict, Any
import torch.nn.functional as F

def upsample_conv3d_weights_auto(conv_small: nn.Conv3d, size: Tuple[int,int,int]):
    # small: (OC_small, IC_small, kT, kH, kW)
    OC, IC, _, _, _ = conv_small.weight.shape
    with torch.no_grad():
        w = F.interpolate(conv_small.weight.data,
                          size=size, mode='trilinear', align_corners=False)
        big = nn.Conv3d(in_channels=IC, out_channels=OC,
                        kernel_size=size, stride=size, padding=0)
        big.weight.copy_(w)
        if conv_small.bias is not None:
            big.bias = nn.Parameter(conv_small.bias.data.clone())
        else:
            big.bias = None
    return big


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)
        self.d = d
        
        self.mask_ratio = 0.3
        self.mask_token = None

        self.patch_embedding_2x  = upsample_conv3d_weights_auto(self.patch_embedding, (1,4,4))
        self.patch_embedding_4x  = upsample_conv3d_weights_auto(self.patch_embedding, (1,8,8))
        self.patch_embedding_8x  = upsample_conv3d_weights_auto(self.patch_embedding, (1,16,16))
        self.patch_embedding_16x = upsample_conv3d_weights_auto(self.patch_embedding, (1,32,32))
        self.patch_embedding_2x_f = nn.Conv3d(
            self.patch_embedding.in_channels,
            self.patch_embedding.in_channels,
            kernel_size=(1,4,4), stride=(1,4,4),
        )


        # initialize weights
        self.init_weights()

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_unkeep = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep


    def forward_side_interpolater(self, x, mask, ids_restore, kwagrs):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x_before = x
        x = self.sideblock(x, **kwagrs)
        
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x*mask + (1-mask)*x_before

        return x
    
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        enable_mask=False,
        y=None,
        latent_frame_zero=8,
        input_ids=None,
        flag = True
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        #print(x[0].shape,"nscdbf09-vmifjw0")
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        if flag:
            x_pack = []
            # According to Eq. 3, compress historical frames at different ratios.
            for u in x:
                u = u.unsqueeze(0)
                f_num = u.shape[2]
                u1 = u[:,:,:-latent_frame_zero]
                u2 = u[:,:,-latent_frame_zero:]
                f_1 = rope_params(1024, self.d - 4 * (self.d // 6)).to(u.device)
                f_2 = rope_params(1024, 2 * (self.d // 6)).to(u.device)
                f_3 = rope_params(1024, 2 * (self.d // 6)).to(u.device)
                if f_num - latent_frame_zero <= 2 + 4:
                    f_zero = u1.shape[2]

                    u_1 = self.patch_embedding(u1[:,:,0].unsqueeze(2))

                    if f_zero - 2 <= 0:
                        u_2 = self.patch_embedding_2x(convpadd(u1[:,:,-1].unsqueeze(2),4))
                    else:
                        u_2 = self.patch_embedding_2x(convpadd(u1[:,:,1:-1],4))

                    u_3 = self.patch_embedding(u1[:,:,-1].unsqueeze(2))
                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    # Generate the corresponding RoPE encoding based on the compressed historical frames.
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2)],dim=0)
                    f_z = f1+f2+u_3.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2)],dim=1)


                elif f_num - latent_frame_zero <= 2 + 4 + 16:
                    
                    f_zero = u1.shape[2]
                    u_1 = self.patch_embedding(u1[:,:,0].unsqueeze(2))
                    
                    if f_zero-6<=0:
                        u_2 = self.patch_embedding_4x(convpadd(u1[:,:,-5].unsqueeze(2),8))
                    else:
                        u_2 = self.patch_embedding_4x(convpadd(u1[:,:,1:-5],8))


                    u_3 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4))
                    u_4 = self.patch_embedding(u1[:,:,-3:])
                    
                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    f3 = u_3.shape[2]
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2,True), up_fre(f_1,f_2,f_3,u_4,f1+f2+f3)],dim=0)
                    f_z = f1+f2+f3+u_4.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2),u_4.flatten(2).transpose(1, 2)],dim=1)


                elif f_num - latent_frame_zero <= 2 + 4 + 16 + 64:
                    f_zero = u1.shape[2]
                    u_1 = self.patch_embedding(u1[:,:,0].unsqueeze(2))
                    

                    if f_zero-22<=0:
                        u_2 = self.patch_embedding_8x(convpadd(u1[:,:,-21].unsqueeze(2),16))
                    else:
                        u_2 = self.patch_embedding_8x(convpadd(u1[:,:,1:-21],16))

                    u_3 = self.patch_embedding_4x(convpadd(u1[:,:,-21:-5],8))
                    u_4 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4)) 
                    u_5 = self.patch_embedding(u1[:,:,-3:])

                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    f3 = u_3.shape[2]
                    f4 = u_4.shape[2]
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2,True), up_fre(f_1,f_2,f_3,u_4,f1+f2+f3,True), \
                                          up_fre(f_1,f_2,f_3,u_5,f1+f2+f3+f4)],dim=0)
                    f_z = f1+f2+f3+f4+u_5.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2),u_4.flatten(2).transpose(1, 2),u_5.flatten(2).transpose(1, 2)],dim=1)


                elif f_num - latent_frame_zero <= 2 + 4 + 16 + 64 + 256: 
                    f_zero = u1.shape[2]
                    u_1 = self.patch_embedding_2x(convpadd(u1[:,:,0].unsqueeze(2),4))

                    if f_zero-86<=0:
                        u_2 = self.patch_embedding_16x(convpadd(u1[:,:,-85].unsqueeze(2),32))
                    else:
                        u_2 = self.patch_embedding_16x(convpadd(u1[:,:,1:-85],32))


                    u_3 = self.patch_embedding_8x(convpadd(u1[:,:,-85:-21],16))
                    u_4 = self.patch_embedding_4x(convpadd(u1[:,:,-21:-5],8))
                    u_5 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4))
                    u_6 = self.patch_embedding(u1[:,:,-3:])

                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    f3 = u_3.shape[2]
                    f4 = u_4.shape[2]
                    f5 = u_5.shape[2]
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2,True), up_fre(f_1,f_2,f_3,u_4,f1+f2+f3,True), \
                                          up_fre(f_1,f_2,f_3,u_5,f1+f2+f3+f4,True),up_fre(f_1,f_2,f_3,u_6,f1+f2+f3+f4+f5)],dim=0)
                    f_z = f1+f2+f3+f4+f5+u_6.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2),u_4.flatten(2).transpose(1, 2), \
                                    u_5.flatten(2).transpose(1, 2),u_6.flatten(2).transpose(1, 2)],dim=1)

                elif f_num - latent_frame_zero <= 2 + 4 + 16 + 64 + 256 + 1024:
                    f_zero = u1.shape[2]

                    u_1 = self.patch_embedding_2x(convpadd(u1[:,:,0].unsqueeze(2),4))

                    if f_zero - 342 <= 0:
                        u_2 = self.patch_embedding_16x(convpadd(self.patch_embedding_2x_f(convpadd(u1[:,:,-341].unsqueeze(2),4)), 32) )
                    else:
                        u_2 = self.patch_embedding_16x(convpadd(self.patch_embedding_2x_f(convpadd(u1[:,:,1:-341],4)), 32) )


                    u_3 = self.patch_embedding_16x(convpadd(u1[:,:,-341:-85],32) )
                    u_4 = self.patch_embedding_8x(convpadd(u1[:,:,-85:-21],16))
                    u_5 = self.patch_embedding_4x(convpadd(u1[:,:,-21:-5],8))
                    u_6 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4))
                    u_7 = self.patch_embedding(u1[:,:,-3:])

                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    f3 = u_3.shape[2]
                    f4 = u_4.shape[2]
                    f5 = u_5.shape[2]
                    f6 = u_6.shape[2]
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2,True), up_fre(f_1,f_2,f_3,u_4,f1+f2+f3,True), \
                                          up_fre(f_1,f_2,f_3,u_5,f1+f2+f3+f4,True),up_fre(f_1,f_2,f_3,u_6,f1+f2+f3+f4+f5,True),\
                                            up_fre(f_1,f_2,f_3,u_7,f1+f2+f3+f4+f5+f6)],dim=0)
                    f_z = f1+f2+f3+f4+f5+f6+u_7.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2),u_4.flatten(2).transpose(1, 2), \
                                    u_5.flatten(2).transpose(1, 2),u_6.flatten(2).transpose(1, 2),u_7.flatten(2).transpose(1, 2)],dim=1)

                u2 = self.patch_embedding(u2)
                freqs_i = torch.cat([freqs_i, up_fre(f_1,f_2,f_3,u2,f_z)],dim=0)
                seq_lens1 = u1.shape[1]
                grid_sizes = torch.stack([torch.tensor(u2.shape[2:], dtype=torch.long)])
                u2 = u2.flatten(2).transpose(1, 2)
                seq_lens = torch.tensor([u1.shape[1]+u2.shape[1]], dtype=torch.long)

                seq_len = int(seq_lens[0])
                u = torch.cat([u1,u2],dim=1)
                x_pack.append(u)
                t = t.squeeze()
                u1_shape = u1.shape[1]
                u2_shape = u2.shape[1]
                t = torch.cat([
                            t[0:1].new_ones(u1.shape[1]) * t[0],
                            t[-1:].new_ones(u2.shape[1]) * t[-1]
                        ])
                t = t.unsqueeze(0)
    
            self.freqs = freqs_i
            x = x_pack
            x = torch.cat(x)
        else:
            self.freqs = torch.cat([
                    rope_params(1024, self.d - 4 * (self.d // 6)),
                    rope_params(1024, 2 * (self.d // 6)),
                    rope_params(1024, 2 * (self.d // 6))
                ],
                                       dim=1).to(device)    
            # embeddings
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            assert seq_lens.max() <= seq_len
            x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                          dim=1) for u in x
            ])
                
        # We referenced the implementation at https://github.com/sail-sg/MDT
        ids_keep = None
        ids_restore = None
        if self.mask_ratio is not None and enable_mask:
            # masking: length -> length * mask_ratio
            rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio # mask_ratio, mask_ratio + 0.2 
            x_ori = x
            x, mask, ids_restore, ids_keep = self.random_masking(
                x, rand_mask_ratio)
            masked_stage = True
            seq_lens = torch.tensor([x.shape[1]], dtype=torch.long)
            seq_len_ori = seq_len
            seq_len = int(seq_lens.item()) 
            
            t_masked = torch.gather(
                t, dim=1, index=ids_keep)
            t_ori = t
            
            
        if self.mask_ratio is not None and enable_mask:
            t = t_masked
        
            with torch.amp.autocast('cuda', dtype=torch.float32):
                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
            seq_len = seq_len_ori
            with torch.amp.autocast('cuda', dtype=torch.float32):
                bt = t_ori.size(0)
                t = t_ori.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t).unflatten(0, (bt, seq_len)).float())
                e0_ori = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            # time embeddings
            if t.dim() == 1:
                t = t.expand(t.size(0), seq_len)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        
        len_blocks = (len(self.blocks)+1)//2
        
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            ids_keep=ids_keep,
            ids_restore=ids_restore,
            mask_token=self.mask_token,
            flag=flag)


        cnt_blocks = 0
        for block in self.blocks:
            cnt_blocks += 1
            if cnt_blocks==len_blocks and enable_mask:
                kwargs["ids_keep"]=None
                kwargs["ids_restore"]=None
                kwargs["mask_token"]=None
                kwargs["seq_lens"] = torch.tensor([x.shape[1]], dtype=torch.long)
                kwargs["e"] = e0_ori
                x = self.forward_side_interpolater(x, mask, ids_restore, kwargs)
                x = block(x, **kwargs)
            else:
                kwargs["seq_lens"] = torch.tensor([x.shape[1]], dtype=torch.long)
                x = block(x, **kwargs)


        # head
        x = self.head(x, e)
        
        if flag:
            # unpatchify
            x = self.unpatchify(x[:,seq_lens1:,:], grid_sizes)
        else:
            # unpatchify
            x = self.unpatchify(x, grid_sizes)

        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

        

def convpadd(tensor,pad_num):
    dim = tensor.dim()
    if dim==4:
        tensor = tensor.unsqueeze(2)
    if dim==6:
        tensor = tensor.squeeze(2)
        
    b,c,f,h,w = tensor.shape

    pad_h = (pad_num - h % pad_num) % pad_num  
    pad_w = (pad_num - w % pad_num) % pad_num  
    tensor = torch.cat([tensor,torch.zeros(b,c,f,pad_h,w).to(tensor.device)],dim=3)
    tensor = torch.cat([tensor,torch.zeros(b,c,f,h+pad_h,pad_w).to(tensor.device)],dim=4)
    return tensor

def up_fre(f_1,f_2,f_3,u,f_z,scale=False):
    b1, c1, f1, h1, w1 = u.shape
    freqs_i = torch.cat([
        f_1[f_z:f_z+f1].view(f1, 1, 1, -1).expand(f1, h1, w1, -1),
        f_2[:h1].view(1, h1, 1, -1).expand(f1, h1, w1, -1),
        f_3[:w1].view(1, 1, w1, -1).expand(f1, h1, w1, -1)],
    dim=-1).reshape(f1*h1*w1, 1, -1)
    return freqs_i
