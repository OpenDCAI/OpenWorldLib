from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_wan import _get_qkv_projections, _get_added_kv_projections

try:
    from ..src.sp_utils.communications import all_to_all_4D
    from ..src.sp_utils.parallel_states import get_parallel_state
    from ..src.general_utils import rank0_log
except ImportError:
    from src.sp_utils.communications import all_to_all_4D
    from src.sp_utils.parallel_states import get_parallel_state
    from src.general_utils import rank0_log
import einops

# Auto-detect Flash Attention availability; priority: flash3 > flash2 > SDPA
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func

    flash_attn_func = flash_attn_3_func
    FLASH_ATTN_AVAILABLE = True
    rank0_log("********************Using Flash Attention 3********************")
except ImportError:
    try:
        from flash_attn import flash_attn_func as flash_attn_2_func

        flash_attn_func = flash_attn_2_func
        FLASH_ATTN_AVAILABLE = True
        rank0_log("********************Using Flash Attention 2********************")
    except ImportError:
        FLASH_ATTN_AVAILABLE = False
        flash_attn_func = None
        rank0_log("********************Flash Attention NOT available, using SDPA********************")


class WanAttnProcessorSP:
    def __init__(self, sp_size=1):
        self.sp_size = sp_size
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        parallel_dims = get_parallel_state()
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            def apply_rotary_emb(
                    hidden_states: torch.Tensor,
                    freqs_cos: torch.Tensor,
                    freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        head_dim = query.shape[-1]

        if self.sp_size > 1:
            # Stack QKV
            qkv = torch.cat([query, key, value], dim=0)  # [3, num_heads, seq_len_per_sp_rank, head_dim]
            # Redistribute heads across sequence dimension
            qkv = all_to_all_4D(qkv, group=parallel_dims.sp_group, scatter_dim=2, gather_dim=1)  # [3, seq_len_per_sp_rank * sp, num_heads//sp, head_dim]
            # Apply backend-specific preprocess_qkv
            query, key, value = qkv.chunk(3, dim=0)

        if FLASH_ATTN_AVAILABLE:
            hidden_states = flash_attn_func(
                query,
                key,
                value,
                softmax_scale=head_dim ** -0.5,
                causal=False)
        else:
            query = query.transpose(1, 2)  # [b,head,l,c]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2)

        if self.sp_size > 1:
            hidden_states = all_to_all_4D(hidden_states, group=parallel_dims.sp_group, scatter_dim=1, gather_dim=2)  # [1, seq_len_per_sp_rank, num_heads, head_dim]

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SimpleAttnProcessor2_0:
    def __init__(self):
        self.sp_size = 1
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SimpleAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            extrinsics=None,
            intrinsics=None,
            patches_x=None,
            patches_y=None,
            image_width=None,
            image_height=None,
            **kwargs
    ) -> torch.Tensor:

        parallel_dims = get_parallel_state()
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))  # [b,l,nhead,c]

        if rotary_emb is not None:
            def apply_rotary_emb(
                    hidden_states: torch.Tensor,
                    freqs_cos: torch.Tensor,
                    freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        head_dim = query.shape[-1]

        if self.sp_size > 1:
            # Stack QKV
            qkv = torch.cat([query, key, value], dim=0)  # [3, seq_len_per_sp_rank, num_heads, head_dim]
            # Redistribute heads across sequence dimension
            qkv = all_to_all_4D(qkv, group=parallel_dims.sp_group, scatter_dim=2, gather_dim=1)  # [3, seq_len_per_sp_rank * sp, num_heads//sp, head_dim]
            # Apply backend-specific preprocess_qkv
            query, key, value = qkv.chunk(3, dim=0)  # [1,l,head/sp,c]

        if FLASH_ATTN_AVAILABLE:
            hidden_states = flash_attn_func(
                query.to(torch.bfloat16),
                key.to(torch.bfloat16),
                value.to(torch.bfloat16),
                softmax_scale=head_dim ** -0.5,
                causal=False)
        else:
            query = query.transpose(1, 2)  # [b,head,l,c]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2)

        if self.sp_size > 1:
            hidden_states = all_to_all_4D(hidden_states, group=parallel_dims.sp_group, scatter_dim=1, gather_dim=2)  # [1, seq_len_per_sp_rank, num_heads, head_dim]

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class SimpleCogVideoXLayerNormZero(nn.Module):
    def __init__(
            self,
            conditioning_dim: int,
            embedding_dim: int,
            elementwise_affine: bool = True,
            eps: float = 1e-5,
            bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 3 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor):
        shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        return hidden_states, gate[:, None, :]


class SingleAttentionBlock(nn.Module):

    def __init__(
            self,
            dim,
            ffn_dim,
            num_heads,
            time_embed_dim=512,
            qk_norm="rms_norm_across_heads",
            eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        processor = SimpleAttnProcessor2_0()

        # layers
        self.norm1 = SimpleCogVideoXLayerNormZero(
            time_embed_dim, dim, elementwise_affine=True, eps=1e-5, bias=True
        )
        self.self_attn = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=processor,
        )
        self.norm2 = SimpleCogVideoXLayerNormZero(
            time_embed_dim, dim, elementwise_affine=True, eps=1e-5, bias=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

    def forward(
            self,
            hidden_states,
            temb,
            rotary_emb,
            extrinsics=None,
            intrinsics=None,
            patches_x=None,
            patches_y=None,
            image_width=None,
            image_height=None,
    ):
        # norm & modulate
        norm_hidden_states, gate_msa = self.norm1(hidden_states, temb)

        # attention
        attn_hidden_states = self.self_attn(hidden_states=norm_hidden_states,
                                            rotary_emb=rotary_emb,
                                            extrinsics=extrinsics,
                                            intrinsics=intrinsics,
                                            patches_x=patches_x,
                                            patches_y=patches_y,
                                            image_width=image_width,
                                            image_height=image_height)

        hidden_states = hidden_states + gate_msa * attn_hidden_states

        # norm & modulate
        norm_hidden_states, gate_ff = self.norm2(hidden_states, temb)

        # feed-forward
        ff_output = self.ffn(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output

        return hidden_states


class RefAttnProcessor2_0:
    _attention_backend = None

    def __init__(self):
        self.sp_size = 1
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SimpleAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            ref_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_emb_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            rotary_emb_ref: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            f=None,
            h=None,
            w=None,
            **kwargs
    ) -> [torch.Tensor, torch.Tensor]:
        parallel_dims = get_parallel_state()

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        ref_query = attn.to_q(ref_states)
        ref_key = attn.to_k(ref_states)
        ref_value = attn.to_v(ref_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
            ref_query = attn.norm_q(ref_query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            ref_key = attn.norm_k(ref_key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))  # [b,l,nhead,c]

        ref_query = ref_query.unflatten(2, (attn.heads, -1))
        ref_key = ref_key.unflatten(2, (attn.heads, -1))
        ref_value = ref_value.unflatten(2, (attn.heads, -1))  # [b,l,nhead,c]

        if rotary_emb_hidden is not None:
            def apply_rotary_emb(
                    hidden_states: torch.Tensor,
                    freqs_cos: torch.Tensor,
                    freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb_hidden)
            key = apply_rotary_emb(key, *rotary_emb_hidden)

            ref_query = apply_rotary_emb(ref_query, *rotary_emb_ref)
            ref_key = apply_rotary_emb(ref_key, *rotary_emb_ref)

        head_dim = query.shape[-1]
        if self.sp_size > 1:
            # Stack QKV
            qkv = torch.cat([query, key, value], dim=0)  # [3, seq_len_per_sp_rank, num_heads, head_dim]
            # Redistribute heads across sequence dimension
            qkv = all_to_all_4D(qkv, group=parallel_dims.sp_group, scatter_dim=2, gather_dim=1)  # [3, seq_len_per_sp_rank * sp, num_heads//sp, head_dim]
            # Apply backend-specific preprocess_qkv
            query, key, value = qkv.chunk(3, dim=0)

            query = einops.rearrange(query, "b (f h w) n c -> b f h w n c", f=f, h=h, w=w)
            key = einops.rearrange(key, "b (f h w) n c -> b f h w n c", f=f, h=h, w=w)
            value = einops.rearrange(value, "b (f h w) n c -> b f h w n c", f=f, h=h, w=w)

            qkv_ref = torch.cat([ref_query, ref_key, ref_value], dim=0)  # [3, seq_len_per_sp_rank, num_heads, head_dim]
            qkv_ref = all_to_all_4D(qkv_ref, group=parallel_dims.sp_group, scatter_dim=2, gather_dim=1)  # [3, seq_len_per_sp_rank * sp, num_heads//sp, head_dim]
            ref_query, ref_key, ref_value = qkv_ref.chunk(3, dim=0)

            ref_query = einops.rearrange(ref_query, "b (f h w) n c -> b f h w n c", f=f, h=h, w=w)
            ref_key = einops.rearrange(ref_key, "b (f h w) n c -> b f h w n c", f=f, h=h, w=w)
            ref_value = einops.rearrange(ref_value, "b (f h w) n c -> b f h w n c", f=f, h=h, w=w)

            combine_query = torch.cat([query, ref_query], dim=3)
            combine_key = torch.cat([key, ref_key], dim=3)
            combine_value = torch.cat([value, ref_value], dim=3)

            combine_query = einops.rearrange(combine_query, "b f h w n c -> (b f) (h w) n c")
            combine_key = einops.rearrange(combine_key, "b f h w n c -> (b f) (h w) n c")
            combine_value = einops.rearrange(combine_value, "b f h w n c -> (b f) (h w) n c")

            if FLASH_ATTN_AVAILABLE:
                hidden_states = flash_attn_func(
                    combine_query,
                    combine_key,
                    combine_value,
                    softmax_scale=head_dim ** -0.5,
                    causal=False)
            else:
                query = query.transpose(1, 2)  # [b,head,l,c]
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2)

            hidden_states = einops.rearrange(hidden_states, "(b f) (h w) n c -> b f h w n c", f=f, h=h, w=2 * w)

            ref_states = hidden_states[:, :, :, w:, :].contiguous()
            hidden_states = hidden_states[:, :, :, :w, :].contiguous()

            hidden_states = einops.rearrange(hidden_states, "b f h w n c -> b (f h w) n c")
            ref_states = einops.rearrange(ref_states, "b f h w n c -> b (f h w) n c")

            hidden_states = all_to_all_4D(hidden_states, group=parallel_dims.sp_group, scatter_dim=1, gather_dim=2)  # [1, seq_len_per_sp_rank, num_heads, head_dim]
            ref_states = all_to_all_4D(ref_states, group=parallel_dims.sp_group, scatter_dim=1, gather_dim=2)  # [1, seq_len_per_sp_rank, num_heads, head_dim]
        else:
            if FLASH_ATTN_AVAILABLE:
                hidden_states = flash_attn_func(
                    query,
                    key,
                    value,
                    softmax_scale=head_dim ** -0.5,
                    causal=False)
            else:
                query = query.transpose(1, 2)  # [b,head,l,c]
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        ref_states = ref_states.flatten(2, 3)
        ref_states = ref_states.type_as(query)

        ref_states = attn.to_out[0](ref_states)
        ref_states = attn.to_out[1](ref_states)
        return hidden_states, ref_states


class SingleAttentionBlockRef(nn.Module):

    def __init__(
            self,
            dim,
            ffn_dim,
            num_heads,
            time_embed_dim=512,
            qk_norm="rms_norm_across_heads",
            eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.norm1 = SimpleCogVideoXLayerNormZero(
            time_embed_dim, dim, elementwise_affine=True, eps=1e-5, bias=True
        )
        self.self_attn = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=RefAttnProcessor2_0(),
        )
        self.norm2 = SimpleCogVideoXLayerNormZero(
            time_embed_dim, dim, elementwise_affine=True, eps=1e-5, bias=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

    def forward(
            self,
            hidden_states,
            ref_states,
            temb,
            rotary_emb_hidden,
            rotary_emb_ref,
            f,
            h,
            w
    ):
        # norm & modulate
        norm_hidden_states, gate_msa = self.norm1(hidden_states, temb)
        norm_ref_states, gate_msa_ref = self.norm1(ref_states, temb)

        # attention
        attn_hidden_states, attn_ref_states = self.self_attn(hidden_states=norm_hidden_states, ref_states=norm_ref_states, rotary_emb_hidden=rotary_emb_hidden, rotary_emb_ref=rotary_emb_ref, f=f, h=h, w=w)

        hidden_states = hidden_states + gate_msa * attn_hidden_states

        # norm & modulate
        norm_hidden_states, gate_ff = self.norm2(hidden_states, temb)

        ref_states = ref_states + gate_msa_ref * attn_ref_states
        norm_ref_states, gate_ff_ref = self.norm2(ref_states, temb)

        # feed-forward
        ff_output = self.ffn(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output

        ff_ref = self.ffn(norm_ref_states)
        ref_states = ref_states + gate_ff_ref * ff_ref

        return hidden_states, ref_states


class WanAttnProcessorSparseSpatialSP:
    def __init__(self, sp_size=1):
        self.sp_size = sp_size
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            ref_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            rotary_emb_ref: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            f=None,
            h=None,
            w=None,
            ref_index=None,
    ) -> [torch.Tensor, torch.Tensor]:
        parallel_dims = get_parallel_state()

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        ref_query = attn.to_q(ref_states)
        ref_key = attn.to_k(ref_states)
        ref_value = attn.to_v(ref_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        ref_query = attn.norm_q(ref_query)
        ref_key = attn.norm_k(ref_key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        ref_query = ref_query.unflatten(2, (attn.heads, -1))
        ref_key = ref_key.unflatten(2, (attn.heads, -1))
        ref_value = ref_value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            def apply_rotary_emb(
                    hidden_states: torch.Tensor,
                    freqs_cos: torch.Tensor,
                    freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

            ref_query = apply_rotary_emb(ref_query, *rotary_emb_ref)
            ref_key = apply_rotary_emb(ref_key, *rotary_emb_ref)

        head_dim = query.shape[-1]

        if self.sp_size > 1:
            # Stack QKV
            qkv = torch.cat([query, key, value], dim=0)  # [3, seq_len_per_sp_rank, num_heads, head_dim]
            qkv = all_to_all_4D(qkv, group=parallel_dims.sp_group, scatter_dim=2, gather_dim=1)  # [3, seq_len_per_sp_rank * sp, num_heads//sp, head_dim]
            query, key, value = qkv.chunk(3, dim=0)  # [1, L, H, C]

            qkv_ref = torch.cat([ref_query, ref_key, ref_value], dim=0)  # [3, seq_len_per_sp_rank, num_heads, head_dim]
            qkv_ref = all_to_all_4D(qkv_ref, group=parallel_dims.sp_group, scatter_dim=2, gather_dim=1)  # [3, seq_len_per_sp_rank * sp, num_heads//sp, head_dim]
            ref_query, ref_key, ref_value = qkv_ref.chunk(3, dim=0)

        ref_f = ref_query.shape[1]

        # Build combine_query/key/value by concatenating ref at corresponding frames
        combine_query = torch.cat([ref_query, query], dim=1)
        combine_key = torch.cat([ref_key, key], dim=1)
        combine_value = torch.cat([ref_value, value], dim=1)

        if FLASH_ATTN_AVAILABLE:
            hidden_states = flash_attn_func(
                combine_query,
                combine_key,
                combine_value,
                softmax_scale=head_dim ** -0.5,
                causal=False)
        else:
            combine_query = combine_query.transpose(1, 2)  # [b,head,l,c]
            combine_key = combine_key.transpose(1, 2)
            combine_value = combine_value.transpose(1, 2)
            hidden_states = F.scaled_dot_product_attention(
                combine_query, combine_key, combine_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2)

        # Split hidden_states back to hidden_states and ref_states
        ref_states = hidden_states[:, :ref_f]
        hidden_states = hidden_states[:, ref_f:]

        if self.sp_size > 1:
            hidden_states = all_to_all_4D(hidden_states, group=parallel_dims.sp_group, scatter_dim=1, gather_dim=2)  # [1, seq_len_per_sp_rank, num_heads, head_dim]
            ref_states = all_to_all_4D(ref_states, group=parallel_dims.sp_group, scatter_dim=1, gather_dim=2)  # [1, seq_len_per_sp_rank, num_heads, head_dim]

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        ref_states = ref_states.flatten(2, 3)
        ref_states = ref_states.type_as(query)

        ref_states = attn.to_out[0](ref_states)
        ref_states = attn.to_out[1](ref_states)

        return hidden_states, ref_states
