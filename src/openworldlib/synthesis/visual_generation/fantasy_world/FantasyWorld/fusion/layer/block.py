# Copyright Alibaba Inc. All Rights Reserved.

import torch.nn as nn
import torch
from torch import Tensor
import warnings
from typing import Callable, Literal, Tuple, Optional
import torch.nn.functional as F
from openworldlib.base_models.diffusion_model.diffsynth.models.wan_video_dit import DiTBlock, modulate
from openworldlib.base_models.three_dimensions.point_clouds.vggt.vggt.layers.block import Block as VGGTBlock
import math
from einops import rearrange
from openworldlib.base_models.diffusion_model.diffsynth.models.wan_video_dit import rope_apply

from torch.utils.checkpoint import checkpoint


class PartialVGGTBlock(nn.Module):
    """Adapter that adds FantasyWorld's partial-forward hooks to the shared VGGT block."""

    def __init__(self, block: VGGTBlock) -> None:
        super().__init__()
        self.norm1 = block.norm1
        self.attn = block.attn
        dim = self.norm1.normalized_shape[0]
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.ls1 = block.ls1
        self.drop_path1 = block.drop_path1
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = block.ls2
        self.drop_path2 = block.drop_path2
        self.sample_drop_ratio = block.sample_drop_ratio

    def attn_residual_func(self, x: Tensor, pos=None, e=None) -> Tensor:
        if e is None:
            return self.ls1(self.attn(self.norm1(x), pos=pos))
        return self.ls1(self.attn(self.norm1(x) * (1 + e[1]) + e[0], pos=pos))

    def ffn_residual_func(self, x: Tensor, e=None) -> Tensor:
        if e is None:
            return self.ls2(self.mlp(self.norm2(x)))
        return self.ls2(self.mlp(self.norm2(x)) * (1 + e[4]) + e[3]) * e[5]

    def forward(
        self,
        x: Tensor,
        pos=None,
        e0=None,
        return_partial: bool = False,
        run_remaining: bool = False,
        modifiers: Tuple[Tensor, ...] | None = None,
    ) -> Tensor | tuple[Tensor, tuple]:
        if run_remaining:
            if modifiers is None:
                raise ValueError("run_remaining requires modifiers")
            return x + self.ffn_residual_func(x, e=modifiers)

        if e0 is not None:
            batch_size = e0.shape[0]
            if batch_size != x.shape[0]:
                e0 = e0.unsqueeze(1).repeat(1, x.shape[0] // batch_size, 1, 1)
                e0 = e0.reshape(x.shape[0], 6, -1)

        if e0 is not None:
            e_mod = (self.modulation + e0).chunk(6, dim=1)
        else:
            e_mod = None

        x = x + self.attn_residual_func(x, pos=pos, e=e_mod)
        if return_partial:
            return x, e_mod

        if modifiers is not None:
            e_mod = modifiers

        return x + self.ffn_residual_func(x, e=e_mod)

    def forward_partial(self, *args, **kwargs):
        return self.forward(*args, **kwargs, return_partial=True)

    def forward_remaining(self, x: Tensor, e=None):
        return self.forward(x, run_remaining=True, modifiers=e)


class IRGBlock(nn.Module):
    def __init__(
        self,
        x_dit_block: DiTBlock,
        x_agg_block: PartialVGGTBlock,
        m1_dim, m2_dim, hidden_size, num_heads,
        drop_path=0.0,
        enable_layernorm_kernel=False,
        enable_flash_attn=False,
        init_values=1e-4,
        bica_mode: Literal['overall', 'temporal'] = 'overall',
    ):
        super().__init__()
        self.x_dit = x_dit_block
        self.x_agg = x_agg_block

        self.bicross_attention = CrossModalityBiAttentionBlock(
            m1_dim, m2_dim, hidden_size, num_heads,
            drop_path=drop_path,
            enable_layernorm_kernel=enable_layernorm_kernel,
            enable_flash_attn=enable_flash_attn,
            init_values=init_values,
            bica_mode=bica_mode,
        )

    def _dit_forward_partial(self, x, context, t_mod, freqs, **kwargs):
        if hasattr(self.x_dit, "forward_partial"):
            return self.x_dit(
                x, context, t_mod, freqs,
                return_partial=True, **kwargs
            )

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.x_dit.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = modulate(self.x_dit.norm1(x), shift_msa, scale_msa)
        x = self.x_dit.gate(x, gate_msa, self.x_dit.self_attn(input_x, freqs))
        x = x + self.x_dit.cross_attn(self.x_dit.norm3(x), context, **kwargs)
        return x, (shift_mlp, scale_mlp, gate_mlp)

    def _dit_forward_remaining(self, x, context, t_mod, freqs, modifiers, **kwargs):
        if hasattr(self.x_dit, "forward_remaining"):
            return self.x_dit(
                x, context, t_mod, freqs,
                run_remaining=True,
                modifiers=modifiers, **kwargs,
            )

        shift_mlp, scale_mlp, gate_mlp = modifiers
        input_x = modulate(self.x_dit.norm2(x), shift_mlp, scale_mlp)
        return self.x_dit.gate(x, gate_mlp, self.x_dit.ffn(input_x))

    def _forward_impl(
        self,
        x_dit: torch.Tensor,
        x_agg: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        freqs_dit: torch.Tensor,
        freqs_agg: torch.Tensor,
        pos: torch.Tensor | None = None,
        e0: torch.Tensor | None = None,
        uncond=False,
        **kwargs,
    ):

        _, P, D = x_agg.shape
        x_dit_p, mod_dit = self._dit_forward_partial(
            x_dit, context, t_mod, freqs, **kwargs
        )
        pos = rearrange(pos, '(b s) p d -> b (s p) d', b=x_dit_p.size(0))
        x_agg = rearrange(x_agg, '(b s) p d -> b (s p) d', b=x_dit_p.size(0))
        B, _, _ = x_agg.shape
        x_agg_p, mod_agg = self.x_agg(
            x_agg, pos=pos, e0=e0,
            return_partial=True
        )
        if uncond is True:
            x_dit_f = x_dit_p
            x_agg_f = x_agg_p
        else:
            x_dit_f, x_agg_f = self.bicross_attention(
                [x_dit_p, x_agg_p], freqs=freqs, freqs_dit=freqs_dit, freqs_agg=freqs_agg,
            )
        x_dit_out = self._dit_forward_remaining(
            x_dit_f, context, t_mod, freqs, mod_dit, **kwargs
        )

        x_agg_out = self.x_agg(
            x_agg_f,
            run_remaining=True,
            modifiers=mod_agg
        )

        intermediates = [x_agg_out.view(B, -1, P, D)]

        del x_dit_p, x_agg_p, x_dit_f, x_agg_f
        del mod_dit, mod_agg

        return x_dit_out, x_agg_out, intermediates

    # ------------------------------------------------------------
    def forward(
        self,
        x_dit: torch.Tensor,
        x_agg: torch.Tensor,
        *,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        freqs_dit: torch.Tensor,
        freqs_agg: torch.Tensor,
        pos: torch.Tensor | None = None,
        e0: torch.Tensor | None = None,
        uncond=False,
        **kwargs,
    ):

        if self.training:
            return checkpoint(
                self._forward_impl,
                x_dit,
                x_agg,
                context,
                t_mod,
                freqs,
                freqs_dit,
                freqs_agg,
                pos,
                e0,
                use_reentrant=False,
                preserve_rng_state=False,
                uncond=uncond,
                **kwargs,
            )
        else:
            return self._forward_impl(
                x_dit,
                x_agg,
                context,
                t_mod,
                freqs,
                freqs_dit,
                freqs_agg,
                pos,
                e0,
                uncond=uncond,
                **kwargs,
            )


class CrossModalityBiAttentionBlock(nn.Module):
    def __init__(self,
                 m1_dim,
                 m2_dim,
                 hidden_size,
                 num_heads,
                 drop_path=0.0,
                 enable_layernorm_kernel=False,
                 enable_flash_attn=False,
                 init_values=1e-4,
                 bica_mode: Literal['overall',
                                    'temporal'] = 'overall'):
        super().__init__()
        self.m1_dim = m1_dim
        self.m2_dim = m2_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.attn_norm_m1 = get_layernorm(
            m1_dim, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn_norm_m2 = get_layernorm(
            m2_dim, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.cross_attn = BiMultiHeadAttention(
            m1_dim, m2_dim, hidden_size, num_heads, dropout=0.0,
            attn_implementation='flash_attn_2' if enable_flash_attn else 'sdpa'
        )

        self.gamma_m1 = nn.Parameter(torch.zeros((m1_dim)), requires_grad=True)
        self.gamma_m2 = nn.Parameter(torch.zeros((m2_dim)), requires_grad=True)

        self.drop_path = nn.Identity()
        self.bica_mode = bica_mode

    def forward(self,
                xs: Tuple[torch.Tensor],
                attention_masks: Optional[Tuple[torch.Tensor]] = (None,
                                                                  None),
                T: int = None,
                S: int = None,
                R: int = None,
                M: int = None,
                freqs=None,
                freqs_dit=None,
                freqs_agg=None):
        # x1: shape(B, T*S, C), x2: shape(B, R*M, C)
        x1, x2 = xs
        attention_mask_1, attention_mask_2 = attention_masks
        if attention_mask_1 is not None or attention_mask_2 is not None:
            raise NotImplementedError(
                'attention mask is currently unsupported for video-audio cross attention')

        x_m1, x_m2 = self.attn_norm_m1(x1), self.attn_norm_m2(x2)
        if self.bica_mode == 'overall':
            dx_m1, dx_m2 = self.cross_attn(
                x_m1, x_m2, attention_mask_1, attention_mask_2, freqs=freqs, freqs_dit=freqs_dit, freqs_agg=freqs_agg)
        elif self.bica_mode == 'temporal':
            assert (B := x1.shape[0]) == x2.shape[0]
            assert (C := x1.shape[-1]) == x2.shape[-1]
            x_m1, x_m2 = x_m1.view(B, T, S, C), x_m2.view(B, R, M, C)
            # shape (B, R, M, C) -> (B, T, M*r, C)
            x_m2, x_m2_pad_mask = self.auto_temporal_slice(
                x_m2, pad_mask=None, window_num=T)
            # shape (B, T, S / M*r, C) -> (B*T, S / M*r, C)
            x_m1, x_m2 = x_m1.flatten(0, 1), x_m2.flatten(0, 1)
            attention_mask_2 = ~x_m2_pad_mask.flatten(0, 1)

            dx_m1, dx_m2 = self.cross_attn(x_m1, x_m2, None, attention_mask_2)

            dx_m1 = dx_m1.view(B, T * S, C)
            dx_m2 = dx_m2[attention_mask_2].view(B, R * M, C)
        else:
            raise NotImplementedError(self.bica_mode)
        x1 = x1 + self.drop_path(self.gamma_m1 * dx_m1)
        x2 = x2 + self.drop_path(self.gamma_m2 * dx_m2)

        return x1, x2

    def auto_temporal_slice(
            self,
            x: torch.Tensor,
            pad_mask: torch.Tensor,
            window_num: int):
        """
        Rearrange 1D padded tensor into a 2D tensor with zeros uniformly distributed.
        Thanks to DeepSeek-R1.

        Args:
            a (Tensor): Input tensor of shape (batch_size, length).
            mask (Tensor): Boolean mask tensor of shape (batch_size, length), True for valid elements.
            cols (int): Number of columns in the output 2D tensor.

        Returns:
            Tensor: Output tensor of shape (batch_size, rows, cols), where rows = length // cols.
        """
        # x: [B, T, S, C], pad_mask: [B, T, S]
        B, T, S, C = x.shape

        pad_len = math.ceil(T / window_num) * window_num - T
        if pad_len > 0:
            if pad_mask is None:
                pad_mask = torch.full(
                    x.shape[:3], 0, dtype=torch.bool, device=x.device)
            x = smart_pad(x, pad_len, dim=1, mode='constant', value=0.)
            pad_mask = smart_pad(
                pad_mask,
                pad_len,
                dim=1,
                mode='constant',
                value=True)
        T += pad_len

        # [B, T] equal for (frequency component)
        valid_mask = ~pad_mask[:, :, 0]
        window_size = T // window_num
        device = x.device

        # Flatten batch and sequence dimensions to handle all elements
        flat_mask = valid_mask.flatten()
        valid_elements = x.view(B * T, S * C)[flat_mask]  # (total_valid, Sa*C)

        # Compute indices for each valid element in the original tensor
        batch_indices = torch.arange(B, device=device).unsqueeze(
            1).expand(-1, T).reshape(-1)
        batch_indices = batch_indices[flat_mask]  # (total_valid,)

        # Calculate the number of valid elements per sample
        n_elements = valid_mask.sum(dim=1)  # (batch_size,)
        valid_n_elements = n_elements[batch_indices]

        # Calculate row and column indices for valid elements
        cum_counts = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), n_elements.cumsum(0)])
        local_indices = torch.arange(
            len(valid_elements),
            device=device) - cum_counts[batch_indices]

        # Compute row & col indices:
        rows_f = float(window_num)
        r = (local_indices.float() * rows_f / valid_n_elements.float()
             ).floor().long()  # (total_valid,)
        k = (local_indices - r * valid_n_elements.float() /
             rows_f).floor().long()      # (total_valid,)

        # Ensure rows/columns do not exceed rows-1/cols-1
        valid_mask = (k < window_size) & (r < window_num)
        final_r = r[valid_mask]
        final_k = k[valid_mask]
        final_values = valid_elements[valid_mask]  # (total_valid, Sa*C)
        final_batch = batch_indices[valid_mask]

        # Create output tensor and scatter values
        output = torch.zeros(
            (B,
             window_num,
             window_size,
             S * C),
            device=device,
            dtype=x.dtype)
        output_pad_mask = torch.ones(
            (B, window_num, window_size, S), device=device, dtype=pad_mask.dtype)
        output[final_batch, final_r, final_k] = final_values
        output_pad_mask[final_batch, final_r, final_k] = 0

        output = output.view(B, window_num, window_size * S, C).contiguous()
        output_pad_mask = output_pad_mask.view(
            B, window_num, window_size * S).contiguous()

        return output, output_pad_mask


class BiMultiHeadAttention(nn.Module):
    def __init__(self,
                 m1_dim,
                 m2_dim,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 attn_implementation: Literal['eager',
                                              'sdpa',
                                              'flash_attn_2'] = 'sdpa'):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.m1_dim = m1_dim
        self.m2_dim = m2_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.m1_proj = nn.Linear(self.m1_dim, self.embed_dim)
        self.m2_proj = nn.Linear(self.m2_dim, self.embed_dim)
        self.values_m1_proj = nn.Linear(self.m1_dim, self.embed_dim)
        self.values_m2_proj = nn.Linear(self.m2_dim, self.embed_dim)

        self.out_m1_proj = nn.Linear(self.embed_dim, self.m1_dim)
        self.out_m2_proj = nn.Linear(self.embed_dim, self.m2_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()
        self.attn_implementation = attn_implementation

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(
            bsz,
            seq_len,
            self.num_heads,
            self.head_dim).transpose(
            1,
            2).contiguous()

    def _reset_parameters(self):
        for proj in [
            self.m1_proj, self.values_m1_proj, self.out_m1_proj,
            self.m2_proj, self.values_m2_proj, self.out_m2_proj
        ]:
            nn.init.xavier_uniform_(proj.weight)
            proj.bias.data.fill_(0)

    def forward(
            self,
            x1,
            x2,
            attention_mask_1: torch.Tensor = None,
            attention_mask_2: torch.Tensor = None,
            freqs=None,
            freqs_dit=None,
            freqs_agg=None):
        """_summary_

        Args:
            x1 (_type_): bs, n_m1, dim
            x2 (_type_): bs, n_m2, dim
            attention_mask_1 (_type_, optional): _description_. bs, n_m1
            attention_mask_2 (_type_, optional): _description_. bs, n_m2

        Returns:
            _type_: _description_
        """

        attn_implementation = getattr(self, 'attn_implementation', 'eager')
        if attn_implementation == 'eager':
            return self.forward_eager(
                x1, x2, attention_mask_1, attention_mask_2)
        elif attn_implementation == 'sdpa':
            return self.forward_sdpa(
                x1,
                x2,
                attention_mask_1,
                attention_mask_2,
                freqs=freqs,
                freqs_dit=freqs_dit,
                freqs_agg=freqs_agg)
        elif attn_implementation == 'flash_attn_2':
            return self.forward_flash_attn_2(
                x1, x2, attention_mask_1, attention_mask_2)
        else:
            raise NotImplementedError(attn_implementation)

    def forward_eager(
            self,
            x1,
            x2,
            attention_mask_1: torch.Tensor = None,
            attention_mask_2: torch.Tensor = None):
        bsz, L1, _ = x1.size()
        device = x1.device

        # shape(B, L1, C)
        query_states = self.m1_proj(x1) * self.scale
        key_states = self._shape(self.m2_proj(x2), -
                                 1, bsz)             # shape(B, h, L2, d)
        value_m1_states = self._shape(
            self.values_m1_proj(x1), -1, bsz)  # shape(B, h, L1, d)
        value_m2_states = self._shape(
            self.values_m2_proj(x2), -1, bsz)  # shape(B, h, L2, d)

        # shape(B*h, L, d)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, L1, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_m1_states = value_m1_states.view(*proj_shape)
        value_m2_states = value_m2_states.view(*proj_shape)

        L2 = key_states.size(1)  # L2
        # shape(B*h, L1, L2)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, L1, L2):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, L1, L2)}, "
                f"but is {attn_weights.size()}")

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # shape(B*h, L2, L1)
        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_2 = attn_weights_T - \
            torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_2 = torch.clamp(
                attn_weights_2, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_2 = torch.clamp(
                attn_weights_2, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        if attention_mask_1 is not None or attention_mask_2 is not None:
            if attention_mask_1 is None:
                attention_mask_1 = torch.ones(
                    (bsz, L1), dtype=attention_mask_2.dtype, device=device)
            if attention_mask_2 is None:
                attention_mask_2 = torch.ones(
                    (bsz, L2), dtype=attention_mask_1.dtype, device=device)
            # shape(L1, L2)
            mask_m2_to_m1 = attention_mask_1[:, :,
                                             None] | attention_mask_2[:, None, :]
            attn_weights.masked_fill_(
                torch.logical_not(mask_m2_to_m1), float("-inf"))
            # shape(L2, L1)
            mask_m1_to_m2 = mask_m2_to_m1.transpose(1, 2)
            attn_weights_2.masked_fill_(
                torch.logical_not(mask_m1_to_m2), float("-inf"))

        attn_probs_1 = attn_weights.softmax(dim=-1)
        attn_probs_2 = attn_weights_2.softmax(dim=-1)

        # shape(B*h, L1, L2)
        attn_probs_1 = F.dropout(
            attn_probs_1,
            p=self.dropout,
            training=self.training)
        # shape(B*h, L2, L1)
        attn_probs_2 = F.dropout(
            attn_probs_2,
            p=self.dropout,
            training=self.training)

        # shape(B*h, L1, L2) @ shape(B*h, L2, d) -> shape(B*h, L1, d)
        attn_output_1 = torch.bmm(attn_probs_1, value_m2_states)
        # shape(B*h, L2, L1) @ shape(B*h, L1, d) -> shape(B*h, L2, d)
        attn_output_2 = torch.bmm(attn_probs_2, value_m1_states)

        if attn_output_1.size() != (bsz * self.num_heads, L1, self.head_dim):
            raise ValueError(
                f"`attn_output_1` should be of size {(bsz, self.num_heads, L1, self.head_dim)}, "
                f"but is {attn_output_1.size()}")

        if attn_output_2.size() != (bsz * self.num_heads, L2, self.head_dim):
            raise ValueError(
                f"`attn_output_2` should be of size {(bsz, self.num_heads, L2, self.head_dim)}, "
                f"but is {attn_output_2.size()}")

        attn_output_1 = attn_output_1.view(
            bsz, self.num_heads, L1, self.head_dim)
        attn_output_1 = attn_output_1.transpose(1, 2)
        attn_output_1 = attn_output_1.reshape(bsz, L1, self.embed_dim)

        attn_output_2 = attn_output_2.view(
            bsz, self.num_heads, L2, self.head_dim)
        attn_output_2 = attn_output_2.transpose(1, 2)
        attn_output_2 = attn_output_2.reshape(bsz, L2, self.embed_dim)

        attn_output_1 = self.out_m1_proj(attn_output_1)
        attn_output_2 = self.out_m2_proj(attn_output_2)

        return attn_output_1, attn_output_2

    def forward_sdpa(
            self,
            x1,
            x2,
            attention_mask_1: torch.Tensor = None,
            attention_mask_2: torch.Tensor = None,
            freqs=None,
            freqs_dit=None,
            freqs_agg=None):
        bsz, L1, _ = x1.size()
        L2 = x2.size(1)
        device = x1.device

        q = self.m1_proj(x1)
        k = self.m2_proj(x2)
        if freqs_dit is not None:
            q = rope_apply(q, freqs=freqs_dit, num_heads=self.num_heads)
            k = rope_apply(k, freqs=freqs_agg, num_heads=self.num_heads)
        query_states = self._shape(
            q, -1, bsz)                # shape(B, h, L1, d)
        # shape(B, h, L2, d)
        key_states = self._shape(k, -1, bsz)

        value_m1_states = self._shape(
            self.values_m1_proj(x1), -1, bsz)      # shape(B, h, L1, d)
        value_m2_states = self._shape(
            self.values_m2_proj(x2), -1, bsz)      # shape(B, h, L2, d)

        if attention_mask_1 is None and attention_mask_2 is None:
            mask_m1_to_m2, mask_m2_to_m1 = None, None
        else:
            if attention_mask_1 is None:
                attention_mask_1 = torch.ones(
                    (bsz, L1), dtype=attention_mask_2.dtype, device=device)
            if attention_mask_2 is None:
                attention_mask_2 = torch.ones(
                    (bsz, L2), dtype=attention_mask_1.dtype, device=device)
            # shape(L1, L2)
            mask_m2_to_m1 = attention_mask_1[:,
                                             None,
                                             :,
                                             None] | attention_mask_2[:,
                                                                      None,
                                                                      None,
                                                                      :]
            # shape(L2, L1)
            mask_m1_to_m2 = mask_m2_to_m1.transpose(-1, -2)

        if self.training:
            def attn_forward(q, k, v1, v2, mask1, mask2):
                attn_output_1 = F.scaled_dot_product_attention(
                    q, k, v2, attn_mask=mask1, dropout_p=self.dropout
                )
                attn_output_2 = F.scaled_dot_product_attention(
                    k, q, v1, attn_mask=mask2, dropout_p=self.dropout
                )
                return attn_output_1, attn_output_2

            attn_output_1, attn_output_2 = checkpoint(
                attn_forward,
                query_states, key_states, value_m1_states, value_m2_states,
                mask_m2_to_m1, mask_m1_to_m2,
                use_reentrant=False,
                preserve_rng_state=False
            )
        else:
            attn_output_1 = F.scaled_dot_product_attention(
                query_states, key_states, value_m2_states,
                attn_mask=mask_m2_to_m1, dropout_p=self.dropout
            )
            attn_output_2 = F.scaled_dot_product_attention(
                key_states, query_states, value_m1_states,
                attn_mask=mask_m1_to_m2, dropout_p=self.dropout
            )

        attn_output_1 = attn_output_1.view(
            bsz, self.num_heads, L1, self.head_dim)
        attn_output_1 = attn_output_1.transpose(1, 2)
        attn_output_1 = attn_output_1.reshape(bsz, L1, self.embed_dim)

        attn_output_2 = attn_output_2.view(
            bsz, self.num_heads, L2, self.head_dim)
        attn_output_2 = attn_output_2.transpose(1, 2)
        attn_output_2 = attn_output_2.reshape(bsz, L2, self.embed_dim)

        attn_output_1 = self.out_m1_proj(attn_output_1)
        attn_output_2 = self.out_m2_proj(attn_output_2)

        # 清理中间变量
        del q, k, query_states, key_states, value_m1_states, value_m2_states
        if attention_mask_1 is not None or attention_mask_2 is not None:
            del mask_m1_to_m2, mask_m2_to_m1

        return attn_output_1, attn_output_2

    def forward_flash_attn_2(
            self,
            x1,
            x2,
            attention_mask_1: torch.Tensor = None,
            attention_mask_2: torch.Tensor = None):
        bsz, L1, _ = x1.size()
        L2 = x2.size(1)

        if L1 <= bsz:  # copy from Attention block
            warnings.warn(
                f'Sequence length {L1} less than batch size {bsz}. Back to sdpa.')
            return self.forward_sdpa(
                x1, x2, attention_mask_1, attention_mask_2)

        if attention_mask_1 is not None and attention_mask_2 is not None:
            assert attention_mask_1.all() or attention_mask_2.all(), \
                'Currently does not support 2-directional mask attention'
        if attention_mask_1 is not None:
            x1 = x1 * attention_mask_1[..., None].type_as(x1)
        if attention_mask_2 is not None:
            x2 = x2 * attention_mask_2[..., None].type_as(x2)

        query_states = self._shape(
            self.m1_proj(x1), -1, bsz)                # shape(B, h, L1, d)
        key_states = self._shape(self.m2_proj(x2), -
                                 1, bsz)                  # shape(B, h, L2, d)
        value_m1_states = self._shape(
            self.values_m1_proj(x1), -1, bsz)      # shape(B, h, L1, d)
        value_m2_states = self._shape(
            self.values_m2_proj(x2), -1, bsz)      # shape(B, h, L2, d)

        from flash_attn import flash_attn_func

        # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
        query_states = query_states.permute(0, 2, 1, 3)
        key_states = key_states.permute(0, 2, 1, 3)
        value_m1_states = value_m1_states.permute(0, 2, 1, 3)
        value_m2_states = value_m2_states.permute(0, 2, 1, 3)

        # (B, N, #heads, #dim) -> (B, #heads, N, #dim)
        attn_output_1 = flash_attn_func(
            query_states, key_states, value_m2_states,
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2)
        attn_output_2 = flash_attn_func(
            key_states, query_states, value_m1_states,
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2)

        attn_output_1 = attn_output_1.view(
            bsz, self.num_heads, L1, self.head_dim)
        attn_output_1 = attn_output_1.transpose(1, 2)
        attn_output_1 = attn_output_1.reshape(bsz, L1, self.embed_dim)

        attn_output_2 = attn_output_2.view(
            bsz, self.num_heads, L2, self.head_dim)
        attn_output_2 = attn_output_2.transpose(1, 2)
        attn_output_2 = attn_output_2.reshape(bsz, L2, self.embed_dim)

        attn_output_1 = self.out_m1_proj(attn_output_1)
        attn_output_2 = self.out_m2_proj(attn_output_2)

        return attn_output_1, attn_output_2


def get_layernorm(
        hidden_size: torch.Tensor,
        eps: float,
        affine: bool,
        use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(
                hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError(
                "FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


def smart_pad(x: torch.Tensor, pad_len, dim=0, mode="constant", value=0,
              pos: Literal["right", "left", "both"] = "right"):
    if dim < 0:
        dim += x.ndim
    assert dim < x.ndim, 'invalid padding dimension'
    pad_dim = [0, 0] * (x.ndim - dim - 1)
    if pos == "right":
        pad_dim += [0, pad_len]
    elif pos == "left":
        pad_dim += [pad_len, 0]
    else:
        pad_dim += [pad_len, pad_len]
    x = F.pad(x, pad_dim, mode=mode, value=value)
    return x
