# Copyright Alibaba Inc. All Rights Reserved.

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


XFORMERS_AVAILABLE = False




class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def attn_residual_func(self,x: Tensor, pos=None, e = None) -> Tensor:
        if e is None:
            return self.ls1(self.attn(self.norm1(x), pos=pos))
        return self.ls1(self.attn(self.norm1(x) * (1 + e[1]) + e[0], pos=pos))

    def ffn_residual_func(self,x: Tensor, e = None) -> Tensor:
        if e is None:
            return self.ls2(self.mlp(self.norm2(x)))
        return self.ls2(self.mlp(self.norm2(x)) * (1 + e[4])+ e[3]) * e[5]
    def forward(
        self, x: Tensor, pos=None, e0=None,
        return_partial: bool = False,
        run_remaining:  bool = False,
        modifiers:      tuple | None = None,   # e_mod
    ) -> Tensor | tuple[Tensor, tuple]:

        if run_remaining:
            assert modifiers is not None, "run_remaining need modifiers"
            e_mod = modifiers

            x = x + self.ffn_residual_func(x, e=e_mod)
            return x

        if e0 is not None:
            B = e0.shape[0]
            if B != x.shape[0]:
                e0 = e0.unsqueeze(1).repeat(1, x.shape[0] // B, 1, 1)
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

        x = x + self.ffn_residual_func(x, e=e_mod)
        return x

    def forward_partial(self, *args, **kwargs):
        return self.forward(*args, **kwargs, return_partial=True)

    def forward_remaining(self, x: Tensor, e=None):
        return self.forward(x,
                            run_remaining=True,
                            modifiers=e)

def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0, pos=None
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)




def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls2.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError

class CamTokenProjector(nn.Module):
    def __init__(self, out_dim: int, hidden: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(36, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, cam: torch.Tensor) -> torch.Tensor:
        B, V_in, _ = cam.shape
        repeat = 3
        pad = cam[:, :1, :].repeat(1, repeat, 1)
        cam = torch.cat([cam, pad], dim=1)
        V_pad = cam.size(1)
        cam = cam.view(B, V_pad // 4, 4, 9).reshape(B, V_pad // 4, 36)

        cam = cam.flatten(0, 1)
        cam = self.mlp(cam).view(-1, 1, self.out_dim )

        return cam


class Projection_Head(nn.Module):

    def __init__(self,
                 mode: str = "conv",
                 in_channels: int = 5120,
                 out_channels: int = 1024,):
        super().__init__()
        assert mode in {"conv", "linear"}
        self.mode = mode

        if mode == "conv":
            self.proj = nn.Conv3d(in_channels, out_channels,
                                  kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.proj = nn.Linear(in_channels, out_channels, bias=False)

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.mode == "conv":
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            x = self.proj(x)
            x = x.permute(0, 2, 3, 4, 1).contiguous()
        else:  # linear
            B, T, H, W, C_in = x.shape
            x = self.proj(x.view(B * T * H * W, C_in))
            x = x.view(B, T, H, W, -1)

        B, T, H, W, C = x.shape
        x = self.norm(x.view(-1, C)).view(B, T, H, W, C)
        return x
