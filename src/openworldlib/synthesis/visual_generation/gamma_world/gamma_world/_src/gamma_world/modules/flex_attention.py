# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor, Partial, Shard
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    create_block_mask,
    flex_attention,
)


@lru_cache
def create_block_mask_cached(
    score_mod: _mask_mod_signature,
    B: Optional[int],
    H: Optional[int],
    M: int,
    N: int,
    device: str = "cuda",
) -> BlockMask:
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    return q_idx >= kv_idx


def rewrite_mask_mod_for_cp(
    mask_mod: _mask_mod_signature,
    rank: int,
    shard_size: int,
) -> _mask_mod_signature:


    return lambda b, h, q_idx, kv_idx: mask_mod(b, h, q_idx + rank * shard_size, kv_idx)


def flex_attention_cp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    process_group: Optional[ProcessGroup] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    block_mask: Optional[BlockMask] = None,
    flex_attention_fn: Callable = flex_attention,
    seq_dim: int = 2,
    **kwargs,
) -> Tensor:


    if process_group is None:
        return flex_attention_fn(query, key, value, score_mod=mask_mod, block_mask=block_mask, **kwargs)
    else:
        assert query.ndim == 4 and key.ndim == 4 and value.ndim == 4, "Only support 4D input."

        world_size = torch.distributed.get_world_size(process_group)
        local_rank = torch.distributed.get_rank(process_group)
        backend = torch.distributed.get_backend(process_group)
        assert backend == "nccl", "Only support NCCL backend."
        device_type = "cuda"

        device_mesh = DeviceMesh.from_group(process_group, device_type=device_type)

        if mask_mod is None:
            assert block_mask is not None, "Either mask_mod or block_mask must be provided."
            mask_mod = block_mask.mask_mod



        query_is_dtensor = isinstance(query, DTensor)
        key_is_dtensor = isinstance(key, DTensor)
        value_is_dtensor = isinstance(value, DTensor)

        if not query_is_dtensor:
            q_local = query
            shard_size = query.shape[seq_dim]
            seq_len = shard_size * world_size
        else:
            q_local = query.to_local()
            seq_len = query.shape[seq_dim]
            shard_size = seq_len // world_size





        if key_is_dtensor:
            k_full = key.full_tensor(grad_placements=[Partial()])
        else:
            k_local = DTensor.from_local(key, device_mesh, [Shard(seq_dim)])
            k_full = k_local.full_tensor(grad_placements=[Partial()])

        if value_is_dtensor:
            v_full = value.full_tensor(grad_placements=[Partial()])
        else:
            v_local = DTensor.from_local(value, device_mesh, [Shard(seq_dim)])
            v_full = v_local.full_tensor(grad_placements=[Partial()])


        cp_mask_mod = rewrite_mask_mod_for_cp(mask_mod, local_rank, shard_size)
        cp_block_mask = create_block_mask_cached(cp_mask_mod, B=1, H=1, M=shard_size, N=seq_len, device=device_type)


        cp_out = flex_attention_fn(
            q_local,
            k_full,
            v_full,
            score_mod=None,
            block_mask=cp_block_mask,
            **kwargs,
        )
        assert isinstance(cp_out, torch.Tensor)

        if query_is_dtensor:

            cp_out_dist = DTensor.from_local(cp_out, device_mesh, [Shard(seq_dim)])
            return cp_out_dist
        else:

            return cp_out
