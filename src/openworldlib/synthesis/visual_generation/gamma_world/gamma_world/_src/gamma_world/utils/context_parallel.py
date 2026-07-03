# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup, all_gather, all_gather_into_tensor, get_process_group_ranks

from gamma_world._src.imaginaire.utils import log


def _all_gather_along_dim(x: Tensor, dim: int, group: ProcessGroup) -> Tensor:

    world_size = group.size()
    if world_size == 1:
        return x




    x_permuted = x.movedim(dim, 0).contiguous()
    gathered_shape = list(x_permuted.shape)
    gathered_shape[0] *= world_size
    gathered = torch.empty(gathered_shape, dtype=x.dtype, device=x.device)

    all_gather_into_tensor(gathered, x_permuted, group=group)


    gathered = gathered.movedim(0, dim)
    return gathered


def _split_along_dim(x: Tensor, dim: int, rank: int, world_size: int) -> Tensor:

    dim_size = x.shape[dim]
    assert dim_size % world_size == 0, (
        f"Dimension {dim} with size {dim_size} must be divisible by world_size {world_size}"
    )
    chunk_size = dim_size // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size


    slices = [slice(None)] * x.ndim
    slices[dim] = slice(start_idx, end_idx)

    return x[tuple(slices)].contiguous()


class _GatherVSplitL(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x: Tensor, group: ProcessGroup) -> Tensor:
        ctx.group = group
        ctx.v_local = x.shape[1]

        world_size = group.size()
        rank = group.rank()


        gathered = _all_gather_along_dim(x, dim=1, group=group)


        output = _split_along_dim(gathered, dim=3, rank=rank, world_size=world_size)

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        group = ctx.group
        world_size = group.size()
        rank = group.rank()



        gathered = _all_gather_along_dim(grad_output, dim=3, group=group)


        grad_input = _split_along_dim(gathered, dim=1, rank=rank, world_size=world_size)

        return grad_input, None


class _GatherLSplitV(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x: Tensor, group: ProcessGroup) -> Tensor:
        ctx.group = group
        ctx.l_local = x.shape[3]

        world_size = group.size()
        rank = group.rank()


        gathered = _all_gather_along_dim(x, dim=3, group=group)


        output = _split_along_dim(gathered, dim=1, rank=rank, world_size=world_size)

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        group = ctx.group
        world_size = group.size()
        rank = group.rank()



        gathered = _all_gather_along_dim(grad_output, dim=1, group=group)


        grad_input = _split_along_dim(gathered, dim=3, rank=rank, world_size=world_size)

        return grad_input, None


def gather_v_split_l(x: Tensor, group: ProcessGroup) -> Tensor:

    world_size = group.size()
    L = x.shape[3]
    assert L % world_size == 0, f"L dimension ({L}) must be divisible by process group size ({world_size})"
    return _GatherVSplitL.apply(x, group)


def gather_l_split_v(x: Tensor, group: ProcessGroup) -> Tensor:

    world_size = group.size()
    V = x.shape[1]
    assert V % world_size == 0, f"V dimension ({V}) must be divisible by process group size ({world_size})"
    return _GatherLSplitV.apply(x, group)


def materialize_split_pattern(x_shape: tuple, dim_names: list, seq_dim: tuple, split_factors: tuple) -> tuple[str, str]:

    rearrange_pattern_from = []
    rearrange_pattern_to = []

    for orig_dim_idx in range(len(x_shape)):
        if orig_dim_idx in seq_dim:
            seq_idx = seq_dim.index(orig_dim_idx)
            split_factor = split_factors[seq_idx]
            if split_factor > 1:

                chunk_name = f"chunk{seq_idx}"
                split_name = f"split{seq_idx}"

                rearrange_pattern_from.append(f"({chunk_name} {split_name})")

                rearrange_pattern_to.append(f"{chunk_name} {split_name}")
            else:

                rearrange_pattern_from.append(dim_names[orig_dim_idx])
                rearrange_pattern_to.append(dim_names[orig_dim_idx])
        else:

            rearrange_pattern_from.append(dim_names[orig_dim_idx])
            rearrange_pattern_to.append(dim_names[orig_dim_idx])

    rearrange_pattern_from_str = " ".join(rearrange_pattern_from)
    rearrange_pattern_to_str = " ".join(rearrange_pattern_to)
    return rearrange_pattern_from_str, rearrange_pattern_to_str


def build_index_pattern(x_shape: tuple, seq_dim: tuple, split_factors: tuple, rank_positions: tuple) -> tuple:
    index_tuple = []

    for orig_dim_idx in range(len(x_shape)):
        if orig_dim_idx in seq_dim:
            seq_idx = seq_dim.index(orig_dim_idx)
            split_factor = split_factors[seq_idx]
            if split_factor > 1:

                index_tuple.append(slice(None))
                index_tuple.append(rank_positions[seq_idx])

            else:
                index_tuple.append(slice(None))

        else:
            index_tuple.append(slice(None))

    return tuple(index_tuple)


def split_inputs_cp_multidim(
    x: Tensor, seq_dim: tuple, maximum_split_factor: tuple, cp_group: ProcessGroup
) -> tuple[Tensor, tuple, tuple]:

    cp_ranks = get_process_group_ranks(cp_group)
    cp_size = len(cp_ranks)


    assert len(seq_dim) == len(maximum_split_factor), (
        f"seq_dim length {len(seq_dim)} must match maximum_split_factor length {len(maximum_split_factor)}"
    )


    split_factors = [1] * len(seq_dim)
    remaining_cp_size = cp_size


    for i in range(len(seq_dim)):
        dim_idx = seq_dim[i]
        max_factor = maximum_split_factor[i]
        dim_size = x.shape[dim_idx]



        actual_factor = min(max_factor, remaining_cp_size)


        while actual_factor > 1 and (dim_size % actual_factor != 0 or remaining_cp_size % actual_factor != 0):
            actual_factor -= 1

        split_factors[i] = actual_factor
        remaining_cp_size //= actual_factor
    log.info(f"[rank {cp_group.rank()}] split_factors: {split_factors}")

    total_split = 1
    for factor in split_factors:
        total_split *= factor
    assert total_split == cp_size, (
        f"Product of split factors {split_factors} (={total_split}) must equal cp_size {cp_size}"
    )


    rank = cp_group.rank()
    rank_positions = []



    remaining_rank = rank
    for i in range(len(seq_dim) - 1, -1, -1):
        rank_positions.insert(0, remaining_rank % split_factors[i])
        remaining_rank //= split_factors[i]







    dim_names = [f"d{i}" for i in range(len(x.shape))]
    rearrange_pattern_from, rearrange_pattern_to_str = materialize_split_pattern(
        x.shape, dim_names, seq_dim, tuple(split_factors)
    )
    log.info(
        f"[rank {cp_group.rank()}] rearrange_pattern_from: {rearrange_pattern_from}, rearrange_pattern_to_str: {rearrange_pattern_to_str}"
    )

    rearrange_dict = {}
    for i, dim_idx in enumerate(seq_dim):
        if split_factors[i] > 1:
            chunk_size = x.shape[dim_idx] // split_factors[i]
            rearrange_dict[f"chunk{i}"] = chunk_size
            rearrange_dict[f"split{i}"] = split_factors[i]


    result = rearrange(x, f"{rearrange_pattern_from} -> {rearrange_pattern_to_str}", **rearrange_dict)



    index_tuple = build_index_pattern(x.shape, seq_dim, tuple(split_factors), tuple(rank_positions))
    log.info(f"[rank {cp_group.rank()}] index_tuple: {index_tuple}")
    result = result[index_tuple]

    return result, tuple(split_factors), tuple(rank_positions)


def cat_outputs_cp_multidim(
    x: Tensor,
    seq_dim: tuple,
    split_factors: tuple,
    x_original_shape: tuple,
    cp_group: ProcessGroup,
    preserve_grad: bool = False,
) -> Tensor:

    cp_size = cp_group.size()
    x = x.contiguous()

    gathered_tensors = [torch.zeros_like(x) for _ in range(cp_size)]
    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to gather tensors: {e}")

    if preserve_grad:
        gathered_tensors[cp_group.rank()] = x



    full_shape = list(x.shape)
    inserted_dims = 0
    for i, dim_idx in enumerate(seq_dim):
        if split_factors[i] > 1:

            adjusted_idx = dim_idx + inserted_dims


            full_shape.insert(adjusted_idx + 1, split_factors[i])


            inserted_dims += 1


    result = torch.zeros(full_shape, dtype=x.dtype, device=x.device)


    for rank in range(cp_size):

        rank_pos = []
        remaining_rank = rank
        for i in range(len(seq_dim) - 1, -1, -1):
            rank_pos.insert(0, remaining_rank % split_factors[i])
            remaining_rank //= split_factors[i]


        index_tuple = []
        dim_counter = 0
        for orig_dim_idx in range(len(x.shape)):


            matched = False
            for seq_idx, s_dim in enumerate(seq_dim):
                if split_factors[seq_idx] > 1:


                    num_prev_splits = sum(1 for j in range(seq_idx) if split_factors[j] > 1)
                    adjusted_s_dim = s_dim + num_prev_splits
                    if dim_counter == adjusted_s_dim:

                        index_tuple.append(slice(None))
                        index_tuple.append(rank_pos[seq_idx])
                        matched = True
                        dim_counter += 2
                        break

            if not matched:
                index_tuple.append(slice(None))
                dim_counter += 1
        log.info(f"gathering [rank {rank}] index_tuple: {index_tuple}")

        result[tuple(index_tuple)] = gathered_tensors[rank]



    dim_names = [f"d{i}" for i in range(len(x_original_shape))]
    rearrange_pattern_from, rearrange_pattern_to_str = materialize_split_pattern(
        x_original_shape, dim_names, seq_dim, split_factors
    )
    log.info(
        f"[rank {cp_group.rank()}] rearrange_pattern_from: {rearrange_pattern_from}, rearrange_pattern_to_str: {rearrange_pattern_to_str}"
    )

    result = rearrange(result, f"{rearrange_pattern_to_str} -> {rearrange_pattern_from}")
    log.info(f"[rank {cp_group.rank()}] result: {result.shape}")
    return result
