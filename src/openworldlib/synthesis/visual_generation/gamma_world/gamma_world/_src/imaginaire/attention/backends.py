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

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Frontend APIs
"""

from functools import lru_cache

import torch

from gamma_world._src.imaginaire.attention.flash2.checks import flash2_attention_check
from gamma_world._src.imaginaire.attention.flash3.checks import flash3_attention_check
from gamma_world._src.imaginaire.attention.masks import CausalType
from gamma_world._src.imaginaire.attention.natten.checks import natten_attention_check, natten_multi_dim_attention_check
from gamma_world._src.imaginaire.attention.utils import get_arch_tag
from gamma_world._src.imaginaire.attention.utils import safe_log as log


BACKEND_CHECK_MAP = {
    "natten": natten_attention_check,
    "flash2": flash2_attention_check,
    "flash3": flash3_attention_check,
}

BACKEND_MULTI_DIM_CHECK_MAP = {
    "natten": natten_multi_dim_attention_check,
}


def is_backend_compatible(
    backend: str,
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType | None,
    is_varlen: bool,
    raise_error: bool = False,
) -> bool:
    """
    Input validation function a specified backend.
    Runs the common and backend-specific checks. Returns False if any checks fail, otherwise True.

    Parameters:
        backend (str): selected backend.

        query_shape (torch.Size): Shape of 4-D query tensor (`[batch, seqlen, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D key tensor (`[batch, seqlen_kv, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D value tensor (`[batch, seqlen_kv, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        is_causal (bool): whether or not causal masking is enabled.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`.

        is_varlen (bool): whether or not a variable length (varlen) use case. Must be inferred
            beforehand based on arguments such as seqlens_{Q,KV} or cumulative_seqlen_{Q,KV} being
            passed.

        raise_error (bool): whether to raise an error if any checks fail or no backend is selected,
            instead of just returning False. Default is False.

    Returns:
        success (bool): whether use case is compatible with the backend.

    """
    if backend is None:
        raise ValueError("Cannot pass None backend to is_backend_compatible.")

    if backend not in BACKEND_CHECK_MAP:
        raise ValueError(f"Unrecognized backend name {backend}.")

    return BACKEND_CHECK_MAP[backend](
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=is_varlen,
        raise_error=raise_error,
    )


def get_backend_list(arch_tag: int) -> list[str]:
    """
    Returns list of supported backends according to arch tag (attention.utils.get_arch_tag).
    Backends are ordered based on their known performance levels, so that the best-performing
    compatible backend is selected.

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        backend_list (list[str]): a list of backend names (string). Empty if device is not supported.

    """

    if arch_tag < 75:
        log.debug(f"Minimum architecture supported for Attention is 75, got {arch_tag=}.")
        return []

    if arch_tag == 90:
        return [
            "flash3",
            "natten",
            "flash2",
        ]

    if arch_tag in [100, 103]:
        return [
            "natten",
            "flash2",
        ]

    if arch_tag >= 80:
        return [
            "flash2",
            "natten",
        ]

    return ["natten"]


@lru_cache
def choose_backend(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType | None,
    is_varlen: bool,
    backend: str | None = None,
    raise_error: bool = True,
) -> str | None:
    """
    Selects a compatible backend, unless one is already selected, which runs its corresponding
    checks.

    Parameters:
        query_shape (torch.Size): Shape of 4-D query tensor (`[batch, seqlen, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D key tensor (`[batch, seqlen_kv, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D value tensor (`[batch, seqlen_kv, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        is_causal (bool): whether or not causal masking is enabled.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`.

        is_varlen (bool): whether or not a variable length (varlen) use case. Must be inferred
            beforehand based on arguments such as seqlens_{Q,KV} or cumulative_seqlen_{Q,KV} being
            passed.

        backend (str | None): selected backend, if any.

        raise_error (bool): whether to raise an error if any checks fail or no backend is selected,
            instead of just returning False. Default is **True**.

    Returns:
        backend (str | None): selected backend, or None if no backends are compatible.

    """
    if backend is not None:
        if is_backend_compatible(
            backend=backend,
            query_shape=query_shape,
            key_shape=key_shape,
            value_shape=value_shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            is_causal=is_causal,
            causal_type=causal_type,
            is_varlen=is_varlen,
            raise_error=raise_error,
        ):
            return backend
        return None

    arch_tag = get_arch_tag(device)
    backend_list = get_backend_list(arch_tag)
    for backend in backend_list:
        if is_backend_compatible(
            backend=backend,
            query_shape=query_shape,
            key_shape=key_shape,
            value_shape=value_shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            is_causal=is_causal,
            causal_type=causal_type,
            is_varlen=is_varlen,
            raise_error=False,
        ):
            return backend

    if not raise_error:
        return None

    raise ValueError(
        "Could not find a compatible Attention backend for this use case / device. "
        "Try running with debug logs to find out why."
    )


def is_multi_dim_backend_compatible(
    backend: str,
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    raise_error: bool = False,
) -> bool:
    """
    Input validation function a specified multi-dimensional backend.
    Runs the common and backend-specific checks. Returns False if any checks fail, otherwise True.

    Parameters:
        backend (str): selected backend.

        query_shape (torch.Size): Shape of 4-D, 5-D, or 6-D query tensor (`[batch, *token_layout_shape, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D, 5-D, or 6-D key tensor (`[batch, *token_layout_shape, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D, 5-D, or 6-D value tensor (`[batch, *token_layout_shape, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        raise_error (bool): whether to raise an error if any checks fail or no backend is selected,
            instead of just returning False. Default is False.

    Returns:
        success (bool): whether use case is compatible with the backend.

    """
    if backend is None:
        raise ValueError("Cannot pass None backend to is_backend_compatible.")

    if backend not in BACKEND_MULTI_DIM_CHECK_MAP:
        raise ValueError(f"Unrecognized backend name {backend}.")

    return BACKEND_MULTI_DIM_CHECK_MAP[backend](
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        raise_error=raise_error,
    )


def get_multi_dim_backend_list(arch_tag: int) -> list[str]:
    """
    Returns list of supported multi-dimensional backends according to arch tag (attention.utils.get_arch_tag).
    Backends are ordered based on their known performance levels, so that the best-performing
    compatible backend is selected.

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        backend_list (list[str]): a list of backend names (string). Empty if device is not supported.

    """

    if arch_tag < 75:
        log.debug(f"Minimum architecture supported for Multi-Dimensional Attention is 75, got {arch_tag=}.")
        return []

    # NATTEN is the only supported backend for now
    return ["natten"]


@lru_cache
def choose_multi_dim_backend(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    backend: str | None = None,
) -> str:
    """
    Selects a compatible multi-dimensional backend, unless one is already selected, which runs its
    corresponding checks.

    Parameters:
        query_shape (torch.Size): Shape of 4-D, 5-D, or 6-D query tensor (`[batch, *token_layout_shape, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D, 5-D, or 6-D key tensor (`[batch, *token_layout_shape, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D, 5-D, or 6-D value tensor (`[batch, *token_layout_shape, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        backend (str | None): selected backend, if any.

    Returns:
        backend (str): selected backend.

    """
    if backend is not None:
        assert is_multi_dim_backend_compatible(
            backend=backend,
            query_shape=query_shape,
            key_shape=key_shape,
            value_shape=value_shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            raise_error=True,
        )
        return backend

    arch_tag = get_arch_tag(device)
    backend_list = get_multi_dim_backend_list(arch_tag)
    for backend in backend_list:
        if is_multi_dim_backend_compatible(
            backend=backend,
            query_shape=query_shape,
            key_shape=key_shape,
            value_shape=value_shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            raise_error=False,
        ):
            return backend

    raise ValueError(
        "Could not find a compatible Multi-Dimensional Attention backend for this use case / device. "
        "Try running with debug logs to find out why."
    )
