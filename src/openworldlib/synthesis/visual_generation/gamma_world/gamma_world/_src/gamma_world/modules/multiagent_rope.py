# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):

    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def compute_simplex_agent_freqs(
    num_agents: int,
    dim_agent: int,
    scale: float = 1.0,
) -> torch.Tensor:

    assert num_agents >= 1, f"num_agents must be >= 1, got {num_agents}"
    assert dim_agent % 2 == 0, f"dim_agent must be even, got {dim_agent}"
    assert scale > 0, f"scale must be positive, got {scale}"

    d = dim_agent // 2
    n = num_agents

    if n == 1:

        return torch.ones(1, d, dtype=torch.complex64)

    if n <= d + 1:

        vecs = torch.eye(n, d, dtype=torch.float64)
    else:


        gen = torch.Generator().manual_seed(0)
        random_mat = torch.randn(d, n, dtype=torch.float64, generator=gen)
        q, _ = torch.linalg.qr(random_mat)
        vecs = q.T[:n].contiguous()

    vecs = vecs - vecs.mean(dim=0, keepdim=True)
    vecs = vecs / vecs.norm(dim=1, keepdim=True)
    angles = vecs * scale

    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64)


def precompute_freqs_cis_4d(
    dim: int,
    max_f: int = 1024,
    max_h: int = 1024,
    max_w: int = 1024,
    max_agents: int = 100,
    theta: float = 10000.0,
    agent_encoding: str = "linear",
    num_agents: int | None = None,
    agent_scale: float = 1.0,
):

    dim_spatial = dim // 3
    dim_temporal = dim - 2 * dim_spatial
    dim_th = (dim_temporal // 2 // 2) * 2
    if dim_th == 0:
        dim_th = 2
    dim_agent = dim_temporal - dim_th

    if dim_agent % 2 == 1:
        dim_th += 1
        dim_agent -= 1

    freqs_th = precompute_freqs_cis(dim_th, max_f, theta)

    if agent_encoding == "simplex":
        assert num_agents is not None, (
            "num_agents must be provided when agent_encoding='simplex'"
        )
        freqs_agent = compute_simplex_agent_freqs(num_agents, dim_agent, agent_scale)
    elif agent_encoding == "linear":
        freqs_agent = precompute_freqs_cis(dim_agent, max_agents, theta)
    else:
        raise ValueError(
            f"Unknown agent_encoding={agent_encoding!r}; expected 'linear' or 'simplex'"
        )

    freqs_h = precompute_freqs_cis(dim_spatial, max_h, theta)
    freqs_w = precompute_freqs_cis(dim_spatial, max_w, theta)
    return freqs_th, freqs_agent, freqs_h, freqs_w


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def build_multi_agent_freqs(
    dit_or_freqs,
    f: int,
    h: int,
    w: int,
    num_agents: int | None = None,
    agent_pool_indices: list[int] | None = None,
):

    layout = getattr(dit_or_freqs, "multi_agent_token_layout", "spatial")
    if num_agents is None:
        num_agents = getattr(dit_or_freqs, "num_agents", 1)



    agent_id_offset = getattr(dit_or_freqs, "agent_id_offset", 0)


    agent_encoding = getattr(dit_or_freqs, "agent_encoding", "linear")

    if layout == "sequence" and getattr(dit_or_freqs, "freqs_4d", None) is not None:


        freqs_th, freqs_agent, freqs_h, freqs_w = dit_or_freqs.freqs_4d

        if agent_encoding == "simplex":
            if agent_pool_indices is None:
                agent_pool_indices = list(range(num_agents))
            assert len(agent_pool_indices) == num_agents, (
                f"agent_pool_indices length {len(agent_pool_indices)} must equal num_agents {num_agents}"
            )
            assert max(agent_pool_indices) < freqs_agent.shape[0], (
                f"agent_pool_indices max {max(agent_pool_indices)} out of range "
                f"for simplex pool of size {freqs_agent.shape[0]}"
            )
            assert len(set(agent_pool_indices)) == num_agents, (
                f"agent_pool_indices must be unique to avoid degenerate identical "
                f"rotations across slots, got {agent_pool_indices}"
            )

        all_freqs = []
        for agent_id in range(num_agents):
            if agent_encoding == "simplex":


                agent_idx = agent_pool_indices[agent_id]
            else:

                agent_idx = agent_id + agent_id_offset

            agent_freqs = torch.cat([
                freqs_th[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_agent[agent_idx:agent_idx+1].view(1, 1, 1, -1).expand(f, h, w, -1),
                freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1)
            all_freqs.append(agent_freqs.reshape(f * h * w, 1, -1))

        freqs = torch.cat(all_freqs, dim=0)
        return freqs


    freqs_3d = dit_or_freqs.freqs
    freqs = torch.cat([
        freqs_3d[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_3d[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_3d[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1)
    return freqs


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)