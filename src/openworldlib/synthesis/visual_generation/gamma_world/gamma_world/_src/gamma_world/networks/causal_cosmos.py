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



import math
from collections import namedtuple

import torch
import torch.amp as amp
import torch.nn as nn
import transformer_engine as te
from einops import rearrange
from torch.distributed import ProcessGroup, get_process_group_ranks, get_rank, is_initialized
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torchvision import transforms

from gamma_world._src.imaginaire.utils import distributed

try:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
except ImportError:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb

from transformer_engine.pytorch.attention import DotProductAttention

from gamma_world._src.imaginaire.utils import log
from gamma_world._src.imaginaire.utils.context_parallel import cat_outputs_cp, cat_outputs_cp_with_grad
from gamma_world._src.predict2.conditioner import DataType
from gamma_world._src.predict2.networks.minimal_v4_dit import (
    Attention,
    CheckpointMode,
    FinalLayer,
    GPT2FeedForward,
    I2VCrossAttention,
    LearnablePosEmbAxis,
    PatchEmbed,
    SACConfig,
    TimestepEmbedding,
    Timesteps,
    VideoRopePosition3DEmb,
)
from gamma_world._src.predict2.networks.model_weights_stats import WeightTrainingStat
from gamma_world._src.gamma_world.modules.action_encoder import (
    MultiAgentActionControlModule,
    MultiAgentActionControlSpatialConcatModule,
    build_multi_agent_action_control,
)
from gamma_world._src.gamma_world.modules.multiagent_rope import precompute_freqs_cis_4d
from gamma_world._src.gamma_world.modules.flex_attention import flex_attention_cp

from gamma_world._src.gamma_world.modules.action_encoder import RMS_norm


flex_attention = torch.compile(flex_attention, dynamic=False)

VideoSize = namedtuple("VideoSize", ["T", "H", "W"])
DEBUG = False


class CausalSelfAttention(nn.Module):


    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
        backend: str = "transformer_engine",
        use_wan_fp32_strategy: bool = False,
        local_attn_size: int = -1,
        sink_size: int = 0,
        attention_mode: str = "dense",
        num_views: int = 1,
        z_num: int = 0,
        use_batched_attn: bool = True,
    ):
        super().__init__()
        log.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{n_heads} heads with a dimension of {head_dim}."
        )
        self.is_selfattn = context_dim is None

        assert backend in ["transformer_engine", "torch", "minimal_a2a"], f"Invalid backend: {backend}"
        self.backend = backend

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        assert attention_mode in ("dense", "sparse_hub"), (
            f"Invalid attention_mode: {attention_mode!r}; expected 'dense' or 'sparse_hub'"
        )
        self.attention_mode = attention_mode
        self.num_views = num_views
        self.z_num = z_num



        self.use_batched_attn = use_batched_attn
        if attention_mode == "sparse_hub":
            assert num_views >= 2, (
                f"sparse_hub attention requires num_views>=2, got {num_views}"
            )
            assert z_num >= 1, f"sparse_hub attention requires z_num>=1, got {z_num}"






            if local_attn_size != -1:
                assert sink_size >= 0 and sink_size < local_attn_size, (
                    f"sparse_hub attention with local_attn_size={local_attn_size} "
                    f"requires 0 <= sink_size < local_attn_size, got sink_size={sink_size}"
                )
            else:
                assert sink_size == 0, (
                    f"sparse_hub attention with global attention (local_attn_size=-1) "
                    f"does not use sink tokens; got sink_size={sink_size}"
                )


        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()


        self.q_norm = te.pytorch.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = te.pytorch.RMSNorm(head_dim, eps=1e-6)
        self.v_norm = nn.Identity()


        self.attn_op = DotProductAttention(
            n_heads,
            head_dim,
            num_gqa_groups=n_heads,
            attention_dropout=0,
            qkv_format=qkv_format,
            attn_mask_type="no_mask",
        )

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim

        self.cp_group: ProcessGroup | None = None

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rope_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        original_dtype = q.dtype
        if self.use_wan_fp32_strategy:
            q = q.to(torch.float32)
            k = k.to(torch.float32)

        q = apply_rotary_pos_emb(q, rope_emb, tensor_format="bshd", fused=True)
        k = apply_rotary_pos_emb(k, rope_emb, tensor_format="bshd", fused=True)

        if self.use_wan_fp32_strategy:
            q = q.to(original_dtype)
            k = k.to(original_dtype)

        return q, k

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
        video_size: VideoSize | None = None,

        block_mask: BlockMask | None = None,
        kv_cache: dict | None = None,
        current_start: int = 0,
        current_end: int = 0,
        disable_kv_cache: bool = False,
        disable_kv_cache_update: bool = False,
    ) -> torch.Tensor:

        del context

        b, s, _ = x.shape
        n, d = self.n_heads, self.head_dim


        q = self.q_norm(self.q_proj(x).view(b, s, n, d))
        k = self.k_norm(self.k_proj(x).view(b, s, n, d))
        v = self.v_proj(x).view(b, s, n, d)

        if kv_cache is None:

            roped_q, roped_k = self._apply_rope(q, k, rope_emb)
            roped_q = roped_q.type_as(v)
            roped_k = roped_k.type_as(v)


            padded_length = math.ceil(s / 128) * 128 - s

            if padded_length > 0:
                pad_shape = [b, padded_length, n, d]
                roped_q = torch.cat([roped_q, torch.zeros(pad_shape, device=q.device, dtype=v.dtype)], dim=1)
                roped_k = torch.cat([roped_k, torch.zeros(pad_shape, device=k.device, dtype=v.dtype)], dim=1)
                v_padded = torch.cat([v, torch.zeros(pad_shape, device=v.device, dtype=v.dtype)], dim=1)
            else:
                v_padded = v



            out = flex_attention_cp(
                query=roped_q.transpose(2, 1),
                key=roped_k.transpose(2, 1),
                value=v_padded.transpose(2, 1),
                block_mask=block_mask,
                process_group=self.cp_group,
                flex_attention_fn=flex_attention,
            )


            if padded_length > 0:
                out = out[:, :, :-padded_length].transpose(2, 1)
            else:
                out = out.transpose(2, 1)

        elif disable_kv_cache:

            roped_q, roped_k = self._apply_rope(q, k, rope_emb)
            out = self.attn_op(roped_q.type_as(v), roped_k.type_as(v), v)

        elif self.attention_mode == "sparse_hub":









            roped_q, roped_k = self._apply_rope(q, k, rope_emb)
            roped_q = roped_q.type_as(v)
            roped_k = roped_k.type_as(v)
            out = self._sparse_hub_inference(
                roped_q=roped_q,
                roped_k=roped_k,
                v=v,
                video_size=video_size,
                kv_cache=kv_cache,
                current_start=current_start,
                current_end=current_end,
                disable_kv_cache_update=disable_kv_cache_update,
            )

        else:

            roped_q, roped_k = self._apply_rope(q, k, rope_emb)
            roped_q = roped_q.type_as(v)
            roped_k = roped_k.type_as(v)

            frame_seqlen = video_size.H * video_size.W
            if self.cp_group is not None:
                assert frame_seqlen % self.cp_group.size() == 0
                frame_seqlen = frame_seqlen // self.cp_group.size()

            assert current_end == current_start + roped_q.shape[1]

            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_q.shape[1]


            if (
                self.local_attn_size != -1
                and (current_end > kv_cache["global_end_index"].item())
                and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size)
            ):

                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["k"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
                kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["v"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()

                local_end_index = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                    - num_evicted_tokens
                )
                local_start_index = local_end_index - num_new_tokens

                if not disable_kv_cache_update:
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_k
                    kv_cache["v"][:, local_start_index:local_end_index] = v
            else:

                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                if torch.is_grad_enabled():

                    kv_cache["k"] = torch.cat(
                        [kv_cache["k"][:, :local_start_index], roped_k, kv_cache["k"][:, local_end_index:]],
                        dim=1,
                    )
                    kv_cache["v"] = torch.cat(
                        [kv_cache["v"][:, :local_start_index], v, kv_cache["v"][:, local_end_index:]],
                        dim=1,
                    )
                else:
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_k
                    kv_cache["v"][:, local_start_index:local_end_index] = v


            if disable_kv_cache_update:
                cached_k = torch.cat([kv_cache["k"][:, :local_start_index], roped_k], dim=1)
                cached_v = torch.cat([kv_cache["v"][:, :local_start_index], v], dim=1)
            else:
                if self.local_attn_size != -1:
                    window_tokens = (self.local_attn_size - self.sink_size) * frame_seqlen
                    cached_k = torch.cat(
                        [
                            kv_cache["k"][:, :sink_tokens],
                            kv_cache["k"][:, max(sink_tokens, local_end_index - window_tokens) : local_end_index],
                        ],
                        dim=1,
                    )
                    cached_v = torch.cat(
                        [
                            kv_cache["v"][:, :sink_tokens],
                            kv_cache["v"][:, max(sink_tokens, local_end_index - window_tokens) : local_end_index],
                        ],
                        dim=1,
                    )
                else:
                    cached_k = kv_cache["k"][:, :local_end_index]
                    cached_v = kv_cache["v"][:, :local_end_index]

            out = self.attn_op(roped_q, cached_k, cached_v)

            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)


        out = out.flatten(2)
        return self.output_dropout(self.output_proj(out))

    def _sparse_hub_inference(
        self,
        roped_q: torch.Tensor,
        roped_k: torch.Tensor,
        v: torch.Tensor,
        video_size: VideoSize,
        kv_cache: dict,
        current_start: int,
        current_end: int,
        disable_kv_cache_update: bool,
    ) -> torch.Tensor:

        b, total_s, n, d = roped_q.shape
        V = self.num_views
        z_num = self.z_num

        frame_seqlen = video_size.H * video_size.W
        if self.cp_group is not None:
            assert frame_seqlen % self.cp_group.size() == 0
            frame_seqlen = frame_seqlen // self.cp_group.size()

        player_tokens_total = current_end - current_start
        assert player_tokens_total % V == 0, (
            f"sparse_hub: player_tokens_total={player_tokens_total} not divisible by V={V}"
        )
        npb_HW = player_tokens_total // V
        assert npb_HW % frame_seqlen == 0, (
            f"sparse_hub: npb_HW={npb_HW} not divisible by frame_seqlen={frame_seqlen}"
        )
        npb = npb_HW // frame_seqlen
        z_tokens_total = npb * z_num
        expected = player_tokens_total + z_tokens_total
        assert total_s == expected, (
            f"sparse_hub: input sequence length {total_s} != "
            f"V*npb*HW + npb*z_num = {expected}"
        )


        p_q = roped_q[:, :player_tokens_total].view(b, V, npb_HW, n, d)
        p_k = roped_k[:, :player_tokens_total].view(b, V, npb_HW, n, d)
        p_v = v[:, :player_tokens_total].view(b, V, npb_HW, n, d)
        z_q = roped_q[:, player_tokens_total:]
        z_k = roped_k[:, player_tokens_total:]
        z_v = v[:, player_tokens_total:]










        use_rolling_write = (
            self.local_attn_size != -1
            and not torch.is_grad_enabled()
            and not disable_kv_cache_update
        )
        use_window_read = self.local_attn_size != -1

        if use_rolling_write:
            cached_k_p, cached_v_p, cached_k_z, cached_v_z = self._sparse_hub_rolling_step(
                p_k=p_k,
                p_v=p_v,
                z_k=z_k,
                z_v=z_v,
                kv_cache=kv_cache,
                current_end=current_end,
                npb=npb,
                npb_HW=npb_HW,
                z_tokens_total=z_tokens_total,
                frame_seqlen=frame_seqlen,
                V=V,
                z_num=z_num,
            )
        else:
            cached_k_p, cached_v_p, cached_k_z, cached_v_z = self._sparse_hub_chunk_addressed_step(
                p_k=p_k,
                p_v=p_v,
                z_k=z_k,
                z_v=z_v,
                kv_cache=kv_cache,
                current_start=current_start,
                current_end=current_end,
                npb_HW=npb_HW,
                z_tokens_total=z_tokens_total,
                frame_seqlen=frame_seqlen,
                V=V,
                z_num=z_num,
                disable_kv_cache_update=disable_kv_cache_update,
                use_window_read=use_window_read,
            )


        if self.use_batched_attn:



            kv_len_p = cached_k_p.shape[2]
            kv_len_z = cached_k_z.shape[1]
            total_kv_len = kv_len_p + kv_len_z

            k_flat = torch.empty((b * V, total_kv_len, n, d), dtype=v.dtype, device=v.device)
            v_flat = torch.empty((b * V, total_kv_len, n, d), dtype=v.dtype, device=v.device)


            k_view = k_flat.view(b, V, total_kv_len, n, d)
            v_view = v_flat.view(b, V, total_kv_len, n, d)


            k_view[:, :, :kv_len_p].copy_(cached_k_p)
            v_view[:, :, :kv_len_p].copy_(cached_v_p)


            k_view[:, :, kv_len_p:].copy_(cached_k_z.unsqueeze(1))
            v_view[:, :, kv_len_p:].copy_(cached_v_z.unsqueeze(1))



            q_flat = p_q.reshape(b * V, npb_HW, n, d)


            out_p_flat = self.attn_op(q_flat, k_flat, v_flat)


            out_p_flat = out_p_flat.reshape(b * V, npb_HW, n, d)
            out_players = out_p_flat.reshape(b, V * npb_HW, n, d)



            total_z_len = V * kv_len_p + kv_len_z
            k_z_full = torch.empty((b, total_z_len, n, d), dtype=v.dtype, device=v.device)
            v_z_full = torch.empty((b, total_z_len, n, d), dtype=v.dtype, device=v.device)
            k_z_full[:, : V * kv_len_p].view(b, V, kv_len_p, n, d).copy_(cached_k_p)
            v_z_full[:, : V * kv_len_p].view(b, V, kv_len_p, n, d).copy_(cached_v_p)
            k_z_full[:, V * kv_len_p :].copy_(cached_k_z)
            v_z_full[:, V * kv_len_p :].copy_(cached_v_z)
            out_z = self.attn_op(z_q, k_z_full, v_z_full)
            out_z = out_z.reshape(b, z_tokens_total, n, d)


            out = torch.cat([out_players, out_z], dim=1)
        else:

            player_outs = []
            for p in range(V):
                q_p = p_q[:, p]
                k_p_full = torch.cat([cached_k_p[:, p], cached_k_z], dim=1)
                v_p_full = torch.cat([cached_v_p[:, p], cached_v_z], dim=1)
                out_p = self.attn_op(q_p, k_p_full, v_p_full)



                out_p = out_p.reshape(b, npb_HW, n, d)
                player_outs.append(out_p)


            cached_k_p_flat = rearrange(cached_k_p, "b v s n d -> b (v s) n d")
            cached_v_p_flat = rearrange(cached_v_p, "b v s n d -> b (v s) n d")
            k_z_full = torch.cat([cached_k_p_flat, cached_k_z], dim=1)
            v_z_full = torch.cat([cached_v_p_flat, cached_v_z], dim=1)
            out_z = self.attn_op(z_q, k_z_full, v_z_full)
            out_z = out_z.reshape(b, z_tokens_total, n, d)


            out_p_concat = torch.stack(player_outs, dim=1)
            out_p_concat = rearrange(out_p_concat, "b v s n d -> b (v s) n d")
            out = torch.cat([out_p_concat, out_z], dim=1)

        return out

    def _sparse_hub_chunk_addressed_step(
        self,
        p_k: torch.Tensor,
        p_v: torch.Tensor,
        z_k: torch.Tensor,
        z_v: torch.Tensor,
        kv_cache: dict,
        current_start: int,
        current_end: int,
        npb_HW: int,
        z_tokens_total: int,
        frame_seqlen: int,
        V: int,
        z_num: int,
        disable_kv_cache_update: bool,
        use_window_read: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        assert current_start % V == 0 and current_end % V == 0, (
            f"sparse_hub: current_start={current_start} or current_end={current_end} "
            f"not divisible by V={V}"
        )
        p_start = current_start // V
        p_end = current_end // V
        assert (p_end - p_start) == npb_HW, (
            f"sparse_hub: derived (p_end - p_start)={p_end - p_start} != npb_HW={npb_HW}"
        )
        assert p_start % frame_seqlen == 0, (
            f"sparse_hub: p_start={p_start} not divisible by frame_seqlen={frame_seqlen}"
        )



        z_start = (p_start // frame_seqlen) * z_num
        z_end = (p_end // frame_seqlen) * z_num
        assert (z_end - z_start) == z_tokens_total, (
            f"sparse_hub: derived (z_end - z_start)={z_end - z_start} != z_tokens_total={z_tokens_total}"
        )

        if not disable_kv_cache_update:
            if torch.is_grad_enabled():

                kv_cache["k_players"] = torch.cat(
                    [kv_cache["k_players"][:, :, :p_start], p_k, kv_cache["k_players"][:, :, p_end:]],
                    dim=2,
                )
                kv_cache["v_players"] = torch.cat(
                    [kv_cache["v_players"][:, :, :p_start], p_v, kv_cache["v_players"][:, :, p_end:]],
                    dim=2,
                )
                kv_cache["k_z"] = torch.cat(
                    [kv_cache["k_z"][:, :z_start], z_k, kv_cache["k_z"][:, z_end:]],
                    dim=1,
                )
                kv_cache["v_z"] = torch.cat(
                    [kv_cache["v_z"][:, :z_start], z_v, kv_cache["v_z"][:, z_end:]],
                    dim=1,
                )
            else:
                kv_cache["k_players"][:, :, p_start:p_end] = p_k
                kv_cache["v_players"][:, :, p_start:p_end] = p_v
                kv_cache["k_z"][:, z_start:z_end] = z_k
                kv_cache["v_z"][:, z_start:z_end] = z_v






        if use_window_read:
            sink_p = self.sink_size * frame_seqlen
            sink_z = self.sink_size * z_num
            window_p = (self.local_attn_size - self.sink_size) * frame_seqlen
            window_z = (self.local_attn_size - self.sink_size) * z_num

            if disable_kv_cache_update:



                cached_k_p = torch.cat(
                    [
                        kv_cache["k_players"][:, :, :sink_p],
                        kv_cache["k_players"][:, :, max(sink_p, p_start - window_p) : p_start],
                        p_k,
                    ],
                    dim=2,
                )
                cached_v_p = torch.cat(
                    [
                        kv_cache["v_players"][:, :, :sink_p],
                        kv_cache["v_players"][:, :, max(sink_p, p_start - window_p) : p_start],
                        p_v,
                    ],
                    dim=2,
                )
                cached_k_z = torch.cat(
                    [
                        kv_cache["k_z"][:, :sink_z],
                        kv_cache["k_z"][:, max(sink_z, z_start - window_z) : z_start],
                        z_k,
                    ],
                    dim=1,
                )
                cached_v_z = torch.cat(
                    [
                        kv_cache["v_z"][:, :sink_z],
                        kv_cache["v_z"][:, max(sink_z, z_start - window_z) : z_start],
                        z_v,
                    ],
                    dim=1,
                )
            else:



                cached_k_p = torch.cat(
                    [
                        kv_cache["k_players"][:, :, :sink_p],
                        kv_cache["k_players"][:, :, max(sink_p, p_end - window_p) : p_end],
                    ],
                    dim=2,
                )
                cached_v_p = torch.cat(
                    [
                        kv_cache["v_players"][:, :, :sink_p],
                        kv_cache["v_players"][:, :, max(sink_p, p_end - window_p) : p_end],
                    ],
                    dim=2,
                )
                cached_k_z = torch.cat(
                    [
                        kv_cache["k_z"][:, :sink_z],
                        kv_cache["k_z"][:, max(sink_z, z_end - window_z) : z_end],
                    ],
                    dim=1,
                )
                cached_v_z = torch.cat(
                    [
                        kv_cache["v_z"][:, :sink_z],
                        kv_cache["v_z"][:, max(sink_z, z_end - window_z) : z_end],
                    ],
                    dim=1,
                )
        else:

            if disable_kv_cache_update:
                cached_k_p = torch.cat([kv_cache["k_players"][:, :, :p_start], p_k], dim=2)
                cached_v_p = torch.cat([kv_cache["v_players"][:, :, :p_start], p_v], dim=2)
                cached_k_z = torch.cat([kv_cache["k_z"][:, :z_start], z_k], dim=1)
                cached_v_z = torch.cat([kv_cache["v_z"][:, :z_start], z_v], dim=1)
            else:
                cached_k_p = kv_cache["k_players"][:, :, :p_end]
                cached_v_p = kv_cache["v_players"][:, :, :p_end]
                cached_k_z = kv_cache["k_z"][:, :z_end]
                cached_v_z = kv_cache["v_z"][:, :z_end]

        if not disable_kv_cache_update:
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(p_end)
            kv_cache["z_local_end_index"].fill_(z_end)

        return cached_k_p, cached_v_p, cached_k_z, cached_v_z

    def _sparse_hub_rolling_step(
        self,
        p_k: torch.Tensor,
        p_v: torch.Tensor,
        z_k: torch.Tensor,
        z_v: torch.Tensor,
        kv_cache: dict,
        current_end: int,
        npb: int,
        npb_HW: int,
        z_tokens_total: int,
        frame_seqlen: int,
        V: int,
        z_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        sink_p = self.sink_size * frame_seqlen
        sink_z = self.sink_size * z_num
        cache_size_p = kv_cache["k_players"].shape[2]
        cache_size_z = kv_cache["k_z"].shape[1]
        num_new_p = npb_HW
        num_new_z = z_tokens_total


        global_end = kv_cache["global_end_index"].item()
        local_end_p_cached = kv_cache["local_end_index"].item()
        local_end_z_cached = kv_cache["z_local_end_index"].item()





        chunk_advance_p = (current_end - global_end) // V

        assert (current_end - global_end) * z_num % (V * frame_seqlen) == 0, (
            f"sparse_hub rolling: chunk advance {(current_end - global_end)} not aligned for "
            f"Z stride z_num={z_num} / V*frame_seqlen={V * frame_seqlen}"
        )
        chunk_advance_z = (current_end - global_end) * z_num // (V * frame_seqlen)

        if current_end > global_end and num_new_p + local_end_p_cached > cache_size_p:


            num_evicted_p = num_new_p + local_end_p_cached - cache_size_p




            assert num_evicted_p % frame_seqlen == 0, (
                f"sparse_hub rolling: num_evicted_p={num_evicted_p} not aligned to "
                f"frame_seqlen={frame_seqlen}; check that local_attn_size, sink_size, "
                f"and num_frame_per_block are all integer-frame quantities"
            )
            num_rolled_p = local_end_p_cached - num_evicted_p - sink_p
            assert num_rolled_p >= 0, (
                f"sparse_hub rolling: num_rolled_p={num_rolled_p} < 0; "
                f"local_attn_size={self.local_attn_size} + sink_size={self.sink_size} "
                f"too small to fit one chunk of npb={npb} frames"
            )

            num_evicted_z = num_evicted_p // frame_seqlen * z_num
            num_rolled_z = local_end_z_cached - num_evicted_z - sink_z
            assert num_rolled_z >= 0, (
                f"sparse_hub rolling: num_rolled_z={num_rolled_z} < 0 (Z desync from player roll)"
            )
            assert num_new_z + local_end_z_cached - num_evicted_z <= cache_size_z, (
                f"sparse_hub rolling: Z cache overflow after eviction; "
                f"num_new_z={num_new_z}, local_end_z_cached={local_end_z_cached}, "
                f"num_evicted_z={num_evicted_z}, cache_size_z={cache_size_z}"
            )



            kv_cache["k_players"][:, :, sink_p : sink_p + num_rolled_p] = kv_cache["k_players"][
                :, :, sink_p + num_evicted_p : sink_p + num_evicted_p + num_rolled_p
            ].clone()
            kv_cache["v_players"][:, :, sink_p : sink_p + num_rolled_p] = kv_cache["v_players"][
                :, :, sink_p + num_evicted_p : sink_p + num_evicted_p + num_rolled_p
            ].clone()

            kv_cache["k_z"][:, sink_z : sink_z + num_rolled_z] = kv_cache["k_z"][
                :, sink_z + num_evicted_z : sink_z + num_evicted_z + num_rolled_z
            ].clone()
            kv_cache["v_z"][:, sink_z : sink_z + num_rolled_z] = kv_cache["v_z"][
                :, sink_z + num_evicted_z : sink_z + num_evicted_z + num_rolled_z
            ].clone()

            new_local_end_p = local_end_p_cached + chunk_advance_p - num_evicted_p
            new_local_end_z = local_end_z_cached + chunk_advance_z - num_evicted_z
        else:


            new_local_end_p = local_end_p_cached + chunk_advance_p
            new_local_end_z = local_end_z_cached + chunk_advance_z
            assert new_local_end_p <= cache_size_p, (
                f"sparse_hub rolling: cache overflow without eviction; "
                f"new_local_end_p={new_local_end_p}, cache_size_p={cache_size_p}"
            )
            assert new_local_end_z <= cache_size_z, (
                f"sparse_hub rolling: Z cache overflow without eviction; "
                f"new_local_end_z={new_local_end_z}, cache_size_z={cache_size_z}"
            )

        new_local_start_p = new_local_end_p - num_new_p
        new_local_start_z = new_local_end_z - num_new_z


        kv_cache["k_players"][:, :, new_local_start_p:new_local_end_p] = p_k
        kv_cache["v_players"][:, :, new_local_start_p:new_local_end_p] = p_v
        kv_cache["k_z"][:, new_local_start_z:new_local_end_z] = z_k
        kv_cache["v_z"][:, new_local_start_z:new_local_end_z] = z_v


        window_p = (self.local_attn_size - self.sink_size) * frame_seqlen
        window_z = (self.local_attn_size - self.sink_size) * z_num
        cached_k_p = torch.cat(
            [
                kv_cache["k_players"][:, :, :sink_p],
                kv_cache["k_players"][:, :, max(sink_p, new_local_end_p - window_p) : new_local_end_p],
            ],
            dim=2,
        )
        cached_v_p = torch.cat(
            [
                kv_cache["v_players"][:, :, :sink_p],
                kv_cache["v_players"][:, :, max(sink_p, new_local_end_p - window_p) : new_local_end_p],
            ],
            dim=2,
        )
        cached_k_z = torch.cat(
            [
                kv_cache["k_z"][:, :sink_z],
                kv_cache["k_z"][:, max(sink_z, new_local_end_z - window_z) : new_local_end_z],
            ],
            dim=1,
        )
        cached_v_z = torch.cat(
            [
                kv_cache["v_z"][:, :sink_z],
                kv_cache["v_z"][:, max(sink_z, new_local_end_z - window_z) : new_local_end_z],
            ],
            dim=1,
        )

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(new_local_end_p)
        kv_cache["z_local_end_index"].fill_(new_local_end_z)

        return cached_k_p, cached_v_p, cached_k_z, cached_v_z

    def set_context_parallel_group(self, process_group: ProcessGroup | None, ranks, stream, cp_comm_type: str = "p2p"):
        self.attn_op.set_context_parallel_group(process_group, ranks, stream, cp_comm_type=cp_comm_type)
        self.cp_group = process_group


class CausalCosmosBlock(nn.Module):


    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "transformer_engine",
        image_context_dim: int | None = None,
        use_wan_fp32_strategy: bool = False,

        local_attn_size: int = -1,
        sink_size: int = 0,
        action_dim: int = 2048,

        attention_mode: str = "dense",
        num_views: int = 1,
        z_num: int = 0,
        use_batched_attn: bool = True,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.use_wan_fp32_strategy = use_wan_fp32_strategy


        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = CausalSelfAttention(
            query_dim=x_dim,
            context_dim=None,
            n_heads=num_heads,
            head_dim=x_dim // num_heads,
            qkv_format="bshd",
            backend=backend,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            attention_mode=attention_mode,
            num_views=num_views,
            z_num=z_num,
            use_batched_attn=use_batched_attn,
        )


        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        if image_context_dim is None:
            self.cross_attn = Attention(
                x_dim,
                context_dim,
                num_heads,
                x_dim // num_heads,
                qkv_format="bshd",
                backend=backend,
            )
        else:
            self.cross_attn = I2VCrossAttention(
                x_dim,
                context_dim,
                num_heads,
                x_dim // num_heads,
                img_latent_dim=image_context_dim,
                qkv_format="bshd",
                backend=backend,
            )


        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))


        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        self.cp_size: int | None = None
        self.action_dim = action_dim
        self.input_encoder = nn.Linear(self.action_dim, x_dim, bias=False)

    def init_weights(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

        torch.nn.init.zeros_(self.input_encoder.weight)

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        self.cp_size = None if ranks is None else len(ranks)
        self.self_attn.set_context_parallel_group(process_group, ranks, stream, cp_comm_type=cp_comm_type)

    def forward(
        self,
        x_B_L_D: torch.Tensor,
        emb_B_L_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_B_L_D: torch.Tensor | None = None,
        adaln_lora_B_L_3D: torch.Tensor | None = None,
        extra_per_block_pos_emb: torch.Tensor | None = None,

        block_mask: BlockMask | None = None,
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        current_end: int = 0,
        disable_kv_cache: bool = False,
        disable_kv_cache_update: bool = False,
        video_size: VideoSize | None = None,
        action_bias_B_L_D: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if extra_per_block_pos_emb is not None:
            x_B_L_D = x_B_L_D + extra_per_block_pos_emb


        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                shift_self, scale_self, gate_self = (
                    self.adaln_modulation_self_attn(emb_B_L_D) + adaln_lora_B_L_3D
                ).chunk(3, dim=-1)
                shift_cross, scale_cross, gate_cross = (
                    self.adaln_modulation_cross_attn(emb_B_L_D) + adaln_lora_B_L_3D
                ).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_B_L_D) + adaln_lora_B_L_3D).chunk(
                    3, dim=-1
                )
            else:
                shift_self, scale_self, gate_self = self.adaln_modulation_self_attn(emb_B_L_D).chunk(3, dim=-1)
                shift_cross, scale_cross, gate_cross = self.adaln_modulation_cross_attn(emb_B_L_D).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_L_D).chunk(3, dim=-1)



        shift_self = shift_self.type_as(x_B_L_D)
        scale_self = scale_self.type_as(x_B_L_D)
        gate_self = gate_self.type_as(x_B_L_D)

        shift_cross = shift_cross.type_as(x_B_L_D)
        scale_cross = scale_cross.type_as(x_B_L_D)
        gate_cross = gate_cross.type_as(x_B_L_D)

        shift_mlp = shift_mlp.type_as(x_B_L_D)
        scale_mlp = scale_mlp.type_as(x_B_L_D)
        gate_mlp = gate_mlp.type_as(x_B_L_D)


        normed_x = self.layer_norm_self_attn(x_B_L_D) * (1 + scale_self) + shift_self

        action_emb = self.input_encoder(action_bias_B_L_D)
        normed_x = normed_x + action_emb

        attn_out = self.self_attn(
            normed_x,
            context=None,
            rope_emb=rope_emb_B_L_D,
            video_size=video_size,
            block_mask=block_mask,
            kv_cache=kv_cache,
            current_start=current_start,
            current_end=current_end,
            disable_kv_cache=disable_kv_cache,
            disable_kv_cache_update=disable_kv_cache_update,
        )
        x_B_L_D = x_B_L_D + gate_self * attn_out



        normed_x = self.layer_norm_cross_attn(x_B_L_D) * (1 + scale_cross) + shift_cross

        if crossattn_cache is not None:
            cross_out = self.cross_attn(normed_x, crossattn_emb, crossattn_cache=crossattn_cache)
        else:
            cross_out = self.cross_attn(normed_x, crossattn_emb)

        x_B_L_D = x_B_L_D + gate_cross * cross_out


        normed_x = self.layer_norm_mlp(x_B_L_D) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(normed_x)
        x_B_L_D = x_B_L_D + gate_mlp * mlp_out

        return x_B_L_D


class CosmosCausalDiT(WeightTrainingStat):


    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        extra_image_context_dim: int | None = None,
        pos_emb_cls: str = "rope3d",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        timestep_scale: float = 1.0,

        local_attn_size: int = -1,
        sink_size: int = 0,
        sac_config: SACConfig = SACConfig(),
        postpone_checkpoint: bool = False,
        on_the_fly_checkpoint: bool = False,
        use_wan_fp32_strategy: bool = False,

        num_views: int = 8,
        enable_action_control: bool = False,
        action_keyboard_dim: int = 23,
        action_camera_dim: int = 2,
        action_use_camera: bool = True,
        action_embed_dim: int = 256,
        action_temporal_downsample: int = 4,
        action_use_conv1d: bool = True,










        enable_view_embedding: bool = False,
        use_multi_agent_rope: bool = False,
        multi_agent_rope_num_agents: int = 2,
        multi_agent_rope_agent_id_offset: int = 0,
        multi_agent_rope_agent_encoding: str = "linear",
        multi_agent_rope_agent_scale: float = 1.0,
        multi_agent_rope_simplex_pool_size: int | None = None,
        multi_agent_rope_share_action_encoder: bool = False,







        action_spatial_concat: bool = False,

        use_sparse_hub: bool = False,
        z_num: int = 8,
        z_init_std: float = 0.02,
        sparse_hub_use_batched_attn: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert not (enable_view_embedding and use_multi_agent_rope), (
            "enable_view_embedding and use_multi_agent_rope are alternative ways to encode "
            "agent identity (view embedding adds a per-frame embedding to patch tokens; "
            "4D RoPE encodes agent in the rotary phase). Enabling both at once is unintended."
        )
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.timestep_scale = timestep_scale

        self.in_channels = in_channels + 1
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_layers = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.on_the_fly_checkpoint = on_the_fly_checkpoint


        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.extra_image_context_dim = extra_image_context_dim

        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.use_crossattn_projection = use_crossattn_projection
        self.crossattn_proj_in_channels = crossattn_proj_in_channels

        if on_the_fly_checkpoint:
            self.postpone_checkpoint = True
        else:
            self.postpone_checkpoint = postpone_checkpoint


        self._build_patch_embed()
        self._build_pos_embed()


        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )
        self.t_embedding_norm = te.pytorch.RMSNorm(model_channels, eps=1e-6)




        self.use_sparse_hub = use_sparse_hub
        self.z_num = z_num if use_sparse_hub else 0
        self.z_init_std = z_init_std
        if use_sparse_hub:
            assert use_multi_agent_rope, (
                "use_sparse_hub=True requires use_multi_agent_rope=True so that "
                "Z tokens can share the temporal RoPE band with player tokens"
            )
            assert multi_agent_rope_num_agents >= 2, (
                f"use_sparse_hub requires multi_agent_rope_num_agents>=2, "
                f"got {multi_agent_rope_num_agents}"
            )
            assert z_num >= 1, f"use_sparse_hub requires z_num>=1, got {z_num}"

        attention_mode_for_blocks = "sparse_hub" if use_sparse_hub else "dense"
        num_views_for_blocks = multi_agent_rope_num_agents if use_sparse_hub else 1


        self.blocks = nn.ModuleList(
            [
                CausalCosmosBlock(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend="transformer_engine",
                    image_context_dim=None if extra_image_context_dim is None else model_channels,
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    action_dim=model_channels,
                    attention_mode=attention_mode_for_blocks,
                    num_views=num_views_for_blocks,
                    z_num=self.z_num,
                    use_batched_attn=sparse_hub_use_batched_attn,
                )
                for _ in range(num_blocks)
            ]
        )


        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )


        if extra_image_context_dim is not None:
            self.img_context_proj = nn.Sequential(
                nn.Linear(extra_image_context_dim, model_channels, bias=True),
                nn.GELU(),
            )

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )







        self.num_views = num_views
        self.view_embedding = nn.Embedding(num_views, model_channels)
        self.enable_view_embedding = enable_view_embedding


        self.use_multi_agent_rope = use_multi_agent_rope
        self.multi_agent_token_layout = "sequence" if use_multi_agent_rope else "spatial"
        self.num_agents = multi_agent_rope_num_agents
        self.agent_id_offset = multi_agent_rope_agent_id_offset
        self.agent_encoding = multi_agent_rope_agent_encoding
        self.agent_scale = multi_agent_rope_agent_scale
        self.simplex_pool_size = (
            multi_agent_rope_simplex_pool_size
            if multi_agent_rope_simplex_pool_size is not None
            else multi_agent_rope_num_agents
        )
        assert self.agent_encoding in ("linear", "simplex"), (
            f"multi_agent_rope_agent_encoding must be 'linear' or 'simplex', got {self.agent_encoding!r}"
        )
        assert self.simplex_pool_size >= self.num_agents, (
            f"multi_agent_rope_simplex_pool_size ({self.simplex_pool_size}) must be "
            f">= multi_agent_rope_num_agents ({self.num_agents})"
        )






        if self.agent_encoding == "simplex":
            assert self.agent_id_offset == 0, (
                f"multi_agent_rope_agent_id_offset must be 0 under simplex encoding "
                f"(simplex vertices are already non-degenerate); got {self.agent_id_offset}. "
                f"Set multi_agent_rope_agent_id_offset=0 explicitly in your simplex experiment "
                f"config to override any value inherited from a linear-RoPE parent."
            )
        self.freqs_4d: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None




        self._override_agent_pool_indices: list[int] | None = None








        if self.use_sparse_hub:
            self.z_tokens = nn.Parameter(torch.zeros(self.z_num, model_channels))



        self.enable_action_control = enable_action_control

        self.action_spatial_concat = action_spatial_concat
        if action_spatial_concat:
            assert not use_multi_agent_rope, (
                "action_spatial_concat=True requires n_views=1, but "
                "use_multi_agent_rope=True requires n_views > 1; these are "
                "mutually exclusive. Disable use_multi_agent_rope for the "
                "spatial-concat ablation."
            )
            assert not multi_agent_rope_share_action_encoder, (
                "action_spatial_concat=True already uses a single shared encoder; "
                "do not also set multi_agent_rope_share_action_encoder=True."
            )
        self.multi_agent_action_control: MultiAgentActionControlModule | None = None
        if enable_action_control:
            self.multi_agent_action_control = build_multi_agent_action_control(
                num_agents=multi_agent_rope_num_agents,
                pool_size=self.simplex_pool_size,
                share_encoder=multi_agent_rope_share_action_encoder,
                spatial_concat=action_spatial_concat,
                keyboard_dim=action_keyboard_dim,
                camera_dim=action_camera_dim,
                use_camera=action_use_camera,
                embed_dim=action_embed_dim,
                dit_dim=model_channels,
                temporal_downsample=action_temporal_downsample,
                use_conv1d=action_use_conv1d,
            )


        self.init_weights()


        self.sac_config = sac_config
        if not postpone_checkpoint:
            self.enable_selective_checkpoint(sac_config, self.blocks)


        self.block_mask_dict: dict[str, BlockMask] = {}
        self.num_frame_per_block = 1


        self.cp_group: ProcessGroup | None = None
        self._is_context_parallel_enabled = False

    def _build_patch_embed(self) -> None:
        in_ch = self.in_channels + 1 if self.concat_padding_mask else self.in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_ch,
            out_channels=self.model_channels,
        )

    def _build_pos_embed(self) -> None:
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        len_h = self.max_img_h // self.patch_spatial
        len_w = self.max_img_w // self.patch_spatial
        len_t = self.max_frames // self.patch_temporal
        head_dim = self.model_channels // self.num_heads

        self.pos_embedder = cls_type(
            head_dim=head_dim,
            len_h=len_h,
            len_w=len_w,
            len_t=len_t,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )

        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = LearnablePosEmbAxis(
                interpolation=self.pos_emb_interpolation,
                model_channels=self.model_channels,
                len_h=len_h,
                len_w=len_w,
                len_t=len_t,
            )

    def init_weights(self) -> None:
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()

        self.t_embedder[1].init_weights()
        self.t_embedding_norm.reset_parameters()

        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()

        if self.extra_image_context_dim is not None:
            self.img_context_proj[0].reset_parameters()

        if hasattr(self, "view_embedding"):
            nn.init.zeros_(self.view_embedding.weight)

        if hasattr(self, "z_tokens"):


            nn.init.trunc_normal_(self.z_tokens, std=self.z_init_std, a=-3 * self.z_init_std, b=3 * self.z_init_std)

        if hasattr(self, "multi_agent_action_control") and self.multi_agent_action_control is not None:
            for m in self.multi_agent_action_control.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    if m.weight is not None:
                        nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, RMS_norm):
                    nn.init.ones_(m.gamma)
                    if isinstance(m.bias, nn.Parameter):
                        nn.init.zeros_(m.bias)

    def _generate_multi_agent_rope(
        self,
        t_per_view: int,
        H: int,
        W: int,
        n_views: int,
        device: torch.device,
        start_frame_for_rope: int = 0,
    ) -> torch.Tensor:

        if self.freqs_4d is None:
            head_dim = self.model_channels // self.num_heads
            self.freqs_4d = precompute_freqs_cis_4d(
                head_dim,
                agent_encoding=self.agent_encoding,
                num_agents=self.simplex_pool_size,
                agent_scale=self.agent_scale,
            )

        freqs_th, freqs_agent, freqs_h, freqs_w = self.freqs_4d
        agent_pool_indices = self._resolve_agent_pool_indices(n_views)
        all_freqs = []
        for agent_id in range(n_views):
            if self.agent_encoding == "simplex":
                agent_idx = agent_pool_indices[agent_id]
            else:
                agent_idx = agent_id + self.agent_id_offset
            t_slice = freqs_th[start_frame_for_rope : start_frame_for_rope + t_per_view]
            agent_freqs = torch.cat(
                [
                    t_slice.view(t_per_view, 1, 1, -1).expand(t_per_view, H, W, -1),
                    freqs_agent[agent_idx : agent_idx + 1].view(1, 1, 1, -1).expand(t_per_view, H, W, -1),
                    freqs_h[:H].view(1, H, 1, -1).expand(t_per_view, H, W, -1),
                    freqs_w[:W].view(1, 1, W, -1).expand(t_per_view, H, W, -1),
                ],
                dim=-1,
            )
            all_freqs.append(agent_freqs.reshape(t_per_view * H * W, 1, -1))
        complex_freqs = torch.cat(all_freqs, dim=0)

        angles = torch.angle(complex_freqs).unsqueeze(2)
        return torch.cat([angles, angles], dim=-1).to(device=device, dtype=torch.float32)

    def _generate_z_rope(
        self,
        t_per_view: int,
        device: torch.device,
        start_frame_for_rope: int = 0,
    ) -> torch.Tensor:

        assert self.freqs_4d is not None, (
            "_generate_z_rope must be called after _generate_multi_agent_rope "
            "so freqs_4d is initialized"
        )
        z_num = self.z_num
        assert z_num >= 1

        freqs_th, freqs_agent, freqs_h, freqs_w = self.freqs_4d
        dim_th = freqs_th.shape[-1]
        dim_agent = freqs_agent.shape[-1]
        dim_h = freqs_h.shape[-1]
        dim_w = freqs_w.shape[-1]

        t_slice = freqs_th[start_frame_for_rope : start_frame_for_rope + t_per_view]

        t_part = t_slice.view(t_per_view, 1, dim_th).expand(t_per_view, z_num, dim_th)



        ones_a = torch.ones(t_per_view, z_num, dim_agent, dtype=t_part.dtype, device=t_part.device)
        ones_h = torch.ones(t_per_view, z_num, dim_h, dtype=t_part.dtype, device=t_part.device)
        ones_w = torch.ones(t_per_view, z_num, dim_w, dtype=t_part.dtype, device=t_part.device)

        z_complex = torch.cat([t_part, ones_a, ones_h, ones_w], dim=-1)
        z_complex = z_complex.reshape(t_per_view * z_num, 1, -1)
        angles = torch.angle(z_complex).unsqueeze(2)
        return torch.cat([angles, angles], dim=-1).to(device=device, dtype=torch.float32)

    def _infer_n_views(self, view_indices_B_T: torch.Tensor | None) -> int:

        if view_indices_B_T is not None:
            max_idx = int(view_indices_B_T.max().item())
            min_idx = int(view_indices_B_T.min().item())
            assert min_idx >= 0 and (max_idx + 1) <= self.num_views, (
                f"view_indices out of range: range=[{min_idx}, {max_idx}], "
                f"num_views={self.num_views}"
            )



            return int(view_indices_B_T[0].unique().numel())
        return self.num_agents

    def _resolve_agent_pool_indices(self, n_views: int) -> list[int]:

        if self._override_agent_pool_indices is not None:
            indices = list(self._override_agent_pool_indices)
            assert len(indices) == n_views, (
                f"_override_agent_pool_indices length {len(indices)} does not "
                f"match n_views={n_views}"
            )
            assert max(indices) < self.simplex_pool_size, (
                f"_override_agent_pool_indices max {max(indices)} out of range "
                f"for simplex_pool_size={self.simplex_pool_size}"
            )
            assert len(set(indices)) == n_views, (
                f"_override_agent_pool_indices must be unique, got {indices}"
            )
            return indices
        assert n_views <= self.simplex_pool_size, (
            f"n_views={n_views} exceeds simplex_pool_size={self.simplex_pool_size}; "
            f"either pass _override_agent_pool_indices or increase the pool"
        )
        return list(range(n_views))

    def _add_view_embedding(self, x_B_T_H_W_D: torch.Tensor, view_indices_B_T: torch.Tensor) -> torch.Tensor:

        view_emb_B_T_D = self.view_embedding(view_indices_B_T)
        return x_B_T_H_W_D + view_emb_B_T_D[:, :, None, None, :]

    def _get_action_bias(
        self,
        x_B_T_H_W_D: torch.Tensor,
        action_inputs: dict,
        n_views: int,
        start_frame: int = 0,
        agent_pool_indices: list[int] | None = None,
    ) -> torch.Tensor:

        if isinstance(self.multi_agent_action_control, MultiAgentActionControlSpatialConcatModule):
            assert n_views == 1, (
                f"spatial-concat action control expects n_views=1, got n_views={n_views}"
            )
        else:
            assert n_views == self.multi_agent_action_control.num_agents, (
                f"action control expects n_views={self.multi_agent_action_control.num_agents}, "
                f"got n_views={n_views}"
            )
        if agent_pool_indices is None:
            agent_pool_indices = self._resolve_agent_pool_indices(n_views)
        B, T, H, W, D = x_B_T_H_W_D.shape
        action_bias = self.multi_agent_action_control(
            actions=action_inputs["actions"],
            x_shape=(B, D, T, H, W),
            device=x_B_T_H_W_D.device,
            dtype=x_B_T_H_W_D.dtype,
            n_views=n_views,
            start_frame=start_frame,
            agent_pool_indices=agent_pool_indices,
        )
        return action_bias.reshape(B, T, H, W, D)

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        num_interleave: int = 0,
        cp_size: int = 1,
    ) -> BlockMask:

        log.info(
            f"Constructing block mask: num_frames={num_frames}, frame_seqlen={frame_seqlen}, "
            f"num_frame_per_block={num_frame_per_block}, cp_size={cp_size}"
        )

        total_length = num_frames * frame_seqlen * (1 + num_interleave)



        local_len = total_length // cp_size
        local_padded_len = math.ceil(local_len / 128) * 128
        total_padded_len = local_padded_len * cp_size
        padded_length = total_padded_len - total_length

        log.info(f"Block maskpadded_length={padded_length}")

        if num_interleave == 0:



            ends = torch.zeros(total_padded_len, device=device, dtype=torch.long)
            frame_indices = torch.arange(
                start=0, end=total_length, step=frame_seqlen * num_frame_per_block, device=device
            )
            for idx in frame_indices:
                ends[idx : idx + frame_seqlen * num_frame_per_block] = idx + frame_seqlen * num_frame_per_block

            def attention_mask(b, h, q_idx, kv_idx):

                q_rank = q_idx // local_padded_len
                q_off = q_idx % local_padded_len
                q_logical = q_rank * local_len + q_off
                q_valid = q_off < local_len

                kv_rank = kv_idx // local_padded_len
                kv_off = kv_idx % local_padded_len
                kv_logical = kv_rank * local_len + kv_off
                kv_valid = kv_off < local_len

                causal_check = (
                    q_valid & kv_valid & ((kv_logical < ends[q_logical.to(torch.long)]) | (q_logical == kv_logical))
                )

                return causal_check

        else:

            num_interleaved_frames_per_time = 1 + num_interleave

            frame_types = torch.zeros(total_padded_len, device=device, dtype=torch.long)
            frame_indices = torch.zeros(total_padded_len, device=device, dtype=torch.long)
            total_indices_per_time = frame_seqlen * num_frame_per_block * num_interleaved_frames_per_time

            position = 0
            time_frame_idx = 0
            while position < total_length:
                for _ in range(num_frame_per_block):
                    for interleave_idx in range(num_interleaved_frames_per_time):
                        end_pos = min(position + frame_seqlen, total_length)
                        frame_types[position:end_pos] = interleave_idx
                        frame_indices[position:end_pos] = time_frame_idx
                        position = end_pos
                        if position >= total_length:
                            break
                time_frame_idx += 1

            attend_frame_type = num_interleaved_frames_per_time - 1

            def attention_mask(b, h, q_idx, kv_idx):

                q_rank = q_idx // local_padded_len
                q_off = q_idx % local_padded_len
                q_logical = q_rank * local_len + q_off
                q_valid = q_off < local_len

                kv_rank = kv_idx // local_padded_len
                kv_off = kv_idx % local_padded_len
                kv_logical = kv_rank * local_len + kv_off
                kv_valid = kv_off < local_len

                q_frame_type = frame_types[q_logical]
                q_frame_idx = frame_indices[q_logical]
                kv_frame_type = frame_types[kv_logical]
                kv_frame_idx = frame_indices[kv_logical]

                same_block = ((q_logical // total_indices_per_time) == (kv_logical // total_indices_per_time)) & (
                    kv_frame_type == q_frame_type
                )
                cross_block_attend = (kv_frame_type == attend_frame_type) & (kv_frame_idx < q_frame_idx)

                return q_valid & kv_valid & (same_block | cross_block_attend)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_padded_len,
            KV_LEN=total_padded_len,
            _compile=True,
            device=device,
        )

        if not is_initialized() or get_rank() == 0:
            log.info(f"Cached block-wise causal mask with block size {num_frame_per_block} frames")
            log.debug(f"Block mask: {block_mask}")

        return block_mask

    @staticmethod
    def _prepare_sparse_hub_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_blocks_per_view: int,
        n_views: int,
        num_frame_per_block: int,
        frame_seqlen: int,
        z_num: int,
        cp_size: int = 1,
    ) -> BlockMask:

        npb_HW = num_frame_per_block * frame_seqlen
        z_seg = num_frame_per_block * z_num
        block_size = n_views * npb_HW + z_seg
        z_offset = n_views * npb_HW

        total_length = num_blocks_per_view * block_size


        local_len = total_length // cp_size
        local_padded_len = math.ceil(local_len / 128) * 128
        total_padded_len = local_padded_len * cp_size

        log.info(
            f"Constructing sparse hub mask: num_blocks={num_blocks_per_view}, "
            f"n_views={n_views}, npb={num_frame_per_block}, "
            f"frame_seqlen={frame_seqlen}, z_num={z_num}, "
            f"block_size={block_size}, total_padded_len={total_padded_len}, cp_size={cp_size}"
        )


        _block_size = block_size
        _z_offset = z_offset
        _npb_HW = npb_HW
        _V = n_views
        _local_padded_len = local_padded_len
        _local_len = local_len

        def attention_mask(b, h, q_idx, kv_idx):

            q_rank = q_idx // _local_padded_len
            q_off = q_idx % _local_padded_len
            q_logical = q_rank * _local_len + q_off
            q_valid = q_off < _local_len

            kv_rank = kv_idx // _local_padded_len
            kv_off = kv_idx % _local_padded_len
            kv_logical = kv_rank * _local_len + kv_off
            kv_valid = kv_off < _local_len

            q_block = q_logical // _block_size
            kv_block = kv_logical // _block_size

            q_in = q_logical % _block_size
            kv_in = kv_logical % _block_size


            q_player = torch.where(q_in < _z_offset, q_in // _npb_HW, torch.full_like(q_in, _V))
            kv_player = torch.where(kv_in < _z_offset, kv_in // _npb_HW, torch.full_like(kv_in, _V))





            causal = q_block >= kv_block
            sparse_ok = (q_player == kv_player) | (q_player == _V) | (kv_player == _V)

            return q_valid & kv_valid & causal & sparse_ok

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_padded_len,
            KV_LEN=total_padded_len,
            _compile=True,
            device=device,
        )

        if not is_initialized() or get_rank() == 0:
            log.info(
                f"Cached sparse-hub block-causal mask: V={n_views}, "
                f"num_blocks={num_blocks_per_view}, npb={num_frame_per_block}, "
                f"z_num={z_num}"
            )
            log.debug(f"Sparse hub mask: {block_mask}")

        return block_mask

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.VIDEO,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int = 0,
        current_end: int = 0,
        start_frame_for_rope: int = 0,
        disable_kv_cache: bool = False,
        num_interleave: int = 0,
        img_context_emb: torch.Tensor | None = None,
        view_indices_B_T: torch.Tensor | None = None,
        action_inputs: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:

        del kwargs
        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
        timesteps_B_T = timesteps_B_T * self.timestep_scale
        if kv_cache is not None:
            return self._forward_inference(
                x_B_C_T_H_W=x_B_C_T_H_W,
                timesteps_B_T=timesteps_B_T,
                crossattn_emb=crossattn_emb,
                fps=fps,
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                start_frame_for_rope=start_frame_for_rope,
                disable_kv_cache=disable_kv_cache,
                img_context_emb=img_context_emb,
                view_indices_B_T=view_indices_B_T,
                action_inputs=action_inputs,
            )
        else:
            return self._forward_train(
                x_B_C_T_H_W=x_B_C_T_H_W,
                timesteps_B_T=timesteps_B_T,
                crossattn_emb=crossattn_emb,
                fps=fps,
                padding_mask=padding_mask,
                data_type=data_type,
                num_interleave=num_interleave,
                img_context_emb=img_context_emb,
                view_indices_B_T=view_indices_B_T,
                action_inputs=action_inputs,
            )

    def _forward_train(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.VIDEO,
        num_interleave: int = 0,
        img_context_emb: torch.Tensor | None = None,
        view_indices_B_T: torch.Tensor | None = None,
        action_inputs: dict | None = None,
    ) -> torch.Tensor:

        device = self.x_embedder.proj[1].weight.device


        frame_seqlen = x_B_C_T_H_W.shape[-2] * x_B_C_T_H_W.shape[-1] // (self.patch_spatial * self.patch_spatial)
        num_frames = x_B_C_T_H_W.shape[2]

        cp_size = 1
        if self._is_context_parallel_enabled and self.cp_group is not None:
            cp_size = self.cp_group.size()

        n_views = self._infer_n_views(view_indices_B_T)
        use_joint_block_layout = n_views > 1


        mask_num_frame_per_block = (
            n_views * self.num_frame_per_block if use_joint_block_layout else self.num_frame_per_block
        )

        if self.use_sparse_hub:



            assert use_joint_block_layout, (
                "use_sparse_hub requires multi-view input (n_views > 1)"
            )
            assert num_interleave == 0, (
                "use_sparse_hub is incompatible with num_interleave > 0"
            )
            t_per_view_full = num_frames // n_views
            assert t_per_view_full % self.num_frame_per_block == 0, (
                f"T_per_view={t_per_view_full} not divisible by "
                f"num_frame_per_block={self.num_frame_per_block}"
            )
            num_blocks_per_view = t_per_view_full // self.num_frame_per_block
            mask_key = (
                f"sparse_hub_mask_v{n_views}_blks{num_blocks_per_view}_npb{self.num_frame_per_block}"
                f"_seqlen{frame_seqlen}_z{self.z_num}_cp{cp_size}"
            )
            if mask_key not in self.block_mask_dict:
                block_mask = self._prepare_sparse_hub_blockwise_causal_attn_mask(
                    device=device,
                    num_blocks_per_view=num_blocks_per_view,
                    n_views=n_views,
                    num_frame_per_block=self.num_frame_per_block,
                    frame_seqlen=frame_seqlen,
                    z_num=self.z_num,
                    cp_size=cp_size,
                )
                self.block_mask_dict[mask_key] = block_mask
            else:
                block_mask = self.block_mask_dict[mask_key]
        else:
            mask_key = f"mask_f{num_frames}_seqlen{frame_seqlen}_block{mask_num_frame_per_block}_cp{cp_size}"
            if mask_key not in self.block_mask_dict:
                block_mask = self._prepare_blockwise_causal_attn_mask(
                    device=device,
                    num_frames=num_frames // (num_interleave + 1),
                    frame_seqlen=frame_seqlen,
                    num_frame_per_block=mask_num_frame_per_block,
                    num_interleave=num_interleave,
                    cp_size=cp_size,
                )
                self.block_mask_dict[mask_key] = block_mask
            else:
                block_mask = self.block_mask_dict[mask_key]


        if self.concat_padding_mask and padding_mask is not None:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )


        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        video_size = VideoSize(T=x_B_T_H_W_D.shape[1], H=x_B_T_H_W_D.shape[2], W=x_B_T_H_W_D.shape[3])





        if self.enable_view_embedding:
            assert view_indices_B_T is not None, (
                "enable_view_embedding=True requires view_indices_B_T to be passed "
                "into forward (the conditioner must emit it)."
            )
            x_B_T_H_W_D = self._add_view_embedding(x_B_T_H_W_D, view_indices_B_T)


        if action_inputs is not None and self.multi_agent_action_control is not None:
            action_bias = self._get_action_bias(x_B_T_H_W_D, action_inputs, n_views)



        if self.use_multi_agent_rope:
            assert num_interleave == 0, "use_multi_agent_rope is incompatible with num_interleave > 0"
            assert n_views > 1, "use_multi_agent_rope requires n_views > 1"
            t_per_view = video_size.T // n_views
            rope_freq = self._generate_multi_agent_rope(
                t_per_view=t_per_view,
                H=video_size.H,
                W=video_size.W,
                n_views=n_views,
                device=x_B_T_H_W_D.device,
                start_frame_for_rope=0,
            )
        elif num_interleave > 0:
            rope_shape = list(x_B_T_H_W_D.shape)
            rope_shape[1] = rope_shape[1] // (num_interleave + 1)
            rope_freq = self.pos_embedder.generate_embeddings(torch.Size(rope_shape), fps=fps)
            rope_freq = rearrange(rope_freq, "(t h w) 1 1 d -> t h w d", h=rope_shape[2], w=rope_shape[3])
            rope_freq = rope_freq.repeat(num_interleave + 1, 1, 1, 1)
            rope_freq = rearrange(rope_freq, "(n t) h w d -> (t n h w) 1 1 d", n=num_interleave + 1)
        else:
            rope_freq = self.pos_embedder.generate_embeddings(x_B_T_H_W_D.shape, fps=fps)



        z_rope_freq = None
        if self.use_sparse_hub:
            t_per_view = video_size.T // n_views
            z_rope_freq = self._generate_z_rope(
                t_per_view=t_per_view,
                device=x_B_T_H_W_D.device,
                start_frame_for_rope=0,
            )


        extra_pos_emb = None
        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)


        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_emb_B_T_D = self.t_embedding_norm(t_emb_B_T_D)





        if use_joint_block_layout:
            assert video_size.T % (n_views * self.num_frame_per_block) == 0, (
                f"video_size.T={video_size.T} must be divisible by n_views*num_frame_per_block="
                f"{n_views * self.num_frame_per_block} for joint-block layout"
            )
            num_blocks_per_view = video_size.T // (n_views * self.num_frame_per_block)
            npb_inner = self.num_frame_per_block
            x_B_T_H_W_D = rearrange(
                x_B_T_H_W_D,
                "b (v blk npb) h w d -> b (blk v npb) h w d",
                v=n_views,
                blk=num_blocks_per_view,
                npb=npb_inner,
            )
            action_bias = rearrange(
                action_bias,
                "b (v blk npb) h w d -> b (blk v npb) h w d",
                v=n_views,
                blk=num_blocks_per_view,
                npb=npb_inner,
            )
            t_emb_B_T_D_for_blocks = rearrange(
                t_emb_B_T_D,
                "b (v blk npb) d -> b (blk v npb) d",
                v=n_views,
                blk=num_blocks_per_view,
                npb=npb_inner,
            )
            if adaln_lora_B_T_3D is not None:
                adaln_lora_B_T_3D_for_blocks = rearrange(
                    adaln_lora_B_T_3D,
                    "b (v blk npb) d -> b (blk v npb) d",
                    v=n_views,
                    blk=num_blocks_per_view,
                    npb=npb_inner,
                )
            else:
                adaln_lora_B_T_3D_for_blocks = None
            rope_freq = rearrange(
                rope_freq,
                "(v blk npb h w) one1 one2 d -> (blk v npb h w) one1 one2 d",
                v=n_views,
                blk=num_blocks_per_view,
                npb=npb_inner,
                h=video_size.H,
                w=video_size.W,
            )
            if extra_pos_emb is not None:
                extra_pos_emb = rearrange(
                    extra_pos_emb,
                    "b (v blk npb) h w d -> b (blk v npb) h w d",
                    v=n_views,
                    blk=num_blocks_per_view,
                    npb=npb_inner,
                )
        else:
            t_emb_B_T_D_for_blocks = t_emb_B_T_D
            adaln_lora_B_T_3D_for_blocks = adaln_lora_B_T_3D


        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None and self.extra_image_context_dim is not None:
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb


        x_B_L_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
        action_bias_B_L_D = rearrange(action_bias, "b t h w d -> b (t h w) d")

        frame_seqlen = video_size.H * video_size.W
        t_emb_B_L_D = torch.repeat_interleave(t_emb_B_T_D_for_blocks, frame_seqlen, dim=1)

        if adaln_lora_B_T_3D_for_blocks is not None:
            adaln_lora_B_L_3D = torch.repeat_interleave(adaln_lora_B_T_3D_for_blocks, frame_seqlen, dim=1)
        else:
            adaln_lora_B_L_3D = None

        if extra_pos_emb is not None:
            extra_pos_emb = rearrange(extra_pos_emb, "b t h w d -> b (t h w) d")








        if self.use_sparse_hub:
            assert use_joint_block_layout
            B = x_B_L_D.shape[0]
            D = x_B_L_D.shape[-1]
            blk = num_blocks_per_view
            npb = npb_inner
            z_num = self.z_num
            t_per_view = video_size.T // n_views

            player_block_size = n_views * npb * frame_seqlen
            z_block_size = npb * z_num


            assert x_B_L_D.shape[1] == blk * player_block_size, (
                f"sparse_hub: expected x_B_L_D length {blk * player_block_size}, "
                f"got {x_B_L_D.shape[1]}"
            )


            x_players_blk = rearrange(x_B_L_D, "b (blk s) d -> b blk s d", blk=blk)
            t_players_blk = rearrange(t_emb_B_L_D, "b (blk s) d -> b blk s d", blk=blk)
            action_players_blk = rearrange(action_bias_B_L_D, "b (blk s) d -> b blk s d", blk=blk)
            if adaln_lora_B_L_3D is not None:
                adaln_players_blk = rearrange(adaln_lora_B_L_3D, "b (blk s) d -> b blk s d", blk=blk)
            else:
                adaln_players_blk = None
            if extra_pos_emb is not None:
                extra_players_blk = rearrange(extra_pos_emb, "b (blk s) d -> b blk s d", blk=blk)
            else:
                extra_players_blk = None





            z_tokens_full = self.z_tokens.to(dtype=x_B_L_D.dtype)
            z_tokens_full = z_tokens_full[None, None].expand(B, t_per_view, z_num, D)

            z_blk = rearrange(
                z_tokens_full, "b (blk npb) zn d -> b blk (npb zn) d", blk=blk, npb=npb
            )



            t_emb_view0 = t_emb_B_T_D[:, :t_per_view]
            t_emb_z_per_frame = t_emb_view0
            t_z_blk = rearrange(
                t_emb_z_per_frame[:, :, None, :].expand(B, t_per_view, z_num, D),
                "b (blk npb) zn d -> b blk (npb zn) d", blk=blk, npb=npb,
            )

            if adaln_players_blk is not None:
                adaln_view0 = adaln_lora_B_T_3D[:, :t_per_view]
                adaln_z_per_frame = adaln_view0
                adaln_z_blk = rearrange(
                    adaln_z_per_frame[:, :, None, :].expand(
                        B, t_per_view, z_num, adaln_view0.shape[-1]
                    ),
                    "b (blk npb) zn d -> b blk (npb zn) d", blk=blk, npb=npb,
                )
            else:
                adaln_z_blk = None



            action_z_blk = torch.zeros(
                B, blk, z_block_size, D, device=x_B_L_D.device, dtype=x_B_L_D.dtype
            )
            if extra_players_blk is not None:
                extra_z_blk = torch.zeros(
                    B, blk, z_block_size, D, device=x_B_L_D.device, dtype=x_B_L_D.dtype
                )
            else:
                extra_z_blk = None


            x_B_L_D = rearrange(
                torch.cat([x_players_blk, z_blk], dim=2), "b blk s d -> b (blk s) d"
            )
            t_emb_B_L_D = rearrange(
                torch.cat([t_players_blk, t_z_blk], dim=2), "b blk s d -> b (blk s) d"
            )
            action_bias_B_L_D = rearrange(
                torch.cat([action_players_blk, action_z_blk], dim=2),
                "b blk s d -> b (blk s) d",
            )
            if adaln_players_blk is not None:
                adaln_lora_B_L_3D = rearrange(
                    torch.cat([adaln_players_blk, adaln_z_blk], dim=2),
                    "b blk s d -> b (blk s) d",
                )
            if extra_players_blk is not None:
                extra_pos_emb = rearrange(
                    torch.cat([extra_players_blk, extra_z_blk], dim=2),
                    "b blk s d -> b (blk s) d",
                )




            assert z_rope_freq is not None
            rope_players_blk = rearrange(rope_freq, "(blk s) o1 o2 d -> blk s o1 o2 d", blk=blk)
            rope_z_blk = rearrange(
                z_rope_freq, "(blk npb zn) o1 o2 d -> blk (npb zn) o1 o2 d", blk=blk, npb=npb
            )
            rope_freq = rearrange(
                torch.cat([rope_players_blk, rope_z_blk], dim=1),
                "blk s o1 o2 d -> (blk s) o1 o2 d",
            )


        cp_enabled = self._is_context_parallel_enabled and self.cp_group is not None
        if cp_enabled and self.cp_group.size() > 1:
            from gamma_world._src.imaginaire.utils.context_parallel import split_inputs_cp

            if self.use_sparse_hub:
                cp_sz = self.cp_group.size()
                assert x_B_L_D.shape[1] % cp_sz == 0, (
                    f"sparse_hub total seq length {x_B_L_D.shape[1]} must be divisible "
                    f"by cp_size={cp_sz}; check that "
                    f"num_blocks_per_view * (V*npb*HW + npb*z_num) is a multiple of cp_size"
                )

            x_B_L_D = split_inputs_cp(x_B_L_D, seq_dim=1, cp_group=self.cp_group)
            t_emb_B_L_D = split_inputs_cp(t_emb_B_L_D, seq_dim=1, cp_group=self.cp_group)
            rope_freq = split_inputs_cp(rope_freq, seq_dim=0, cp_group=self.cp_group)

            if adaln_lora_B_L_3D is not None:
                adaln_lora_B_L_3D = split_inputs_cp(adaln_lora_B_L_3D, seq_dim=1, cp_group=self.cp_group)

            if extra_pos_emb is not None:
                extra_pos_emb = split_inputs_cp(extra_pos_emb, seq_dim=1, cp_group=self.cp_group)

            if distributed.get_rank() == 0 and DEBUG:
                print(f"CP split shapes (train): x={x_B_L_D.shape}, t_emb={t_emb_B_L_D.shape}, rope={rope_freq.shape}")
                if adaln_lora_B_L_3D is not None:
                    print(f"adaln_lora={adaln_lora_B_L_3D.shape}")
                if extra_pos_emb is not None:
                    print(f"extra_pos_emb={extra_pos_emb.shape}")


        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.on_the_fly_checkpoint:
                x_B_L_D = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x_B_L_D,
                    t_emb_B_L_D,
                    context_input,
                    rope_freq,
                    adaln_lora_B_L_3D,
                    extra_pos_emb,
                    block_mask,
                    None,
                    None,
                    0,
                    0,
                    False,
                    False,
                    video_size,
                    use_reentrant=False,
                    action_bias_B_L_D=action_bias_B_L_D,
                )
            else:
                x_B_L_D = block(
                    x_B_L_D,
                    t_emb_B_L_D,
                    context_input,
                    rope_emb_B_L_D=rope_freq,
                    adaln_lora_B_L_3D=adaln_lora_B_L_3D,
                    extra_per_block_pos_emb=extra_pos_emb,
                    block_mask=block_mask,
                    video_size=video_size,
                    action_bias_B_L_D=action_bias_B_L_D,
                )


        if cp_enabled and self.cp_group is not None:

            x_B_L_D = cat_outputs_cp_with_grad(x_B_L_D, seq_dim=1, cp_group=self.cp_group)




        if self.use_sparse_hub:
            blk = num_blocks_per_view
            npb = npb_inner
            player_block_size = n_views * npb * frame_seqlen
            z_block_size = npb * self.z_num
            x_blk = rearrange(
                x_B_L_D, "b (blk s) d -> b blk s d", blk=blk, s=player_block_size + z_block_size
            )
            x_players = x_blk[:, :, :player_block_size, :]
            x_B_L_D = rearrange(x_players, "b blk s d -> b (blk s) d")


        x_B_T_H_W_D = rearrange(x_B_L_D, "b (t h w) d -> b t h w d", t=video_size.T, h=video_size.H, w=video_size.W)




        if use_joint_block_layout:
            x_B_T_H_W_D = rearrange(
                x_B_T_H_W_D,
                "b (blk v npb) h w d -> b (v blk npb) h w d",
                v=n_views,
                blk=num_blocks_per_view,
                npb=npb_inner,
            )


        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_emb_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)


        t, h, w = video_size
        x_B_C_T_H_W = rearrange(
            x_B_T_H_W_O,
            "b t h w (nt nh nw d) -> b d (t nt) (h nh) (w nw)",
            nt=self.patch_temporal,
            nh=self.patch_spatial,
            nw=self.patch_spatial,
            t=t,
            h=h,
            w=w,
            d=self.out_channels,
        )

        return x_B_C_T_H_W

    def _forward_inference(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int = 0,
        current_end: int = 0,
        start_frame_for_rope: int = 0,
        disable_kv_cache: bool = False,
        img_context_emb: torch.Tensor | None = None,
        view_indices_B_T: torch.Tensor | None = None,
        action_inputs: dict | None = None,
    ) -> torch.Tensor:


        if self.concat_padding_mask and padding_mask is not None:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )


        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        video_size = VideoSize(T=x_B_T_H_W_D.shape[1], H=x_B_T_H_W_D.shape[2], W=x_B_T_H_W_D.shape[3])

        n_views = self._infer_n_views(view_indices_B_T)





        if self.enable_view_embedding:
            assert view_indices_B_T is not None, (
                "enable_view_embedding=True requires view_indices_B_T to be passed "
                "into forward (the conditioner must emit it)."
            )
            x_B_T_H_W_D = self._add_view_embedding(x_B_T_H_W_D, view_indices_B_T)


        action_bias_B_L_D = None
        if action_inputs is not None and self.multi_agent_action_control is not None:
            action_bias = self._get_action_bias(x_B_T_H_W_D, action_inputs, n_views, start_frame=start_frame_for_rope)
            action_bias_B_L_D = rearrange(action_bias, "b t h w d -> b (t h w) d")


        if self.use_multi_agent_rope:
            assert n_views > 1, "use_multi_agent_rope requires n_views > 1"
            t_per_view = video_size.T // n_views
            rope_freq = self._generate_multi_agent_rope(
                t_per_view=t_per_view,
                H=video_size.H,
                W=video_size.W,
                n_views=n_views,
                device=x_B_T_H_W_D.device,
                start_frame_for_rope=start_frame_for_rope,
            )
        else:
            rope_freq = self.pos_embedder.generate_embeddings(
                x_B_T_H_W_D.shape,
                fps=fps,
            )

            if start_frame_for_rope > 0:
                full_shape = list(x_B_T_H_W_D.shape)
                full_shape[1] = start_frame_for_rope + full_shape[1]
                full_rope = self.pos_embedder.generate_embeddings(torch.Size(full_shape), fps=fps)
                start_idx = start_frame_for_rope * video_size.H * video_size.W
                end_idx = start_idx + video_size.T * video_size.H * video_size.W
                rope_freq = full_rope[start_idx:end_idx]


        z_rope_freq = None
        if self.use_sparse_hub:
            t_per_view = video_size.T // n_views
            z_rope_freq = self._generate_z_rope(
                t_per_view=t_per_view,
                device=x_B_T_H_W_D.device,
                start_frame_for_rope=start_frame_for_rope,
            )


        extra_pos_emb = None
        if self.extra_per_block_abs_pos_emb:
            if start_frame_for_rope > 0:
                full_shape = list(x_B_T_H_W_D.shape)
                full_shape[1] = start_frame_for_rope + full_shape[1]
                full_extra_pos_emb = self.extra_pos_embedder.generate_embeddings(torch.Size(full_shape), fps=fps)
                extra_pos_emb = full_extra_pos_emb[:, start_frame_for_rope:, :, :, :]
            else:
                extra_pos_emb = self.extra_pos_embedder.generate_embeddings(x_B_T_H_W_D.shape, fps=fps)

            extra_pos_emb = rearrange(extra_pos_emb, "b t h w d -> b (t h w) d")


        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_emb_B_T_D = self.t_embedding_norm(t_emb_B_T_D)


        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None and self.extra_image_context_dim is not None:
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb


        x_B_L_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")

        frame_seqlen = video_size.H * video_size.W
        t_emb_B_L_D = torch.repeat_interleave(t_emb_B_T_D, frame_seqlen, dim=1)

        if adaln_lora_B_T_3D is not None:
            adaln_lora_B_L_3D = torch.repeat_interleave(adaln_lora_B_T_3D, frame_seqlen, dim=1)
        else:
            adaln_lora_B_L_3D = None







        if self.use_sparse_hub:
            assert n_views >= 2
            B = x_B_L_D.shape[0]
            D = x_B_L_D.shape[-1]
            t_per_view = video_size.T // n_views
            z_num = self.z_num
            z_seg_len = t_per_view * z_num


            z_tokens_full = self.z_tokens.to(dtype=x_B_L_D.dtype)
            z_tokens_full = z_tokens_full[None, None].expand(B, t_per_view, z_num, D)
            z_tokens_flat = rearrange(z_tokens_full, "b t zn d -> b (t zn) d")


            t_emb_view0 = t_emb_B_T_D[:, :t_per_view]
            t_z_flat = rearrange(
                t_emb_view0[:, :, None, :].expand(B, t_per_view, z_num, D),
                "b t zn d -> b (t zn) d",
            )
            if adaln_lora_B_T_3D is not None:
                adaln_view0 = adaln_lora_B_T_3D[:, :t_per_view]
                adaln_z_flat = rearrange(
                    adaln_view0[:, :, None, :].expand(
                        B, t_per_view, z_num, adaln_view0.shape[-1]
                    ),
                    "b t zn d -> b (t zn) d",
                )
                adaln_lora_B_L_3D = torch.cat([adaln_lora_B_L_3D, adaln_z_flat], dim=1)


            if action_bias_B_L_D is not None:
                z_action = torch.zeros(
                    B, z_seg_len, D, device=x_B_L_D.device, dtype=x_B_L_D.dtype
                )
                action_bias_B_L_D = torch.cat([action_bias_B_L_D, z_action], dim=1)
            if extra_pos_emb is not None:
                z_extra = torch.zeros(
                    B, z_seg_len, D, device=x_B_L_D.device, dtype=x_B_L_D.dtype
                )
                extra_pos_emb = torch.cat([extra_pos_emb, z_extra], dim=1)

            x_B_L_D = torch.cat([x_B_L_D, z_tokens_flat], dim=1)
            t_emb_B_L_D = torch.cat([t_emb_B_L_D, t_z_flat], dim=1)
            assert z_rope_freq is not None
            rope_freq = torch.cat([rope_freq, z_rope_freq], dim=0)


        cp_enabled = self._is_context_parallel_enabled and self.cp_group is not None
        if cp_enabled and self.cp_group.size() > 1:
            from gamma_world._src.imaginaire.utils.context_parallel import split_inputs_cp

            assert not self.use_sparse_hub, (
                "sparse_hub inference does not support cp_size > 1 yet: "
                "contiguous CP split would break the [v0, v1, ..., v(V-1), Z] "
                "in-chunk layout assumed by _sparse_hub_inference / its KV cache"
            )

            x_B_L_D = split_inputs_cp(x_B_L_D, seq_dim=1, cp_group=self.cp_group)
            t_emb_B_L_D = split_inputs_cp(t_emb_B_L_D, seq_dim=1, cp_group=self.cp_group)
            rope_freq = split_inputs_cp(rope_freq, seq_dim=0, cp_group=self.cp_group)

            if adaln_lora_B_L_3D is not None:
                adaln_lora_B_L_3D = split_inputs_cp(adaln_lora_B_L_3D, seq_dim=1, cp_group=self.cp_group)

            if extra_pos_emb is not None:
                extra_pos_emb = split_inputs_cp(extra_pos_emb, seq_dim=1, cp_group=self.cp_group)

            if distributed.get_rank() == 0 and DEBUG:
                print(
                    f"CP split shapes (inference): x={x_B_L_D.shape}, t_emb={t_emb_B_L_D.shape}, rope={rope_freq.shape}"
                )
                if adaln_lora_B_L_3D is not None:
                    print(f"adaln_lora={adaln_lora_B_L_3D.shape}")
                if extra_pos_emb is not None:
                    print(f"extra_pos_emb={extra_pos_emb.shape}")


        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block_idx, block in enumerate(self.blocks):
            block_kv_cache = kv_cache[block_idx] if kv_cache is not None and not disable_kv_cache else None
            block_crossattn_cache = crossattn_cache[block_idx] if crossattn_cache is not None else None

            if torch.is_grad_enabled() and self.on_the_fly_checkpoint:
                x_B_L_D = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x_B_L_D,
                    t_emb_B_L_D,
                    context_input,
                    rope_freq,
                    adaln_lora_B_L_3D,
                    extra_pos_emb,
                    None,
                    block_kv_cache,
                    block_crossattn_cache,
                    current_start,
                    current_end,
                    disable_kv_cache,
                    False,
                    video_size,
                    use_reentrant=False,
                    action_bias_B_L_D=action_bias_B_L_D,
                )
            else:
                x_B_L_D = block(
                    x_B_L_D,
                    t_emb_B_L_D,
                    context_input,
                    rope_emb_B_L_D=rope_freq,
                    adaln_lora_B_L_3D=adaln_lora_B_L_3D,
                    block_mask=None,
                    kv_cache=block_kv_cache,
                    crossattn_cache=block_crossattn_cache,
                    current_start=current_start,
                    current_end=current_end,
                    disable_kv_cache=disable_kv_cache,
                    video_size=video_size,
                    action_bias_B_L_D=action_bias_B_L_D,
                )


        if cp_enabled and self.cp_group is not None:

            if torch.is_grad_enabled():
                x_B_L_D = cat_outputs_cp_with_grad(x_B_L_D, seq_dim=1, cp_group=self.cp_group)
            else:
                x_B_L_D = cat_outputs_cp(x_B_L_D, seq_dim=1, cp_group=self.cp_group)



        if self.use_sparse_hub:
            t_per_view = video_size.T // n_views
            player_total = video_size.T * frame_seqlen
            x_B_L_D = x_B_L_D[:, :player_total, :]
            del t_per_view


        x_B_T_H_W_D = rearrange(x_B_L_D, "b (t h w) d -> b t h w d", t=video_size.T, h=video_size.H, w=video_size.W)


        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_emb_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)


        t, h, w = video_size
        x_B_C_T_H_W = rearrange(
            x_B_T_H_W_O,
            "b t h w (nt nh nw d) -> b d (t nt) (h nh) (w nw)",
            nt=self.patch_temporal,
            nh=self.patch_spatial,
            nw=self.patch_spatial,
            t=t,
            h=h,
            w=w,
            d=self.out_channels,
        )

        return x_B_C_T_H_W

    def init_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[dict]:

        kv_caches = []
        for _ in range(self.num_blocks):
            cache = {
                "k": torch.zeros(
                    batch_size,
                    max_seq_len,
                    self.num_heads,
                    self.model_channels // self.num_heads,
                    device=device,
                    dtype=dtype,
                ),
                "v": torch.zeros(
                    batch_size,
                    max_seq_len,
                    self.num_heads,
                    self.model_channels // self.num_heads,
                    device=device,
                    dtype=dtype,
                ),
                "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
                "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
            }
            kv_caches.append(cache)
        return kv_caches

    def reset_kv_cache(self, kv_caches: list[dict]) -> None:

        for cache in kv_caches:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["global_end_index"].zero_()
            cache["local_end_index"].zero_()

    def enable_selective_checkpoint(self, sac_config: SACConfig, blocks: nn.ModuleList) -> None:
        if sac_config.mode == CheckpointMode.NONE:
            return

        log.info(
            f"Enable selective checkpoint with {sac_config.mode}, "
            f"for every {sac_config.every_n_blocks} blocks. Total blocks: {len(blocks)}"
        )
        _context_fn = sac_config.get_context_fn()

        for block_id, block in blocks.named_children():
            if int(block_id) % sac_config.every_n_blocks == 0:
                log.info(f"Enable selective checkpoint for block {block_id}")
                block = ptd_checkpoint_wrapper(
                    block,

                    context_fn=_context_fn,
                    preserve_rng_state=False,
                )
                blocks.register_module(block_id, block)

        self.register_module(
            "final_layer",
            ptd_checkpoint_wrapper(
                self.final_layer,

                context_fn=_context_fn,
                preserve_rng_state=False,
            ),
        )

    def fully_shard(self, mesh, **fsdp_kwargs) -> None:

        for block in self.blocks:
            fully_shard(block, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.x_embedder, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)







    def enable_context_parallel(self, process_group: ProcessGroup | None = None) -> None:

        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )




        self._is_context_parallel_enabled = True
        self.cp_group = process_group

    def disable_context_parallel(self) -> None:

        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )
        self.pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.disable_context_parallel()
        self._is_context_parallel_enabled = False
        self.cp_group = None

    @property
    def is_context_parallel_enabled(self) -> bool:
        return self._is_context_parallel_enabled
