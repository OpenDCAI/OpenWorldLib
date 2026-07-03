# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from .attention import attention
from .model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory

from utils.debug_option import DEBUG

from typing import List, Dict, Tuple

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 record_interval=3,
                 SMA=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.record_interval = record_interval
        self.SMA = SMA

        # Support list/tuple local_attn_size by converting to list first (handles OmegaConf ListConfig)
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def dynamic_topk_routing_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        chunk_size: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements dynamic sparse attention via top-k chunk routing.
        Args:
            query (torch.Tensor): Query tensor of shape [B, L_q, H, D].
            key (torch.Tensor): Key tensor of shape [B, L_kv, H, D].
            value (torch.Tensor): Value tensor of shape [B, L_kv, H, D].
            chunk_size (int): The size of each chunk to divide the key/value tensors into.
            top_k (int): The number of top relevant chunks to select for each query.

        Returns:
            Returns K and V subsets compatible with a (B, L, H, D) attention function.
        """
        B, L_q, H, D = query.shape
        _, L_kv, _, _ = key.shape

        # 1. Divide Key and Value into chunks
        # Ensure L_kv is divisible by chunk_size, or handle padding
        assert L_kv % chunk_size == 0, f"Key sequence length {L_kv} must be divisible by chunk_size {chunk_size}."
        num_chunks = L_kv // chunk_size
        # 6

        # Reshape K and V to expose the chunk dimension
        # K: [B, num_chunks, chunk_size, H, D]
        key_chunks = key.view(B, num_chunks, chunk_size, H, D)
        value_chunks = value.view(B, num_chunks, chunk_size, H, D)

        # 2. Compute descriptors (phi) for Q and K chunks using mean pooling
        # Query descriptor (phi_q): [B, L_q, H, D] -> mean pool -> [B, 1, H, D]
        # We compute one descriptor for the entire query sequence for simplicity,
        # as described for q_i in the paper, where q_i is a single query token.
        # To make it more granular, one could compute a descriptor per query token.
        # For now, let's use one descriptor for all queries in the sequence.
        phi_q = query.mean(dim=1, keepdim=True).permute(0, 2, 1, 3) # [B, H, 1, D]
        # Key chunk descriptors (phi_k): [B, num_chunks, chunk_size, H, D] -> mean pool -> [B, num_chunks, H, D]
        phi_k_chunks = key_chunks.mean(dim=2).permute(0, 2, 1, 3)  # [B, H, num_chunks, D]
        # 3. Compute relevance scores (inner product)
        # We need to compute scores between each query and each key chunk.
        # Reshape for batch matrix multiplication:
        # phi_q: [B, H, 1, D]
        # phi_k_chunks: [B, H, num_chunks, D]
        relevance_scores = torch.matmul(phi_q, phi_k_chunks.transpose(-2, -1)).squeeze(2) # [B, H, num_chunks]
        # 3. Select top-k chunk indices
        k_selected = min(top_k, num_chunks)
        # Get indices of the top-k chunks for each head and batch item
        _, top_k_indices = torch.topk(relevance_scores, k=k_selected, dim=-1) # [B, H, k_selected]
        topk_indices_sorted, _ = torch.sort(top_k_indices, dim=-1)
        # 4. Gather the selected Key and Value chunks

        # 4.1. Permute K/V chunks to align with indices' shape [B, H, ...] for gathering
        # [B, num_chunks, chunk_size, H, D] -> [B, H, num_chunks, chunk_size, D]
        key_chunks_permuted = key_chunks.permute(0, 3, 1, 2, 4)
        value_chunks_permuted = value_chunks.permute(0, 3, 1, 2, 4)

        # 4.2. Expand indices to match the dimensions of the chunks we want to gather
        # [B, H, k_selected] -> [B, H, k_selected, chunk_size, D]
        expanded_indices = topk_indices_sorted.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, chunk_size, D)

        # 4.3. Gather along the 'num_chunks' dimension (dim=2)
        selected_k = torch.gather(key_chunks_permuted, 2, expanded_indices)
        selected_v = torch.gather(value_chunks_permuted, 2, expanded_indices)
        # Shape of selected_k/v is now [B, H, k_selected, chunk_size, D]

        # 5. Reshape and permute to the final target format

        # 5.1. Reshape to combine k_selected and chunk_size into a new length dimension
        # [B, H, k_selected, chunk_size, D] -> [B, H, (k_selected * chunk_size), D]
        L_selected = k_selected * chunk_size
        final_k_bhld = selected_k.reshape(B, H, L_selected, D)
        final_v_bhld = selected_v.reshape(B, H, L_selected, D)

        # 5.2. Permute from (B, H, L, D) to your desired (B, L, H, D) format
        # [B, H, L_selected, D] -> [B, L_selected, H, D]
        final_k = final_k_bhld.permute(0, 2, 1, 3)
        final_v = final_v_bhld.permute(0, 2, 1, 3)

        return final_k, final_v

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        kv_bank=None,
        update_bank=None,
        is_recache=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            current_end_bank = ((current_end//(1560*3)-1)//self.record_interval+1)*1560
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            kv_bank_size = kv_bank["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            #     print("***********before attention***********")
            #     print(f"kv_cache_size = {kv_cache_size / frame_seqlen}")
            #     print(f"torch.is_grad_enabled() = {torch.is_grad_enabled()}")
            #     print(f"current_end = {current_end / frame_seqlen}")
            #     print(f"current_start = {current_start / frame_seqlen}")
            #     print(f"kv_cache['global_end_index'] = {kv_cache['global_end_index']}")
            #     print(f"kv_cache['local_end_index'] = {kv_cache['local_end_index']}")
            #     print(f"num_new_tokens = {num_new_tokens}")

            # Compute cache update parameters without modifying kv_cache directly
            cache_update_info = None
            is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                #     print(f"need roll")
                #     print(f"num_rolled_tokens: {num_rolled_tokens / frame_seqlen}")
                #     print(f"num_evicted_tokens: {num_evicted_tokens / frame_seqlen}")
                #     print(f"sink_tokens: {sink_tokens / frame_seqlen}")

                # Compute updated local indices
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation (without modifying the original cache)
                # Create temporary k, v for computation
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()

                local_end_index_bank = kv_bank["local_end_index"].item() + current_end_bank - kv_bank["global_end_index"].item()
                local_start_index_bank = local_end_index_bank - num_new_tokens//3

                bank_k = kv_bank["k"].clone()
                bank_v = kv_bank["v"].clone()

                # Apply rolling update to the temporary cache
                temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                # Insert new key/value into the temporary cache
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                write_start_index_bank = local_start_index_bank
                if write_len > 0:
                    temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # Save cache update info for later use
                cache_update_info = {
                    "action": "roll_and_insert",
                    "sink_tokens": sink_tokens,
                    "num_rolled_tokens": num_rolled_tokens,
                    "num_evicted_tokens": num_evicted_tokens,
                    "local_start_index": local_start_index,
                    "local_start_index_bank": local_start_index_bank,
                    "local_end_index": local_end_index,
                    "local_end_index_bank": local_end_index_bank,
                    "write_start_index": write_start_index,
                    "write_start_index_bank": write_start_index_bank,
                    "write_end_index": local_end_index,
                    "write_end_index_bank": local_end_index_bank,
                    "new_k": roped_key[:, roped_offset:roped_offset + write_len],
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "current_end_bank": current_end_bank,
                    "is_recompute": is_recompute
                }

                # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                #     print(f"used kv cache size: local_end_index - local_start_index = {local_end_index - local_start_index}")
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation (without modifying the original cache)
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()

                local_end_index_bank = kv_bank["local_end_index"].item() + current_end_bank - kv_bank["global_end_index"].item()
                local_start_index_bank = local_end_index_bank - num_new_tokens//3

                bank_k = kv_bank["k"].clone()
                bank_v = kv_bank["v"].clone()
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                write_start_index_bank = local_start_index_bank
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # Save cache update info for later use
                cache_update_info = {
                    "action": "direct_insert",
                    "local_start_index": local_start_index,
                    "local_start_index_bank": local_start_index_bank,
                    "local_end_index": local_end_index,
                    "local_end_index_bank": local_end_index_bank,
                    "write_start_index": write_start_index,
                    "write_start_index_bank": write_start_index_bank,
                    "write_end_index": local_end_index,
                    "write_end_index_bank": local_end_index_bank,
                    "new_k": roped_key[:, roped_offset:roped_offset + write_len],
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "current_end_bank": current_end_bank,
                    "is_recompute": is_recompute
                }

            # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            #     print(f"local_start_index: {local_start_index}, local_end_index: {local_end_index}")

            # Use temporary k, v to compute attention
            if sink_tokens > 0:
                # Concatenate sink tokens and local window tokens, keeping total length strictly below max_attention_size
                local_budget = self.max_attention_size - sink_tokens
                k_sink = temp_k[:, :sink_tokens]
                v_sink = temp_v[:, :sink_tokens]
                # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                #     print(f"local_budget: {local_budget}")
                if local_budget > 0:
                    local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                    k_local = temp_k[:, local_start_for_window:local_end_index]
                    v_local = temp_v[:, local_start_for_window:local_end_index]

                    if not is_recache:
                        is_update_bank = (current_start//(3*1560))%self.record_interval==0
                    else:
                        is_update_bank = False
                    if is_update_bank:
                        local_end_index_bank_ = min(local_start_index_bank, kv_bank_size)
                    else:
                        local_end_index_bank_ = min(local_end_index_bank, kv_bank_size)

                    k_bank = bank_k[:, :local_end_index_bank_]
                    v_bank = bank_v[:, :local_end_index_bank_]
                    if not self.SMA:
                        k_cat = torch.cat([k_sink, k_bank, k_local], dim=1)
                        v_cat = torch.cat([v_sink, v_bank, v_local], dim=1)
                    else:
                        k_global = torch.cat([k_sink, k_bank], dim=1)
                        v_global = torch.cat([v_sink, v_bank], dim=1)

                        k_global, v_global = self.dynamic_topk_routing_attention(
                            query=roped_query,
                            key=k_global,
                            value=v_global,
                            chunk_size=1560,
                            top_k=3
                        )
                        k_cat = torch.cat([k_global, k_local], dim=1)
                        v_cat = torch.cat([v_global, v_local], dim=1)

                else:
                    k_cat = k_sink
                    v_cat = v_sink
                x = attention(
                    roped_query,
                    k_cat,
                    v_cat
                )
            else:
                window_start = max(0, local_end_index - self.max_attention_size)
                x = attention(
                    roped_query,
                    temp_k[:, window_start:local_end_index],
                    temp_v[:, window_start:local_end_index]
                )

        # output
        x = x.flatten(2)
        x = self.o(x)

        # del k_sink, v_sink, k_local, v_local, k_cat, v_cat
        # torch.cuda.empty_cache()
        # Return both output and cache update info
        if kv_cache is not None:
            return x, (current_end, local_end_index, cache_update_info)
        else:
            return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 bank_size=3,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 record_interval=3,
                 SMA=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps, record_interval, SMA)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
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
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        kv_bank=None,
        update_bank=None,
        q_bank=None,
        is_recache=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        self_attn_result = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start, kv_bank, update_bank, is_recache)

        if kv_cache is not None:
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)

        if cache_update_info is not None:
            # cache_update_info is already in the format (current_end, local_end_index, cache_update_info)
            return x, cache_update_info
        else:
            return x


class CausalHead(nn.Module):

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
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

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
                 local_attn_size=-1,
                 sink_size=0,
                 bank_size=3,
                 record_interval=3,
                 SMA=False,
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
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
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
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.bank_size = bank_size

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, bank_size, qk_norm, cross_attn_norm, eps, record_interval, SMA)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            pass

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # # debug
        # DEBUG = False
        # if DEBUG:
        #     num_frames = 9
        #     frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            pass

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _apply_cache_updates(self, kv_cache, cache_update_infos, kv_bank, update_bank=True):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                bank = kv_bank[block_index]

                if update_info["action"] == "roll_and_insert":
                    # Apply rolling update
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_start_index_bank = update_info["local_start_index_bank"]
                    local_end_index = update_info["local_end_index"]
                    local_end_index_bank = update_info["local_end_index_bank"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_start_index_bank = update_info.get("write_start_index_bank", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    write_end_index_bank = update_info.get("write_end_index_bank", local_end_index_bank)
                    current_end_bank = update_info["current_end_bank"]
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]

                    # Perform the rolling operation
                    cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                        if update_bank:
                            bank["k_new"][:,:] = new_k[:, :1560]
                            bank["v_new"][:,:] = new_v[:, :1560]

                elif update_info["action"] == "direct_insert":
                    # Direct insert
                    local_start_index = update_info["local_start_index"]
                    local_start_index_bank = update_info["local_start_index_bank"]
                    local_end_index = update_info["local_end_index"]
                    local_end_index_bank = update_info["local_end_index_bank"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_start_index_bank = update_info.get("write_start_index_bank", local_start_index_bank)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    write_end_index_bank = update_info.get("write_end_index_bank", local_end_index_bank)
                    current_end_bank = update_info["current_end_bank"]
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]

                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                        if update_bank:
                            bank["k_new"][:,:] = new_k[:, :1560]
                            bank["v_new"][:,:] = new_v[:, :1560]

            # Update indices: do not roll back pointers during recomputation
            is_recompute = False if update_info is None else update_info.get("is_recompute", False)
            if not is_recompute:
                kv_cache[block_index]["global_end_index"].fill_(current_end)
                kv_cache[block_index]["local_end_index"].fill_(local_end_index)
                if update_bank:
                    kv_bank[block_index]["global_end_index"].fill_(current_end_bank)
                    kv_bank[block_index]["local_end_index"].fill_(local_end_index_bank)


    def _apply_cache_updates_before(self, kv_bank, crossattn_cache):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, block in enumerate(self.blocks):
            bank = kv_bank[block_index]
            crossattn_cache_block = crossattn_cache[block_index]
            write_end_index_bank = kv_bank[block_index]["local_end_index"]
            if write_end_index_bank >= 1560:
                write_start_index_bank = write_end_index_bank - 1560
                new_k = bank["k_new"].clone()
                new_v = bank["v_new"].clone()
                with torch.no_grad():
                    if write_end_index_bank <= bank["k"].shape[1]:
                        bank["k"][:, write_start_index_bank:write_end_index_bank] = new_k
                        bank["v"][:, write_start_index_bank:write_end_index_bank] = new_v
                    else:
                        new_compressed_kv_cache = self.compress_kv_bank(
                            kv_cache=bank,
                            new_k=new_k,
                            new_v=new_v,
                            crossattn_cache=crossattn_cache_block,
                            tokens_per_block=1560,
                            memory_budget_in_blocks=self.bank_size,
                            num_prototypes_in_blocks=1,
                        )
                        bank["k"][:,:] = new_compressed_kv_cache["k"]
                        bank["v"][:,:] = new_compressed_kv_cache["v"]

    def compress_kv_bank(
        self,
        kv_cache: Dict[str, torch.Tensor],
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        crossattn_cache: Dict[str, torch.Tensor],
        tokens_per_block: int = 1560,
        memory_budget_in_blocks: int = 3,
        num_prototypes_in_blocks: int = 1, # Represents the number of new blocks
    ) -> Dict[str, torch.Tensor]:
        """
        Compresses historical KV cache by selecting top-k salient blocks based on full
        attention scores, and then concatenates them with all new blocks.

        Args:
            kv_cache (Dict): Historical KV cache. 'k'/'v' shape: [B, L_hist, H, D].
            new_k, new_v (torch.Tensor): New K/V for generated chunks. Shape: [B, L_new, H, D].
            crossattn_cache (Dict): Text prompt KV cache, 'k' is used as query.
            tokens_per_block (int): Number of tokens per video block.
            memory_budget_in_blocks (int): Total number of blocks to keep in the new cache.
            num_prototypes_in_blocks (int): The number of new blocks being added.

        Returns:
            Dict[str, torch.Tensor]: The new, compressed KV cache dictionary.
        """
        # --- Step 0: Prepare Tensors and Information ---
        hist_k, hist_v = kv_cache['k'].clone(), kv_cache['v'].clone()
        text_q = crossattn_cache["k"].clone()

        B, L_hist, H, D = hist_k.shape
        L_new = new_k.shape[1]

        if L_hist % tokens_per_block != 0 or L_new % tokens_per_block != 0:
            raise ValueError("Cache lengths must be multiples of tokens_per_block.")

        num_hist_blocks = L_hist // tokens_per_block
        num_new_blocks = L_new // tokens_per_block

        # num_prototypes_in_blocks seems to represent num_new_blocks in this logic
        num_hist_to_keep = memory_budget_in_blocks - num_new_blocks

        # --- Step 1: Compute Saliency Scores for Historical Blocks (Full Attention) ---

        # Prepare Q (text) and K (historical visual) for batch matrix multiplication
        q_reshaped = text_q.permute(0, 2, 1, 3).reshape(B * H, -1, D)
        k_reshaped = hist_k.permute(0, 2, 1, 3).reshape(B * H, L_hist, D)

        # Compute attention scores: [B*H, L_text, L_hist]
        attn_scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) * (D ** -0.5)

        # Aggregate scores to get a single saliency value per historical block
        # 1. Average over text tokens: [B*H, L_hist]
        # 2. Reshape and average over heads: [B, L_hist]
        # 3. Reshape and average over tokens within each block: [B, num_hist_blocks]
        importance_scores_per_block = attn_scores.mean(dim=1) \
                                                .view(B, H, L_hist) \
                                                .mean(dim=1) \
                                                .view(B, num_hist_blocks, tokens_per_block) \
                                                .mean(dim=2)

        # --- Step 2: Select Top-K Salient Historical Blocks ---

        k_to_select = min(num_hist_to_keep, num_hist_blocks)
        _, topk_indices = torch.topk(importance_scores_per_block, k=k_to_select, dim=1)

        # Gather the most salient historical blocks
        hist_k_blocks = hist_k.view(B, num_hist_blocks, tokens_per_block, H, D)
        hist_v_blocks = hist_v.view(B, num_hist_blocks, tokens_per_block, H, D)

        expanded_indices = topk_indices.view(B, k_to_select, 1, 1, 1).expand(-1, -1, tokens_per_block, H, D)

        salient_k_blocks = torch.gather(hist_k_blocks, 1, expanded_indices)
        salient_v_blocks = torch.gather(hist_v_blocks, 1, expanded_indices)

        # --- Step 3: Construct the Final Compressed Cache ---

        # Reshape new blocks and concatenate with salient historical blocks
        new_k_blocks = new_k.view(B, num_new_blocks, tokens_per_block, H, D)
        new_v_blocks = new_v.view(B, num_new_blocks, tokens_per_block, H, D)

        # Concatenate in block view, then reshape to final token sequence
        final_k_blocks = torch.cat([salient_k_blocks, new_k_blocks], dim=1)
        final_v_blocks = torch.cat([salient_v_blocks, new_v_blocks], dim=1)

        final_num_tokens = (k_to_select + num_new_blocks) * tokens_per_block
        final_k = final_k_blocks.reshape(B, final_num_tokens, H, D)
        final_v = final_v_blocks.reshape(B, final_num_tokens, H, D)

        return {
            "k": final_k,
            "v": final_v,
        }

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        kv_bank: dict = None,
        update_bank: bool = True,
        q_bank: bool = False,
        update_cache: bool = True,
        is_recache: bool = False,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # print(f"x.device: {x[0].device}, t.device: {t.device}, context.device: {context.device}, seq_len: {seq_len}")

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # print("patch embedding done")
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # print("time embedding done")
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        # print("text embedding done")
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            is_recache=is_recache,
            update_bank=update_bank
        )
        # print("kwargs done")
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward
        if kv_bank is not None and q_bank:
            self._apply_cache_updates_before(kv_bank, crossattn_cache)

        cache_update_info = None
        cache_update_infos = []  # Collect cache update info for all blocks
        for block_index, block in enumerate(self.blocks):
            # print(f"block_index: {block_index}")
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "kv_bank": kv_bank[block_index],
                    }
                )
                # print(f"forward checkpointing")
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "kv_bank": kv_bank[block_index],
                    }
                )
                # print(f"forward no checkpointing")
                result = block(x, **kwargs)
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
        # log_gpu_memory(f"in _forward_inference: {x[0].device}")
        # After all blocks are processed, apply cache updates in a single pass
        if kv_cache is not None and cache_update_infos and update_cache:
            self._apply_cache_updates(kv_cache, cache_update_infos, kv_bank, update_bank=update_bank)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
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
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        pass
        raise NotImplementedError()

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

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
