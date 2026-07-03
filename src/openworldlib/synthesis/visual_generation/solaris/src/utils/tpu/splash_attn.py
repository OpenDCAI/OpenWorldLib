from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax.experimental.pallas.ops.tpu.splash_attention import (
    BlockSizes,
    FullMask,
    SegmentIds,
)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import (
    _make_splash_attention,
)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
    MultiHeadMask,
    _ComputableMask,
    make_chunk_attention_mask,
)

chunked_attn = make_chunk_attention_mask((16, 16), 8)

V5P_OPTIMAL_BLOCK_SIZES = BlockSizes(
    block_q=1024,
    block_kv=1024,
    block_kv_compute=512,
    block_q_dkv=1024,
    block_kv_dkv=1024,
    block_kv_dkv_compute=512,
    block_q_dq=1024,
    block_kv_dq=1024,
)


class CausalBlockMask(_ComputableMask):
    """Bidirectional within a block and causal across blocks."""

    chunk_size = None

    def __init__(
        self,
        shape,
        block_size,
        shard_count=1,
        window_size=6,
    ):
        if block_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.block_size = block_size

        def chunked_causal_mask_function(q_ids, kv_ids):
            q_block_id = q_ids // block_size
            kv_block_id = kv_ids // block_size
            sliding_window_causal = (q_block_id >= kv_block_id) & (
                kv_block_id > (q_block_id - window_size)
            )
            return sliding_window_causal

        super().__init__(
            shape=shape,
            mask_function=chunked_causal_mask_function,
            shard_count=shard_count,
        )

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.block_size == other.block_size
            and np.array_equal(self.q_sequence, other.q_sequence)
        )

    def __hash__(self):
        return hash(
            (
                type(self),
                self.shape,
                self.block_size,
                self.q_sequence.tobytes() if self.q_sequence is not None else None,
            )
        )


class TeacherForcingBlockMask(_ComputableMask):
    """Bidirectional within a block and causal across blocks."""

    chunk_size = None

    def __init__(
        self,
        shape,
        block_size,
        seq_len,  # the sequence length of the first
        shard_count=1,
        window_size=6,
    ):
        if block_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.block_size = block_size
        self.seq_len = seq_len

        def chunked_causal_mask_function(q_ids, kv_ids):
            q_block_abs_id = q_ids // block_size
            kv_block_abs_id = kv_ids // block_size
            num_blocks = seq_len // block_size
            q_is_noisy = q_block_abs_id >= num_blocks
            kv_is_noisy = kv_block_abs_id >= num_blocks
            q_block_id = q_block_abs_id % num_blocks
            kv_block_id = kv_block_abs_id % num_blocks

            teacher_forcing_block_causal = (
                (q_is_noisy & kv_is_noisy & (q_block_id == kv_block_id))
                | (q_is_noisy & ~kv_is_noisy & (q_block_id > kv_block_id))
                | (~q_is_noisy & ~kv_is_noisy & (q_block_id >= kv_block_id))
            )

            sliding_window_causal = kv_block_id > (q_block_id - window_size)

            return teacher_forcing_block_causal & sliding_window_causal

        super().__init__(
            shape=shape,
            mask_function=chunked_causal_mask_function,
            shard_count=shard_count,
        )

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.block_size == other.block_size
            and np.array_equal(self.q_sequence, other.q_sequence)
        )

    def __hash__(self):
        return hash(
            (
                type(self),
                self.shape,
                self.block_size,
                self.q_sequence.tobytes() if self.q_sequence is not None else None,
            )
        )


def _pad_to_splash_block_size(x_BTHD, block_sizes=V5P_OPTIMAL_BLOCK_SIZES):
    block_size = (
        block_sizes.block_q
    )  # block_q and block_kv should match for self-attention
    current_T = x_BTHD.shape[1]
    next_T = ((current_T + block_size - 1) // block_size) * block_size
    pad_amount = next_T - current_T
    if pad_amount == 0:
        return x_BTHD
    return jnp.pad(
        x_BTHD,
        ((0, 0), (0, pad_amount), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def full_splash_attn(q_BTHD, k_BTHD, v_BTHD, block_sizes=V5P_OPTIMAL_BLOCK_SIZES):
    Tq, Tk = q_BTHD.shape[1], k_BTHD.shape[1]
    q_BTHD = _pad_to_splash_block_size(q_BTHD, block_sizes)
    k_BTHD = _pad_to_splash_block_size(k_BTHD, block_sizes)
    v_BTHD = _pad_to_splash_block_size(v_BTHD, block_sizes)
    _, padded_T, _, _ = q_BTHD.shape
    mask = FullMask(_shape=(padded_T, padded_T))
    o_BTHD = _wrapped_splash_attn(
        q_BTHD, k_BTHD, v_BTHD, Tq, Tk, mask, block_sizes=block_sizes
    )[:, :Tq, :, :]
    return o_BTHD


def block_causal_splash_attn(
    q_BTHD,
    k_BTHD,
    v_BTHD,
    block_size,  # Required! No default to prevent forgetting num_players
    window_size=6,
    block_sizes=V5P_OPTIMAL_BLOCK_SIZES,
):
    """Block-causal splash attention.

    Args:
        q_BTHD: Query tensor of shape (B, T, H, D)
        k_BTHD: Key tensor of shape (B, T, H, D)
        v_BTHD: Value tensor of shape (B, T, H, D)
        block_size: Size of each causal block. For multiplayer models, this should
            be spatial_size * num_players to ensure all players in a frame are in
            the same block.
        window_size: Sliding window size in blocks (default: 6)
        block_sizes: TPU block sizes for splash attention kernel
    """
    Tq, Tk = q_BTHD.shape[1], k_BTHD.shape[1]
    q_BTHD = _pad_to_splash_block_size(q_BTHD, block_sizes)
    k_BTHD = _pad_to_splash_block_size(k_BTHD, block_sizes)
    v_BTHD = _pad_to_splash_block_size(v_BTHD, block_sizes)
    _, padded_T, _, _ = q_BTHD.shape
    mask = CausalBlockMask(
        shape=(padded_T, padded_T), block_size=block_size, window_size=window_size
    )
    o_BTHD = _wrapped_splash_attn(
        q_BTHD, k_BTHD, v_BTHD, Tq, Tk, mask, block_sizes=block_sizes
    )[:, :Tq, :, :]
    return o_BTHD


def teacher_forcing_block_causal_splash_attn(
    q_BTHD,
    k_BTHD,
    v_BTHD,
    block_size,  # Required! No default to prevent forgetting num_players
    window_size=6,
    block_sizes=V5P_OPTIMAL_BLOCK_SIZES,
):
    """Teacher-forcing block-causal splash attention.

    In teacher forcing, the sequence is split into clean (first half) and noisy
    (second half) tokens. Attention patterns:
    - Clean tokens attend causally to clean tokens (frame-level blocks)
    - Noisy tokens attend to previous clean blocks and same noisy block only
    - Clean tokens do NOT attend to noisy tokens

    Args:
        q_BTHD: Query tensor of shape (B, T, H, D) where T = 2 * clean_seq_len
        k_BTHD: Key tensor of shape (B, T, H, D)
        v_BTHD: Value tensor of shape (B, T, H, D)
        block_size: Size of each causal block. For multiplayer models, this should
            be spatial_size * num_players to ensure all players in a frame are in
            the same block.
        window_size: Sliding window size in blocks (default: 6)
        block_sizes: TPU block sizes for splash attention kernel
    """
    Tq, Tk = q_BTHD.shape[1], k_BTHD.shape[1]
    seq_len = Tq // 2  # clean sequence length
    q_BTHD = _pad_to_splash_block_size(q_BTHD, block_sizes)
    k_BTHD = _pad_to_splash_block_size(k_BTHD, block_sizes)
    v_BTHD = _pad_to_splash_block_size(v_BTHD, block_sizes)
    _, padded_T, _, _ = q_BTHD.shape
    mask = TeacherForcingBlockMask(
        shape=(padded_T, padded_T),
        block_size=block_size,
        seq_len=seq_len,
        window_size=window_size,
    )
    o_BTHD = _wrapped_splash_attn(
        q_BTHD, k_BTHD, v_BTHD, Tq, Tk, mask, block_sizes=block_sizes
    )[:, :Tq, :, :]
    return o_BTHD


# equivalent to jax.nn.dot_product_attention
def _wrapped_splash_attn(
    q_BTHD, k_BTHD, v_BTHD, q_len, k_len, mask=None, block_sizes=V5P_OPTIMAL_BLOCK_SIZES
):
    _, _, H, D = q_BTHD.shape
    multi_head_mask = MultiHeadMask(masks=(mask,) * H)
    splash_attn_kernel = _make_splash_attention(
        multi_head_mask,
        is_mqa=False,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
    )
    q_BTHD = q_BTHD / jnp.sqrt(D)
    q_BHTD = rearrange(q_BTHD, "b t h d -> b h t d")
    k_BHTD = rearrange(k_BTHD, "b t h d -> b h t d")
    v_BHTD = rearrange(v_BTHD, "b t h d -> b h t d")
    Tq, Tk = q_BHTD.shape[2], k_BHTD.shape[2]
    q_segment = jnp.ones(Tq, dtype=jnp.int32)
    k_segment = jnp.ones(Tk, dtype=jnp.int32)
    q_segment = q_segment.at[q_len:].set(0)
    k_segment = k_segment.at[k_len:].set(0)
    segment_ids = SegmentIds(q=q_segment, kv=k_segment)
    o_BHTD = jax.vmap(
        partial(splash_attn_kernel, segment_ids=segment_ids),
        in_axes=(0, 0, 0),
        out_axes=0,
    )(q_BHTD, k_BHTD, v_BHTD)
    return rearrange(o_BHTD, "b h t d -> b t h d")
