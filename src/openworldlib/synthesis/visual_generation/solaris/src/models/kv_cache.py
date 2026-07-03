from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass
class KVCache:
    k: jax.Array
    v: jax.Array
    length: jax.Array  # store as an array instead of int

    def __init__(self, k: jax.Array, v: jax.Array, length: int | jax.Array = 0):
        self.k = k
        self.v = v
        # make sure length is a JAX scalar
        if isinstance(length, int):
            self.length = jnp.array(length, dtype=jnp.int32)
        else:
            self.length = length

    def tree_flatten(self):
        # length goes in children, not aux_data
        children = (self.k, self.v, self.length)
        aux_data = ()  # no static data needed
        return children, aux_data

    def zeros_like(self):
        return KVCache(
            jnp.zeros_like(self.k), jnp.zeros_like(self.v), jnp.zeros_like(self.length)
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        k, v, length = children
        return cls(k, v, length)

    def update(self, k_blhd: jax.Array, v_blhd: jax.Array):

        kv_cache_size = self.k.shape[1]
        new_k = jax.lax.concatenate([self.k, k_blhd], dimension=1)
        new_v = jax.lax.concatenate([self.v, v_blhd], dimension=1)
        new_k = new_k[:, -kv_cache_size:]
        new_v = new_v[:, -kv_cache_size:]
        new_length = jnp.minimum(self.length + k_blhd.shape[1], kv_cache_size)
        return KVCache(new_k, new_v, new_length)


@jax.tree_util.register_pytree_node_class
@dataclass
class KVCacheDict:
    kv_cache: KVCache
    kv_cache_mouse: KVCache
    kv_cache_keyboard: KVCache

    def zeros_like(self):
        return KVCacheDict(
            kv_cache=self.kv_cache.zeros_like(),
            kv_cache_mouse=self.kv_cache_mouse.zeros_like(),
            kv_cache_keyboard=self.kv_cache_keyboard.zeros_like(),
        )

    def tree_flatten(self):
        # Flatten into a tuple of WanKVCache objects
        children = (self.kv_cache, self.kv_cache_mouse, self.kv_cache_keyboard)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kv_cache, kv_cache_mouse, kv_cache_keyboard = children
        return cls(kv_cache, kv_cache_mouse, kv_cache_keyboard)
