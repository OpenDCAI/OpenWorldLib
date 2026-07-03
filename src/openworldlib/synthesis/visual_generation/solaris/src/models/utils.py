import jax.numpy as jnp
import numpy as np
import torch


def torch_to_jax(x):
    # Preserve original dtype
    return jnp.array(x.cpu().numpy())


def jax_to_torch(x):
    # Preserve original dtype
    return torch.from_numpy(np.asarray(x))
