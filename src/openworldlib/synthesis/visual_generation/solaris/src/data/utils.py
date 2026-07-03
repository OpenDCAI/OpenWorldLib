import hashlib

import cv2
import jax
import torch

"""
Source: https://github.com/willisma/jax_measure_transport/blob/main/data/utils.py
"""


def torch_pytree_to_numpy(xs):
    def _prepare(x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return x

    return jax.tree.map(_prepare, xs)


def anything_to_seed(*args):
    serialized_args = []
    for arg in args:
        if isinstance(arg, int):
            type_code = "int"
            value_repr = str(arg)
        elif isinstance(arg, float):
            type_code = "float"
            value_repr = repr(arg)
        elif isinstance(arg, bool):
            type_code = "bool"
            value_repr = str(arg)
        elif isinstance(arg, str):
            type_code = "str"
            value_repr = repr(arg)
        else:
            raise TypeError(f"Unsupported type: {type(arg).__name__}")
        serialized_arg = f"{type_code}:{value_repr}"
        serialized_args.append(serialized_arg)

    serialized_str = "|".join(serialized_args)
    serialized_bytes = serialized_str.encode("utf-8")
    hash_bytes = hashlib.sha256(serialized_bytes).digest()
    seed_int = int.from_bytes(hash_bytes, "big")
    return seed_int % (1 << 64)


def resize_letterbox(image, desired_height, desired_width):
    h, w = image.shape[:2]
    if h == desired_height and w == desired_width:
        return image

    resized = cv2.resize(
        image, (desired_width, desired_height), interpolation=cv2.INTER_AREA
    )
    letterboxed = resized
    return letterboxed
