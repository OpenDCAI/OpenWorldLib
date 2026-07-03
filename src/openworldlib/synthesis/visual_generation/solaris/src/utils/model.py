import logging

from flax import nnx


def flatten_state(state, path=()):
    """Recursively traverse an NNX VariableState, yielding (path, VariableState)."""
    if isinstance(state, nnx.VariableState):
        # Join path components into a string name (e.g. "Encoder/Layer_0/kernel")
        name = "/".join(str(p) for p in path)
        yield name, state
    elif hasattr(state, "items"):  # state behaves like a dict of submodules/vars
        for key, subtree in state.items():
            yield from flatten_state(subtree, path + (key,))
    elif isinstance(state, (list, tuple)):
        for idx, subtree in enumerate(state):
            yield from flatten_state(subtree, path + (str(idx),))


def log_num_params(tag, state):
    """
    Logs the total number of model parameters (and total size in MB).
    """
    total_params = 0
    total_bytes = 0
    for name, var in flatten_state(state):
        arr = var.value if isinstance(var, nnx.VariableState) else var
        if arr is None:
            continue
        total_params += arr.size
        total_bytes += arr.size * arr.dtype.itemsize

    logging.info(
        "[%s] Total parameters: %s (%.2f MB)",
        tag,
        f"{total_params:,}",
        total_bytes / 1024**2,
    )
