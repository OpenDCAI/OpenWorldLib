import collections.abc

from itertools import repeat

import contextlib
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


def all_gather_sum(running_value, device):
    value = torch.tensor(running_value, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value.item()


class EventsMonitor(object):
    def __init__(self, events_root, rank):
        self.rank = rank
        if rank == 0:
            self.writer = SummaryWriter(log_dir=events_root)
        else:
            self.writer = None

    def write_events(self, events):
        for event in events:
            name, val, count = event
            if self.rank == 0:
                self.writer.add_scalar(name, val, global_step=count)


def profiler_context(enable, exp_dir, worker_name):
    if enable:
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=10,
                wait=5,
                warmup=1,
                active=3,
                repeat=2,
            ),
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                exp_dir, worker_name=worker_name
            ),
        )
    else:
        # return empty python context manager
        return contextlib.nullcontext()


def set_reproducibility(enable, global_seed=None):
    if enable:
        # Configure the seed for reproducibility
        set_manual_seed(global_seed)
    # Set following debug environment variable
    # See the link for details: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Cudnn benchmarking
    torch.backends.cudnn.benchmark = not enable
    # Use deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(enable)

    # LSTM and RNN networks are not deterministic


def set_manual_seed(global_seed):
    # Seed the RNG for Python
    random.seed(global_seed)
    # Seed the RNG for Numpy
    np.random.seed(global_seed)
    # Seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(global_seed)
    # Seed cuda
    torch.cuda.manual_seed_all(global_seed)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def as_tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    if x is None or isinstance(x, (int, float, str)):
        return (x,)
    else:
        raise ValueError(f"Unknown type {type(x)}")


def as_list_of_2tuple(x):
    x = as_tuple(x)
    if len(x) == 1:
        x = (x[0], x[0])
    assert len(x) % 2 == 0, f"Expect even length, got {len(x)}."
    lst = []
    for i in range(0, len(x), 2):
        lst.append((x[i], x[i + 1]))
    return lst
