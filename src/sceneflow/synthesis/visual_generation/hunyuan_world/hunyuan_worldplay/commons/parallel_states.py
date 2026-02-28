import os
from dataclasses import dataclass

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


@dataclass
class ParallelDims:
    sp: int = 1
    world_size: int = -1

    def __post_init__(self):
        if self.world_size == -1:
            if dist.is_initialized():
                self.world_size = dist.get_world_size()
            else:
                self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.build_mesh("cuda")

    def build_mesh(self, device_type):
        assert self.world_size % self.sp == 0, "world_size must be divisible by sp"
        mesh = init_device_mesh(
            device_type,
            [self.world_size // self.sp, self.sp],
            mesh_dim_names=["dp", "sp"],
        )
        self.world_mesh = mesh
        return mesh

    @property
    def sp_enabled(self):
        return self.sp > 1

    @property
    def sp_group(self):
        return self.world_mesh["sp"].get_group()

    @property
    def sp_mesh(self):
        return self.world_mesh["sp"]

    @property
    def sp_rank(self):
        if self.sp_enabled:
            return self.world_mesh["sp"].get_local_rank()
        else:
            return dist.get_rank()

    @property
    def dp_enabled(self):
        return self.sp > 1


__parallel_dims = None


def initialize_parallel_state(
    sp: int = 1,
):
    global __parallel_dims
    __parallel_dims = ParallelDims(sp=sp)
    return __parallel_dims


def get_parallel_state():
    if __parallel_dims is None:
        # create default parallel states (without enabling any parallelism)
        initialize_parallel_state()
    return __parallel_dims
