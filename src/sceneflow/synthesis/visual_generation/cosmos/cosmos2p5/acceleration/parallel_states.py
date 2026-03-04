from typing import Optional

import torch.distributed as dist

_SEQUENCE_PARALLEL_GROUPS: dict[str, dist.ProcessGroup] = dict()


def initialize_sequence_parallel_group(sp_size: int) -> None:
    """Create and set the sequence-parallel group for this rank.

    Args:
        sp_size: Number of ranks per sequence-parallel group.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % sp_size == 0, 'world_size must be divisible by sequence_parallel_size'
    num_sequence_parallel_groups = world_size // sp_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sp_size, (i + 1) * sp_size)
        if rank in ranks:
            group = dist.new_group(ranks)
            set_sequence_parallel_group(group)
            break


def set_sequence_parallel_group(group: dist.ProcessGroup) -> None:
    """Register the sequence group for this process."""
    _SEQUENCE_PARALLEL_GROUPS['sequence'] = group


def get_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """Return the sequence-parallel group for this rank, if any."""
    return _SEQUENCE_PARALLEL_GROUPS.get('sequence', None)
