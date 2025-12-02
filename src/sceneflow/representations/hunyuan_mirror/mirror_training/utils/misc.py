from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def compose_batches_from_list(batch: List[Dict[str, torch.Tensor]], device: torch.device, validation: bool = False) -> Dict[str, torch.Tensor]:
    batched_inputs = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            if validation:
                assert batch[0][key].shape[0] == 1, 'batch size must be 1 for validation'
            batched_inputs[key] = torch.stack([b[key] for b in batch], dim=1).to(device, non_blocking=True)
        elif isinstance(batch[0][key], np.ndarray):
            if validation:
                assert batch[0][key].shape[0] == 1, 'batch size must be 1 for validation'
            batched_inputs[key] = np.stack([b[key] for b in batch], axis=1)
        elif isinstance(batch[0][key], (int, float, str, bool)):
            batched_inputs[key] = [b[key] for b in batch]
        elif isinstance(batch[0][key], list):
            if validation:
                assert len(batch[0][key]) == 1, 'batch size must be 1 for validation'
                batched_inputs[key] = [item for b in batch for item in b[key]]
            else:
                batched_inputs[key] = [b[key] for b in batch]
        else:
            continue
    return batched_inputs

def convert_defaultdict_to_dict(obj):
    """Recursively convert defaultdict to regular dict"""
    if isinstance(obj, defaultdict):
        # Convert to regular dict and recursively process all values
        obj = {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        # Recursively process all values
        obj = {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
    return obj

def deep_merge_dict(target_dict, source_dict):
    for key, value in source_dict.items():
        if key in target_dict:
            if isinstance(target_dict[key], dict) and isinstance(value, dict):
                deep_merge_dict(target_dict[key], value)
            else:
                target_dict[key] = value
        else:
            target_dict[key] = value