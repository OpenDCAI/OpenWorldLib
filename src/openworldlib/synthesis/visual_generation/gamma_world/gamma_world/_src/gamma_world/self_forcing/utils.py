# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import copy
from typing import Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.tensor import DTensor, distribute_tensor

from gamma_world._src.imaginaire.lazy_config import LazyDict
from gamma_world._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from gamma_world._src.imaginaire.utils import log, misc
from gamma_world._src.predict2.checkpointer.dcp import ModelWrapper
from gamma_world._src.predict2.utils.dtensor_helper import broadcast_dtensor_model_states
from gamma_world._src.gamma_world.checkpointer.dcp import get_storage_reader
from gamma_world._src.gamma_world.utils.misc import sync_timer


def build_net(
    net_config: LazyDict,
    device_mesh=None,
    mixed_precision_policy_root_module: Optional[torch.distributed.fsdp.MixedPrecisionPolicy] = None,
    mixed_precision_policy_internal_layers: Optional[torch.distributed.fsdp.MixedPrecisionPolicy] = None,
):

    if mixed_precision_policy_root_module is not None:
        root_fsdp_kwargs = {"mp_policy": mixed_precision_policy_root_module}
    else:
        root_fsdp_kwargs = {}

    if mixed_precision_policy_internal_layers is not None:
        internal_fsdp_kwargs = {"mp_policy": mixed_precision_policy_internal_layers}
    else:
        internal_fsdp_kwargs = {}

    init_device = "meta"
    with misc.timer("Creating PyTorch model"):
        with sync_timer("net instantiate"):
            with torch.device(init_device):
                net: torch.nn.Module = lazy_instantiate(net_config)

        if device_mesh:
            net.fully_shard(mesh=device_mesh, **internal_fsdp_kwargs)
            net = fully_shard(net, mesh=device_mesh, reshard_after_forward=True, **root_fsdp_kwargs)

        with misc.timer("meta to cuda and broadcast model states"):
            net.to_empty(device="cuda")

            net.init_weights()

        if device_mesh:

            broadcast_dtensor_model_states(net, device_mesh)
            for name, param in net.named_parameters():
                assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"
    return net


def load_self_forcing_public_ckpt_to_net(
    net,
    checkpoint_path: str,
    message: str = "",
    model_key: str | None = None,
):

    assert checkpoint_path.endswith(".pt"), f"checkpoint_path should end with .pt, got {checkpoint_path}"

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if model_key is not None:
        state_dict = state_dict[model_key]
    elif "generator" in state_dict:
        state_dict = state_dict["generator"]
    elif "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]


    _state_dict = get_model_state_dict(net)
    for key, v in state_dict.items():
        tgt_key = key.replace("model.", "") if key.startswith("model.") else key
        tgt = _state_dict[tgt_key]


        if isinstance(tgt, DTensor) and not isinstance(v, DTensor):

            v = v.to(tgt.device, dtype=tgt.dtype, copy=False)


            if tgt_key == "patch_embedding.weight":
                v = v.reshape_as(tgt)
            v = distribute_tensor(v, tgt.device_mesh, tgt.placements)
            _state_dict[tgt_key] = v

        elif not isinstance(tgt, DTensor) and isinstance(v, DTensor):
            v = v.to_local().to(tgt.device, dtype=tgt.dtype, copy=False)


            if tgt_key == "patch_embedding.weight":
                v = v.reshape_as(tgt)
            _state_dict[tgt_key] = v
        else:
            if tgt_key == "patch_embedding.weight":
                v = v.reshape_as(tgt)
            _state_dict[tgt_key] = v


    log.critical(
        f"{message}: " + str(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=True))),
        rank0_only=False,
    )




def load_consolidated_pt_to_net(
    net,
    checkpoint_path: str,
    message: str = "",
    net_prefix: str = "net.",
):

    assert checkpoint_path.endswith(".pt"), f"checkpoint_path should end with .pt, got {checkpoint_path}"

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)


    if isinstance(state_dict, dict):
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

    _state_dict = get_model_state_dict(net)

    loaded_count = 0
    for key, v in state_dict.items():
        if not key.startswith(net_prefix):

            continue
        tgt_key = key[len(net_prefix) :]
        if tgt_key not in _state_dict:
            continue
        tgt = _state_dict[tgt_key]


        if isinstance(tgt, DTensor) and not isinstance(v, DTensor):
            v = v.to(tgt.device, dtype=tgt.dtype, copy=False)
            v = distribute_tensor(v, tgt.device_mesh, tgt.placements)
        elif not isinstance(tgt, DTensor) and isinstance(v, DTensor):
            v = v.to_local().to(tgt.device, dtype=tgt.dtype, copy=False)
        else:
            v = v.to(tgt.device, dtype=tgt.dtype, copy=False)
        _state_dict[tgt_key] = v
        loaded_count += 1

    if loaded_count == 0:
        raise ValueError(
            f"No keys with prefix {net_prefix!r} found in {checkpoint_path}. "
            f"Top-level keys (sample): {list(state_dict.keys())[:5]}"
        )

    log.critical(
        f"{message} (loaded {loaded_count}/{len(_state_dict)} from {checkpoint_path}): "
        + str(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=True))),
        rank0_only=False,
    )


def load_internal_dcp_checkpoint_to_net(
    net,
    checkpoint_path: str,
    net_prefix: str = "net.",
    patch_embedding_reshape: bool = True,
    allow_partial_load: bool = True,
    credential_path="credentials/s3_inference.secret",
):
    assert checkpoint_path.endswith("/model"), f"checkpoint_path should end with /model, got {checkpoint_path}"

    storage_reader = get_storage_reader(checkpoint_path, credential_path)
    _state_dict = get_model_state_dict(net)
    _new_state_dict = collections.OrderedDict()

    _key_to_check = None
    _value_to_check = None
    loaded_st_to_st_mapping = {}
    for k in _state_dict.keys():

        if _value_to_check is None and "weight" in k and "lora" not in k:
            _key_to_check = k
            _value_to_check = torch.clone(_state_dict[k])


        if "_extra_state" in k:
            continue
        _name_to_load = f"{net_prefix}{k}"
        _new_state_dict[_name_to_load] = _state_dict[k]
        loaded_st_to_st_mapping[_name_to_load] = k

    dcp.load(
        _new_state_dict,
        storage_reader=storage_reader,
        planner=DefaultLoadPlanner(allow_partial_load=allow_partial_load),
    )
    for k in _new_state_dict.keys():
        _state_dict[loaded_st_to_st_mapping[k]] = _new_state_dict[k]
    log.info(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=True)))


    if _key_to_check is not None:
        _current_value = get_model_state_dict(net)[_key_to_check]
        if (_current_value == _value_to_check).all():
            log.warning(
                f"The value of {_key_to_check} remain unchanged, please double check!"
                f"before: {_value_to_check}, after: {_current_value}"
            )
    del _state_dict, _new_state_dict, _current_value, _key_to_check, _value_to_check
