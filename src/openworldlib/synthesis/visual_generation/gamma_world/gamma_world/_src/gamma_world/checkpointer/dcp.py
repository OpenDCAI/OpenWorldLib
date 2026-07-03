# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Distributed checkpoint (DCP) directory structure and storage backends.

See gamma_world/_src/predict2/checkpointer/dcp.py for full documentation.
"""

import functools
import os
import warnings
from typing import Any, Dict, List, Optional, Union, cast

import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint._storage_utils import _storage_setup
from torch.distributed.checkpoint.default_planner import LoadPlan
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.storage import StorageReader
from torch.distributed.checkpoint.utils import _DistWrapper, _profile

from gamma_world._src.imaginaire.utils import log, misc
from gamma_world._src.predict2.checkpointer.dcp import (
    AsyncMode,
    DefaultLoadPlanner,
    OptimizerWrapper,
    SaveDone,
    StateDictItemPath,
    Terminate,
    create_default_local_load_plan,
    dcp_load_state_dict,
    save_checkpoint_in_background,
)
from gamma_world._src.predict2.checkpointer.dcp import DistributedCheckpointer as _DistributedCheckpointer
from gamma_world._src.predict2.checkpointer.dcp import ModelWrapper as _ModelWrapper


def load(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_reader: Optional[StorageReader] = None,
    planner: Optional[DefaultLoadPlanner] = None,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
    no_dist: bool = False,
) -> None:
    """
    Load a checkpoint into a distributed state dict in SPMD style.

    Each rank must have the same keys in their ``state_dict`` provided to this
    API. Mismatched keys may result in hangs or errors. If unsure, you can use
    the ``utils._assert_same_keys`` API to check (but may incur communication
    costs).

    Each rank will try to read the least amount of data necessary
    to fullfill the requested `state_dict`. When loading :class:`ShardedTensor`
    or :class:`DTensor` instances, each rank only reads data for their local shards.

    For each ``Stateful`` object (having both a ``state_dict`` and a ``load_state_dict``),
    load will first call ``state_dict`` before attempting deserialization, followed by
    ``load_state_dict`` once the deserialization is complete.
    For each non-``Stateful`` object, load will deserailize the object, and then replace
    it in the ``state_dict`` with the deserialized object.

    .. warning::
        All tensors in ``state_dict`` must be allocated on their
        destination device *prior to* calling this function.

        All non-tensor data is loaded using `torch.load()` and modified in place
        on state_dict.

    .. warning::
        Users must call `load_state_dict` on the root module to ensure load
        pos-processing and non-tensor data properly propagates.

    .. note:
        If no process group is initialized, this function will assume the intent
        is to load a checkpoint into the local process. This can be useful in the
        case of local inference, and when using regular Tensors (as opposed to DTensor
        or ShardedTensor)

    .. note:
        Rank 0 is assumed to be the coordinator rank.

    Args:
        state_dict (Dict[str, Any]): The state_dict to load the checkpoint into.
        checkpoint_id (Union[str, os.PathLike, None]):
            The ID of this checkpoint instance. The meaning of the checkpoint_id
            depends on the storage. It can be a path to a folder or to a file.
            It can also be a key if the storage is a key-value store.
            (Default: ``None``)
        storage_reader (Optional[StorageReader]):
            Instance of StorageWriter used to perform reads. If this is not
            specified, DCP will automatically infer the reader based on the
            checkpoint_id. If checkpoint_id is also None, an exception will
            be raised. (Default: ``None``)
        planner (Optional[LoadPlanner]):
            Instance of LoadPlanner. If this is not specificed, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)
        no_dist (bool): If ``True``, this function will assume the intent is to load
            a checkpoint without using cross-rank synchronization. (Default: ``False``)
    Returns:
        None.

    Examples
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()
        >>> optimizer = Adagrad(my_model.parameters())
        >>> model_state_dict = my_model.state_dict()
        >>> fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(
        ...     "/checkpoint/1"
        ... )

        >>> torch.distributed.checkpoint.load_state_dict(
        >>>     state_dict=model_state_dict,
        >>>     storage_reader=fs_storage_reader,
        >>> )

        >>> # module.load_state_dict() function might have customized steps
        >>> # to flush the state_dict, must call it to
        >>> # ensure correct behavior.
        >>> my_model.load_state_dict(model_state_dict)

    .. note::
        load_state_dict uses collectives to coordinate reads across ranks.
        For NCCL-based process groups, internal tensor representations of
        objects must be moved to the GPU device before communication takes place.
        In this case, the device used is given by ``torch.cuda.current_device()``
        and it is the user's responsibility to ensure that this is set so that each
        rank has an individual GPU, via ``torch.cuda.set_device()``.
    """

    no_dist = no_dist or (not torch.distributed.is_available()) or (not torch.distributed.is_initialized())
    if no_dist:
        warnings.warn(
            "torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process."
        )

    with _profile():
        storage_reader = cast(StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True))

        # All ranks must have the same keys in their `state_dict` provided to
        # this API.  See documentation for more details.
        # Here we simply sort the keys to ensure that all ranks load values in
        # the same order.
        keys = sorted(state_dict.keys())

        statetful_sd = {}
        for key in keys:
            if key not in state_dict:
                continue
            elem = state_dict[key]
            statetful_sd[key] = elem.state_dict() if isinstance(elem, Stateful) else elem

        _load_state_dict(
            state_dict=statetful_sd,
            storage_reader=storage_reader,
            process_group=process_group,
            no_dist=no_dist,
            planner=planner,
        )
        for key in keys:
            if key not in state_dict:
                continue
            elem = state_dict[key]
            if isinstance(elem, Stateful):
                # If the state_dict is a Stateful object,
                # DCP does an in-place load in the original state dict.
                elem.load_state_dict(statetful_sd[key])
            else:
                # Otherwise, replace the state_dict with the loaded state_dict.
                state_dict[key] = statetful_sd[key]


def _load_state_dict(
    state_dict: dict[str, Any],
    storage_reader: StorageReader,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[DefaultLoadPlanner] = None,
) -> None:
    torch._C._log_api_usage_once("torch.distributed.checkpoint.load_state_dict")

    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultLoadPlanner()

    ckpt_kwargs = {}
    if (ckpt_id := getattr(storage_reader, "checkpoint_id", None)) is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id
        ckpt_kwargs["process_group"] = distW.group

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        assert planner is not None
        metadata = storage_reader.read_metadata()
        planner.set_up_planner(state_dict, metadata, distW.is_coordinator)
        storage_reader.set_up_storage_reader(metadata, distW.is_coordinator)

        local_plan = planner.create_local_plan()
        local_plan = storage_reader.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        assert planner is not None
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan: LoadPlan = distW.reduce_scatter("plan", local_step, global_step)
    if distW.is_coordinator:
        # Compare central_plan items with storage_reader.storage_data keys
        dest_fqns = set()
        storage_fqns = set()

        # Extract FQNs from central_plan items
        for item in central_plan.items:
            if hasattr(item, "dest_index") and hasattr(item.dest_index, "fqn"):
                dest_fqns.add(item.dest_index.fqn)
            if hasattr(item, "storage_index") and hasattr(item.storage_index, "fqn"):
                storage_fqns.add(item.storage_index.fqn)

        # Get storage data keys
        storage_data_keys = set()
        if hasattr(storage_reader, "storage_data") and storage_reader.storage_data is not None:
            storage_data_keys = set(item[0].fqn for item in storage_reader.storage_data.items())
        state_dict_keys = set(state_dict.keys())
        # Compare sets and log differences
        # Remove any item that has "_extra_state" as substring in the sets
        state_dict_keys = {fqn for fqn in state_dict_keys if "_extra_state" not in fqn}
        dest_fqns = {fqn for fqn in dest_fqns if "_extra_state" not in fqn}
        storage_fqns = {fqn for fqn in storage_fqns if "_extra_state" not in fqn}
        storage_data_keys = {fqn for fqn in storage_data_keys if "_extra_state" not in fqn}

        log.info("=== Load Plan FQN Analysis ===")
        log.info(f"State Dict FQNs count: {len(state_dict_keys)}")
        log.info(f"Destination FQNs count (without _extra_state): {len(dest_fqns)}")
        log.info(f"Loaded FQNs count (without _extra_state): {len(storage_fqns)}")
        log.info(f"In Storage keys count (without _extra_state): {len(storage_data_keys)}")

        # Find missing keys in each direction
        state_dict_missing_from_dest = state_dict_keys - dest_fqns
        # dest_missing_from_storage = dest_fqns - storage_data_keys
        # storage_missing_from_dest = storage_data_keys - dest_fqns
        # storage_fqns_missing_from_storage_data = storage_fqns - storage_data_keys
        storage_data_missing_from_storage_fqns = storage_data_keys - storage_fqns

        if state_dict_missing_from_dest:
            log.info(
                f"State Dict FQNs missing from load plan ({len(state_dict_missing_from_dest)} items): {sorted(state_dict_missing_from_dest)}"
            )
        else:
            log.info("✓ All State Dict FQNs found in storage_data")

        # if storage_missing_from_dest:
        #     log.info(f"Storage data keys missing from State Dict FQNs ({len(storage_missing_from_dest)} items): {sorted(storage_missing_from_dest)}")
        # else:
        #     log.info("✓ All storage data keys found in State Dict FQNs")

        # if storage_fqns_missing_from_storage_data:
        #     log.info(f"Loaded FQNs missing from storage_data ({len(storage_fqns_missing_from_storage_data)} items): {sorted(storage_fqns_missing_from_storage_data)}")
        # else:
        #     log.info("✓ All storage FQNs found in storage_data")

        if storage_data_missing_from_storage_fqns:
            # If there are more than 100 "net_ema" keys in storage_data_missing_from_storage_fqns, summarize them
            net_ema_keys = {k for k in storage_data_missing_from_storage_fqns if "net_ema" in k}
            if len(net_ema_keys) > 100:
                storage_data_missing_from_storage_fqns = storage_data_missing_from_storage_fqns - net_ema_keys
                storage_data_missing_from_storage_fqns = set(storage_data_missing_from_storage_fqns)  # ensure set type
                storage_data_missing_from_storage_fqns.add("net_ema")
                log.info(f"Summarized {len(net_ema_keys)} 'net_ema' keys as 'net_ema' in missing storage data keys.")
            log.info(
                f"Storage data keys not loaded by load plan ({len(storage_data_missing_from_storage_fqns)} items): {sorted(storage_data_missing_from_storage_fqns)}"
            )
        else:
            log.info("✓ All storage data keys found in Loaded FQNs")

        log.info("=== End Load Plan FQN Analysis ===")

    @_dcp_method_logger(**ckpt_kwargs)
    def read_data():
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_reads = storage_reader.read_data(final_local_plan, planner)

        all_reads.wait()
        return None

    _ = distW.all_gather("read", read_data)


__all__ = [
    "AsyncMode",
    "DefaultLoadPlanner",
    "DistributedCheckpointer",
    "ModelWrapper",
    "OptimizerWrapper",
    "SaveDone",
    "StateDictItemPath",
    "Terminate",
    "create_default_local_load_plan",
    "dcp_load_state_dict",
    "save_checkpoint_in_background",
    "DistributedCheckpointerS3Auto",
    "get_storage_reader",
]


def get_storage_reader(checkpoint_path: str, credential_path: Optional[str] = None):
    """Return an S3StorageReader for s3:// paths or a FileSystemReader otherwise.

    Free-function counterpart of `DistributedCheckpointerS3Auto.get_storage_reader`
    for callers that only need the reader (no checkpointer config).
    """
    from gamma_world._src.predict2.checkpointer.dcp import FileSystemReader, S3StorageReader

    if "s3://" in checkpoint_path:
        return S3StorageReader(credential_path=credential_path, path=checkpoint_path)
    return FileSystemReader(checkpoint_path)


class ModelWrapper(_ModelWrapper):
    """Wrapper for model state dict handling with causal multiview model support."""

    def __init__(self, model: Union[nn.Module, List[nn.Module]], load_ema_to_reg: bool = False):
        self.model = [model] if isinstance(model, nn.Module) else model
        self.load_ema_to_reg = load_ema_to_reg
        if self.load_ema_to_reg:
            supported_model_types = []
            from gamma_world._src.predict2.models.text2world_model import DiffusionModel as predict2_DiffusionModel

            supported_model_types.append(predict2_DiffusionModel)
            from gamma_world._src.predict2.models.text2world_model_rectified_flow import (
                Text2WorldModelRectifiedFlow as predict2_DiffusionModel_rectified_flow,
            )

            supported_model_types.append(predict2_DiffusionModel_rectified_flow)
            from gamma_world._src.predict2.models.text2world_wan2pt1_model import (
                WANDiffusionModel as wan2pt1_DiffusionModel,
            )

            supported_model_types.append(wan2pt1_DiffusionModel)
            from gamma_world._src.gamma_world.models.joint_causal_cosmos_model import (
                CausalJointCosmosModel as causal_cosmos_model,
            )

            supported_model_types.append(causal_cosmos_model)
            assert any(isinstance(model, cls) for cls in supported_model_types), (
                f"ModelWrapper only supports DiffusionModel when load_ema_to_reg is True, but got {type(model)}"
            )

    def state_dict(self, mapping_keys: dict[str, str] = {}) -> Dict[str, Any]:
        _state_dict = {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}
        if self.load_ema_to_reg:
            if not self.model[0].config.ema.enabled:
                all_keys = list(_state_dict.keys())
                assert all(k.startswith("net.") for k in all_keys), "All keys must start with net."
                for k in all_keys:
                    _state_dict[k.replace("net.", "net_ema.")] = _state_dict.pop(k)
            else:
                log.warning("EMA is enabled, will only load EMA weights from checkpoint file.")
                all_keys = list(_state_dict.keys())
                for k in all_keys:
                    if k.startswith("net_ema."):
                        break
                else:
                    raise ValueError("No EMA keys found in state_dict")
                # do not load .net keys, since we do not need them anyway.
                _state_dict = {k: _state_dict[k] for k in all_keys if not k.startswith("net.")}

        if hasattr(self.model[0].config, "lora_config") and self.model[0].config.lora_config.enabled:
            """
            When using LoRA, `inject_adapter_in_model` modifies the target modules in place.
            For example, `blocks[0].attn.q_proj.weight` will be modified to `blocks[0].attn.q_proj.base_layer.weight`.
            This means that the model will have the key `blocks[0].attn.q_proj.base_layer.weight`,
            but the checkpoint will have the key `blocks[0].attn.q_proj.weight`.
            We need to map the model key to the checkpoint key.
            """
            self.checkpoint_to_model_key = {}
            mapping_keys.update(
                {
                    "base_layer.": "",
                    "base_model.model.": "",
                }
            )
            keys_to_update = []
            for k in _state_dict.keys():
                new_key = k
                for from_key, to_key in mapping_keys.items():
                    new_key = new_key.replace(from_key, to_key)
                if new_key != k:
                    keys_to_update.append((k, new_key))
                    self.checkpoint_to_model_key[new_key] = k
            for k, new_key in keys_to_update:
                _state_dict[new_key] = _state_dict.pop(k)

        return _state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if hasattr(self.model[0].config, "lora_config") and self.model[0].config.lora_config.enabled:
            if hasattr(self, "checkpoint_to_model_key"):
                for checkpoint_key, model_key in self.checkpoint_to_model_key.items():
                    state_dict[model_key] = state_dict.pop(checkpoint_key)
            else:
                raise ValueError("checkpoint_to_model_key is not set by `state_dict`")
        if self.load_ema_to_reg:
            if not self.model[0].config.ema.enabled:
                all_keys = list(state_dict.keys())
                assert all(k.startswith("net_ema.") for k in all_keys), "All keys must start with net_ema."
                for k in all_keys:
                    state_dict[k.replace("net_ema.", "net.")] = state_dict.pop(k)
            else:
                log.warning("EMA is enabled, will load EMA weights to regular model weights")
                all_keys = list(state_dict.keys())
                assert all(not k.startswith("net.") for k in all_keys), "No .net keys should be in state_dict"
                for k in all_keys:
                    if k.startswith("net_ema."):
                        state_dict[k.replace("net_ema.", "net.")] = torch.clone(state_dict[k])

        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class DistributedCheckpointer(_DistributedCheckpointer):
    """DistributedCheckpointer for causal multiview model."""

    @misc.timer("checkpoint loading")
    def load(
        self,
        model,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        resume_keys, checkpoint_path = self.keys_to_resume_during_load()
        resume_keys = sorted(resume_keys)
        log.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

        iteration = 0


        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            for key in resume_keys:
                load_planner = DefaultLoadPlanner(allow_partial_load=True)
                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                log.critical(f"Start loading checkpoint from {checkpoint_path}")
                storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                torch.distributed.barrier()
                log.critical(f"starting {cur_key_ckpt_full_path}", rank0_only=False)
                if key == "model":
                    log.info("- Loading the model...")
                    _model_wrapper = ModelWrapper(model)
                    _state_dict = _model_wrapper.state_dict()

                    load(_state_dict, storage_reader=storage_reader, planner=load_planner)
                    _model_wrapper.load_state_dict(_state_dict)
                elif key == "optim":
                    log.info("- Loading the optimizer...")
                    _optim_wrapper = OptimizerWrapper(model, optimizer)
                    _state_dict = _optim_wrapper.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    _optim_wrapper.load_state_dict(_state_dict)
                elif key == "scheduler":
                    log.info("- Loading the scheduler...")
                    _state_dict = scheduler.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    scheduler.load_state_dict(_state_dict)
                elif key == "trainer":
                    log.info("- Loading the trainer...")
                    _state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    iteration = _state_dict["iteration"]
                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")
            if self.callbacks is not None:
                self.callbacks.on_load_checkpoint(model, state_dict=_state_dict)
            log.critical(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        else:
            log.info(
                "Training from scratch (no DCP checkpoint to resume). "
                "If you loaded a consolidated .pt via load_path, the model was already loaded earlier; "
                "only optimizer/scheduler/iteration start from zero."
            )
        torch.cuda.empty_cache()

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)
        return iteration


class DistributedCheckpointerS3Auto(DistributedCheckpointer):
    """Subclass that auto-detects S3 paths in get_storage_reader().

    This allows loading from S3 paths without explicitly setting load_from_object_store=True
    in the config.
    """

    def get_storage_reader(self, checkpoint_path: str):
        """Get storage reader with auto-detection of S3 paths.

        If checkpoint_path starts with 's3://', automatically use S3StorageReader
        even if load_from_object_store is not explicitly configured.
        """
        from gamma_world._src.predict2.checkpointer.dcp import FileSystemReader, S3StorageReader

        # Auto-detect S3 paths even if load_from_object_store isn't explicitly configured
        if self.load_from_object_store or checkpoint_path.startswith("s3://"):
            credential_path = (
                self.config_checkpoint.load_from_object_store.credentials
                if self.load_from_object_store
                else "credentials/s3_training.secret"  # default credential path
            )
            return S3StorageReader(
                credential_path=credential_path,
                path=checkpoint_path,
            )
        return FileSystemReader(os.path.join(checkpoint_path, "model"))
