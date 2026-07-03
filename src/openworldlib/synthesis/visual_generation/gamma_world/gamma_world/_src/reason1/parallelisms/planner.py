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

from typing import Optional

from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata

from gamma_world._src.imaginaire.utils import log
from gamma_world._src.reason1.utils.checkpoint import remap_model_state_dict


class RenameLoadPlanner(DefaultLoadPlanner):
    """
    RenameLoadPlanner that renames variables during checkpoint load.
    """

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        super().set_up_planner(
            state_dict=state_dict,
            metadata=metadata,
            is_coordinator=is_coordinator,
        )
        # Do an early check to see if the checkpoint is valid and print the state dict if not
        # The reason is the original defauly planner's error message is not helpful enough when the keys are mismatched
        missing_keys = []
        for fqn, obj in state_dict.items():
            # ignore state_dict keys which do not exist in `state_dict` if strict=False
            if fqn not in metadata.state_dict_metadata:
                missing_keys.append(fqn)
        if missing_keys:
            log.critical(f"Missing keys in checkpoint: {missing_keys}...")
            log.critical(f"Checkpoint keys: {list(metadata.state_dict_metadata)}...")

        if need_remapping(metadata):
            log.critical("Old checkpoint, requires remapping of tensors")
            self.state_dict = remap_model_state_dict(self.state_dict)


def need_remapping(metadata: Metadata) -> bool:
    # Check if there is substring "mlp.down_projs" in any key of metadata.state_dict_metadata
    # If yes, do a remapping of state_dict keys
    for key in metadata.state_dict_metadata.keys():
        if "mlp.down_projs" in key:
            # Means this is old checkpoint
            return True
    return False
