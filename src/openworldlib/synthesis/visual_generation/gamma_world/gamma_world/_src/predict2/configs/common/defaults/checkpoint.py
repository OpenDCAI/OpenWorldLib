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

from hydra.core.config_store import ConfigStore

from gamma_world._src.imaginaire.config import CheckpointConfig, ObjectStoreConfig

pbss_object_store = ObjectStoreConfig(
    enabled=False,
    credentials="credentials/pbss_checkpoint.secret",
    bucket="checkpoints",
)
s3_object_store = ObjectStoreConfig(
    enabled=False,
    credentials="credentials/s3_checkpoint.secret",
    bucket="bucket",
)
gcp_object_store = ObjectStoreConfig(
    enabled=False,
    credentials="credentials/gcp_checkpoint.secret",
    bucket="bucket",
)

CHECKPOINT_PBSS = CheckpointConfig(
    save_to_object_store=pbss_object_store,
    save_iter=1000,
    load_from_object_store=pbss_object_store,
    load_path="",
    load_training_state=False,
    strict_resume=True,
)

CHECKPOINT_S3 = CheckpointConfig(
    save_to_object_store=s3_object_store,
    save_iter=1000,
    load_from_object_store=s3_object_store,
    load_path="",
    load_training_state=False,
    strict_resume=True,
)

CHECKPOINT_GCP = CheckpointConfig(
    save_to_object_store=gcp_object_store,
    save_iter=1000,
    load_from_object_store=gcp_object_store,
    load_path="",
    load_training_state=False,
    strict_resume=True,
)

def register_checkpoint():
    cs = ConfigStore.instance()
    cs.store(group="checkpoint", package="checkpoint", name="pbss", node=CHECKPOINT_PBSS)
    cs.store(group="checkpoint", package="checkpoint", name="s3", node=CHECKPOINT_S3)
    cs.store(group="checkpoint", package="checkpoint", name="gcp", node=CHECKPOINT_GCP)
