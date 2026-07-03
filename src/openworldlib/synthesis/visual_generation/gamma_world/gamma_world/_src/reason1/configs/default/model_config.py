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

from typing import Optional, Union

import attrs

@attrs.define
class TrainingConfig:

    compile: bool = False
    data_parallel_shard_degree: int = -1
    data_parallel_replicate_degree: int = 1
    tensor_parallel_degree: int = 1
    context_parallel_degree: int = 1

    disable_loss_parallel: bool = False
    mixed_precision_param: str = "bfloat16"
    mixed_precision_reduce: str = "float32"
    enable_cpu_offload: bool = False
    warmup_steps: int = 1000
    steps: int = 400_000
    use_linear_decay: bool = True
    use_cosine_decay: bool = False
    fsdp_reshard_after_forward: str = "default"

@attrs.define
class ExperimentalConfig:

    pipeline_parallel_degree: int = 1
    enable_async_tensor_parallel: bool = False
    enable_compiled_autograd: bool = False

@attrs.define
class OptimizerConfig:

    name: str = "AdamW"
    lr: float = 3e-4
    init_lr: float = 1e-5
    end_lr: float = 2.5e-5
    fused: bool = False
    early_step_in_backward: bool = False
    lr_multiplier_vision_encoder: float = 0.1
    lr_multiplier_mm_projector: float = 1.0
    lr_multiplier_llm: float = 1.0

@attrs.define
class ActivationCheckpointConfig:

    mode: str = "selective"
    models: str = "vlm"
    selective_ac_option: str = "op"

@attrs.define
class Float8Config:

    enable_float8_linear: bool = False

@attrs.define
class CheckpointConfig:

    enable_checkpoint: bool = False
    folder: str = "checkpoint"
    interval_type: str = "steps"
    interval: int = 500
    model_weights_only: bool = False
    export_dtype: str = "float32"
    async_mode: str = "disabled"
    create_seed_checkpoint: bool = False

@attrs.define
class CommConfig:

    init_timeout_seconds: int = 300
    train_timeout_seconds: int = 100
    trace_buf_size: int = 20000

@attrs.define
class VisionEncoderConfig:

    dim: int = 1024
    num_channels: int = 3
    image_size: int = 1024
    patch_size: int = 16
    rope_theta: float = 10000
    hidden_dim: int = 4096
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    image_token_id: Optional[int] = None
    head_dim: Union[int, None] = None
    use_rope_from_torchtitan: bool = False

    multiple_of: Optional[int] = None
    ffn_dim_multiplier: Optional[int] = None
    depth_init: bool = True
    hidden_act: Optional[str] = None
    qkv_bias: Optional[bool] = None
    proj_bias: Optional[bool] = None
    use_cache: bool = (
        False
    )

@attrs.define
class FSDP2ModelConfig:

    tokenizer_type: str

    max_batch_size: int = 1
    max_seq_len: int = 128000

    training_seq_len: int = 4096

    use_fsdp2: bool = True
    use_rope_from_torchtitan: bool = False

    vision_encoder: str = "openai/clip-vit-base-patch32"
    vision_encoder_in_channels: int = 3
    vision_encoder_config: VisionEncoderConfig = VisionEncoderConfig()
    mm_projector: str = None

    ckpt_dir: str = None
    ckpt_path: str = None
    s3_credential_path: str = "credentials/pbss_dir.secret"
    cache_dir: str = None
    precision: str = "bfloat16"

    fsdp_enabled: bool = False
    z_loss_coeff: float = 0.0

    freeze_vision_encoder: bool = False
    freeze_mm_projector: bool = False
    freeze_llm: bool = False

    training: TrainingConfig = TrainingConfig()
    experimental: ExperimentalConfig = ExperimentalConfig()
    activation_checkpoint: ActivationCheckpointConfig = ActivationCheckpointConfig()
    float8: Float8Config = Float8Config()
    checkpoint: CheckpointConfig = CheckpointConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    comm: CommConfig = CommConfig()
    seed: int = 0
    deterministic: bool = False

    num_tiles: int = 1
    add_tile_tag: bool = False
    add_image_start_end_tag: bool = False
    add_answer_tag: bool = True
    tile_tag_type: Union[str, None] = "space_separated"

    use_cache: bool = False

    cp_size: Union[int, None] = None
    ep_size: Union[int, None] = None

    loss_per_token: bool = True

    aux_loss_coeff: float = 0.0
    prepend_padding: bool = False

    def __getitem__(self, item):
        return getattr(self, item)
