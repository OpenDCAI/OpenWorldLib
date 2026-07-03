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

import sys

import torch
from hydra.core.config_store import ConfigStore

from gamma_world._src.imaginaire.callbacks.manual_gc import ManualGarbageCollection
from gamma_world._src.imaginaire.config import Config
from gamma_world._src.imaginaire.lazy_config import PLACEHOLDER
from gamma_world._src.imaginaire.lazy_config import LazyCall as L
from gamma_world._src.imaginaire.model import ImaginaireModel
from gamma_world._src.imaginaire.trainer import ImaginaireTrainer as Trainer
from gamma_world._src.imaginaire.utils import log
from gamma_world._src.imaginaire.utils.callback import LowPrecisionCallback as BaseCallback
from gamma_world._src.predict2.callbacks.compile_tokenizer import CompileTokenizer
from gamma_world._src.predict2.callbacks.device_monitor import DeviceMonitor
from gamma_world._src.predict2.callbacks.grad_clip import GradClip
from gamma_world._src.predict2.callbacks.heart_beat import HeartBeat
from gamma_world._src.predict2.callbacks.iter_speed import IterSpeed
from gamma_world._src.predict2.callbacks.validation_draw_sample import ValidationDrawSample
from gamma_world._src.gamma_world.callbacks.model_param_stats import ModelParamStats
from gamma_world._src.gamma_world.callbacks.val_loss_computation import ValLossComputation
from gamma_world._src.gamma_world.callbacks.wandb_log_dmd import WandbCallback

WANDB_CALLBACK = dict(
    wandb=L(WandbCallback)(
        save_s3="${upload_reproducible_setup}",
        logging_iter_multipler=1,
        save_logging_iter_multipler=10,
    ),
    wandb_10x=L(WandbCallback)(
        logging_iter_multipler=10,
        save_logging_iter_multipler=1,
        save_s3="${upload_reproducible_setup}",
    ),
)

class CameraLowPrecisionCallback(BaseCallback):

    def __init__(self, config: Config, trainer: Trainer, update_iter: int):
        self.config = config
        self.trainer = trainer
        self.update_iter = update_iter
        self.skip_tensor_name = [
            "camera",
            "depth",
            "intrinsics",
            "buffer_depths",
            "buffer_w2cs",
            "target_w2cs",
            "buffer_intrinsics",
            "target_intrinsics",
            "buffer_points",
            "buffer_masks",
        ]

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if model.precision == torch.float32:
            log.critical("Using fp32, should disable master weights.")
            self.update_iter = sys.maxsize
        else:
            assert model.precision in [
                torch.bfloat16,
                torch.float16,
                torch.half,
            ], "LowPrecisionCallback must use a low precision dtype."
        self.precision_type = model.precision

    def on_training_step_start(self, model, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data[k]):
                if k not in self.skip_tensor_name:
                    data[k] = v.to(dtype=self.precision_type)

CAM_BASIC_CALLBACKS = dict(
    grad_clip=L(GradClip)(),
    low_prec=L(CameraLowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
    iter_speed=L(IterSpeed)(
        every_n="${trainer.logging_iter}",
        save_s3="${upload_reproducible_setup}",
        save_s3_every_log_n=10,
    ),
    param_count=L(ModelParamStats)(
        save_s3="${upload_reproducible_setup}",
    ),
    heart_beat=L(HeartBeat)(
        every_n=10,
        update_interval_in_minute=20,
        save_s3="${upload_reproducible_setup}",
    ),
    device_monitor=L(DeviceMonitor)(
        every_n="${trainer.logging_iter}",
        save_s3="${upload_reproducible_setup}",
        upload_every_n_mul=10,
    ),
    manual_gc=L(ManualGarbageCollection)(every_n=200),
    compile_tokenizer=L(CompileTokenizer)(
        enabled=True,
        compile_after_iterations=4,
        dynamic=False,
    ),
)

VALIDATION_CALLBACKS = dict(
    validation_draw_sample=L(ValidationDrawSample)(
        num_batches=3,
        guidance=5.0,
        num_steps=20,
        fps=0,
        save_dir="val_videos",
        compute_fvd=False,
        compute_temporal_consistency=False,
        compute_hpsv3=False,
    ),
    val_loss_computation=L(ValLossComputation)(
        enabled=True,
    ),
)

VALIDATION_WITH_METRICS_CALLBACKS = dict(
    validation_draw_sample=L(ValidationDrawSample)(
        num_batches=3,
        guidance=5.0,
        num_steps=20,
        fps=0,
        save_dir="val_videos",
        compute_fvd=True,
        compute_temporal_consistency=True,
        compute_hpsv3=True,
        fvd_feature_extractor="styleganv",
        fvd_s3_credential_path="credentials/s3_sil_videogen.secret",
        temporal_consistency_clip_model="openai/clip-vit-base-patch32",
    ),
    val_loss_computation=L(ValLossComputation)(
        enabled=True,
    ),
)

def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(group="callbacks", package="trainer.callbacks", name="wandb_dmd", node=WANDB_CALLBACK)
    cs.store(group="callbacks", package="trainer.callbacks", name="camera_basic", node=CAM_BASIC_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="validation", node=VALIDATION_CALLBACKS)
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="validation_with_metrics",
        node=VALIDATION_WITH_METRICS_CALLBACKS,
    )
