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

from gamma_world._src.imaginaire.callbacks.manual_gc import ManualGarbageCollection
from gamma_world._src.imaginaire.lazy_config import PLACEHOLDER
from gamma_world._src.imaginaire.lazy_config import LazyCall as L
from gamma_world._src.imaginaire.utils.callback import LowPrecisionCallback
from gamma_world._src.predict2.callbacks.compile_tokenizer import CompileTokenizer
from gamma_world._src.predict2.callbacks.dataloading_monitor import DetailedDataLoadingSpeedMonitor
from gamma_world._src.predict2.callbacks.device_monitor import DeviceMonitor
from gamma_world._src.predict2.callbacks.grad_clip import GradClip
from gamma_world._src.predict2.callbacks.heart_beat import HeartBeat
from gamma_world._src.predict2.callbacks.iter_speed import IterSpeed
from gamma_world._src.predict2.callbacks.wandb_log import WandbCallback

BASIC_CALLBACKS = dict(
    grad_clip=L(GradClip)(),
    low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
    iter_speed=L(IterSpeed)(
        every_n="${trainer.logging_iter}",
        save_s3="${upload_reproducible_setup}",
        save_s3_every_log_n=10,
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
    manual_gc=L(ManualGarbageCollection)(every_n=5),
    compile_tokenizer=L(CompileTokenizer)(
        enabled=True,
        compile_after_iterations=4,
        dynamic=False,
    ),
)

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

SPEED_CALLBACKS = dict(
    dataloader_speed=L(DetailedDataLoadingSpeedMonitor)(
        every_n="${trainer.logging_iter}",
        save_s3="${upload_reproducible_setup}",
    ),
)
