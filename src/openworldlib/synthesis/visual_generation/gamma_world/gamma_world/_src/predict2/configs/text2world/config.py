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

from typing import Any, List

import attrs

from gamma_world._src.imaginaire import config
from gamma_world._src.imaginaire.trainer import ImaginaireTrainer as Trainer
from gamma_world._src.imaginaire.utils.config_helper import import_all_modules_from_package
from gamma_world._src.predict2.configs.common.defaults.checkpoint import register_checkpoint
from gamma_world._src.predict2.configs.common.defaults.ckpt_type import register_ckpt_type
from gamma_world._src.predict2.configs.common.defaults.dataloader import register_training_and_val_data
from gamma_world._src.predict2.configs.common.defaults.ema import register_ema
from gamma_world._src.predict2.configs.common.defaults.optimizer import register_optimizer
from gamma_world._src.predict2.configs.common.defaults.scheduler import register_scheduler
from gamma_world._src.predict2.configs.common.defaults.tokenizer import register_tokenizer
from gamma_world._src.predict2.configs.text2world.defaults.callbacks import register_callbacks
from gamma_world._src.predict2.configs.text2world.defaults.conditioner import register_conditioner
from gamma_world._src.predict2.configs.text2world.defaults.model import register_model
from gamma_world._src.predict2.configs.text2world.defaults.net import register_net

@attrs.define(slots=False)
class Config(config.Config):

    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock"},
            {"data_val": "mock"},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"model": "ddp"},
            {"callbacks": "basic"},
            {"net": None},
            {"conditioner": "add_fps_padding_mask"},
            {"ema": "power"},
            {"tokenizer": "cosmos_tokenizer_causal_cv8x8x8_c16_res720_t121_it121_v1_0"},
            {"checkpoint": "s3"},
            {"ckpt_type": "dummy"},

            {"experiment": None},
        ]
    )

def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    c.job.project = "cosmos_diffusion_v2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    register_training_and_val_data()
    register_optimizer()
    register_scheduler()
    register_model()
    register_callbacks()
    register_net()
    register_conditioner()
    register_ema()
    register_tokenizer()
    register_checkpoint()
    register_ckpt_type()

    import_all_modules_from_package("gamma_world._src.predict2.configs.text2world.experiment", reload=True)
    return c
