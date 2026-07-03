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

from typing import Dict

from gamma_world._src.imaginaire.model import ImaginaireModel
from gamma_world._src.imaginaire.utils import log
from gamma_world._src.imaginaire.utils.callback import Callback
from gamma_world._src.imaginaire.utils.distributed import rank0_only
from gamma_world._src.imaginaire.utils.easy_io import easy_io


class ModelParamStats(Callback):
    def __init__(
        self,
        save_s3: bool = False,
    ):
        self.save_s3 = save_s3
        self.name = self.__class__.__name__

    @rank0_only
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        try:
            model_stat: Dict = model.model_param_stats()
        except AttributeError:
            raise AttributeError("Model does not have model_param_stats method. Please implement it.")

        log_str = ""
        for k, v in model_stat.items():
            log_str += f"{k}: {v}\n"
        log.info(f"Model param Stats:\n{log_str}")

        if self.save_s3:
            easy_io.dump(model_stat, f"s3://rundir/{self.name}.yaml")
