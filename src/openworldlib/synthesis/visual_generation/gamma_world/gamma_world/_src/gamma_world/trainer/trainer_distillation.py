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

import signal

import torch
import torch.distributed as dist
import torch.utils.data

from gamma_world._src.imaginaire.model import ImaginaireModel
from gamma_world._src.imaginaire.trainer import ImaginaireTrainer as BaseImaginaireTrainer
from gamma_world._src.imaginaire.utils import distributed, log, misc
from gamma_world._src.imaginaire.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling

SHOULD_LOAD_NEW_BATCH_KEY = "should_load_new_batch"


class ImaginaireTrainer(BaseImaginaireTrainer):


    def __init__(self, config):

        super().__init__(config)

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:


        model = model.to("cuda", memory_format=self.config.trainer.memory_format)
        model.on_train_start(self.config.trainer.memory_format)


        self.callbacks.on_optimizer_init_start()
        model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()

        iteration = self.checkpointer.load(model, model.optimizer_dict, model.scheduler_dict, grad_scaler)
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":

            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        log.info("Starting training...")
        self.callbacks.on_train_start(model, iteration=iteration)

        if self.config.trainer.run_validation and iteration == 0:
            self.validate(model, dataloader_val, iteration=iteration)
        with (
            maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler,
            maybe_enable_memory_snapshot(self.config, global_step=iteration) as memory_profiler,
        ):
            dataloader_train_iter = iter(dataloader_train)
            should_load_new_batch = True
            while True:

                if iteration >= self.config.trainer.max_iter:
                    break

                if should_load_new_batch:
                    self.callbacks.on_before_dataloading(iteration)
                    try:
                        with (
                            self.training_timer("dataloader_train"),
                            self.straggler_detector.profile_section(
                                "dataloading",
                                self.config.trainer.straggler_detection.analyze_dataloading,
                                profile_cuda=False,
                            ),
                        ):
                            data_batch, stop_signal = self._fetch_and_broadcast_data(
                                model,
                                dataloader_train_iter,
                                iteration,
                            )
                            if stop_signal:
                                raise StopIteration
                    except StopIteration:
                        break
                    finally:
                        self.callbacks.on_after_dataloading(iteration)


                    data_batch = misc.to(data_batch, device="cuda")


                self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                self.callbacks.on_training_step_batch_start(model, data_batch, iteration=iteration)
                if not model.training:
                    model_ddp.train()
                assert model_ddp.training, "model_ddp is not in training mode."
                assert model.training, "model is not in training mode."
                output_batch, loss, grad_accum_iter = self.training_step(
                    model_ddp,
                    grad_scaler,
                    data_batch,
                    iteration=iteration,
                    grad_accum_iter=grad_accum_iter,
                )
                should_load_new_batch = output_batch.get(SHOULD_LOAD_NEW_BATCH_KEY, True)
                self.callbacks.on_training_step_batch_end(model, data_batch, output_batch, loss, iteration=iteration)

                if grad_accum_iter != 0:
                    continue

                iteration += 1

                if iteration % self.config.checkpoint.save_iter == 0:
                    self.checkpointer.save(
                        model, model.optimizer_dict, model.scheduler_dict, grad_scaler, iteration=iteration
                    )
                self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)

                if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                    self.validate(model, dataloader_val, iteration=iteration)

                signal.alarm(self.config.trainer.timeout_period)
                self.straggler_detector.generate_report(iteration)
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

        log.success("Done with training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, model.optimizer_dict, model.scheduler_dict, grad_scaler, iteration=iteration)
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def training_step(
        self,
        model_ddp: torch.nn.Module | distributed.DistributedDataParallel,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:

        if self.config.trainer.distributed_parallelism == "ddp":
            model = model_ddp.module
        else:
            model = model_ddp

        with distributed.ddp_sync_grad(model_ddp, grad_accum_iter == self.config.trainer.grad_accum_iter - 1):
            self.callbacks.on_before_forward(iteration=iteration)
            with self.training_timer("forward"):
                with self.straggler_detector.profile_section(
                    "fwd", self.config.trainer.straggler_detection.analyze_forward
                ):
                    output_batch, loss = model_ddp.training_step(data, iteration)
            self.callbacks.on_after_forward(iteration=iteration)
            self.callbacks.on_before_backward(model_ddp, loss, iteration=iteration)
            with self.training_timer("backward"):
                with self.straggler_detector.profile_section(
                    "bwd", self.config.trainer.straggler_detection.analyze_backward
                ):
                    loss_scaled = grad_scaler.scale(loss / self.config.trainer.grad_accum_iter)
                    loss_scaled.backward()
                    model.on_after_backward()
            self.callbacks.on_after_backward(model_ddp, iteration=iteration)
        grad_accum_iter += 1
        if grad_accum_iter == self.config.trainer.grad_accum_iter:
            with self.training_timer("optimizer_step"):
                with self.straggler_detector.profile_section(
                    "opt", self.config.trainer.straggler_detection.analyze_optimizer
                ):
                    self.callbacks.on_before_optimizer_step(
                        model_ddp,
                        model.optimizer_dict["net"],
                        model.scheduler_dict["net"],
                        grad_scaler,
                        iteration=iteration,
                    )
                    model.optimizers_schedulers_step(grad_scaler, iteration=iteration)
                    self.callbacks.on_before_zero_grad(
                        model_ddp, model.optimizer_dict["net"], model.scheduler_dict["net"], iteration=iteration
                    )
                    model.on_before_zero_grad(
                        model.optimizer_dict["net"], model.scheduler_dict["net"], iteration=iteration
                    )
                    model.optimizers_zero_grad(iteration=iteration)
            grad_accum_iter = 0
        return output_batch, loss, grad_accum_iter
