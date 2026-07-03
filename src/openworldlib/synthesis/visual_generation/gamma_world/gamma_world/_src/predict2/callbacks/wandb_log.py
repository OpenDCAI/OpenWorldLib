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

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.distributed as dist
import torch.utils.data
import wandb

from gamma_world._src.imaginaire.model import ImaginaireModel
from gamma_world._src.imaginaire.utils import distributed, log, misc, wandb_util
from gamma_world._src.imaginaire.utils.callback import Callback
from gamma_world._src.imaginaire.utils.easy_io import easy_io


@dataclass
class _LossRecord:
    loss: float = 0
    iter_count: int = 0
    edm_loss: float = 0

    def reset(self) -> None:
        self.loss = 0
        self.iter_count = 0
        self.edm_loss = 0

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            avg_loss = self.loss / self.iter_count
            avg_edm_loss = self.edm_loss / self.iter_count
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(avg_edm_loss, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss.item()
            avg_edm_loss = avg_edm_loss.item()
        else:
            avg_loss = 0
            avg_edm_loss = 0
        self.reset()
        return avg_loss, avg_edm_loss


class WandbCallback(Callback):
    """
    This callback is used to log the loss, average loss over logging_iter_multipler, and unstable counts of image and video to wandb.
    """

    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.train_image_log = _LossRecord()
        self.train_video_log = _LossRecord()
        self.final_loss_log = _LossRecord()

        self.img_unstable_count = torch.zeros(1, device="cuda")
        self.video_unstable_count = torch.zeros(1, device="cuda")

        self.logging_iter_multipler = logging_iter_multipler
        self.save_logging_iter_multipler = save_logging_iter_multipler
        assert self.logging_iter_multipler > 0, "logging_iter_multipler should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = f"@{logging_iter_multipler}" if logging_iter_multipler > 1 else ""
        self.name = "wandb_loss_log" + self.wandb_extra_tag

    @distributed.rank0_only
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        wandb_util.init_wandb(self.config, model=model)
        config = self.config
        job_local_path = config.job.path_local
        # read optional job_env saved by `log_reproducible_setup`
        if os.path.exists(f"{job_local_path}/job_env.yaml"):
            job_info = easy_io.load(f"{job_local_path}/job_env.yaml")
            if wandb.run:
                wandb.run.config.update({f"JOB_INFO/{k}": v for k, v in job_info.items()}, allow_val_change=True)

        if os.path.exists(f"{config.job.path_local}/config.yaml") and "SLURM_LOG_DIR" in os.environ:
            easy_io.copyfile(
                f"{config.job.path_local}/config.yaml",
                os.path.join(os.environ["SLURM_LOG_DIR"], "config.yaml"),
            )

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:  # Log the curent learning rate.
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0():
            info = {}
            info["sample_counter"] = getattr(self.trainer, "sample_counter", iteration)

            for i, param_group in enumerate(optimizer.param_groups):
                info[f"optim/lr_{i}"] = param_group["lr"]
                info[f"optim/weight_decay_{i}"] = param_group["weight_decay"]

            wandb.log(info, step=iteration)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        skip_update_due_to_unstable_loss = False
        if torch.isnan(loss) or torch.isinf(loss):
            skip_update_due_to_unstable_loss = True
            log.critical(
                f"Unstable loss {loss} at iteration {iteration} with is_image_batch: {model.is_image_batch(data_batch)}",
                rank0_only=False,
            )

        if not skip_update_due_to_unstable_loss:
            if model.is_image_batch(data_batch):
                self.train_image_log.loss += loss.detach().float()
                self.train_image_log.iter_count += 1
                self.train_image_log.edm_loss += output_batch["edm_loss"].detach().float()
            else:
                self.train_video_log.loss += loss.detach().float()
                self.train_video_log.iter_count += 1
                self.train_video_log.edm_loss += output_batch["edm_loss"].detach().float()

            self.final_loss_log.loss += loss.detach().float()
            self.final_loss_log.iter_count += 1
            self.final_loss_log.edm_loss += output_batch["edm_loss"].detach().float()
        else:
            if model.is_image_batch(data_batch):
                self.img_unstable_count += 1
            else:
                self.video_unstable_count += 1

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            if self.logging_iter_multipler > 1:
                timer_results = {}
            else:
                timer_results = self.trainer.training_timer.compute_average_results()
            avg_image_loss, avg_image_edm_loss = self.train_image_log.get_stat()
            avg_video_loss, avg_video_edm_loss = self.train_video_log.get_stat()
            avg_final_loss, avg_final_edm_loss = self.final_loss_log.get_stat()

            dist.all_reduce(self.img_unstable_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.video_unstable_count, op=dist.ReduceOp.SUM)

            if distributed.is_rank0():
                info = {f"timer/{key}": value for key, value in timer_results.items()}
                info.update(
                    {
                        f"train{self.wandb_extra_tag}/image_loss": avg_image_loss,
                        f"train{self.wandb_extra_tag}/image_edm_loss": avg_image_edm_loss,
                        f"train{self.wandb_extra_tag}/video_loss": avg_video_loss,
                        f"train{self.wandb_extra_tag}/video_edm_loss": avg_video_edm_loss,
                        f"train{self.wandb_extra_tag}/loss": avg_final_loss,
                        f"train{self.wandb_extra_tag}/edm_loss": avg_final_edm_loss,
                        f"train{self.wandb_extra_tag}/img_unstable_count": self.img_unstable_count.item(),
                        f"train{self.wandb_extra_tag}/video_unstable_count": self.video_unstable_count.item(),
                        "iteration": iteration,
                        "sample_counter": getattr(self.trainer, "sample_counter", iteration),
                    }
                )
                if self.save_s3:
                    if (
                        iteration
                        % (
                            self.config.trainer.logging_iter
                            * self.logging_iter_multipler
                            * self.save_logging_iter_multipler
                        )
                        == 0
                    ):
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Train_Iter{iteration:09d}.json",
                        )

                if wandb:
                    wandb.log(info, step=iteration)
            if self.logging_iter_multipler == 1:
                self.trainer.training_timer.reset()

            # reset unstable count
            self.img_unstable_count.zero_()
            self.video_unstable_count.zero_()

    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        # Cache for collecting data/output batches.
        self._val_cache: dict[str, Any] = dict(
            data_batches=[],
            output_batches=[],
            loss=torch.tensor(0.0, device="cuda"),
            sample_size=torch.tensor(0, device="cuda"),
        )

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Collect the validation batch and aggregate the overall loss.
        # Collect the validation batch and aggregate the overall loss.
        batch_size = misc.get_data_batch_size(data_batch)
        self._val_cache["loss"] += loss * batch_size
        self._val_cache["sample_size"] += batch_size

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        # Compute the average validation loss across all devices.
        dist.all_reduce(self._val_cache["loss"], op=dist.ReduceOp.SUM)
        dist.all_reduce(self._val_cache["sample_size"], op=dist.ReduceOp.SUM)
        loss = self._val_cache["loss"].item() / self._val_cache["sample_size"]
        # Log data/stats of validation set to W&B.
        if distributed.is_rank0():
            log.info(f"Validation loss (iteration {iteration}): {loss}")
            wandb.log({"val/loss": loss}, step=iteration)

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        wandb.finish()
