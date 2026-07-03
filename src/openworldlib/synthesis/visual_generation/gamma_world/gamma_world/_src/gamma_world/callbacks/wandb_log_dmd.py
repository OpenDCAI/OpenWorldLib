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

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.utils.data
import wandb

from gamma_world._src.imaginaire.model import ImaginaireModel
from gamma_world._src.imaginaire.utils import distributed, log
from gamma_world._src.imaginaire.utils.easy_io import easy_io
from gamma_world._src.predict2.callbacks.wandb_log import WandbCallback as WandBCallbackBase


@dataclass
class _LossRecord:
    loss: float = 0
    iter_count: int = 0

    def reset(self) -> None:
        self.loss = 0
        self.iter_count = 0

    def get_stat(self) -> Optional[float]:
        if self.iter_count > 0:
            avg_loss = self.loss / self.iter_count
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss.item()
        else:
            avg_loss = None
        self.reset()
        return avg_loss


class WandbCallback(WandBCallbackBase):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.generator_log = _LossRecord()
        self.discriminator_log = _LossRecord()
        self.generator_other_log = defaultdict(_LossRecord)
        self.discriminator_other_log = defaultdict(_LossRecord)

        self.generator_unstable_count = torch.zeros(1, device="cuda")
        self.discriminator_unstable_count = torch.zeros(1, device="cuda")

        self.logging_iter_multipler = logging_iter_multipler
        self.save_logging_iter_multipler = save_logging_iter_multipler
        assert self.logging_iter_multipler > 0, "logging_iter_multipler should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = f"@{logging_iter_multipler}" if logging_iter_multipler > 1 else ""
        self.name = "wandb_loss_log" + self.wandb_extra_tag

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

        assert not model.is_image_batch(data_batch)

        if not skip_update_due_to_unstable_loss:
            if model.is_student_phase(iteration - 1):
                self.generator_log.loss += loss.detach().float()
                self.generator_log.iter_count += 1
                if "raw_dmd_loss" in output_batch:
                    self.generator_other_log["raw_dmd_loss"].loss += output_batch["raw_dmd_loss"].detach().float()
                    self.generator_other_log["raw_dmd_loss"].iter_count += 1
                if "raw_gan_loss" in output_batch:
                    self.generator_other_log["raw_gan_loss"].loss += output_batch["raw_gan_loss"].detach().float()
                    self.generator_other_log["raw_gan_loss"].iter_count += 1
            else:
                self.discriminator_log.loss += loss.detach().float()
                self.discriminator_log.iter_count += 1
                if "raw_gan_loss_d" in output_batch:
                    self.discriminator_other_log["raw_gan_loss_d"].loss += (
                        output_batch["raw_gan_loss_d"].detach().float()
                    )
                    self.discriminator_other_log["raw_gan_loss_d"].iter_count += 1
                if "raw_fake_score_loss" in output_batch:
                    self.discriminator_other_log["raw_fake_score_loss"].loss += (
                        output_batch["raw_fake_score_loss"].detach().float()
                    )
                    self.discriminator_other_log["raw_fake_score_loss"].iter_count += 1
        else:
            if model.is_student_phase(iteration - 1):
                self.generator_unstable_count += 1
            else:
                self.discriminator_unstable_count += 1

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            if self.logging_iter_multipler > 1:
                timer_results = {}
            else:
                timer_results = self.trainer.training_timer.compute_average_results()
            avg_generator_loss = self.generator_log.get_stat()
            avg_discriminator_loss = self.discriminator_log.get_stat()
            generator_other_log = {key: value.get_stat() for key, value in self.generator_other_log.items()}
            discriminator_other_log = {key: value.get_stat() for key, value in self.discriminator_other_log.items()}

            dist.all_reduce(self.generator_unstable_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.discriminator_unstable_count, op=dist.ReduceOp.SUM)

            if distributed.is_rank0():
                info = {f"timer/{key}": value for key, value in timer_results.items()}
                if avg_generator_loss:
                    info.update(
                        {
                            f"train{self.wandb_extra_tag}/generator_loss": avg_generator_loss,
                        }
                    )
                if avg_discriminator_loss:
                    info.update(
                        {
                            f"train{self.wandb_extra_tag}/discriminator_loss": avg_discriminator_loss,
                        }
                    )
                info.update(
                    {
                        f"train{self.wandb_extra_tag}/img_unstable_count": self.img_unstable_count.item(),
                        f"train{self.wandb_extra_tag}/video_unstable_count": self.video_unstable_count.item(),
                        f"train{self.wandb_extra_tag}/generator_unstable_count": self.generator_unstable_count.item(),
                        f"train{self.wandb_extra_tag}/discriminator_unstable_count": self.discriminator_unstable_count.item(),
                        "iteration": iteration,
                        "sample_counter": getattr(self.trainer, "sample_counter", iteration),
                    }
                )
                for key, value in generator_other_log.items():
                    info[f"train{self.wandb_extra_tag}/generator_{key}"] = value
                for key, value in discriminator_other_log.items():
                    info[f"train{self.wandb_extra_tag}/discriminator_{key}"] = value

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


            self.generator_unstable_count.zero_()
            self.discriminator_unstable_count.zero_()

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del optimizer, scheduler, grad_scaler
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0():
            info = {}
            info["sample_counter"] = getattr(self.trainer, "sample_counter", iteration)

            for k, v in model_ddp.optimizer_dict.items():
                for i, param_group in enumerate(v.param_groups):
                    info[f"optim/{k}/lr_{i}"] = param_group["lr"]
                    info[f"optim/{k}/weight_decay_{i}"] = param_group["weight_decay"]
                    if k == "net":
                        info[f"optim/lr_{i}"] = param_group["lr"]
                        info[f"optim/weight_decay_{i}"] = param_group["weight_decay"]

            wandb.log(info, step=iteration)

    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:

        self._val_generator_loss = torch.tensor(0.0, device="cuda")
        self._val_critic_loss = torch.tensor(0.0, device="cuda")
        self._val_sample_count = torch.tensor(0, device="cuda")


        if hasattr(model, "reset_fvd_metric"):
            model.reset_fvd_metric()


        if hasattr(model, "reset_temporal_consistency_metric"):
            model.reset_temporal_consistency_metric()


        if hasattr(model, "reset_hpsv3_metric"):
            model.reset_hpsv3_metric()

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:

        if not (torch.isnan(loss) or torch.isinf(loss)):
            self._val_generator_loss += loss.detach().float()
            self._val_sample_count += 1


            if "val_critic_loss" in output_batch:
                self._val_critic_loss += output_batch["val_critic_loss"].detach().float()

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:


        if hasattr(model.config, "compute_val_loss") and not model.config.compute_val_loss:

            pass
        else:

            if self._val_sample_count > 0:
                dist.all_reduce(self._val_generator_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(self._val_critic_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(self._val_sample_count, op=dist.ReduceOp.SUM)

                avg_val_generator_loss = (self._val_generator_loss / self._val_sample_count).item()
                avg_val_critic_loss = (self._val_critic_loss / self._val_sample_count).item()

                if distributed.is_rank0():
                    info = {
                        f"val{self.wandb_extra_tag}/generator_loss": avg_val_generator_loss,
                        f"val{self.wandb_extra_tag}/critic_loss": avg_val_critic_loss,
                        f"val{self.wandb_extra_tag}/sample_count": self._val_sample_count.item(),
                        "iteration": iteration,
                    }

                    if self.save_s3:
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Val_Iter{iteration:09d}.json",
                        )

                    if wandb:
                        wandb.log(info, step=iteration)

                    log.info(
                        f"Validation @ iter {iteration}: "
                        f"generator_loss={avg_val_generator_loss:.4f}, "
                        f"critic_loss={avg_val_critic_loss:.4f}"
                    )


        if hasattr(model, "compute_fvd") and hasattr(model.config, "compute_fvd") and model.config.compute_fvd:
            fvd_score = model.compute_fvd()
            if fvd_score is not None and distributed.is_rank0():
                fvd_info = {
                    f"val{self.wandb_extra_tag}/fvd": fvd_score,
                    "iteration": iteration,
                }

                if self.save_s3:
                    easy_io.dump(
                        fvd_info,
                        f"s3://rundir/{self.name}/FVD_Iter{iteration:09d}.json",
                    )

                if wandb:
                    wandb.log(fvd_info, step=iteration)

                log.info(f"Validation @ iter {iteration}: FVD={fvd_score:.4f}")


        if (
            hasattr(model, "compute_temporal_consistency")
            and hasattr(model.config, "compute_temporal_consistency")
            and model.config.compute_temporal_consistency
        ):
            tc_scores = model.compute_temporal_consistency()
            if tc_scores is not None and distributed.is_rank0():
                tc_info = {
                    f"val{self.wandb_extra_tag}/temporal_consistency_acm": tc_scores["acm"],
                    f"val{self.wandb_extra_tag}/temporal_consistency_tji": tc_scores["tji"],
                    f"val{self.wandb_extra_tag}/temporal_consistency_tji_score": tc_scores["tji_score"],
                    "iteration": iteration,
                }

                if self.save_s3:
                    easy_io.dump(
                        tc_info,
                        f"s3://rundir/{self.name}/TemporalConsistency_Iter{iteration:09d}.json",
                    )

                if wandb:
                    wandb.log(tc_info, step=iteration)

                log.info(
                    f"Validation @ iter {iteration}: "
                    f"Temporal Consistency ACM={tc_scores['acm']:.4f}, TJI={tc_scores['tji']:.4f}, "
                    f"TJI_score={tc_scores['tji_score']:.4f}"
                )


        if hasattr(model, "compute_hpsv3") and hasattr(model.config, "compute_hpsv3") and model.config.compute_hpsv3:
            hpsv3_scores = model.compute_hpsv3()
            if hpsv3_scores is not None and distributed.is_rank0():
                hpsv3_info = {
                    f"val{self.wandb_extra_tag}/hpsv3": hpsv3_scores["hpsv3"],
                    "iteration": iteration,
                }

                if self.save_s3:
                    easy_io.dump(
                        hpsv3_info,
                        f"s3://rundir/{self.name}/HPSv3_Iter{iteration:09d}.json",
                    )

                if wandb:
                    wandb.log(hpsv3_info, step=iteration)

                log.info(f"Validation @ iter {iteration}: HPSv3={hpsv3_scores['hpsv3']:.4f}")
