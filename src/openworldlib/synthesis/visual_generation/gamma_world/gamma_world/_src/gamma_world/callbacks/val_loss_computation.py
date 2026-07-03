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



import torch
import torch.distributed as dist
import wandb
from einops import rearrange

from gamma_world._src.imaginaire.model import ImaginaireModel
from gamma_world._src.imaginaire.utils import distributed, log
from gamma_world._src.imaginaire.utils.callback import Callback


class ValLossComputation(Callback):


    def __init__(self, enabled: bool = True):
        self.enabled = enabled


        self._val_loss_sum = torch.tensor(0.0)
        self._val_sample_count = torch.tensor(0)

    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:

        self._val_loss_sum = torch.tensor(0.0, device="cuda")
        self._val_sample_count = torch.tensor(0, device="cuda")

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:

        if not self.enabled:
            return

        try:
            val_loss = self._compute_validation_loss(model, data_batch)
            if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                self._val_loss_sum += val_loss.detach()
                self._val_sample_count += 1
        except Exception as e:
            log.warning(f"Failed to compute validation loss: {e}")

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:

        if not self.enabled:
            return

        if self._val_sample_count > 0:

            dist.all_reduce(self._val_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._val_sample_count, op=dist.ReduceOp.SUM)

            avg_val_loss = (self._val_loss_sum / self._val_sample_count).item()

            if distributed.is_rank0():
                log.info(f"Validation loss at iteration {iteration}: {avg_val_loss:.6f}")

                if wandb.run is not None:
                    wandb.log({"val/flow_loss": avg_val_loss}, step=iteration)

    @torch.no_grad()
    def _compute_validation_loss(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:


        _, x0_B_C_T_H_W, condition = model.get_data_and_condition(data_batch)


        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **model.flow_matching_kwargs)
        batch_size = x0_B_C_T_H_W.size()[0]
        num_frames = x0_B_C_T_H_W.size()[2]

        noise_scheme = getattr(model, "noise_scheme", "consistent_noise")
        num_frame_per_block = getattr(model, "num_frame_per_block", 1)

        if noise_scheme == "consistent_noise":
            return self._compute_consistent_noise_loss(model, x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, batch_size)
        elif noise_scheme in ("diffusion_forcing", "teacher_forcing"):
            return self._compute_diffusion_forcing_loss(
                model, x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, batch_size, num_frames, num_frame_per_block
            )
        else:
            raise NotImplementedError(f"Validation not implemented for noise_scheme: {noise_scheme}")

    def _compute_consistent_noise_loss(
        self,
        model: ImaginaireModel,
        x0_B_C_T_H_W: torch.Tensor,
        condition: object,
        epsilon_B_C_T_H_W: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:

        t_B = model.rectified_flow.sample_train_time(batch_size).to(**model.flow_matching_kwargs)
        t_B_1 = rearrange(t_B, "b -> b 1")
        x0_loss, condition_loss, epsilon_loss, t_B_1 = model.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B_1
        )
        timesteps = model.rectified_flow.get_discrete_timestamp(t_B_1, model.flow_matching_kwargs)
        sigmas = model.rectified_flow.get_sigmas(timesteps, model.flow_matching_kwargs)
        xt_B_C_T_H_W, vt_B_C_T_H_W = model.rectified_flow.get_interpolation(epsilon_loss, x0_loss, sigmas)

        vt_pred_B_C_T_H_W = model.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**model.tensor_kwargs),
            timesteps_B_T=timesteps.to(**model.tensor_kwargs),
            **condition_loss.to_dict(),
        )

        time_weights_B = model.rectified_flow.train_time_weight(timesteps, model.flow_matching_kwargs)
        per_instance_loss = torch.mean(
            (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=list(range(1, vt_pred_B_C_T_H_W.dim()))
        )
        return torch.mean(time_weights_B * per_instance_loss)

    def _compute_diffusion_forcing_loss(
        self,
        model: ImaginaireModel,
        x0_B_C_T_H_W: torch.Tensor,
        condition: object,
        epsilon_B_C_T_H_W: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_frame_per_block: int,
    ) -> torch.Tensor:


        t_B_T = (
            model.rectified_flow.sample_train_time(batch_size * num_frames)
            .to(**model.flow_matching_kwargs)
            .reshape(batch_size, num_frames)
        )
        t_B_T = t_B_T.reshape(t_B_T.shape[0], -1, num_frame_per_block)
        t_B_T[:, :, 1:] = t_B_T[:, :, 0:1]
        t_B_T = t_B_T.reshape(t_B_T.shape[0], -1)

        x0_loss, condition_loss, epsilon_loss, t_B_T = model.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B_T
        )

        split_cp_in_model = getattr(model.config, "split_cp_in_model", False)
        if not split_cp_in_model:
            assert x0_loss.shape[2] == num_frames
            num_frames_loss = num_frames
        else:
            num_frames_loss = x0_loss.shape[2]

        timesteps_B_T = model.rectified_flow.get_discrete_timestamp(t_B_T, model.flow_matching_kwargs)
        sigmas_B_T = model.rectified_flow.get_sigmas(
            timesteps_B_T.reshape(-1, 1),
            model.flow_matching_kwargs,
        ).reshape(batch_size, num_frames_loss)
        xt_B_C_T_H_W, vt_B_C_T_H_W = model.rectified_flow.get_interpolation_multiple_timesteps(
            epsilon_loss, x0_loss, sigmas_B_T
        )

        vt_pred_B_C_T_H_W = model.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**model.tensor_kwargs),
            timesteps_B_T=timesteps_B_T.to(**model.tensor_kwargs),
            **condition_loss.to_dict(),
        )

        time_weights_B_T = model.rectified_flow.train_time_weight(
            timesteps_B_T.reshape(-1, 1), model.flow_matching_kwargs
        ).reshape(batch_size, num_frames_loss)
        per_instance_loss = torch.mean((vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=[1, 3, 4])
        return torch.mean(time_weights_B_T * per_instance_loss)
