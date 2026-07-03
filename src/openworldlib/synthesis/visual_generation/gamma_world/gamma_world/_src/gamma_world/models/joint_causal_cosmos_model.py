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

import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import attrs
import torch
import torch.distributed as dist
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed import get_process_group_ranks
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from tqdm import tqdm

from gamma_world._src.imaginaire.utils import log, misc
from gamma_world._src.imaginaire.utils.context_parallel import (
    broadcast,
    broadcast_split_tensor,
)
from gamma_world._src.predict2.conditioner import DataType
from gamma_world._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from gamma_world._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldCondition,
    Text2WorldModelRectifiedFlow,
    Text2WorldModelRectifiedFlowConfig,
)
from gamma_world._src.predict2.schedulers.rectified_flow import RectifiedFlow
from gamma_world._src.predict2.utils.dtensor_helper import DTensorFastEmaModelUpdater, broadcast_dtensor_model_states
from gamma_world._src.predict2_multiview.models.multiview_vid2vid_model_rectified_flow import preprocess_databatch
from gamma_world._src.gamma_world.utils.misc import sync_timer

NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


@attrs.define(slots=False)
class I4LoraConfig:
    enabled: bool = False
    pretrained_lora_path: str = ""
    lora_rank: int = -1
    adapter_name: str = "default"
    lora_target_modules: list[str] = []
    init_lora_weights: str = "kaiming"


class CausalRectifiedFlow(RectifiedFlow):


    def get_discrete_timestamp(self, u, tensor_kwargs):

        original_shape = u.shape
        timesteps = self.shift * u / (1 + (self.shift - 1) * u)
        timesteps = timesteps * self.num_train_timesteps
        return timesteps.reshape(original_shape)

    def get_sigmas(self, timesteps, tensor_kwargs):

        sigmas = (timesteps.to(**tensor_kwargs)) / self.num_train_timesteps
        return sigmas

    def get_interpolation_multiple_timesteps(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        sigmas: torch.Tensor | None,
        t: torch.Tensor | None = None,
    ):

        if sigmas is None:
            raise NotImplementedError("sigmas must be provided.")
        else:
            assert t is None, "t must be None when sigmas is provided."
            assert sigmas.ndim == 2, "sigmas must be a tensor with shape `(B, D2)`."
            sigmas = sigmas.to(device=self.device, dtype=self.dtype)

        assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape."
        assert x_0.shape[0] == x_1.shape[0], "Batch size of x_0 and x_1 must match."
        assert sigmas.shape[0] == x_1.shape[0], "Batch size of sigmas must match x_1."

        sigmas = sigmas.view(sigmas.shape[0], 1, sigmas.shape[1], *([1] * (x_1.ndim - 3)))
        x_t = x_0 * sigmas + x_1 * (1 - sigmas)
        dot_x_t = x_0 - x_1
        return x_t, dot_x_t


@attrs.define(slots=False)
class CausalJointCosmosModelConfig(Text2WorldModelRectifiedFlowConfig):

    num_frame_per_block: int = 1
    noise_scheme: str = "diffusion_forcing"
    history_noise: float = 0
    force_teacher_t0: bool = False
    model_type: str = "t2v"
    i2v_zero_latent_condition: bool = False
    max_latent_frames_per_gpu: int = 21
    i2v_use_original_condition: bool = False
    split_cp_in_model: bool = True

    lora_config: I4LoraConfig = I4LoraConfig()

    min_num_conditional_frames: int = 0
    max_num_conditional_frames: int = 2
    conditional_frame_timestep: float = (
        -1.0
    )
    conditioning_strategy: str = "frame_replace"
    denoise_replace_gt_frames: bool = False
    conditional_frames_probs: Optional[Dict[int, float]] = None

    use_action_control: bool = True


class CausalJointCosmosModel(Text2WorldModelRectifiedFlow):
    def __init__(self, config: CausalJointCosmosModelConfig):

        config.net.num_layers = config.net.num_blocks
        super().__init__(config)


        self.flow_matching_kwargs = self.tensor_kwargs_fp32


        self.rectified_flow = CausalRectifiedFlow(
            velocity_field=self.net,
            train_time_distribution=config.train_time_distribution,
            use_dynamic_shift=config.use_dynamic_shift,
            shift=config.shift,
            train_time_weight_method=config.train_time_weight,
            device=torch.device("cuda"),
            dtype=self.tensor_kwargs_fp32["dtype"],
        )


        self.noise_scheme = config.noise_scheme
        assert self.noise_scheme in ["diffusion_forcing", "consistent_noise", "teacher_forcing"]


        self.num_transformer_blocks = config.net.num_layers
        self.num_transformer_heads = config.net.num_heads
        self.frame_seq_length = 1560
        self.max_latent_frames_per_gpu = getattr(
            config, "max_latent_frames_per_gpu", 21
        )
        self.cp_size = None

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.num_frame_per_block = getattr(config, "num_frame_per_block", 1)
        self.net.num_frame_per_block = self.num_frame_per_block
        self.history_noise = getattr(config, "history_noise", 0)
        self.force_teacher_t0 = getattr(config, "force_teacher_t0", False)
        if self.history_noise > 0:
            assert self.noise_scheme == "teacher_forcing", (
                "history_noise is only supported for teacher_forcing noise scheme"
            )

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if config.lora_config.enabled:
            self.set_lora_trainable()


        from gamma_world._src.imaginaire.utils.count_params import count_params

        self.use_camera_cond = getattr(self.net, "use_camera_cond", False)
        if self.use_camera_cond:
            self.net.freeze_parameters_camera_cond()
            self._param_count = count_params(self.net, verbose=False)

    def _extract_action_inputs(self, data_batch: dict[str, torch.Tensor]) -> Optional[dict]:

        if not self.config.use_action_control:
            return None

        if "action_0_keyboard" in data_batch:
            actions: list[dict[str, torch.Tensor]] = []
            i = 0
            while f"action_{i}_keyboard" in data_batch:
                actions.append(
                    {
                        "keyboard": data_batch[f"action_{i}_keyboard"],
                        "camera": data_batch.get(f"action_{i}_camera"),
                    }
                )
                i += 1
            return {"actions": actions}

        if "action_left_keyboard" not in data_batch:
            return None
        return {
            "actions": [
                {
                    "keyboard": data_batch["action_left_keyboard"],
                    "camera": data_batch.get("action_left_camera"),
                },
                {
                    "keyboard": data_batch["action_right_keyboard"],
                    "camera": data_batch.get("action_right_camera"),
                },
            ]
        }

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Video2WorldCondition]:

        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch_with_latent_view_indices)


        state_t = int(
            (data_batch["num_video_frames_per_view"].cpu().item() - 1) // self.tokenizer.temporal_compression_factor + 1
        )
        condition = condition.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=None,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition

    def apply_fsdp(self, dp_mesh: DeviceMesh) -> None:


        self.net.fully_shard(mesh=dp_mesh)
        self.net = fully_shard(self.net, mesh=dp_mesh, reshard_after_forward=True)
        broadcast_dtensor_model_states(self.net, dp_mesh)
        if hasattr(self, "net_ema") and self.net_ema:

            if next(self.net_ema.parameters()).device.type == "cpu":
                with misc.timer("Moving EMA model from CPU to CUDA"):
                    self.net_ema.to(device="cuda")

            self.net_ema.fully_shard(mesh=dp_mesh)
            self.net_ema = fully_shard(self.net_ema, mesh=dp_mesh, reshard_after_forward=True)
            broadcast_dtensor_model_states(self.net_ema, dp_mesh)
            self.net_ema_worker = DTensorFastEmaModelUpdater()

            self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)

    def broadcast_split_for_model_parallelsim(self, x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T):

        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            if x0_B_C_T_H_W is not None:
                if self.config.split_cp_in_model:
                    x0_B_C_T_H_W = broadcast_split_tensor(x0_B_C_T_H_W, seq_dim=2, process_group=cp_group)
                else:
                    x0_B_C_T_H_W = broadcast(x0_B_C_T_H_W, cp_group)
            if epsilon_B_C_T_H_W is not None:
                if self.config.split_cp_in_model:
                    epsilon_B_C_T_H_W = broadcast_split_tensor(epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group)
                else:
                    epsilon_B_C_T_H_W = broadcast(epsilon_B_C_T_H_W, cp_group)
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if (
                    sigma_B_T.shape[-1] == 1 or not self.config.split_cp_in_model
                ):
                    sigma_B_T = broadcast(sigma_B_T, cp_group)
                else:
                    sigma_B_T = broadcast_split_tensor(sigma_B_T, seq_dim=1, process_group=cp_group)
            if condition is not None:
                condition = condition.broadcast(cp_group, split=self.config.split_cp_in_model)
            self.net.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()

        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    def set_lora_trainable(self):
        log.info("Setting LoRA trainable")
        for name, param in self.net.blocks.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                if "lora_B" in name:

                    param.data.zero_()

    @property
    def text_encoder_class(self) -> str:
        return self.config.text_encoder_class

    def get_data_batch_with_latent_view_indices(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
        n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view
        view_indices_B_V_T = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=n_views)

        latent_view_indices_B_V_T = view_indices_B_V_T[:, :, 0 : self.config.state_t]
        latent_view_indices_B_T = rearrange(latent_view_indices_B_V_T, "B V T -> B (V T)")
        data_batch_with_latent_view_indices = data_batch.copy()
        data_batch_with_latent_view_indices["latent_view_indices_B_T"] = latent_view_indices_B_T
        return data_batch_with_latent_view_indices


    def sample_train_time(self, batch_size=None, num_frames=None, state_shape=None) -> torch.Tensor:
        if state_shape is not None:
            batch_size = state_shape[0]
            num_frames = state_shape[2]
        else:
            assert batch_size is not None and num_frames is not None, "batch_size and num_frames must be provided"
        t_B_T = self.rectified_flow.sample_train_time(batch_size * num_frames).to(**self.flow_matching_kwargs)
        t_B_T = t_B_T.reshape(batch_size, -1, self.num_frame_per_block)
        t_B_T[:, :, 1:] = t_B_T[:, :, 0:1]
        t_B_T = t_B_T.reshape(batch_size, num_frames)
        return t_B_T

    def inplace_compute_text_embeddings_online(self, data_batch: dict[str, torch.Tensor]):
        text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
        data_batch["t5_text_embeddings"] = text_embeddings
        data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:



        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            self.inplace_compute_text_embeddings_online(data_batch)

        data_batch = preprocess_databatch(data_batch, None)
        self._update_train_stats(data_batch)

        action_inputs = self._extract_action_inputs(data_batch)


        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)


        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.flow_matching_kwargs)
        batch_size = x0_B_C_T_H_W.size()[0]
        num_frames = x0_B_C_T_H_W.size()[2]

        cp_group = self.get_context_parallel_group()

        if self.noise_scheme == "consistent_noise":
            t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.flow_matching_kwargs)
            t_B_1 = rearrange(t_B, "b -> b 1")
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B_1 = self.broadcast_split_for_model_parallelsim(
                x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B_1
            )

            if not self.config.split_cp_in_model:
                assert x0_B_C_T_H_W.shape[2] == num_frames, "x0_B_C_T_H_W shape should be the same as before broadcast"
            timesteps = self.rectified_flow.get_discrete_timestamp(t_B_1, self.flow_matching_kwargs)
            sigmas = self.rectified_flow.get_sigmas(
                timesteps,
                self.flow_matching_kwargs,
            )
            xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas)

            vt_pred_B_C_T_H_W = self.denoise(
                xt_B_C_T_H_W, timesteps, condition, noise=epsilon_B_C_T_H_W, action_inputs=action_inputs
            )

            time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.flow_matching_kwargs)
            per_instance_loss = torch.mean(
                (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=list(range(1, vt_pred_B_C_T_H_W.dim()))
            )
            loss = torch.mean(time_weights_B * per_instance_loss)
            output_batch = {"edm_loss": loss}

        elif self.noise_scheme == "diffusion_forcing" or self.noise_scheme == "teacher_forcing":

            t_B_T = self.sample_train_time(batch_size, num_frames, x0_B_C_T_H_W.shape)

            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B_T = self.broadcast_split_for_model_parallelsim(
                x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B_T
            )

            if not self.config.split_cp_in_model:
                assert x0_B_C_T_H_W.shape[2] == num_frames, "x0_B_C_T_H_W shape should be the same as before broadcast"
                assert t_B_T.shape[1] == num_frames, "t_B_T shape should be the same as before broadcast"
            timesteps_B_T = self.rectified_flow.get_discrete_timestamp(t_B_T, self.flow_matching_kwargs)
            sigmas_B_T = self.rectified_flow.get_sigmas(
                timesteps_B_T.reshape(-1, 1),
                self.flow_matching_kwargs,
            ).reshape(batch_size, -1)
            xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation_multiple_timesteps(
                epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas_B_T
            )

            if self.noise_scheme == "teacher_forcing":
                num_interleave = 1

                if self.history_noise > 0:
                    aug_sigmas_B_T = torch.rand_like(timesteps_B_T) * self.history_noise
                    if not self.force_teacher_t0:
                        aug_sigmas_B_T[:, 1:] = aug_sigmas_B_T[:, :1]
                    aug_timesteps_B_T = aug_sigmas_B_T * 1000
                    x0_B_C_T_H_W, _ = self.rectified_flow.get_interpolation_multiple_timesteps(
                        torch.randn_like(epsilon_B_C_T_H_W), x0_B_C_T_H_W, aug_sigmas_B_T
                    )
                else:
                    aug_timesteps_B_T = torch.zeros_like(timesteps_B_T)

                if self.force_teacher_t0:
                    aug_timesteps_B_T = torch.zeros_like(aug_timesteps_B_T)

                if cp_group is not None and cp_group.size() > 1:

                    aug_timesteps_B_T = broadcast(aug_timesteps_B_T, cp_group)

                concated_x_B_C_T_H_W = torch.cat([xt_B_C_T_H_W, x0_B_C_T_H_W], dim=2)
                concated_x_B_C_T_H_W = rearrange(
                    concated_x_B_C_T_H_W, "b c (n t) h w -> b c (t n) h w", n=1 + num_interleave
                )
                concated_timesteps_B_T = torch.cat([timesteps_B_T, aug_timesteps_B_T], dim=1)
                concated_timesteps_B_T = rearrange(concated_timesteps_B_T, "b (n t) -> b (t n)", n=1 + num_interleave)



                all_condition = condition.to_dict()
                if self.config.model_type == "i2v":
                    if not self.config.i2v_zero_latent_condition:




                        condition_mask = torch.ones_like(x0_B_C_T_H_W)[:, :4]
                        assert "y_B_C_T_H_W" in all_condition
                        concated_y_condition_B_C_T_H_W = torch.cat(
                            [all_condition["y_B_C_T_H_W"], torch.cat([condition_mask, x0_B_C_T_H_W], dim=1)], dim=2
                        )
                    else:




                        condition_mask = torch.zeros_like(x0_B_C_T_H_W)[:, :4]
                        zero_latent = self.zero_latent[:, :, :1].repeat(1, 1, x0_B_C_T_H_W.shape[2], 1, 1)
                        concated_y_condition_B_C_T_H_W = torch.cat(
                            [all_condition["y_B_C_T_H_W"], torch.cat([condition_mask, zero_latent], dim=1)], dim=2
                        )
                    concated_y_condition_B_C_T_H_W = rearrange(
                        concated_y_condition_B_C_T_H_W, "b c (n t) h w -> b c (t n) h w", n=1 + num_interleave
                    )
                    all_condition["y_B_C_T_H_W"] = concated_y_condition_B_C_T_H_W.to(**self.tensor_kwargs)



                vt_pred_B_C_T_H_W = self.denoise(
                    concated_x_B_C_T_H_W,
                    concated_timesteps_B_T,
                    all_condition,
                    noise=epsilon_B_C_T_H_W,
                    num_interleave=num_interleave,
                    action_inputs=action_inputs,
                )
                vt_pred_B_C_T_H_W = rearrange(vt_pred_B_C_T_H_W, "b c (t n) h w -> n b c t h w", n=1 + num_interleave)[
                    0
                ]
            else:
                vt_pred_B_C_T_H_W = self.denoise(
                    xt_B_C_T_H_W, timesteps_B_T, condition, noise=epsilon_B_C_T_H_W, action_inputs=action_inputs
                )

            time_weights_B_T = self.rectified_flow.train_time_weight(
                timesteps_B_T.reshape(-1, 1), self.flow_matching_kwargs
            ).reshape(batch_size, -1)
            per_instance_loss = torch.mean((vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=[1, 3, 4])
            loss = torch.mean(time_weights_B_T * per_instance_loss)
            output_batch = {"edm_loss": loss}

        else:
            raise NotImplementedError

        return output_batch, loss

    @torch.no_grad()
    def validation_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:

        self.eval()


        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            self.inplace_compute_text_embeddings_online(data_batch)


        data_batch = preprocess_databatch(data_batch, (1, 1))

        action_inputs = self._extract_action_inputs(data_batch)


        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)


        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.flow_matching_kwargs)
        batch_size = x0_B_C_T_H_W.size()[0]
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.flow_matching_kwargs)
        t_B_1 = rearrange(t_B, "b -> b 1")


        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B_1 = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B_1
        )

        timesteps = self.rectified_flow.get_discrete_timestamp(t_B_1, self.flow_matching_kwargs)
        sigmas = self.rectified_flow.get_sigmas(timesteps, self.flow_matching_kwargs)


        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas)


        vt_pred_B_C_T_H_W = self.denoise(
            xt_B_C_T_H_W, timesteps, condition, noise=epsilon_B_C_T_H_W, action_inputs=action_inputs
        )


        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.flow_matching_kwargs)
        per_instance_loss = torch.mean(
            (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=list(range(1, vt_pred_B_C_T_H_W.dim()))
        )
        loss = torch.mean(time_weights_B * per_instance_loss)

        output_batch = {
            "val_flow_loss": loss.detach(),
        }

        return output_batch, loss


    def _initialize_kv_cache(self, batch_size, dtype, device, n_steps=1, use_uncond_kvcache=False):

        local_attn_size = getattr(self.net, "local_attn_size", -1)
        if local_attn_size == -1:

            kv_cache_size = self.frame_seq_length * self.max_latent_frames_per_gpu
        else:
            if local_attn_size > self.max_latent_frames_per_gpu:
                raise ValueError(
                    f"local_attn_size {local_attn_size} is larger than max_latent_frames_per_gpu {self.max_latent_frames_per_gpu}, "
                    f"which is not supported"
                )
            kv_cache_size = self.frame_seq_length * local_attn_size

        if self.cp_size is not None:
            assert kv_cache_size % self.cp_size == 0, "kv_cache_size must be divisible by cp_size"
            kv_cache_size = kv_cache_size // self.cp_size

        if n_steps > 1:
            print("Using step-dependent KV cache with step number:", n_steps)
        else:
            print("Using step-independent KV cache.")

        self.kv_cache1 = dict()
        for step_index in range(n_steps):
            kv_cache1 = []
            for _ in range(self.num_transformer_blocks):
                kv_cache1.append(
                    {
                        "k": torch.zeros(
                            [
                                batch_size,
                                int(kv_cache_size),
                                self.num_transformer_heads,
                                self.config.net.model_channels // self.num_transformer_heads,
                            ],
                            dtype=dtype,
                            device=device,
                        ),
                        "v": torch.zeros(
                            [
                                batch_size,
                                int(kv_cache_size),
                                self.num_transformer_heads,
                                self.config.net.model_channels // self.num_transformer_heads,
                            ],
                            dtype=dtype,
                            device=device,
                        ),
                        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    }
                )
            self.kv_cache1[step_index] = kv_cache1

        if use_uncond_kvcache:
            self.kv_cache2 = dict()
            for step_index in range(n_steps):
                kv_cache2 = []
                for _ in range(self.num_transformer_blocks):
                    kv_cache2.append(
                        {
                            "k": torch.zeros(
                                [
                                    batch_size,
                                    int(kv_cache_size),
                                    self.num_transformer_heads,
                                    self.config.net.model_channels // self.num_transformer_heads,
                                ],
                                dtype=dtype,
                                device=device,
                            ),
                            "v": torch.zeros(
                                [
                                    batch_size,
                                    int(kv_cache_size),
                                    self.num_transformer_heads,
                                    self.config.net.model_channels // self.num_transformer_heads,
                                ],
                                dtype=dtype,
                                device=device,
                            ),
                            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        }
                    )
                self.kv_cache2[step_index] = kv_cache2

    def _initialize_crossattn_cache(self, batch_size, dtype, device):

        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, 512, 12, self.config.net.model_channels // self.num_transformer_heads],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, 512, 12, self.config.net.model_channels // self.num_transformer_heads],
                        dtype=dtype,
                        device=device,
                    ),
                    "is_init": False,
                }
            )

        self.crossattn_cache = crossattn_cache

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:

        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch_with_latent_view_indices:
            num_conditional_frames = data_batch_with_latent_view_indices[NUM_CONDITIONAL_FRAMES_KEY]
            log.debug(f"Using {num_conditional_frames=} from data batch")
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(
                data_batch_with_latent_view_indices
            )
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch_with_latent_view_indices)

        is_image_batch = self.is_image_batch(data_batch_with_latent_view_indices)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        _, x0, _ = self.get_data_and_condition(data_batch_with_latent_view_indices)


        state_t = int(
            (data_batch["num_video_frames_per_view"].cpu().item() - 1) // self.tokenizer.temporal_compression_factor + 1
        )


        condition = condition.set_video_condition(
            state_t=state_t,
            gt_frames=x0,
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            state_t=state_t,
            gt_frames=x0,
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(
            is_cfg_conditional=True,
            condition_locations=["first_random_n"],
            num_conditional_frames_per_view=num_conditional_frames,
        )
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False,
            condition_locations=["first_random_n"],
            num_conditional_frames_per_view=num_conditional_frames,
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        if guidance != 1.0:
            _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)


        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        action_inputs = self._extract_action_inputs(data_batch)

        def velocity_fn(
            noise_x: torch.Tensor,
            timestep: torch.Tensor,
            skip_uncond: bool = False,
            kv_cache: Optional[List[dict]] = None,
            kv_cache_uncond: Optional[List[dict]] = None,
            use_uncond_kvcache: bool = False,
            noise: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
            if use_uncond_kvcache:
                assert kv_cache_uncond is not None
            else:
                kv_cache_uncond = kv_cache



            new_condition = condition
            new_uncondition = uncondition

            if condition.gt_frames is not None and condition.gt_frames.shape[2] != noise_x.shape[2]:
                assert kwargs.get("start_frame_for_rope", None) is not None, "start_frame_for_rope is not provided"
                start_frame = kwargs.get("start_frame_for_rope")
                end_frame = start_frame + noise_x.shape[2]


                new_condition_dict = condition.to_dict()
                new_condition_dict["gt_frames"] = condition.gt_frames[:, :, start_frame:end_frame, :, :]
                if condition.condition_video_input_mask_B_C_T_H_W is not None:
                    new_condition_dict["condition_video_input_mask_B_C_T_H_W"] = (
                        condition.condition_video_input_mask_B_C_T_H_W[:, :, start_frame:end_frame, :, :]
                    )
                if hasattr(condition, "view_indices_B_T") and condition.view_indices_B_T is not None:
                    new_condition_dict["view_indices_B_T"] = condition.view_indices_B_T[:, start_frame:end_frame]
                new_condition = type(condition)(**new_condition_dict)


                if guidance != 1.0:
                    new_uncondition_dict = uncondition.to_dict()
                    new_uncondition_dict["gt_frames"] = uncondition.gt_frames[:, :, start_frame:end_frame, :, :]
                    if uncondition.condition_video_input_mask_B_C_T_H_W is not None:
                        new_uncondition_dict["condition_video_input_mask_B_C_T_H_W"] = (
                            uncondition.condition_video_input_mask_B_C_T_H_W[:, :, start_frame:end_frame, :, :]
                        )
                    if hasattr(uncondition, "view_indices_B_T") and uncondition.view_indices_B_T is not None:
                        new_uncondition_dict["view_indices_B_T"] = uncondition.view_indices_B_T[
                            :, start_frame:end_frame
                        ]
                    new_uncondition = type(uncondition)(**new_uncondition_dict)

            cond_v = self.denoise(
                xt_B_C_T_H_W=noise_x,
                timesteps_B_T=timestep,
                condition=new_condition,
                kv_cache=kv_cache,
                noise=noise,
                action_inputs=action_inputs,
                **kwargs,
            )

            if guidance != 1.0 and not skip_uncond:
                uncond_v = self.denoise(
                    xt_B_C_T_H_W=noise_x,
                    timesteps_B_T=timestep,
                    condition=new_uncondition,
                    kv_cache=kv_cache_uncond,
                    noise=noise,
                    action_inputs=action_inputs,
                    **kwargs,
                )
                velocity_pred = uncond_v + guidance * (cond_v - uncond_v)
            else:
                velocity_pred = cond_v

            return velocity_pred

        return velocity_fn

    @sync_timer("CausalT2V2pt1Model: generate_samples_from_batch")
    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        start_latents: Optional[torch.Tensor] = None,
        verbose: bool = False,
        use_uncond_kvcache: bool = False,
        use_step_dependent_kv_cache: bool = False,
        disable_rollout: bool = False,
        compute_separate_kvcache: bool = True,
        separate_kvcache_timestep_int: int = None,
        **kwargs,
    ) -> torch.Tensor:


        if disable_rollout and use_step_dependent_kv_cache:
            print("Rollout disabled, using parent class inference function.")
            return super().generate_samples_from_batch(
                data_batch=data_batch,
                guidance=guidance,
                seed=seed,
                state_shape=state_shape,
                n_sample=n_sample,
                is_negative_prompt=is_negative_prompt,
                num_steps=num_steps,
                shift=shift,
                **kwargs,
            )

        if use_uncond_kvcache is None:
            use_uncond_kvcache = False if self.noise_scheme == "teacher_forcing" else True

        from gamma_world._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0

        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        noise_B_C_T_H_W = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )
        self.frame_seq_length = int(state_shape[-1] * state_shape[-2] / 4)
        misc.set_random_seed(seed=seed, by_rank=False)

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)


        cp_group = self.get_context_parallel_group()
        self.cp_size = 1 if cp_group is None else len(get_process_group_ranks(cp_group))
        if cp_group is not None and cp_group.size() > 1:
            noise_B_C_T_H_W = broadcast(noise_B_C_T_H_W.contiguous(), cp_group)
            if start_latents is not None:
                start_latents = broadcast(start_latents.contiguous(), cp_group)
        else:

            assert not getattr(self.net, "is_context_parallel_enabled", False), (
                "context parallel should be disabled if parallel_state is not initialized"
            )

        if cp_group is not None and not is_tp_cp_pp_rank0():
            verbose = False

        velocity_fn = self.get_velocity_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        def denoise_fn(
            noisy_image_or_video: torch.Tensor,
            timestep: torch.Tensor,
            kv_cache: Optional[List[dict]] = None,
            kv_cache_uncond: Optional[List[dict]] = None,
            crossattn_cache: Optional[List[dict]] = None,
            current_start: Optional[int] = None,
            current_end: Optional[int] = None,
            start_frame_for_rope: Optional[int] = None,
            skip_uncond: bool = False,
            noise: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            velocity_pred_B_C_T_H_W = velocity_fn(
                noisy_image_or_video,
                timestep,
                kv_cache=kv_cache,
                kv_cache_uncond=kv_cache_uncond,
                use_uncond_kvcache=use_uncond_kvcache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                start_frame_for_rope=start_frame_for_rope,
                skip_uncond=skip_uncond,
                noise=noise,
            )
            return velocity_pred_B_C_T_H_W


        batch_size, num_channels, num_frames, height, width = noise_B_C_T_H_W.shape
        output_B_C_T_H_W = torch.zeros(
            [batch_size, num_channels, num_frames, height, width],
            device=noise_B_C_T_H_W.device,
            dtype=noise_B_C_T_H_W.dtype,
        )


        if self.kv_cache1 is None:
            n_kvcache_steps = num_steps if use_step_dependent_kv_cache else 1
            if use_step_dependent_kv_cache and compute_separate_kvcache:
                n_kvcache_steps += 1
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=self.tensor_kwargs["dtype"],
                device=self.tensor_kwargs["device"],
                n_steps=n_kvcache_steps,
                use_uncond_kvcache=use_uncond_kvcache,
            )
            if guidance == 1.0:
                self._initialize_crossattn_cache(
                    batch_size=batch_size, dtype=self.tensor_kwargs["dtype"], device=self.tensor_kwargs["device"]
                )
            else:
                self.crossattn_cache = None
        else:
            if guidance == 1.0:

                for block_index in range(self.num_transformer_blocks):
                    self.crossattn_cache[block_index]["is_init"] = False
            else:
                self.crossattn_cache = None


            for step_index in list(self.kv_cache1.keys()):
                for block_index in range(len(self.kv_cache1[step_index])):
                    self.kv_cache1[step_index][block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise_B_C_T_H_W.device
                    )
                    self.kv_cache1[step_index][block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise_B_C_T_H_W.device
                    )

            if use_uncond_kvcache:
                for step_index in list(self.kv_cache2.keys()):
                    for block_index in range(len(self.kv_cache2[step_index])):
                        self.kv_cache2[step_index][block_index]["global_end_index"] = torch.tensor(
                            [0], dtype=torch.long, device=noise_B_C_T_H_W.device
                        )
                        self.kv_cache2[step_index][block_index]["local_end_index"] = torch.tensor(
                            [0], dtype=torch.long, device=noise_B_C_T_H_W.device
                        )


        num_blocks = num_frames // self.num_frame_per_block
        for block_index in tqdm(range(num_blocks), desc="Denoising blocks", disable=not verbose):
            if verbose:
                time_block_start = time.time()

            latent_model_input = noise_B_C_T_H_W[
                :, :, block_index * self.num_frame_per_block : (block_index + 1) * self.num_frame_per_block
            ]
            self.sample_scheduler.config.shift = shift

            self.sample_scheduler.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=shift)
            timesteps = self.sample_scheduler.timesteps

            for index, current_timestep in enumerate(timesteps):
                if verbose:
                    time_denoising_start = time.time()


                timestep = (
                    torch.ones([batch_size, self.num_frame_per_block], device=noise_B_C_T_H_W.device, dtype=torch.int64)
                    * current_timestep
                )
                kv_cache_step_index = index if use_step_dependent_kv_cache else 0
                velocity_field_pred = denoise_fn(
                    latent_model_input,
                    timestep,
                    kv_cache=self.kv_cache1[kv_cache_step_index],
                    kv_cache_uncond=self.kv_cache2[kv_cache_step_index] if use_uncond_kvcache else None,
                    crossattn_cache=self.crossattn_cache,
                    current_start=block_index * self.num_frame_per_block * self.frame_seq_length // self.cp_size,
                    current_end=(block_index + 1) * self.num_frame_per_block * self.frame_seq_length // self.cp_size,
                    start_frame_for_rope=block_index * self.num_frame_per_block,
                    noise=noise_B_C_T_H_W[
                        :, :, block_index * self.num_frame_per_block : (block_index + 1) * self.num_frame_per_block
                    ],
                )

                temp_x0 = self.sample_scheduler.step(
                    velocity_field_pred.unsqueeze(0),
                    current_timestep,
                    latent_model_input[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latent_model_input = temp_x0.squeeze(0)

                if verbose:
                    print(f"[Step {index}] Finish one denoising step in {time.time() - time_denoising_start} seconds")


            output_B_C_T_H_W[
                :, :, block_index * self.num_frame_per_block : (block_index + 1) * self.num_frame_per_block
            ] = latent_model_input

            if compute_separate_kvcache:
                if self.noise_scheme == "teacher_forcing":

                    t_kv_cache = 0
                else:

                    t_kv_cache = current_timestep

                if separate_kvcache_timestep_int is not None:
                    t_kv_cache = separate_kvcache_timestep_int

                timestep_kv_cache = (
                    torch.ones([batch_size, self.num_frame_per_block], device=noise_B_C_T_H_W.device, dtype=torch.int64)
                    * t_kv_cache
                )
                kv_cache_step_index = num_steps if use_step_dependent_kv_cache else 0

                if self.noise_scheme == "teacher_forcing" or t_kv_cache == 0:

                    noised_latent_model_input = latent_model_input
                else:
                    noise_input = misc.arch_invariant_rand(
                        noise_B_C_T_H_W[
                            :, :, block_index * self.num_frame_per_block : (block_index + 1) * self.num_frame_per_block
                        ].shape,
                        torch.float32,
                        self.tensor_kwargs["device"],
                        seed + block_index * 42,
                    )
                    noised_latent_model_input = self.sample_scheduler.add_noise(
                        latent_model_input,
                        noise_input,
                        torch.tensor([t_kv_cache], device=noise_B_C_T_H_W.device, dtype=torch.int64),
                    )

                denoise_fn(
                    noised_latent_model_input,
                    timestep_kv_cache,
                    kv_cache=self.kv_cache1[kv_cache_step_index],
                    kv_cache_uncond=self.kv_cache2[kv_cache_step_index] if use_uncond_kvcache else None,
                    crossattn_cache=self.crossattn_cache,
                    current_start=block_index * self.num_frame_per_block * self.frame_seq_length // self.cp_size,
                    current_end=(block_index + 1) * self.num_frame_per_block * self.frame_seq_length // self.cp_size,
                    start_frame_for_rope=block_index * self.num_frame_per_block,
                    skip_uncond=False if use_uncond_kvcache else True,
                )

                if (use_uncond_kvcache and self.noise_scheme == "teacher_forcing") or (
                    self.noise_scheme == "teacher_forcing" and not use_uncond_kvcache
                ):
                    print(
                        f"[Warning] Using {self.noise_scheme} and use_uncond_kvcache={use_uncond_kvcache}. "
                        f"This can lead to degraded results."
                    )

            if verbose:
                print(
                    f"[Block {block_index}] Finish one frame block generation ({int(self.num_frame_per_block * 4)} frames) in {time.time() - time_block_start} seconds (KV cached for {block_index * self.num_frame_per_block * 4} history frames)"
                )

        return output_B_C_T_H_W

    def denoise(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
        noise: torch.Tensor | None = None,
        kv_cache: Optional[List[dict]] = None,
        action_inputs: Optional[dict] = None,
        **kwargs,
    ):

        condition_video_mask = None

        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:

                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )


            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )

                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T * (
                    1 - condition_video_mask_B_1_T_1_1
                )

                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                timesteps_B_T = (
                    timesteps_B_T.unsqueeze(0) if timesteps_B_T.ndim == 1 else timesteps_B_T
                )


        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T.to(**self.tensor_kwargs),
            kv_cache=kv_cache,
            action_inputs=action_inputs,
            **condition.to_dict(),
            **kwargs,
        ).float()

        if condition.is_video and self.config.denoise_replace_gt_frames and noise is not None:








            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )

        return net_output_B_C_T_H_W
