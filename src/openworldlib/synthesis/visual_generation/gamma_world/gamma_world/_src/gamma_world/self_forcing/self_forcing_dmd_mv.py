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

import copy
from typing import Callable, Dict, List, Optional, Tuple, cast

import attrs
import torch
import torch.distributed as dist
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed import get_process_group_ranks

from gamma_world._src.imaginaire.utils import log, misc
from gamma_world._src.imaginaire.utils.context_parallel import (
    broadcast,
    broadcast_split_tensor,
)
from gamma_world._src.predict2.conditioner import DataType
from gamma_world._src.predict2_multiview.configs.vid2vid.defaults.conditioner import MultiViewCondition
from gamma_world._src.predict2_multiview.models.multiview_vid2vid_model_rectified_flow import (
    compute_empty_and_negative_text_embeddings,
    compute_text_embeddings_online_multiview,
    preprocess_databatch,
)
from gamma_world._src.gamma_world.self_forcing.self_forcing_dmd import (
    DMDSelfForcingModel,
    DMDSelfForcingModelConfig,
)

NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


@attrs.define(slots=False)
class DMDSelfForcingMVModelConfig(DMDSelfForcingModelConfig):
    train_sample_views_range: tuple[int, int] | None = None
    use_action_control: bool = False





    shuffle_agents: bool = False


class DMDSelfForcingMVModel(DMDSelfForcingModel):
    def __init__(self, config: DMDSelfForcingMVModelConfig):
        super().__init__(config)
        self.state_t = config.state_t
        self.num_frame_per_block = config.num_frame_per_block
        self.cp_size = None
        self.empty_string_text_embeddings = None
        self.neg_text_embeddings = None
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            compute_empty_and_negative_text_embeddings(self)

    def _extract_action_inputs(self, data_batch: dict[str, torch.Tensor]) -> dict | None:

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

    def _extract_extra_conditional_kwargs(self, data_batch: dict[str, torch.Tensor]) -> dict:

        action_inputs = self._extract_action_inputs(data_batch)
        return {"action_inputs": action_inputs} if action_inputs is not None else {}


    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        pixel_frames_per_view = int(self.tokenizer.get_pixel_num_frames(self.state_t))
        n_views = state.shape[2] // pixel_frames_per_view
        cp_group = self.get_context_parallel_group()
        cp_size = len(get_process_group_ranks(cp_group)) if cp_group is not None else 1
        if cp_group is not None and n_views > 1 and n_views <= cp_size:
            return self.encode_cp(state)
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=n_views)
        encoded_state = super().encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        n_views = latent.shape[2] // self.state_t
        cp_group = self.get_context_parallel_group()
        cp_size = len(get_process_group_ranks(cp_group)) if cp_group is not None else 1
        if cp_group is not None and n_views > 1 and n_views <= cp_size:
            return self.decode_cp(latent)
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=n_views)
        decoded_state = super().decode(latent)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return decoded_state

    @torch.no_grad()
    def encode_cp(self, state: torch.Tensor) -> torch.Tensor:
        cp_group = self.get_context_parallel_group()
        assert cp_group is not None
        cp_size = len(get_process_group_ranks(cp_group))
        get_pixel_frames = cast(Callable[[int], int], self.tokenizer.get_pixel_num_frames)
        pixel_frames_per_view = int(get_pixel_frames(self.state_t))
        n_views = state.shape[2] // pixel_frames_per_view
        assert n_views <= cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        state_V_B_C_T_H_W = rearrange(state, "B C (V T) H W -> V B C T H W", V=n_views)
        state_input = torch.zeros((cp_size, *state_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        state_input[0:n_views] = state_V_B_C_T_H_W
        local_state_V_B_C_T_H_W = broadcast_split_tensor(state_input, seq_dim=0, process_group=cp_group)
        local_state = rearrange(local_state_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        encoded_state = super().encode(local_state)
        encoded_state_list = [torch.empty_like(encoded_state) for _ in range(cp_size)]
        dist.all_gather(encoded_state_list, encoded_state, group=cp_group)
        encoded_state = torch.cat(encoded_state_list[0:n_views], dim=2)
        return encoded_state

    @torch.no_grad()
    def decode_cp(self, latent: torch.Tensor) -> torch.Tensor:
        cp_group = self.get_context_parallel_group()
        assert cp_group is not None
        cp_size = len(get_process_group_ranks(cp_group))
        n_views = latent.shape[2] // self.state_t
        assert n_views <= cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        latent_V_B_C_T_H_W = rearrange(latent, "B C (V T) H W -> V B C T H W", V=n_views)
        latent_input = torch.zeros((cp_size, *latent_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        latent_input[0:n_views] = latent_V_B_C_T_H_W
        local_latent_V_B_C_T_H_W = broadcast_split_tensor(latent_input, seq_dim=0, process_group=cp_group)
        local_latent = rearrange(local_latent_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        decoded_state = super().decode(local_latent)
        decoded_state_list = [torch.empty_like(decoded_state) for _ in range(cp_size)]
        dist.all_gather(decoded_state_list, decoded_state, group=cp_group)
        decoded_state = torch.cat(decoded_state_list[0:n_views], dim=2)
        return decoded_state

    def broadcast_split_for_model_parallelsim(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: MultiViewCondition,
        epsilon_B_C_T_H_W: torch.Tensor | None,
        sigma_B_T: torch.Tensor | None,
    ) -> tuple[torch.Tensor, MultiViewCondition, torch.Tensor | None, torch.Tensor | None]:
        n_views = x0_B_C_T_H_W.shape[2] // self.state_t
        x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
        if epsilon_B_C_T_H_W is not None:
            epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
        reshape_sigma_B_T = False
        if sigma_B_T is not None:
            assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
            if sigma_B_T.shape[-1] != 1:
                assert sigma_B_T.shape[-1] % n_views == 0, (
                    f"sigma_B_T temporal dimension T must either be 1 or a multiple of sample_n_views. Got T={sigma_B_T.shape[-1]} and sample_n_views={n_views}"
                )
                sigma_B_T = rearrange(sigma_B_T, "B (V T) -> (B V) T", V=n_views).contiguous()
                reshape_sigma_B_T = True
        (
            x0_B_C_T_H_W,
            condition,
            epsilon_B_C_T_H_W,
            sigma_B_T,
        ) = super().broadcast_split_for_model_parallelsim(x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T)

        x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
        if epsilon_B_C_T_H_W is not None:
            epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
        if reshape_sigma_B_T:
            sigma_B_T = rearrange(cast(torch.Tensor, sigma_B_T), "(B V) T -> B (V T)", V=n_views)
        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    def get_data_batch_with_latent_view_indices(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
        n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view
        view_indices_B_V_T = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=n_views)
        latent_view_indices_B_V_T = view_indices_B_V_T[:, :, 0 : self.state_t]
        latent_view_indices_B_T = rearrange(latent_view_indices_B_V_T, "B V T -> B (V T)")
        data_batch_with_latent_view_indices = data_batch.copy()
        data_batch_with_latent_view_indices["latent_view_indices_B_T"] = latent_view_indices_B_T
        return data_batch_with_latent_view_indices

    def _normalize_video_databatch_inplace(
        self, data_batch: dict[str, torch.Tensor], input_key: str | None = None
    ) -> None:
        input_key = self.input_data_key if input_key is None else input_key
        is_preprocessed = "is_preprocessed" in data_batch and data_batch["is_preprocessed"] is True
        num_video_frames_per_view = (
            cast(Callable[[int], int], self.tokenizer.get_pixel_num_frames)(self.state_t)
            if is_preprocessed
            else data_batch["num_video_frames_per_view"]
        )
        if isinstance(num_video_frames_per_view, torch.Tensor):
            num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
        n_views = data_batch[input_key].shape[2] // num_video_frames_per_view
        if input_key in data_batch:
            data_batch[input_key] = rearrange(data_batch[input_key], "B C (V T) H W -> (B V) C T H W", V=n_views)
            super()._normalize_video_databatch_inplace(data_batch, input_key)
            data_batch[input_key] = rearrange(data_batch[input_key], "(B V) C T H W -> B C (V T) H W", V=n_views)

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor], with_uncondition: bool = False):
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        return super().get_data_and_condition(data_batch_with_latent_view_indices, with_uncondition)

    def _shuffle_agents_inplace(self, data_batch: dict[str, torch.Tensor]) -> None:

        if not self.config.shuffle_agents:
            return

        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
        input_key = self.input_data_key
        assert input_key in data_batch, f"shuffle_agents: input key {input_key!r} missing from data_batch"

        video = data_batch[input_key]
        n_views = video.shape[2] // num_video_frames_per_view
        if n_views <= 1:
            return

        perm = torch.randperm(n_views)

        video_BCVTHW = rearrange(video, "B C (V T) H W -> B C V T H W", V=n_views)
        video_BCVTHW = video_BCVTHW[:, :, perm].contiguous()
        data_batch[input_key] = rearrange(video_BCVTHW, "B C V T H W -> B C (V T) H W")

        assert "view_indices" in data_batch, "shuffle_agents: data_batch missing 'view_indices'"
        vi = data_batch["view_indices"]
        vi_BVT = rearrange(vi, "B (V T) -> B V T", V=n_views)
        vi_BVT = vi_BVT[:, perm].contiguous()
        data_batch["view_indices"] = rearrange(vi_BVT, "B V T -> B (V T)")

        if self.config.use_action_control:
            perm_list = perm.tolist()
            if "action_0_keyboard" in data_batch:

                for action_kind in ("keyboard", "camera"):
                    keys = [f"action_{i}_{action_kind}" for i in range(n_views)]
                    if not all(k in data_batch for k in keys):
                        if action_kind == "camera":
                            continue
                        raise AssertionError(
                            f"shuffle_agents: expected action keys {keys} but some are missing"
                        )
                    originals = [data_batch[k] for k in keys]
                    for i, k in enumerate(keys):
                        data_batch[k] = originals[perm_list[i]]
            elif "action_left_keyboard" in data_batch:



                assert n_views == 2, (
                    f"shuffle_agents: legacy action_left/right layout requires "
                    f"n_views==2, got n_views={n_views}"
                )
                if perm_list[0] != 0:

                    for action_kind in ("keyboard", "camera"):
                        lk = f"action_left_{action_kind}"
                        rk = f"action_right_{action_kind}"
                        if lk in data_batch and rk in data_batch:
                            data_batch[lk], data_batch[rk] = data_batch[rk], data_batch[lk]
                        elif action_kind == "camera":
                            continue
                        else:
                            raise AssertionError(
                                f"shuffle_agents: expected both '{lk}' and '{rk}' but at least one is missing"
                            )
            else:
                raise AssertionError(
                    "shuffle_agents: data_batch has no recognized action keys "
                    "(neither 'action_0_keyboard' nor 'action_left_keyboard')"
                )

    def _sample_agent_pool_indices(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> list[int] | None:

        pool_size = getattr(self.net, "simplex_pool_size", None)
        if pool_size is None:
            return None

        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
        n_views = data_batch[self.input_data_key].shape[2] // num_video_frames_per_view
        if pool_size <= n_views:
            return None

        if parallel_state.is_initialized():
            dp_rank = parallel_state.get_data_parallel_rank()
        else:
            dp_rank = 0
        seed = ((int(iteration) * 1_000_003) ^ (int(dp_rank) * 7919) ^ 0x9E3779B9) & 0x7FFFFFFF
        gen = torch.Generator()
        gen.manual_seed(seed)
        perm = torch.randperm(pool_size, generator=gen)
        return perm[:n_views].tolist()

    def _set_all_nets_agent_pool_indices(self, indices: list[int] | None) -> None:

        targets: list[tuple[str, torch.nn.Module]] = [("net", self.net)]
        if hasattr(self, "net_ema") and self.net_ema is not None:
            targets.append(("net_ema", self.net_ema))
        for attr in ("net_real_score", "net_fake_score"):
            net = getattr(self, attr, None)
            if net is not None:
                targets.append((attr, net))

        if indices is not None:
            for name, net in targets:
                assert hasattr(net, "_override_agent_pool_indices"), (
                    f"SF simplex pool sampling requires '{name}' to expose "
                    f"_override_agent_pool_indices (sparse_hub / 4DRoPE simplex "
                    f"variant). Configure all nets — student and teacher — "
                    f"with multi_agent_rope_simplex_pool_size, otherwise "
                    f"student and teacher will see different agent permutations."
                )

        for name, net in targets:
            if hasattr(net, "_override_agent_pool_indices"):
                net._override_agent_pool_indices = indices

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        data_batch = preprocess_databatch(data_batch, self.config.train_sample_views_range)





        self._shuffle_agents_inplace(data_batch)








        indices = self._sample_agent_pool_indices(data_batch, iteration)
        self._set_all_nets_agent_pool_indices(indices)

        try:
            output_batch, loss = super().training_step(data_batch, iteration)
        finally:
            self._set_all_nets_agent_pool_indices(None)
        return output_batch, loss

    def inplace_compute_text_embeddings_online(
        self, data_batch: dict[str, torch.Tensor], use_negative_prompt: bool = True
    ):
        output_text_embeddings, output_neg_text_embeddings, dropout_text_embeddings = (
            compute_text_embeddings_online_multiview(self, data_batch)
        )
        t5_text_embeddings = {
            "text_embeddings": output_text_embeddings,
            "dropout_text_embeddings": dropout_text_embeddings,
        }
        neg_t5_text_embeddings = {
            "text_embeddings": output_neg_text_embeddings,
            "dropout_text_embeddings": dropout_text_embeddings,
        }
        data_batch["t5_text_embeddings"] = t5_text_embeddings["text_embeddings"]
        data_batch["neg_t5_text_embeddings"] = neg_t5_text_embeddings["text_embeddings"]

        data_batch["t5_text_mask"] = torch.ones(
            output_text_embeddings.shape[0], output_text_embeddings.shape[1], device="cuda"
        )

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        n_views: int,
        guidance: float = 1.0,
        is_negative_prompt: bool = False,
        conditional_dict: dict = None,
    ) -> Callable:
        assert data_batch is not None or conditional_dict is not None, "data_batch or conditional_dict must be provided"

        if data_batch is not None:
            data_batch = self.get_data_batch_with_latent_view_indices(data_batch)
            if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
                num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
                log.debug(f"Using {num_conditional_frames=} from data batch")
            else:
                num_conditional_frames = 1

        if conditional_dict is None:
            _, latent_state, _ = self.get_data_and_condition(
                data_batch, with_uncondition=False
            )
            is_image_batch = self.is_image_batch(data_batch)
            if is_negative_prompt:
                condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
            else:
                condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

            condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

            state_t = int(
                (data_batch["num_video_frames_per_view"].cpu().item() - 1) // self.tokenizer.temporal_compression_factor
                + 1
            )

            condition = condition.set_video_condition(
                state_t=state_t,
                gt_frames=latent_state.to(**self.tensor_kwargs),
                condition_locations=["first_random_n"],
                random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
                random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
                num_conditional_frames_per_view=num_conditional_frames,
                view_condition_dropout_max=0,
                conditional_frames_probs=self.config.conditional_frames_probs,
            )

            extra_cond = self._extract_extra_conditional_kwargs(data_batch)
            conditional_dict = {**condition.to_dict(), **extra_cond}







            _, condition, _, _ = self.broadcast_split_for_model_parallelsim(latent_state, condition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def x0_fn(
            noise_x: torch.Tensor,
            timestep: torch.Tensor,
            kv_cache: Optional[List[dict]] = None,
            **kwargs,
        ) -> torch.Tensor:

            def _unfold(tensor_flat: torch.Tensor) -> torch.Tensor:
                return rearrange(tensor_flat, "B C (V T) H W -> B V C T H W", V=n_views)

            def _fold(tensor_unfold: torch.Tensor) -> torch.Tensor:
                return rearrange(tensor_unfold, "B V C T H W -> B C (V T) H W")

            noise_x = noise_x.permute(0, 2, 1, 3, 4)

            noise_unfold = _unfold(noise_x)
            gt_frames_unfold: torch.Tensor | None = None
            if conditional_dict["gt_frames"] is not None:
                gt_frames_unfold = _unfold(conditional_dict["gt_frames"])

            start_frame = kwargs.get("start_frame_for_rope", 0)
            end_frame = start_frame + noise_unfold.shape[3]
            new_condition_dict = copy.deepcopy(conditional_dict)

            if gt_frames_unfold is not None and gt_frames_unfold.shape[3] != noise_unfold.shape[3]:
                assert kwargs.get("start_frame_for_rope", None) is not None, "start_frame_for_rope is not provided"
                sliced_gt = gt_frames_unfold[:, :, :, start_frame:end_frame, :, :]
                new_condition_dict["gt_frames"] = _fold(sliced_gt)
                if new_condition_dict["condition_video_input_mask_B_C_T_H_W"] is not None:
                    mask_unfold = _unfold(new_condition_dict["condition_video_input_mask_B_C_T_H_W"])
                    sliced_mask = mask_unfold[:, :, :, start_frame:end_frame, :, :]
                    new_condition_dict["condition_video_input_mask_B_C_T_H_W"] = _fold(sliced_mask)

            if new_condition_dict["view_indices_B_T"] is not None:
                view_indices_unfold = rearrange(new_condition_dict["view_indices_B_T"], "B (V T) -> B V T", V=n_views)
                if view_indices_unfold.shape[2] != noise_unfold.shape[3]:
                    new_condition_dict["view_indices_B_T"] = rearrange(
                        view_indices_unfold[:, :, start_frame:end_frame], "B V T -> B (V T)"
                    )

            noise_fold = _fold(noise_unfold)

            _, denoised_pred = self.generator(
                noisy_image_or_video=noise_fold.permute(0, 2, 1, 3, 4),
                conditional_dict=new_condition_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                n_views=n_views,
                **kwargs,
            )
            return denoised_pred

        return x0_fn

    def _initialize_kv_cache(
        self,
        batch_size: int,
        n_views: int,
        dtype: torch.dtype,
        device: torch.device | str,
        num_training_frames: int = None,
        is_training: bool = False,
    ):

        if num_training_frames is None:
            num_training_frames = self.config.num_training_frames

        local_attn_size = getattr(self.net, "local_attn_size", -1)
        use_sparse_hub = getattr(self.net, "use_sparse_hub", False)
        z_num = getattr(self.net, "z_num", 0)

        log.info(f"Initializing multiview KV cache (joint-block layout):")
        log.info(f"  batch_size: {batch_size}, n_views: {n_views}")
        log.info(f"  local_attn_size: {local_attn_size}")
        log.info(f"  use_sparse_hub: {use_sparse_hub}, z_num: {z_num}")
        log.info(f"  frame_seq_length: {self.frame_seq_length}")



        if local_attn_size == -1 or is_training:
            per_view_frames = num_training_frames
        else:
            if local_attn_size > num_training_frames:
                raise ValueError(
                    f"local_attn_size {local_attn_size} is larger than num_training_frames "
                    f"{num_training_frames}, which is not supported"
                )
            per_view_frames = local_attn_size

        head_dim = self.net.model_channels // self.net.num_heads


        if use_sparse_hub:
            cp_group = self.get_context_parallel_group()
            cp_size = 1 if cp_group is None else cp_group.size()
            assert cp_size == 1, (
                f"sparse_hub does not support CP yet (cp_size={cp_size}); "
                f"the player/Z cache layout would be broken by contiguous CP split"
            )
            assert z_num >= 1, f"sparse_hub requires z_num>=1, got {z_num}"








            max_player_tokens_per_view = self.frame_seq_length * per_view_frames
            max_z_tokens = z_num * per_view_frames

            log.info(
                f"  sparse_hub cache: V={n_views}, per_view_frames={per_view_frames}, "
                f"max_player_tokens_per_view={max_player_tokens_per_view}, "
                f"max_z_tokens={max_z_tokens}, head_dim={head_dim}"
            )

            kv_cache1 = []
            for _ in range(self.net.num_layers):
                kv_cache1.append(
                    {
                        "k_players": torch.zeros(
                            [batch_size, n_views, max_player_tokens_per_view, self.net.num_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "v_players": torch.zeros(
                            [batch_size, n_views, max_player_tokens_per_view, self.net.num_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "k_z": torch.zeros(
                            [batch_size, max_z_tokens, self.net.num_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "v_z": torch.zeros(
                            [batch_size, max_z_tokens, self.net.num_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "z_local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    }
                )
            self.kv_cache1 = kv_cache1
            return



        kv_cache_size = self.frame_seq_length * per_view_frames * n_views


        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1 and not v_split_mode:
            assert kv_cache_size % cp_size == 0, "kv_cache_size must be divisible by cp_size"
            kv_cache_size = kv_cache_size // cp_size


        effective_batch_size = batch_size
        log.info(f"  effective_batch_size for KV cache: {effective_batch_size}")
        log.info(f"  kv_cache_size (sequence length): {kv_cache_size}")

        kv_cache1 = []
        for _ in range(self.net.num_layers):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [effective_batch_size, int(kv_cache_size), self.net.num_heads, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [effective_batch_size, int(kv_cache_size), self.net.num_heads, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache1 = kv_cache1

    def generate_samples_from_batch(
        self,
        data_batch: dict[str, torch.Tensor] | None = None,
        guidance: float = 1.0,
        seed: int = 1,
        state_shape: tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        start_latents: Optional[torch.Tensor] = None,
        verbose: bool = False,
        conditional_dict: dict = None,
        image_or_video_shape: Tuple | None = None,
        noise_B_T_C_H_W: Optional[torch.Tensor] = None,
        is_training: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if data_batch is not None:
            self._normalize_video_databatch_inplace(data_batch)
            if hasattr(self, "_augment_image_dim_inplace"):
                self._augment_image_dim_inplace(data_batch)

            is_image_batch = self.is_image_batch(data_batch)
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            if n_sample is None:
                n_sample = data_batch[input_key].shape[0]

            num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
            n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view

            if state_shape is None:
                _T, _H, _W = data_batch[input_key].shape[-3:]
                state_shape = [
                    self.tokenizer.get_latent_num_frames(_T // n_views),
                    self.config.state_ch,
                    _H // self.tokenizer.spatial_compression_factor,
                    _W // self.tokenizer.spatial_compression_factor,
                ]
            else:
                state_shape = [state_shape[1] // n_views, state_shape[0], state_shape[2], state_shape[3]]

        assert state_shape is not None or image_or_video_shape is not None, (
            "data_batch or image_or_video_shape must be provided"
        )

        if state_shape is not None:
            assert data_batch is not None, "data_batch must be provided when state_shape is provided"

            latent_frames_per_view = state_shape[0]


            flat_state_shape = (
                latent_frames_per_view * n_views,
                state_shape[1],
                state_shape[2],
                state_shape[3],
            )
        else:
            flat_state_shape = image_or_video_shape[1:]
            latent_frames_per_view = self.state_t
            n_views = flat_state_shape[0] // latent_frames_per_view

        if noise_B_T_C_H_W is None:
            noise_B_T_C_H_W = misc.arch_invariant_rand(
                (n_sample,) + tuple(flat_state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            misc.set_random_seed(seed=seed, by_rank=False)


        self.frame_seq_length = int(noise_B_T_C_H_W.shape[-1] * noise_B_T_C_H_W.shape[-2] / 4)


        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            self.net.enable_context_parallel(cp_group)

            noise_B_T_C_H_W = broadcast(noise_B_T_C_H_W.contiguous(), cp_group)
            if start_latents is not None:
                start_latents = broadcast(start_latents.contiguous(), cp_group)
        else:

            assert not getattr(self.net, "is_context_parallel_enabled", False), (
                "context parallel should be disabled if parallel_state is not initialized"
            )

        flow_pred_fn = self.get_x0_fn_from_batch(
            data_batch,
            n_views=n_views,
            guidance=guidance,
            is_negative_prompt=is_negative_prompt,
            conditional_dict=conditional_dict,
        )

        def x0_fn(
            noisy_image_or_video: torch.Tensor,
            timestep: torch.Tensor,
            kv_cache: Optional[List[dict]] = None,
            crossattn_cache: Optional[List[dict]] = None,
            current_start: Optional[int] = None,
            current_end: Optional[int] = None,
            start_frame_for_rope: Optional[int] = None,
        ) -> torch.Tensor:
            return flow_pred_fn(
                noisy_image_or_video,
                timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                start_frame_for_rope=start_frame_for_rope,
            )


        batch_size, num_frames_total, num_channels, height, width = noise_B_T_C_H_W.shape

        num_input_frames = 0
        num_output_frames = num_frames_total + num_input_frames

        output_B_T_C_H_W = torch.zeros(
            [batch_size, num_frames_total, num_channels, height, width],
            device=noise_B_T_C_H_W.device,
            dtype=noise_B_T_C_H_W.dtype,
        )



        frames_per_temporal_block = n_views * self.num_frame_per_block
        num_temporal_blocks = latent_frames_per_view // self.num_frame_per_block




        n_views_local = n_views

        self._initialize_kv_cache(
            batch_size=batch_size,
            n_views=n_views_local,
            dtype=self.tensor_kwargs["dtype"],
            device=self.tensor_kwargs["device"],
            num_training_frames=latent_frames_per_view,
            is_training=is_training,
        )





        token_stride_per_frame = self.frame_seq_length * n_views // cp_size

        self.crossattn_cache = None


        current_start_frame = 0


        all_num_frames = [self.config.num_frame_per_block] * num_temporal_blocks

        exit_flags = []
        if self.config.independent_denoising_step_list:
            for block_index in range(len(all_num_frames)):
                if is_training:
                    exit_flag = self.generate_and_sync_list(
                        1, len(self.denoising_step_list[block_index]), device=noise_B_T_C_H_W.device
                    )
                else:
                    exit_flag = [len(self.denoising_step_list[block_index]) - 1]
                exit_flags.append(exit_flag[0])
        else:
            num_denoising_steps = len(self.denoising_step_list)
            if is_training:
                exit_flags = self.generate_and_sync_list(
                    len(all_num_frames), num_denoising_steps, device=noise_B_T_C_H_W.device
                )
            else:
                exit_flags = [num_denoising_steps - 1] * len(all_num_frames)



        start_gradient_frame_index = 0

        for block_index, current_num_frames in enumerate(all_num_frames):
            current_end_frame = current_start_frame + current_num_frames

            noise_B_V_T_C_H_W = rearrange(noise_B_T_C_H_W, "b (v t) c h w -> b v t c h w", v=n_views)
            noisy_input = noise_B_V_T_C_H_W[
                :,
                :,
                current_start_frame - num_input_frames : current_end_frame - num_input_frames,
            ]
            noisy_input = rearrange(noisy_input, "b v t c h w -> b (v t) c h w")


            denoising_step_list = (
                self.denoising_step_list[block_index]
                if self.config.independent_denoising_step_list
                else self.denoising_step_list
            )
            for index, current_timestep in enumerate(denoising_step_list):
                if self.config.same_step_across_blocks:
                    exit_flag = index == exit_flags[0]
                else:
                    exit_flag = (
                        index == exit_flags[block_index]
                    )
                timestep = (
                    torch.ones(
                        [batch_size, frames_per_temporal_block],
                        device=noise_B_T_C_H_W.device,
                        dtype=torch.int64,
                    )
                    * current_timestep
                )

                if not exit_flag:
                    with torch.no_grad():
                        denoised_pred = x0_fn(
                            noisy_image_or_video=noisy_input,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * token_stride_per_frame,
                            current_end=current_end_frame * token_stride_per_frame,
                            start_frame_for_rope=current_start_frame,
                        )
                        next_timestep = denoising_step_list[index + 1]
                        current_noise = torch.randn_like(denoised_pred.flatten(0, 1))
                        if cp_size > 1:
                            current_noise = broadcast(current_noise.contiguous(), cp_group)
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            current_noise,
                            next_timestep
                            * torch.ones(
                                [batch_size * frames_per_temporal_block],
                                device=noise_B_T_C_H_W.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])

                else:

                    if current_start_frame < start_gradient_frame_index or not is_training:
                        with torch.no_grad():
                            denoised_pred = x0_fn(
                                noisy_image_or_video=noisy_input,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * token_stride_per_frame,
                                current_end=current_end_frame * token_stride_per_frame,
                                start_frame_for_rope=current_start_frame,
                            )
                    else:
                        denoised_pred = x0_fn(
                            noisy_image_or_video=noisy_input,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * token_stride_per_frame,
                            current_end=current_end_frame * token_stride_per_frame,
                            start_frame_for_rope=current_start_frame,
                        )
                    break


            output_B_V_T_C_H_W = rearrange(output_B_T_C_H_W, "b (v t) c h w -> b v t c h w", v=n_views)
            output_B_V_T_C_H_W[:, :, current_start_frame:current_end_frame] = rearrange(
                denoised_pred, "b (v t) c h w -> b v t c h w", v=n_views
            )
            output_B_T_C_H_W = rearrange(output_B_V_T_C_H_W, "b v t c h w -> b (v t) c h w")


            context_timestep = torch.ones_like(timestep) * self.config.context_noise
            if self.config.context_noise > 0:

                current_noise = torch.randn_like(denoised_pred.flatten(0, 1))
                if cp_size > 1:
                    current_noise = broadcast(current_noise.contiguous(), cp_group)
                denoised_pred = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    current_noise,
                    context_timestep
                    * torch.ones(
                        [batch_size * frames_per_temporal_block],
                        device=noise_B_T_C_H_W.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])

            with torch.no_grad():
                x0_fn(
                    noisy_image_or_video=denoised_pred,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * token_stride_per_frame,
                    current_end=current_end_frame * token_stride_per_frame,
                    start_frame_for_rope=current_start_frame,
                )


            current_start_frame = current_end_frame



        denoised_timestep_from, denoised_timestep_to = None, None

        if is_training:
            return output_B_T_C_H_W, denoised_timestep_from, denoised_timestep_to
        else:
            return output_B_T_C_H_W.permute(0, 2, 1, 3, 4)
