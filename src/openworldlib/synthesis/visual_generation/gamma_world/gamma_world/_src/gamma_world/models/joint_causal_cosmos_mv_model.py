# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Callable, Dict, List, Optional, cast

import attrs
import torch
import torch.distributed as dist
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed import get_process_group_ranks

from gamma_world._src.imaginaire.utils import log, misc
from gamma_world._src.imaginaire.utils.context_parallel import broadcast_split_tensor
from gamma_world._src.predict2.conditioner import DataType
from gamma_world._src.predict2_multiview.configs.vid2vid.defaults.conditioner import (
    ConditionLocation,
    ConditionLocationList,
    MultiViewCondition,
)
from gamma_world._src.predict2_multiview.models.multiview_vid2vid_model_rectified_flow import (
    compute_empty_and_negative_text_embeddings,
    compute_text_embeddings_online_multiview,
    preprocess_databatch,
)
from gamma_world._src.gamma_world.models.joint_causal_cosmos_model import (
    CausalJointCosmosModel,
    CausalJointCosmosModelConfig,
)


@attrs.define(slots=False)
class CausalJointCosmosMVModelConfig(CausalJointCosmosModelConfig):
    min_num_conditional_frames_per_view: int = 0
    max_num_conditional_frames_per_view: int = 2
    train_sample_views_range: tuple[int, int] | None = None
    condition_locations: ConditionLocationList = field(
        default_factory=lambda: ConditionLocationList([ConditionLocation.FIRST_RANDOM_N])
    )
    state_t: int = 0
    view_condition_dropout_max: int = 0
    conditional_frames_probs: Optional[Dict[int, float]] = None




    shuffle_agents: bool = False












    view_id_pool_size: int = 0


class CausalJointCosmosMVModel(CausalJointCosmosModel):
    def __init__(self, config: CausalJointCosmosMVModelConfig):
        super().__init__(config)
        self.config: CausalJointCosmosMVModelConfig = config
        self.state_t = config.state_t
        self.empty_string_text_embeddings = None
        self.neg_text_embeddings = None
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            compute_empty_and_negative_text_embeddings(self)

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

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, MultiViewCondition]:
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        raw_state, latent_state, condition = super(CausalJointCosmosModel, self).get_data_and_condition(
            data_batch_with_latent_view_indices
        )
        condition = cast(MultiViewCondition, condition)
        condition = condition.set_video_condition(
            state_t=self.state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=None,
            view_condition_dropout_max=self.config.view_condition_dropout_max,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition

    def sample_train_time(self, batch_size=None, num_frames=None, state_shape=None) -> torch.Tensor:
        if state_shape is not None:
            batch_size = state_shape[0]
            num_frames = state_shape[2]
        else:
            assert batch_size is not None and num_frames is not None, "batch_size and num_frames must be provided"
        n_views = state_shape[2] // self.state_t
        t_B_T = self.rectified_flow.sample_train_time(batch_size * num_frames // n_views).to(
            **self.flow_matching_kwargs
        )
        t_B_T = t_B_T.reshape(batch_size, 1, -1, self.num_frame_per_block)
        t_B_T = t_B_T.repeat(1, n_views, 1, 1)
        t_B_T[:, :, :, 1:] = t_B_T[:, :, :, 0:1]
        t_B_T = t_B_T.reshape(batch_size, -1)
        return t_B_T

    def inplace_compute_text_embeddings_online(self, data_batch: dict[str, torch.Tensor]):
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

    def _remap_view_ids_from_pool_inplace(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> None:

        pool_size = self.config.view_id_pool_size
        if pool_size <= 0:
            return
        if "view_indices" not in data_batch:
            return

        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
        n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view
        if pool_size <= n_views:
            return

        net_for_check = self.net
        view_emb = getattr(net_for_check, "view_embedding", None)
        if view_emb is not None:
            assert view_emb.num_embeddings >= pool_size, (
                f"view_id_pool_size={pool_size} exceeds view_embedding table size "
                f"{view_emb.num_embeddings}; increase net.num_views or lower pool_size."
            )

        if parallel_state.is_initialized():
            dp_rank = parallel_state.get_data_parallel_rank()
        else:
            dp_rank = 0
        seed = ((int(iteration) * 1_000_037) ^ (int(dp_rank) * 7907) ^ 0xDEADBEEF) & 0x7FFFFFFF
        gen = torch.Generator()
        gen.manual_seed(seed)
        chosen = torch.randperm(pool_size, generator=gen)[:n_views]

        vi = data_batch["view_indices"]
        chosen_dev = chosen.to(device=vi.device, dtype=vi.dtype)
        data_batch["view_indices"] = chosen_dev[vi]

    def _set_net_agent_pool_indices(self, indices: list[int] | None) -> None:

        targets = [self.net]
        if hasattr(self, "net_ema") and self.net_ema is not None:
            targets.append(self.net_ema)
        for net in targets:
            assert hasattr(net, "_override_agent_pool_indices"), (
                "Underlying net does not support simplex agent pool sampling "
                "(missing `_override_agent_pool_indices` attribute)."
            )
            net._override_agent_pool_indices = indices

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        data_batch = preprocess_databatch(data_batch, self.config.train_sample_views_range)
        self._shuffle_agents_inplace(data_batch)
        self._remap_view_ids_from_pool_inplace(data_batch, iteration)
        indices = self._sample_agent_pool_indices(data_batch, iteration)
        self._set_net_agent_pool_indices(indices)
        try:
            output_batch, loss = super().training_step(data_batch, iteration)
        finally:
            self._set_net_agent_pool_indices(None)
        return output_batch, loss

    @torch.no_grad()
    def validation_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        data_batch = preprocess_databatch(data_batch, self.config.train_sample_views_range)



        self._set_net_agent_pool_indices(None)
        output_batch, loss = super().validation_step(data_batch, iteration)
        return output_batch, loss

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        n_views: int,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)

        if "num_conditional_frames" in data_batch_with_latent_view_indices:
            num_conditional_frames = data_batch_with_latent_view_indices["num_conditional_frames"]
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

        condition = condition.set_video_condition(
            state_t=self.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            state_t=self.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(
            is_cfg_conditional=True,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False,
            condition_locations=self.config.condition_locations,
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


            def _unfold(tensor_flat: torch.Tensor) -> torch.Tensor:
                return rearrange(tensor_flat, "B C (V T) H W -> B V C T H W", V=n_views)

            def _fold(tensor_unfold: torch.Tensor) -> torch.Tensor:
                return rearrange(tensor_unfold, "B V C T H W -> B C (V T) H W")

            noise_unfold = _unfold(noise_x)
            gt_frames_unfold: torch.Tensor | None = None
            if condition.gt_frames is not None:
                gt_frames_unfold = _unfold(condition.gt_frames)

            new_condition = condition
            new_uncondition = uncondition
            start_frame = kwargs.get("start_frame_for_rope", 0)
            end_frame = start_frame + noise_unfold.shape[3]
            new_condition_dict = condition.to_dict()
            new_uncondition_dict = uncondition.to_dict()

            if gt_frames_unfold is not None and gt_frames_unfold.shape[3] != noise_unfold.shape[3]:
                assert kwargs.get("start_frame_for_rope", None) is not None, "start_frame_for_rope is not provided"
                sliced_gt = gt_frames_unfold[:, :, :, start_frame:end_frame, :, :]
                new_condition_dict["gt_frames"] = _fold(sliced_gt)
                if condition.condition_video_input_mask_B_C_T_H_W is not None:
                    mask_unfold = _unfold(condition.condition_video_input_mask_B_C_T_H_W)
                    sliced_mask = mask_unfold[:, :, :, start_frame:end_frame, :, :]
                    new_condition_dict["condition_video_input_mask_B_C_T_H_W"] = _fold(sliced_mask)
                if guidance != 1.0:
                    if uncondition.gt_frames is not None:
                        un_gt_unfold = _unfold(uncondition.gt_frames)
                        sliced_un_gt = un_gt_unfold[:, :, :, start_frame:end_frame, :, :]
                        new_uncondition_dict["gt_frames"] = _fold(sliced_un_gt)
                    if uncondition.condition_video_input_mask_B_C_T_H_W is not None:
                        un_mask_unfold = _unfold(uncondition.condition_video_input_mask_B_C_T_H_W)
                        sliced_un_mask = un_mask_unfold[:, :, :, start_frame:end_frame, :, :]
                        new_uncondition_dict["condition_video_input_mask_B_C_T_H_W"] = _fold(sliced_un_mask)

            if hasattr(condition, "view_indices_B_T") and condition.view_indices_B_T is not None:
                view_indices_unfold = rearrange(condition.view_indices_B_T, "B (V T) -> B V T", V=n_views)
                if view_indices_unfold.shape[2] != noise_unfold.shape[3]:
                    new_condition_dict["view_indices_B_T"] = rearrange(
                        view_indices_unfold[:, :, start_frame:end_frame], "B V T -> B (V T)"
                    )

                    if guidance != 1.0:
                        if hasattr(uncondition, "view_indices_B_T") and uncondition.view_indices_B_T is not None:
                            flat_view_indices_B_V_T = rearrange(
                                uncondition.view_indices_B_T, "B (V T) -> B V T", V=n_views
                            )
                            new_uncondition_dict["view_indices_B_T"] = rearrange(
                                flat_view_indices_B_V_T[:, :, start_frame:end_frame], "B V T -> B (V T)"
                            )

            new_condition = type(condition)(**new_condition_dict)
            if guidance != 1.0:
                new_uncondition = type(uncondition)(**new_uncondition_dict)

            noise_fold = _fold(noise_unfold)

            cond_v = self.denoise(
                xt_B_C_T_H_W=noise_fold,
                timesteps_B_T=timestep,
                condition=new_condition,
                kv_cache=kv_cache,
                noise=noise,
                action_inputs=action_inputs,
                n_views=n_views,
                **kwargs,
            )

            if guidance != 1.0 and not skip_uncond:
                uncond_v = self.denoise(
                    xt_B_C_T_H_W=noise_fold,
                    timesteps_B_T=timestep,
                    condition=new_uncondition,
                    kv_cache=kv_cache_uncond,
                    noise=noise,
                    action_inputs=action_inputs,
                    n_views=n_views,
                    **kwargs,
                )
                velocity_pred = uncond_v + guidance * (cond_v - uncond_v)
            else:
                velocity_pred = cond_v

            return velocity_pred

        return velocity_fn

    def _initialize_kv_cache(
        self,
        batch_size: int,
        n_views: int,
        dtype: torch.dtype,
        device: torch.device | str,
        n_steps: int = 1,
        use_uncond_kvcache: bool = False,
    ):

        local_attn_size = getattr(self.net, "local_attn_size", -1)
        v_split_mode = getattr(self.net, "v_split_mode", False)
        use_sparse_hub = getattr(self.net, "use_sparse_hub", False)
        z_num = getattr(self.net, "z_num", 0)

        log.info(f"Initializing multiview KV cache:")
        log.info(f"  batch_size: {batch_size}, n_views: {n_views}")
        log.info(f"  local_attn_size: {local_attn_size}")
        log.info(f"  v_split_mode: {v_split_mode}")
        log.info(f"  use_sparse_hub: {use_sparse_hub}, z_num: {z_num}")
        log.info(f"  frame_seq_length: {self.frame_seq_length}")

        if use_sparse_hub:












            assert not v_split_mode, "sparse_hub is not compatible with v_split_mode"
            assert self.cp_size is None or self.cp_size == 1, (
                f"sparse_hub inference does not support CP yet (cp_size={self.cp_size}); "
                f"the player/Z layout would be broken by contiguous CP split"
            )

            head_dim = self.config.net.model_channels // self.num_transformer_heads
            if local_attn_size != -1:
                per_view_frames = local_attn_size
            else:
                per_view_frames = self.max_latent_frames_per_gpu
            max_player_tokens_per_view = self.frame_seq_length * per_view_frames
            max_z_tokens = z_num * per_view_frames

            log.info(
                f"  sparse_hub cache: V={n_views}, per_view_frames={per_view_frames}, "
                f"max_player_tokens_per_view={max_player_tokens_per_view}, "
                f"max_z_tokens={max_z_tokens}"
            )

            def _make_sparse_cache():
                return {
                    "k_players": torch.zeros(
                        [batch_size, n_views, max_player_tokens_per_view, self.num_transformer_heads, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "v_players": torch.zeros(
                        [batch_size, n_views, max_player_tokens_per_view, self.num_transformer_heads, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "k_z": torch.zeros(
                        [batch_size, max_z_tokens, self.num_transformer_heads, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "v_z": torch.zeros(
                        [batch_size, max_z_tokens, self.num_transformer_heads, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "z_local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }

            self.kv_cache1 = dict()
            for step_index in range(n_steps):
                self.kv_cache1[step_index] = [
                    _make_sparse_cache() for _ in range(self.num_transformer_blocks)
                ]

            if use_uncond_kvcache:
                self.kv_cache2 = dict()
                for step_index in range(n_steps):
                    self.kv_cache2[step_index] = [
                        _make_sparse_cache() for _ in range(self.num_transformer_blocks)
                    ]
            return




        if local_attn_size == -1:

            kv_cache_size = self.frame_seq_length * self.max_latent_frames_per_gpu * n_views
        else:
            if local_attn_size > self.max_latent_frames_per_gpu:
                raise ValueError(
                    f"local_attn_size {local_attn_size} is larger than max_latent_frames_per_gpu "
                    f"{self.max_latent_frames_per_gpu}, which is not supported"
                )
            kv_cache_size = self.frame_seq_length * local_attn_size * n_views


        if self.cp_size is not None and not v_split_mode:
            assert kv_cache_size % self.cp_size == 0, "kv_cache_size must be divisible by cp_size"
            kv_cache_size = kv_cache_size // self.cp_size



        effective_batch_size = batch_size
        log.info(f"  effective_batch_size for KV cache: {effective_batch_size}")
        log.info(f"  kv_cache_size (sequence length): {kv_cache_size}")

        if n_steps > 1:
            log.info(f"Using step-dependent KV cache with {n_steps} steps")
        else:
            log.info("Using step-independent KV cache")

        head_dim = self.config.net.model_channels // self.num_transformer_heads

        self.kv_cache1 = dict()
        for step_index in range(n_steps):
            kv_cache1 = []
            for _ in range(self.num_transformer_blocks):
                kv_cache1.append(
                    {
                        "k": torch.zeros(
                            [effective_batch_size, int(kv_cache_size), self.num_transformer_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "v": torch.zeros(
                            [effective_batch_size, int(kv_cache_size), self.num_transformer_heads, head_dim],
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
                                [effective_batch_size, int(kv_cache_size), self.num_transformer_heads, head_dim],
                                dtype=dtype,
                                device=device,
                            ),
                            "v": torch.zeros(
                                [effective_batch_size, int(kv_cache_size), self.num_transformer_heads, head_dim],
                                dtype=dtype,
                                device=device,
                            ),
                            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        }
                    )
                self.kv_cache2[step_index] = kv_cache2

    def _initialize_crossattn_cache(
        self,
        batch_size: int,
        n_views: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):

        del n_views
        effective_batch_size = batch_size
        head_dim = self.config.net.model_channels // self.num_transformer_heads

        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [effective_batch_size, 512, 12, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [effective_batch_size, 512, 12, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "is_init": False,
                }
            )

        self.crossattn_cache = crossattn_cache

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: tuple | None = None,
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
        separate_kvcache_timestep_int: int | None = None,
        **kwargs,
    ) -> torch.Tensor:

        import time

        from tqdm import tqdm

        from gamma_world._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0

        self._normalize_video_databatch_inplace(data_batch)
        if hasattr(self, "_augment_image_dim_inplace"):
            self._augment_image_dim_inplace(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]

        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
        n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view
        original_state_t = self.state_t

        self.net.state_t = self.num_frame_per_block

        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T // n_views),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]
        else:
            state_shape = [state_shape[0], state_shape[1] // n_views, state_shape[2], state_shape[3]]

        latent_frames_per_view = state_shape[1]


        flat_state_shape = (
            state_shape[0],
            latent_frames_per_view * n_views,
            state_shape[2],
            state_shape[3],
        )

        noise_B_C_VT_H_W = misc.arch_invariant_rand(
            (n_sample,) + tuple(flat_state_shape),
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
        if cp_group is not None:
            from gamma_world._src.imaginaire.utils.context_parallel import broadcast

            noise_B_C_VT_H_W = broadcast(noise_B_C_VT_H_W.contiguous(), cp_group)
            if start_latents is not None:
                start_latents = broadcast(start_latents.contiguous(), cp_group)
        else:
            assert not getattr(self.net, "is_context_parallel_enabled", False), (
                "context parallel should be disabled if parallel_state is not initialized"
            )

        if cp_group is not None and not is_tp_cp_pp_rank0():
            verbose = False


        if use_uncond_kvcache is None:
            use_uncond_kvcache = False if self.noise_scheme == "teacher_forcing" else True

        velocity_fn = self.get_velocity_fn_from_batch(
            data_batch, n_views=n_views, guidance=guidance, is_negative_prompt=is_negative_prompt
        )

        def denoise_fn(
            noisy_latent: torch.Tensor,
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
            velocity_pred = velocity_fn(
                noisy_latent,
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
            return velocity_pred


        batch_size, num_channels, num_frames_total, height, width = noise_B_C_VT_H_W.shape
        output_B_C_VT_H_W = torch.zeros(
            [batch_size, num_channels, num_frames_total, height, width],
            device=noise_B_C_VT_H_W.device,
            dtype=noise_B_C_VT_H_W.dtype,
        )



        frames_per_temporal_block = n_views * self.num_frame_per_block
        num_temporal_blocks = latent_frames_per_view // self.num_frame_per_block



        v_split_mode = getattr(self.net, "v_split_mode", False)
        if v_split_mode and self.cp_size is not None and self.cp_size > 1:
            n_views_local = n_views // self.cp_size
        else:
            n_views_local = n_views

        if self.kv_cache1 is None:
            n_kvcache_steps = num_steps if use_step_dependent_kv_cache else 1
            if use_step_dependent_kv_cache and compute_separate_kvcache:
                n_kvcache_steps += 1
            self._initialize_kv_cache(
                batch_size=batch_size,
                n_views=n_views_local,
                dtype=self.tensor_kwargs["dtype"],
                device=self.tensor_kwargs["device"],
                n_steps=n_kvcache_steps,
                use_uncond_kvcache=use_uncond_kvcache,
            )









            self.crossattn_cache = None
        else:
            if guidance == 1.0 and hasattr(self, "crossattn_cache") and self.crossattn_cache is not None:

                for block_idx in range(self.num_transformer_blocks):
                    self.crossattn_cache[block_idx]["is_init"] = False
            else:
                self.crossattn_cache = None


            def _reset_one_cache(cache_dict: dict) -> None:
                cache_dict["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise_B_C_VT_H_W.device
                )
                cache_dict["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise_B_C_VT_H_W.device
                )
                if "z_local_end_index" in cache_dict:
                    cache_dict["z_local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise_B_C_VT_H_W.device
                    )

            for step_index in list(self.kv_cache1.keys()):
                for block_idx in range(len(self.kv_cache1[step_index])):
                    _reset_one_cache(self.kv_cache1[step_index][block_idx])

            if use_uncond_kvcache and self.kv_cache2 is not None:
                for step_index in list(self.kv_cache2.keys()):
                    for block_idx in range(len(self.kv_cache2[step_index])):
                        _reset_one_cache(self.kv_cache2[step_index][block_idx])



        for temporal_block_idx in tqdm(
            range(num_temporal_blocks), desc="Denoising temporal blocks", disable=not verbose
        ):
            if verbose:
                time_block_start = time.time()




            block_start_per_view = temporal_block_idx * self.num_frame_per_block
            block_end_per_view = (temporal_block_idx + 1) * self.num_frame_per_block



            noise_reshaped = rearrange(noise_B_C_VT_H_W, "b c (v t) h w -> b c v t h w", v=n_views)
            block_noise = noise_reshaped[:, :, :, block_start_per_view:block_end_per_view, :, :]
            block_noise_flat = rearrange(block_noise, "b c v t h w -> b c (v t) h w")

            latent_model_input = block_noise_flat.clone()

            self.sample_scheduler.config.shift = shift
            self.sample_scheduler.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=shift)
            timesteps = self.sample_scheduler.timesteps


            for step_idx, current_timestep in enumerate(timesteps):
                if verbose:
                    time_denoising_start = time.time()


                timestep = (
                    torch.ones(
                        [batch_size, frames_per_temporal_block], device=noise_B_C_VT_H_W.device, dtype=torch.int64
                    )
                    * current_timestep
                )
                kv_cache_step_index = step_idx if use_step_dependent_kv_cache else 0






                frames_per_temporal_block_token_stride = (
                    n_views * self.num_frame_per_block * self.frame_seq_length
                )
                current_start = temporal_block_idx * frames_per_temporal_block_token_stride
                current_end = (temporal_block_idx + 1) * frames_per_temporal_block_token_stride

                start_frame_for_rope = temporal_block_idx * self.num_frame_per_block

                velocity_field_pred = denoise_fn(
                    latent_model_input,
                    timestep,
                    kv_cache=self.kv_cache1[kv_cache_step_index],
                    kv_cache_uncond=self.kv_cache2[kv_cache_step_index] if use_uncond_kvcache else None,
                    crossattn_cache=self.crossattn_cache if hasattr(self, "crossattn_cache") else None,
                    current_start=current_start,
                    current_end=current_end,
                    start_frame_for_rope=start_frame_for_rope,
                    noise=block_noise_flat,
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
                    log.info(
                        f"[Step {step_idx}] Finish one denoising step in {time.time() - time_denoising_start:.2f}s"
                    )



            output_reshaped = rearrange(output_B_C_VT_H_W, "b c (v t) h w -> b c v t h w", v=n_views)
            output_block = rearrange(latent_model_input, "b c (v t) h w -> b c v t h w", v=n_views)
            output_reshaped[:, :, :, block_start_per_view:block_end_per_view, :, :] = output_block
            output_B_C_VT_H_W = rearrange(output_reshaped, "b c v t h w -> b c (v t) h w")

            if compute_separate_kvcache:
                if self.noise_scheme == "teacher_forcing":
                    t_kv_cache = 0
                else:
                    t_kv_cache = current_timestep

                if separate_kvcache_timestep_int is not None:
                    t_kv_cache = separate_kvcache_timestep_int

                timestep_kv_cache = (
                    torch.ones(
                        [batch_size, frames_per_temporal_block], device=noise_B_C_VT_H_W.device, dtype=torch.int64
                    )
                    * t_kv_cache
                )
                kv_cache_step_index = num_steps if use_step_dependent_kv_cache else 0

                if self.noise_scheme == "teacher_forcing" or t_kv_cache == 0:
                    noised_latent_model_input = latent_model_input
                else:
                    noise_input = misc.arch_invariant_rand(
                        block_noise_flat.shape,
                        torch.float32,
                        self.tensor_kwargs["device"],
                        seed + temporal_block_idx * 42,
                    )
                    noised_latent_model_input = self.sample_scheduler.add_noise(
                        latent_model_input,
                        noise_input,
                        torch.tensor([t_kv_cache], device=noise_B_C_VT_H_W.device, dtype=torch.int64),
                    )

                denoise_fn(
                    noised_latent_model_input,
                    timestep_kv_cache,
                    kv_cache=self.kv_cache1[kv_cache_step_index],
                    kv_cache_uncond=self.kv_cache2[kv_cache_step_index] if use_uncond_kvcache else None,
                    crossattn_cache=self.crossattn_cache if hasattr(self, "crossattn_cache") else None,
                    current_start=current_start,
                    current_end=current_end,
                    start_frame_for_rope=start_frame_for_rope,
                    skip_uncond=False if use_uncond_kvcache else True,
                )

            if verbose:
                log.info(
                    f"[Block {temporal_block_idx}] Finished temporal block ({n_views} views x "
                    f"{self.num_frame_per_block} frames) in {time.time() - time_block_start:.2f}s"
                )


        self.net.state_t = original_state_t

        return output_B_C_VT_H_W
