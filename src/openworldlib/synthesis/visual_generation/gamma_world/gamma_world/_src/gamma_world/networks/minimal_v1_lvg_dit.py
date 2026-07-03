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

import math
from typing import List, Optional, Tuple

import torch
import torch.amp as amp
from einops import rearrange
from torch import nn
from torch.distributed._composable.fsdp import fully_shard

from gamma_world._src.imaginaire.utils import log
from gamma_world._src.predict2.conditioner import DataType
from gamma_world._src.predict2.networks.minimal_v4_dit import Block, MiniTrainDIT, SACConfig, VideoSize
from gamma_world._src.gamma_world.modules.multiagent_rope import build_multi_agent_freqs, precompute_freqs_cis_4d
from gamma_world._src.gamma_world.modules.action_encoder import (
    MultiAgentActionControlModule,
    build_multi_agent_action_control,
)

from gamma_world._src.gamma_world.modules.action_encoder import RMS_norm


class ActionBlock(Block):


    def __init__(
        self,
        *args,
        action_dim: int = 2048,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.action_dim = action_dim
        self.input_encoder = nn.Linear(action_dim, self.x_dim, bias=False)

    def init_weights(self) -> None:
        super().init_weights()
        std = 1.0 / math.sqrt(self.x_dim)
        nn.init.trunc_normal_(self.input_encoder.weight, std=std, a=-3 * std, b=3 * std)

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        kv_cache_cfg=None,
        action_bias_B_T_H_W_D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                    self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
            else:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
                )
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
                )
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                    self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)
                )

        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _fn(_x, _norm, _scale, _shift):
            return _norm(_x) * (1 + _scale) + _shift

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D, self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D, shift_self_attn_B_T_1_1_D,
        )

        assert action_bias_B_T_H_W_D is not None, "Action bias is None!!!"
        if action_bias_B_T_H_W_D is not None:
            normalized_x_B_T_H_W_D = normalized_x_B_T_H_W_D + self.input_encoder(action_bias_B_T_H_W_D)

        video_size = VideoSize(T=T, H=H, W=W)
        if self.cp_size is not None and self.cp_size > 1:
            video_size = VideoSize(T=T * self.cp_size, H=H, W=W)

        result_B_T_H_W_D = rearrange(
            self.self_attn(
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
                video_size=video_size,
                kv_cache_cfg=kv_cache_cfg,
            ),
            "b (t h w) d -> b t h w d", t=T, h=H, w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        def _x_fn(_x, _ln, _scale, _shift, _gate):
            _norm_x = _fn(_x, _ln, _scale, _shift)
            _result = rearrange(
                self.cross_attn(
                    rearrange(_norm_x, "b t h w d -> b (t h w) d"),
                    crossattn_emb,
                    rope_emb=rope_emb_L_1_1_D,
                ),
                "b (t h w) d -> b t h w d", t=T, h=H, w=W,
            )
            return _result

        result_B_T_H_W_D = _x_fn(
            x_B_T_H_W_D, self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D, shift_cross_attn_B_T_1_1_D, gate_cross_attn_B_T_1_1_D,
        )
        x_B_T_H_W_D = result_B_T_H_W_D * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D, self.layer_norm_mlp,
            scale_mlp_B_T_1_1_D, shift_mlp_B_T_1_1_D,
        )
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
        return x_B_T_H_W_D


class MinimalV1LVGDiT(MiniTrainDIT):
    def __init__(
        self,
        *args,
        timestep_scale: float = 1.0,
        num_views: int = 8,
        enable_action_control: bool = False,
        action_keyboard_dim: int = 23,
        action_camera_dim: int = 2,
        action_use_camera: bool = True,
        action_embed_dim: int = 256,
        action_temporal_downsample: int = 4,
        action_use_conv1d: bool = True,

        enable_view_embedding: bool = False,
        use_multi_agent_rope: bool = False,
        multi_agent_rope_num_agents: int = 2,
        multi_agent_rope_agent_id_offset: int = 0,
        multi_agent_rope_agent_encoding: str = "linear",
        multi_agent_rope_agent_scale: float = 1.0,
        multi_agent_rope_simplex_pool_size: int | None = None,
        multi_agent_rope_share_action_encoder: bool = False,







        action_spatial_concat: bool = False,
        **kwargs,
    ):
        assert not (enable_view_embedding and use_multi_agent_rope), (
            "enable_view_embedding and use_multi_agent_rope are alternative ways to encode "
            "agent identity (view embedding adds a per-frame embedding to patch tokens; "
            "4D RoPE encodes agent in the rotary phase). Enabling both at once is unintended."
        )
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1
        self.timestep_scale = timestep_scale


        if "num_layers" in kwargs:
            del kwargs["num_layers"]


        if enable_action_control:
            self._crossattn_emb_channels = kwargs.get("crossattn_emb_channels", 1024)
            self._mlp_ratio = kwargs.get("mlp_ratio", 4.0)
            saved_sac_config = kwargs.pop("sac_config", SACConfig())
            kwargs["sac_config"] = SACConfig(mode="none")

        super().__init__(*args, **kwargs)

        if enable_action_control:
            self.blocks = nn.ModuleList([
                ActionBlock(
                    x_dim=self.model_channels,
                    context_dim=self._crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self._mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=self.atten_backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                    action_dim=self.model_channels,
                )
                for _ in range(self.num_blocks)
            ])
            for block in self.blocks:
                block.init_weights()
            self.enable_selective_checkpoint(saved_sac_config, self.blocks)

        self.enable_view_embedding = enable_view_embedding
        self.view_embedding = nn.Embedding(num_views, self.model_channels)



        nn.init.zeros_(self.view_embedding.weight)

        self.use_multi_agent_rope = use_multi_agent_rope
        self.simplex_pool_size = (
            multi_agent_rope_simplex_pool_size
            if multi_agent_rope_simplex_pool_size is not None
            else multi_agent_rope_num_agents
        )
        assert self.simplex_pool_size >= multi_agent_rope_num_agents, (
            f"multi_agent_rope_simplex_pool_size ({self.simplex_pool_size}) must be "
            f">= multi_agent_rope_num_agents ({multi_agent_rope_num_agents})"
        )


        self._override_agent_pool_indices: list[int] | None = None

        if use_multi_agent_rope:
            self.freqs_4d = None
            self.multi_agent_token_layout = "sequence"
            self.num_agents = multi_agent_rope_num_agents
            self.agent_id_offset = multi_agent_rope_agent_id_offset
            self.agent_encoding = multi_agent_rope_agent_encoding
            self.agent_scale = multi_agent_rope_agent_scale
            assert self.agent_encoding in ("linear", "simplex"), (
                f"multi_agent_rope_agent_encoding must be 'linear' or 'simplex', got {self.agent_encoding!r}"
            )





            if self.agent_encoding == "simplex":
                assert self.agent_id_offset == 0, (
                    f"multi_agent_rope_agent_id_offset must be 0 under simplex encoding "
                    f"(simplex vertices are already non-degenerate); got {self.agent_id_offset}. "
                    f"Set multi_agent_rope_agent_id_offset=0 explicitly in your simplex experiment "
                    f"config to override any value inherited from a linear-RoPE parent."
                )
        else:
            self.num_agents = multi_agent_rope_num_agents

        self.enable_action_control = enable_action_control

        self.action_spatial_concat = action_spatial_concat
        if action_spatial_concat:
            assert not use_multi_agent_rope, (
                "action_spatial_concat=True requires n_views=1, but "
                "use_multi_agent_rope=True requires n_views > 1; these are "
                "mutually exclusive. Disable use_multi_agent_rope for the "
                "spatial-concat ablation."
            )
            assert not multi_agent_rope_share_action_encoder, (
                "action_spatial_concat=True already uses a single shared encoder; "
                "do not also set multi_agent_rope_share_action_encoder=True."
            )
        self.multi_agent_action_control: MultiAgentActionControlModule | None = None
        if enable_action_control:
            self.multi_agent_action_control = build_multi_agent_action_control(
                num_agents=multi_agent_rope_num_agents,
                pool_size=self.simplex_pool_size,
                share_encoder=multi_agent_rope_share_action_encoder,
                spatial_concat=action_spatial_concat,
                keyboard_dim=action_keyboard_dim,
                camera_dim=action_camera_dim,
                use_camera=action_use_camera,
                embed_dim=action_embed_dim,
                dit_dim=self.model_channels,
                temporal_downsample=action_temporal_downsample,
                use_conv1d=action_use_conv1d,
            )

    def init_weights(self) -> None:

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                m.reset_parameters()
            elif isinstance(m, nn.Embedding):
                nn.init.zeros_(m.weight)


        super().init_weights()

        for block in self.blocks:
            block.init_weights()


        if hasattr(self, "view_embedding"):
            nn.init.zeros_(self.view_embedding.weight)
        if hasattr(self, "multi_agent_action_control") and self.multi_agent_action_control is not None:
            for m in self.multi_agent_action_control.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    if m.weight is not None:
                        nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, RMS_norm):
                    nn.init.ones_(m.gamma)
                    if isinstance(m.bias, nn.Parameter):
                        nn.init.zeros_(m.bias)




    def _resolve_agent_pool_indices(self, n_views: int) -> list[int]:

        if self._override_agent_pool_indices is not None:
            indices = list(self._override_agent_pool_indices)
            assert len(indices) == n_views, (
                f"_override_agent_pool_indices length {len(indices)} does not "
                f"match n_views={n_views}"
            )
            assert max(indices) < self.simplex_pool_size, (
                f"_override_agent_pool_indices max {max(indices)} out of range "
                f"for simplex_pool_size={self.simplex_pool_size}"
            )
            assert len(set(indices)) == n_views, (
                f"_override_agent_pool_indices must be unique, got {indices}"
            )
            return indices
        assert n_views <= self.simplex_pool_size, (
            f"n_views={n_views} exceeds simplex_pool_size={self.simplex_pool_size}; "
            f"either pass _override_agent_pool_indices or increase the pool"
        )
        return list(range(n_views))

    def _generate_multi_agent_rope(
        self,
        B_T_H_W_D: tuple[int, int, int, int, int],
        n_views: int,
        device: torch.device,
        fps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B, T, H, W, D = B_T_H_W_D
        t_per_view = T // n_views


        if self.freqs_4d is None:
            head_dim = self.model_channels // self.num_heads
            self.freqs_4d = precompute_freqs_cis_4d(
                head_dim,
                agent_encoding=self.agent_encoding,
                num_agents=self.simplex_pool_size,
                agent_scale=self.agent_scale,
            )

        agent_pool_indices = self._resolve_agent_pool_indices(n_views)
        complex_freqs = build_multi_agent_freqs(
            self, t_per_view, H, W, num_agents=n_views, agent_pool_indices=agent_pool_indices,
        )


        angles = torch.angle(complex_freqs)
        angles = angles.unsqueeze(2)
        return torch.cat([angles, angles], dim=-1).to(device=device, dtype=torch.float32)

    def _generate_per_view_rope(
        self,
        B_T_H_W_D: tuple[int, int, int, int, int],
        view_indices_B_T: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B, T, H, W, D = B_T_H_W_D
        n_views = view_indices_B_T.max().item() + 1

        if self.use_multi_agent_rope:
            return self._generate_multi_agent_rope(
                B_T_H_W_D, n_views=n_views, device=view_indices_B_T.device, fps=fps,
            )

        t_per_view = T // n_views
        single_view_rope = self.pos_embedder.generate_embeddings((B, t_per_view, H, W, D), fps=fps)

        single_view_rope = rearrange(single_view_rope, "(t hw) 1 1 d -> t hw 1 1 d", t=t_per_view)
        rope_emb = single_view_rope.repeat(n_views, 1, 1, 1, 1)
        return rearrange(rope_emb, "t hw 1 1 d -> (t hw) 1 1 d")

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
        action_inputs: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs
        assert isinstance(data_type, DataType)

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )

        timesteps_B_T = timesteps_B_T * self.timestep_scale

        x_B_T_H_W_D, base_rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask,
        )

        use_view_embedding = self.enable_view_embedding
        use_action = action_inputs is not None and self.multi_agent_action_control is not None

        if use_view_embedding:
            assert view_indices_B_T is not None, (
                "enable_view_embedding=True requires view_indices_B_T to be passed "
                "(produced by MultiViewCausalConditioner from latent_view_indices_B_T)."
            )
            max_idx = view_indices_B_T.max().item()
            min_idx = view_indices_B_T.min().item()
            num_views = self.view_embedding.num_embeddings
            assert 0 <= min_idx and max_idx < num_views, (
                f"view_indices out of range: min={min_idx}, max={max_idx}, num_views={num_views}, "
                f"shape={view_indices_B_T.shape}, dtype={view_indices_B_T.dtype}"
            )
            view_emb_B_T_D = self.view_embedding(view_indices_B_T)
            x_B_T_H_W_D = x_B_T_H_W_D + view_emb_B_T_D[:, :, None, None, :]

        action_bias_B_T_H_W_D = None
        if use_action:
            B, T, H, W, D = x_B_T_H_W_D.shape




            if self.action_spatial_concat:
                n_views = 1
            else:
                n_views = self.num_agents
                assert n_views == self.multi_agent_action_control.num_agents, (
                    f"action control expects n_views={self.multi_agent_action_control.num_agents}, "
                    f"got n_views={n_views}"
                )
            agent_pool_indices = self._resolve_agent_pool_indices(n_views)
            action_bias = self.multi_agent_action_control(
                actions=action_inputs["actions"],
                x_shape=(B, D, T, H, W),
                device=x_B_T_H_W_D.device,
                dtype=x_B_T_H_W_D.dtype,
                n_views=n_views,
                agent_pool_indices=agent_pool_indices,
            )
            action_bias_B_T_H_W_D = action_bias.reshape(B, T, H, W, D)

        if use_view_embedding:
            rope_emb_L_1_1_D = self._generate_per_view_rope(
                x_B_T_H_W_D.shape, view_indices_B_T, fps=fps,
            )
        elif self.use_multi_agent_rope:
            rope_emb_L_1_1_D = self._generate_multi_agent_rope(
                x_B_T_H_W_D.shape,
                n_views=self.num_agents,
                device=x_B_T_H_W_D.device,
                fps=fps,
            )
        else:
            rope_emb_L_1_1_D = base_rope_emb_L_1_1_D

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        action_kwargs = {}
        if action_bias_B_T_H_W_D is not None:
            action_kwargs["action_bias_B_T_H_W_D"] = action_bias_B_T_H_W_D

        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                **action_kwargs,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped_for_disc)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        if intermediate_feature_ids:
            if len(intermediate_features_outputs) != len(intermediate_feature_ids):
                log.warning(
                    f"Collected {len(intermediate_features_outputs)} intermediate features, "
                    f"but expected {len(intermediate_feature_ids)}. "
                    f"Requested IDs: {intermediate_feature_ids}"
                )
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

        return x_B_C_Tt_Hp_Wp

    def fully_shard(self, mesh, **fsdp_kwargs) -> None:
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)


        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        if self.extra_per_block_abs_pos_emb:
            fully_shard(self.extra_pos_embedder, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)


        if self.extra_image_context_dim is not None:
            fully_shard(self.img_context_proj, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)
