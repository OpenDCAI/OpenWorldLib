# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        orig_dtype = x.dtype
        x_norm = F.normalize(x.float(), dim=(1 if self.channel_first else -1)).to(orig_dtype)
        return x_norm * self.scale * self.gamma + self.bias


class SolarisActionEncoder(nn.Module):
    def __init__(
        self,
        keyboard_dim=23,
        camera_dim=2,
        hidden_size=128,
        embed_dim=256,
        out_dim=3072,
        temporal_downsample=4,
        use_camera=True,
        use_conv1d=False,
    ):
        super().__init__()

        self.keyboard_embed = nn.Sequential(
            nn.Linear(keyboard_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.camera_embed = nn.Sequential(
            nn.Linear(camera_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.use_conv1d = use_conv1d
        if self.use_conv1d:
            self.temporal_conv = nn.Conv1d(
                embed_dim, embed_dim,
                kernel_size=temporal_downsample,
                stride=temporal_downsample,
                padding=0,
            )
        else:
            self.temporal_conv = nn.Sequential(
                RMS_norm(embed_dim, images=False),
                nn.SiLU(),
                nn.Conv3d(embed_dim, embed_dim, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
                RMS_norm(embed_dim, images=False),
                nn.SiLU(),
                nn.Conv3d(embed_dim, embed_dim, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            )

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, keyboard, camera):
        kb_emb = self.keyboard_embed(keyboard)
        cam_emb = self.camera_embed(camera)
        fused = self.fusion(torch.cat([kb_emb, cam_emb], dim=-1))

        if self.use_conv1d:
            fused = fused.transpose(1, 2)
            fused = self.temporal_conv(fused)
            fused = fused.transpose(1, 2)
        else:
            fused = fused.transpose(1, 2)
            fused = fused.unsqueeze(-1).unsqueeze(-1)
            fused = self.temporal_conv(fused)
            fused = fused.squeeze(-1).squeeze(-1)
            fused = fused.transpose(1, 2)

        return self.proj(fused)

def _resample_action_emb_to_latent_t(
    emb: torch.Tensor, latent_f_per_view: int, use_conv1d: bool
) -> torch.Tensor:
    if use_conv1d:
        return F.interpolate(
            emb.transpose(1, 2), size=latent_f_per_view, mode="linear", align_corners=False
        ).transpose(1, 2)
    assert emb.shape[1] == latent_f_per_view, (
        f"emb.shape[1]: {emb.shape[1]} != latent_f_per_view: {latent_f_per_view}"
    )
    return emb


class MultiAgentActionControlModule(nn.Module):
    def __init__(
        self,
        keyboard_dim: int = 23,
        camera_dim: int = 2,
        use_camera: bool = True,
        embed_dim: int = 256,
        dit_dim: int = 2048,
        temporal_downsample: int = 4,
        use_conv1d: bool = False,
    ):
        super().__init__()
        self.dit_dim = dit_dim
        self.num_agents = 2

        self.left_action_encoder = SolarisActionEncoder(
            keyboard_dim=keyboard_dim,
            camera_dim=camera_dim,
            use_camera=use_camera,
            embed_dim=embed_dim,
            out_dim=dit_dim,
            temporal_downsample=temporal_downsample,
            use_conv1d=use_conv1d,
        )
        self.right_action_encoder = SolarisActionEncoder(
            keyboard_dim=keyboard_dim,
            camera_dim=camera_dim,
            use_camera=use_camera,
            embed_dim=embed_dim,
            out_dim=dit_dim,
            temporal_downsample=temporal_downsample,
            use_conv1d=use_conv1d,
        )

        self.use_conv1d = use_conv1d

    def forward(
        self,
        actions: list[dict[str, torch.Tensor]],
        x_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        n_views: int | None = None,
        start_frame: int = 0,
        agent_pool_indices: list[int] | None = None,
    ) -> torch.Tensor:
        assert len(actions) == self.num_agents, (
            f"MultiAgentActionControlModule expects {self.num_agents} agents, "
            f"got len(actions)={len(actions)}"
        )
        assert n_views is not None and n_views == self.num_agents, (
            f"n_views must equal num_agents={self.num_agents}, got n_views={n_views}"
        )
        if agent_pool_indices is not None and list(agent_pool_indices) != [0, 1]:
            raise AssertionError(
                f"MultiAgentActionControlModule (2-player legacy) does not support "
                f"agent_pool_indices={agent_pool_indices}; rebuild with the N-agents "
                f"module if a pool > 2 is required."
            )
        left_action, right_action = actions[0], actions[1]

        batch_size = left_action["keyboard"].shape[0]
        _, _, f, h, w = x_shape

        left_emb = self.left_action_encoder(
            left_action["keyboard"].to(device=device, dtype=dtype),
            left_action["camera"].to(device=device, dtype=dtype),
        )
        right_emb = self.right_action_encoder(
            right_action["keyboard"].to(device=device, dtype=dtype),
            right_action["camera"].to(device=device, dtype=dtype),
        )

        latent_f_per_view = (left_action["keyboard"].shape[1] - 1) // 4 + 1
        left_emb = _resample_action_emb_to_latent_t(left_emb, latent_f_per_view, self.use_conv1d)
        right_emb = _resample_action_emb_to_latent_t(right_emb, latent_f_per_view, self.use_conv1d)

        return self._apply_per_view(
            [left_emb, right_emb], f, h, w, batch_size, device, dtype, start_frame
        )

    def _apply_per_view(
        self,
        view_embs: list[torch.Tensor],
        f: int,
        h: int,
        w: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        start_frame: int = 0,
    ) -> torch.Tensor:
        n_views = len(view_embs)
        assert f % n_views == 0, f"f={f} not divisible by n_views={n_views}"
        f_per_view = f // n_views
        end_frame = start_frame + f_per_view

        action_bias = torch.zeros(batch_size, f, h, w, self.dit_dim, device=device, dtype=dtype)
        for i, emb in enumerate(view_embs):
            sliced = emb[:, start_frame:end_frame, :]
            expanded = sliced.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w, -1)
            action_bias[:, i * f_per_view : (i + 1) * f_per_view, :, :, :] = expanded

        return rearrange(action_bias, "b f h w c -> b (f h w) c")


class MultiAgentActionControlNAgentsModule(MultiAgentActionControlModule):
    def __init__(
        self,
        keyboard_dim: int = 23,
        camera_dim: int = 2,
        use_camera: bool = True,
        embed_dim: int = 256,
        dit_dim: int = 2048,
        temporal_downsample: int = 4,
        use_conv1d: bool = False,
        num_agents: int = 4,
        pool_size: int | None = None,
    ):
        nn.Module.__init__(self)
        if num_agents < 2:
            raise ValueError(f"num_agents must be >= 2, got {num_agents}")
        if pool_size is None:
            pool_size = num_agents
        if pool_size < num_agents:
            raise ValueError(
                f"pool_size ({pool_size}) must be >= num_agents ({num_agents})"
            )
        self.dit_dim = dit_dim
        self.num_agents = num_agents
        self.pool_size = pool_size
        self.use_conv1d = use_conv1d

        self.action_encoders = nn.ModuleList(
            [
                SolarisActionEncoder(
                    keyboard_dim=keyboard_dim,
                    camera_dim=camera_dim,
                    use_camera=use_camera,
                    embed_dim=embed_dim,
                    out_dim=dit_dim,
                    temporal_downsample=temporal_downsample,
                    use_conv1d=use_conv1d,
                )
                for _ in range(pool_size)
            ]
        )

    def forward(
        self,
        actions: list[dict[str, torch.Tensor]],
        x_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        n_views: int | None = None,
        start_frame: int = 0,
        agent_pool_indices: list[int] | None = None,
    ) -> torch.Tensor:
        assert len(actions) == self.num_agents, (
            f"MultiAgentActionControlNAgentsModule expects {self.num_agents} agents, "
            f"got len(actions)={len(actions)}"
        )
        assert n_views is not None and n_views == self.num_agents, (
            f"n_views must equal num_agents={self.num_agents}, got n_views={n_views}"
        )

        if agent_pool_indices is None:
            agent_pool_indices = list(range(self.num_agents))
        assert len(agent_pool_indices) == self.num_agents, (
            f"agent_pool_indices length {len(agent_pool_indices)} must equal "
            f"num_agents {self.num_agents}"
        )
        assert max(agent_pool_indices) < self.pool_size, (
            f"agent_pool_indices max {max(agent_pool_indices)} out of range "
            f"for pool of size {self.pool_size}"
        )
        assert len(set(agent_pool_indices)) == self.num_agents, (
            f"agent_pool_indices must be unique, got {agent_pool_indices}"
        )

        batch_size = actions[0]["keyboard"].shape[0]
        _, _, f, h, w = x_shape
        latent_f_per_view = (actions[0]["keyboard"].shape[1] - 1) // 4 + 1

        view_embs: list[torch.Tensor] = []
        for slot, action in enumerate(actions):
            encoder = self.action_encoders[agent_pool_indices[slot]]
            emb = encoder(
                action["keyboard"].to(device=device, dtype=dtype),
                action["camera"].to(device=device, dtype=dtype),
            )
            emb = _resample_action_emb_to_latent_t(emb, latent_f_per_view, self.use_conv1d)
            view_embs.append(emb)

        return self._apply_per_view(view_embs, f, h, w, batch_size, device, dtype, start_frame)


class MultiAgentActionControlSharedEncoderModule(MultiAgentActionControlModule):
    def __init__(
        self,
        keyboard_dim: int = 23,
        camera_dim: int = 2,
        use_camera: bool = True,
        embed_dim: int = 256,
        dit_dim: int = 2048,
        temporal_downsample: int = 4,
        use_conv1d: bool = False,
        num_agents: int = 4,
    ):
        nn.Module.__init__(self)
        if num_agents < 2:
            raise ValueError(f"num_agents must be >= 2, got {num_agents}")
        self.dit_dim = dit_dim
        self.num_agents = num_agents
        self.use_conv1d = use_conv1d

        self.shared_action_encoder = SolarisActionEncoder(
            keyboard_dim=keyboard_dim,
            camera_dim=camera_dim,
            use_camera=use_camera,
            embed_dim=embed_dim,
            out_dim=dit_dim,
            temporal_downsample=temporal_downsample,
            use_conv1d=use_conv1d,
        )

    def forward(
        self,
        actions: list[dict[str, torch.Tensor]],
        x_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        n_views: int | None = None,
        start_frame: int = 0,
        agent_pool_indices: list[int] | None = None,
    ) -> torch.Tensor:
        del agent_pool_indices
        assert len(actions) == self.num_agents, (
            f"MultiAgentActionControlSharedEncoderModule expects {self.num_agents} agents, "
            f"got len(actions)={len(actions)}"
        )
        assert n_views is not None and n_views == self.num_agents, (
            f"n_views must equal num_agents={self.num_agents}, got n_views={n_views}"
        )

        batch_size = actions[0]["keyboard"].shape[0]
        _, _, f, h, w = x_shape
        latent_f_per_view = (actions[0]["keyboard"].shape[1] - 1) // 4 + 1

        view_embs: list[torch.Tensor] = []
        for action in actions:
            emb = self.shared_action_encoder(
                action["keyboard"].to(device=device, dtype=dtype),
                action["camera"].to(device=device, dtype=dtype),
            )
            emb = _resample_action_emb_to_latent_t(emb, latent_f_per_view, self.use_conv1d)
            view_embs.append(emb)

        return self._apply_per_view(view_embs, f, h, w, batch_size, device, dtype, start_frame)


class MultiAgentActionControlSpatialConcatModule(MultiAgentActionControlModule):
    def __init__(
        self,
        keyboard_dim: int = 23,
        camera_dim: int = 2,
        use_camera: bool = True,
        embed_dim: int = 256,
        dit_dim: int = 2048,
        temporal_downsample: int = 4,
        use_conv1d: bool = False,
        num_agents: int = 2,
    ):
        nn.Module.__init__(self)
        if num_agents < 2:
            raise ValueError(f"num_agents must be >= 2, got {num_agents}")
        self.dit_dim = dit_dim
        self.num_agents = num_agents
        self.use_conv1d = use_conv1d

        self.shared_action_encoder = SolarisActionEncoder(
            keyboard_dim=keyboard_dim,
            camera_dim=camera_dim,
            use_camera=use_camera,
            embed_dim=embed_dim,
            out_dim=dit_dim,
            temporal_downsample=temporal_downsample,
            use_conv1d=use_conv1d,
        )

    def forward(
        self,
        actions: list[dict[str, torch.Tensor]],
        x_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        n_views: int | None = None,
        start_frame: int = 0,
        agent_pool_indices: list[int] | None = None,
    ) -> torch.Tensor:
        del agent_pool_indices
        assert len(actions) == self.num_agents, (
            f"MultiAgentActionControlSpatialConcatModule expects {self.num_agents} agents, "
            f"got len(actions)={len(actions)}"
        )
        assert n_views is not None and n_views == 1, (
            f"MultiAgentActionControlSpatialConcatModule requires n_views=1 "
            f"(spatial-concat layout); got n_views={n_views}"
        )

        batch_size = actions[0]["keyboard"].shape[0]
        _, _, f, h, w = x_shape
        assert w % self.num_agents == 0, (
            f"latent width {w} must be divisible by num_agents {self.num_agents} "
            f"so each agent's action bias maps to an equal W slab"
        )
        w_per_agent = w // self.num_agents
        latent_f_per_view = (actions[0]["keyboard"].shape[1] - 1) // 4 + 1
        assert f == latent_f_per_view, (
            f"MultiAgentActionControlSpatialConcatModule: latent f={f} from x_shape "
            f"does not match action latent_f_per_view={latent_f_per_view} (from "
            f"actions[0]['keyboard'].shape[1]={actions[0]['keyboard'].shape[1]}). "
            f"Spatial-concat requires the wide single-view latent T to equal one "
            f"player's latent T. If you are running inference, make sure you pass "
            f"``--spatial_concat`` (not ``--split_width_for_multiview``) so the "
            f"data path keeps n_views=1."
        )
        end_frame = start_frame + f

        action_bias = torch.zeros(
            batch_size, f, h, w, self.dit_dim, device=device, dtype=dtype
        )
        for i, action in enumerate(actions):
            emb = self.shared_action_encoder(
                action["keyboard"].to(device=device, dtype=dtype),
                action["camera"].to(device=device, dtype=dtype),
            )
            emb = _resample_action_emb_to_latent_t(emb, latent_f_per_view, self.use_conv1d)
            sliced = emb[:, start_frame:end_frame, :]
            expanded = sliced.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w_per_agent, -1)
            action_bias[:, :, :, i * w_per_agent : (i + 1) * w_per_agent, :] = expanded

        return rearrange(action_bias, "b f h w c -> b (f h w) c")


def build_multi_agent_action_control(
    *,
    num_agents: int,
    keyboard_dim: int = 23,
    camera_dim: int = 2,
    use_camera: bool = True,
    embed_dim: int = 256,
    dit_dim: int = 2048,
    temporal_downsample: int = 4,
    use_conv1d: bool = False,
    pool_size: int | None = None,
    share_encoder: bool = False,
    spatial_concat: bool = False,
) -> MultiAgentActionControlModule:
    if spatial_concat:
        if share_encoder:
            raise ValueError(
                "spatial_concat=True is mutually exclusive with share_encoder=True; "
                "the spatial-concat module already uses a single shared encoder."
            )
        return MultiAgentActionControlSpatialConcatModule(
            keyboard_dim=keyboard_dim,
            camera_dim=camera_dim,
            use_camera=use_camera,
            embed_dim=embed_dim,
            dit_dim=dit_dim,
            temporal_downsample=temporal_downsample,
            use_conv1d=use_conv1d,
            num_agents=num_agents,
        )

    if share_encoder:
        return MultiAgentActionControlSharedEncoderModule(
            keyboard_dim=keyboard_dim,
            camera_dim=camera_dim,
            use_camera=use_camera,
            embed_dim=embed_dim,
            dit_dim=dit_dim,
            temporal_downsample=temporal_downsample,
            use_conv1d=use_conv1d,
            num_agents=num_agents,
        )

    effective_pool = pool_size if pool_size is not None else num_agents
    use_legacy_2p = num_agents == 2 and effective_pool == 2
    if use_legacy_2p:
        return MultiAgentActionControlModule(
            keyboard_dim=keyboard_dim,
            camera_dim=camera_dim,
            use_camera=use_camera,
            embed_dim=embed_dim,
            dit_dim=dit_dim,
            temporal_downsample=temporal_downsample,
            use_conv1d=use_conv1d,
        )
    return MultiAgentActionControlNAgentsModule(
        keyboard_dim=keyboard_dim,
        camera_dim=camera_dim,
        use_camera=use_camera,
        embed_dim=embed_dim,
        dit_dim=dit_dim,
        temporal_downsample=temporal_downsample,
        use_conv1d=use_conv1d,
        num_agents=num_agents,
        pool_size=effective_pool,
    )
