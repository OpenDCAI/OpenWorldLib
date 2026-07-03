import copy
from typing import Any, Dict, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanRotaryPosEmbed
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from huggingface_hub import hf_hub_download

try:
    from ..src.sp_utils.parallel_states import get_parallel_state
    from ..src.sp_utils.communications import all_gather
    from ..src.general_utils import rank0_log
except ImportError:
    from src.sp_utils.parallel_states import get_parallel_state
    from src.sp_utils.communications import all_gather
    from src.general_utils import rank0_log
from .camera import camera_center_normalization
from .controlnet import zero_module, WanXControlNet, WanTransformerSparseSpatialBlock

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MaskCamEmbed(nn.Module):
    def __init__(self, controlnet_cfg) -> None:
        super().__init__()

        add_channels = controlnet_cfg.get("add_channels", 1)
        mid_channels = controlnet_cfg.get("mid_channels", 64)
        self.mask_downsample = controlnet_cfg.get("mask_downsample", 4)

        if self.mask_downsample > 1:
            # padding bug fixed
            if controlnet_cfg.get("interp", False):
                self.mask_padding = [0, 0, 0, 0, 3, 3]  # [left, right, top, bottom, front, back] for I2V-interp (first and last frames)
            else:
                self.mask_padding = [0, 0, 0, 0, 3, 0]  # [left, right, top, bottom, front, back] for I2V
        else:
            self.mask_padding = None

        if "5B" in controlnet_cfg.get("base_model", ""):
            self.mask_proj = nn.Sequential(nn.Conv3d(add_channels, mid_channels, kernel_size=(4, 16, 16), stride=(4, 16, 16)),
                                           nn.GroupNorm(mid_channels // 8, mid_channels), nn.SiLU())
        else:
            self.mask_proj = nn.Sequential(nn.Conv3d(add_channels, mid_channels,
                                                     kernel_size=(self.mask_downsample, 8, 8),
                                                     stride=(self.mask_downsample, 8, 8)),
                                           nn.GroupNorm(mid_channels // 8, mid_channels), nn.SiLU())
        self.mask_zero_proj = zero_module(nn.Conv3d(mid_channels, controlnet_cfg.conv_out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)))

    def forward(self, add_inputs: torch.Tensor):
        # render_mask.shape [b,c,f,h,w]
        if self.mask_downsample > 1:
            warp_add_pad = F.pad(add_inputs, self.mask_padding, mode="constant", value=0)
        else:
            warp_add_pad = add_inputs
        add_embeds = self.mask_proj(warp_add_pad)  # [B,C,F,H,W]
        add_embeds = self.mask_zero_proj(add_embeds)
        add_embeds = einops.rearrange(add_embeds, "b c f h w -> b (f h w) c")

        return add_embeds


class _WorldStereoCommonMixin:
    def _init_worldstereo_base(
            self,
            patch_size: Tuple[int],
            num_attention_heads: int,
            attention_head_dim: int,
            in_channels: int,
            out_channels: int,
            text_dim: int,
            freq_dim: int,
            ffn_dim: int,
            num_layers: int,
            cross_attn_norm: bool,
            qk_norm: Optional[str],
            eps: float,
            image_dim: Optional[int],
            added_kv_proj_dim: Optional[int],
            rope_max_seq_len: int,
            pos_embed_seq_len: Optional[int],
            controlnet_cfg,
            base_model: str,
    ) -> None:
        WanTransformer3DModel.__init__(
            self,
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        self.controlnet_cfg = controlnet_cfg
        if self.controlnet_cfg is not None:
            self.controlnet_cfg.base_model = base_model
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.rope_max_seq_len = rope_max_seq_len
        self.sp_size = 1

    def _init_controlnet_base(self) -> None:
        self.controlnet = WanXControlNet(self.controlnet_cfg)
        self.controlnet_rope = WanRotaryPosEmbed(
            self.controlnet_cfg.dim // self.controlnet_cfg.num_heads,
            self.patch_size,
            self.rope_max_seq_len,
        )

    def _load_or_create_controlnet_embeddings(self, load_uni3c: bool) -> None:
        if not load_uni3c:
            self.controlnet.controlnet_patch_embedding = nn.Conv3d(
                self.in_channels, self.controlnet_cfg.conv_out_dim, kernel_size=self.patch_size, stride=self.patch_size
            )
            self.controlnet.controlnet_mask_embedding = MaskCamEmbed(self.controlnet_cfg)
            self.controlnet.controlnet_patch_embedding.weight.data.copy_(self.patch_embedding.weight.data.clone())
            self.controlnet.controlnet_patch_embedding.bias.data.copy_(self.patch_embedding.bias.data.clone())
            return

        # Hardcoded for backward compatibility with open-source uni3c.
        self.controlnet_patch_embedding = nn.Conv3d(
            self.in_channels, self.controlnet_cfg.conv_out_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.controlnet_mask_embedding = MaskCamEmbed(self.controlnet_cfg)
        model_path = hf_hub_download(repo_id="ewrfcas/Uni3C", filename="controlnet.pth", repo_type="model")
        state_dict = torch.load(model_path, map_location="cpu")

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        self.controlnet.controlnet_patch_embedding = copy.deepcopy(self.controlnet_patch_embedding)
        self.controlnet.controlnet_mask_embedding = copy.deepcopy(self.controlnet_mask_embedding)
        del self.controlnet_patch_embedding
        del self.controlnet_mask_embedding
        rank0_log(f"Unexpected keys: {unexpected_keys}")

    def _freeze_backbone_for_controlnet(self, freeze_backbone: bool) -> None:
        if freeze_backbone:
            self.requires_grad_(False)
            self.controlnet.requires_grad_(True)

    def _prepare_lora_scale(self, attention_kwargs, use_rank0_warning: bool = False):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            message = "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            if use_rank0_warning:
                rank0_log(message, level="WARNING")
            else:
                logger.warning(message)

        return attention_kwargs, lora_scale

    def _unscale_lora_if_needed(self, lora_scale: float) -> None:
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

    def _get_patch_context(self, hidden_states: torch.Tensor):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        if self.controlnet_cfg is None or "5B" not in self.controlnet_cfg.get("base_model", ""):
            image_width, image_height = width * 8, height * 8
        else:
            image_width, image_height = width * 16, height * 16

        return (
            batch_size,
            num_channels,
            num_frames,
            height,
            width,
            p_t,
            p_h,
            p_w,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            image_width,
            image_height,
        )

    def _embed_timestep_and_conditions(
            self,
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_image: Optional[torch.Tensor],
    ):
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        return temb, timestep_proj, encoder_hidden_states, ts_seq_len

    def _make_controlnet_add_infos(
            self,
            *,
            extrinsics,
            intrinsics,
            post_patch_width: int,
            post_patch_height: int,
            image_width: int,
            image_height: int,
    ):
        return {
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "patches_x": post_patch_width,
            "patches_y": post_patch_height,
            "image_width": image_width,
            "image_height": image_height,
        }

    def _prepare_controlnet_inputs(
            self,
            *,
            hidden_states,
            render_latent,
            render_mask,
            camera_embedding,
            concat_hidden_prefix: bool,
    ):
        if not hasattr(self, "controlnet"):
            return None, None

        if concat_hidden_prefix:
            render_latent = torch.cat([hidden_states[:, :20], render_latent], dim=1)
        controlnet_rotary_emb = self.controlnet_rope(render_latent)
        controlnet_inputs = self.controlnet.controlnet_patch_embedding(render_latent)
        controlnet_inputs = controlnet_inputs.flatten(2).transpose(1, 2)

        if camera_embedding is not None:
            add_inputs = torch.cat([render_mask, camera_embedding], dim=1)
        else:
            add_inputs = render_mask
        add_inputs = self.controlnet.controlnet_mask_embedding(add_inputs)

        return controlnet_inputs + add_inputs, controlnet_rotary_emb

    def _run_controlnet(
            self,
            *,
            controlnet_inputs,
            controlnet_rotary_emb,
            temb,
            add_infos,
            parallel_dims,
            use_5b_last_temb: bool,
    ):
        if not hasattr(self, "controlnet"):
            return []

        if self.sp_size > 1:
            assert controlnet_inputs.shape[1] % self.sp_size == 0
            controlnet_inputs = torch.chunk(controlnet_inputs, self.sp_size, dim=1)[parallel_dims.sp_rank]
            controlnet_rotary_emb = (
                torch.chunk(controlnet_rotary_emb[0], self.sp_size, dim=1)[parallel_dims.sp_rank],
                torch.chunk(controlnet_rotary_emb[1], self.sp_size, dim=1)[parallel_dims.sp_rank],
            )

        with torch.autocast("cuda", dtype=self.dtype, enabled=True):
            if use_5b_last_temb and "5B" in self.controlnet_cfg.get("base_model", ""):
                return self.controlnet(
                    hidden_states=controlnet_inputs, temb=temb[:, -1], rotary_emb=controlnet_rotary_emb, **add_infos
                )
            return self.controlnet(hidden_states=controlnet_inputs, temb=temb, rotary_emb=controlnet_rotary_emb, **add_infos)

    def _apply_output_projection(
            self,
            *,
            hidden_states,
            temb,
            parallel_dims,
            batch_size: int,
            post_patch_num_frames: int,
            post_patch_height: int,
            post_patch_width: int,
            p_t: int,
            p_h: int,
            p_w: int,
    ):
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
            if self.sp_size > 1:
                shift = torch.chunk(shift, self.sp_size, dim=1)[parallel_dims.sp_rank]
                scale = torch.chunk(scale, self.sp_size, dim=1)[parallel_dims.sp_rank]
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if self.sp_size > 1:
            hidden_states = all_gather(hidden_states, dim=1, group=parallel_dims.sp_group)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


class WorldStereoModel(_WorldStereoCommonMixin, WanTransformer3DModel):
    r"""
    A Transformer model for video-like data used in the Wan model.
    """

    def __init__(
            self,
            patch_size: Tuple[int] = (1, 2, 2),
            num_attention_heads: int = 40,
            attention_head_dim: int = 128,
            in_channels: int = 16,
            out_channels: int = 16,
            text_dim: int = 4096,
            freq_dim: int = 256,
            ffn_dim: int = 13824,
            num_layers: int = 40,
            cross_attn_norm: bool = True,
            qk_norm: Optional[str] = "rms_norm_across_heads",
            eps: float = 1e-6,
            image_dim: Optional[int] = None,
            added_kv_proj_dim: Optional[int] = None,
            rope_max_seq_len: int = 1024,
            pos_embed_seq_len: Optional[int] = None,
            controlnet_cfg=None,
            base_model="",
    ) -> None:
        self._init_worldstereo_base(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
            controlnet_cfg=controlnet_cfg,
            base_model=base_model,
        )

    def build_controlnet(self, load_uni3c=False, freeze_backbone=True):
        self._init_controlnet_base()
        self._load_or_create_controlnet_embeddings(load_uni3c)
        self._freeze_backbone_for_controlnet(freeze_backbone)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_image: Optional[torch.Tensor] = None,
            render_latent=None,
            render_mask=None,
            camera_embedding=None,
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            extrinsics=None,
            intrinsics=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param render_latent: [b,c,f,h,w]
        :param render_mask: [b,1,f,h,w]
        :param camera_embedding: [b,6,f,h,w]
        """
        parallel_dims = get_parallel_state()
        attention_kwargs, lora_scale = self._prepare_lora_scale(attention_kwargs)

        (
            batch_size,
            num_channels,
            num_frames,
            height,
            width,
            p_t,
            p_h,
            p_w,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            image_width,
            image_height,
        ) = self._get_patch_context(hidden_states)

        # TODO: some data has invalid camera values — guard against non-finite extrinsics
        if extrinsics is not None and (~torch.isfinite(extrinsics)).sum() > 0:
            print("Error extrinsics!!!")
            extrinsics = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device)[None].repeat(extrinsics.shape[0], 1, 1)  # [N,4,4]
        else:
            if extrinsics is not None:
                extrinsics = camera_center_normalization(extrinsics, extrinsics.shape[0], camera_scale=1.0)

        if intrinsics is not None and (~torch.isfinite(intrinsics)).sum() > 0:
            print("Error intrinsics!!!")
            intrinsics = torch.eye(3, dtype=intrinsics.dtype, device=intrinsics.device)[None].repeat(intrinsics.shape[0], 1, 1)  # [N,3,3]

        # WAN2.1 and WAN2.2-14B concatenate condition latent; WAN2.2-5B blends condition latent.
        controlnet_inputs, controlnet_rotary_emb = self._prepare_controlnet_inputs(
            hidden_states=hidden_states,
            render_latent=render_latent,
            render_mask=render_mask,
            camera_embedding=camera_embedding,
            concat_hidden_prefix="5B" not in self.controlnet_cfg.get("base_model", "") if hasattr(self, "controlnet") else False,
        )

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, ts_seq_len = self._embed_timestep_and_conditions(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )

        if controlnet_inputs is not None:
            ### additional infos ###
            add_infos = self._make_controlnet_add_infos(
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                post_patch_width=post_patch_width,
                post_patch_height=post_patch_height,
                image_width=image_width,
                image_height=image_height,
            )

            controlnet_states = self._run_controlnet(
                controlnet_inputs=controlnet_inputs,
                controlnet_rotary_emb=controlnet_rotary_emb,
                temb=temb,
                add_infos=add_infos,
                parallel_dims=parallel_dims,
                use_5b_last_temb=True,
            )
            ### controlnet encoding over ###
        else:
            controlnet_states = []

        ### sp
        if self.sp_size > 1:
            assert hidden_states.shape[1] % self.sp_size == 0
            hidden_states = torch.chunk(hidden_states, self.sp_size, dim=1)[parallel_dims.sp_rank]
            rotary_emb = (torch.chunk(rotary_emb[0], self.sp_size, dim=1)[parallel_dims.sp_rank],
                          torch.chunk(rotary_emb[1], self.sp_size, dim=1)[parallel_dims.sp_rank])
            if ts_seq_len is not None:  # 5B
                timestep_proj = torch.chunk(timestep_proj, self.sp_size, dim=1)[parallel_dims.sp_rank]

        # 4. Transformer blocks
        for i, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

            # adding control features
            if i < len(controlnet_states):
                hidden_states += controlnet_states[i]

        output = self._apply_output_projection(
            hidden_states=hidden_states,
            temb=temb,
            parallel_dims=parallel_dims,
            batch_size=batch_size,
            post_patch_num_frames=post_patch_num_frames,
            post_patch_height=post_patch_height,
            post_patch_width=post_patch_width,
            p_t=p_t,
            p_h=p_h,
            p_w=p_w,
        )
        self._unscale_lora_if_needed(lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class WorldStereoRefSModel(_WorldStereoCommonMixin, WanTransformer3DModel):
    r"""
    A Transformer model for video-like data used in the Wan model.
    """

    def __init__(
            self,
            patch_size: Tuple[int] = (1, 2, 2),
            num_attention_heads: int = 40,
            attention_head_dim: int = 128,
            in_channels: int = 16,
            out_channels: int = 16,
            text_dim: int = 4096,
            freq_dim: int = 256,
            ffn_dim: int = 13824,
            num_layers: int = 40,
            cross_attn_norm: bool = True,
            qk_norm: Optional[str] = "rms_norm_across_heads",
            eps: float = 1e-6,
            image_dim: Optional[int] = None,
            added_kv_proj_dim: Optional[int] = None,
            rope_max_seq_len: int = 1024,
            pos_embed_seq_len: Optional[int] = None,
            controlnet_cfg=None,
            base_model="",
            **kwargs
    ) -> None:
        self._init_worldstereo_base(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
            controlnet_cfg=controlnet_cfg,
            base_model=base_model,
        )
        self.ref_patch_size = patch_size

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.blocks = nn.ModuleList(
            [
                WanTransformerSparseSpatialBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

    def build_controlnet(self, load_uni3c=False, freeze_backbone=True):
        self._init_controlnet_base()
        if self.controlnet_cfg.get("ref_embedding", False):
            # Embedding layer for ref_index, zero initialized
            self.ref_index_embedding = nn.Embedding(21, self.inner_dim)
            nn.init.zeros_(self.ref_index_embedding.weight)
        else:
            self.ref_index_embedding = None

        if self.controlnet_cfg.get("main_camera_embedding", False):
            self.camera_embedding = nn.Sequential(
                nn.Linear(self.controlnet_cfg.camera_embedding_dim, self.inner_dim // 2),
                nn.SiLU(),
                nn.Linear(self.inner_dim // 2, self.inner_dim),
                nn.SiLU(),
                zero_module(nn.Linear(self.inner_dim, self.inner_dim)),
            )
        else:
            self.camera_embedding = None

        self._load_or_create_controlnet_embeddings(load_uni3c)
        self._freeze_backbone_for_controlnet(freeze_backbone)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_image: Optional[torch.Tensor] = None,
            render_latent=None,
            render_mask=None,
            reference_latent=None,
            camera_embedding=None,
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            ref_index=None,
            camera_qt=None,
            camera_qt_ref=None,
            **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param render_latent: [b,c,f,h,w]
        :param render_mask: [b,1,f,h,w]
        :param camera_embedding: [b,6,f,h,w]
        """
        parallel_dims = get_parallel_state()
        attention_kwargs, lora_scale = self._prepare_lora_scale(attention_kwargs, use_rank0_warning=True)

        (
            batch_size,
            num_channels,
            num_frames,
            height,
            width,
            p_t,
            p_h,
            p_w,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            image_width,
            image_height,
        ) = self._get_patch_context(hidden_states)

        controlnet_inputs, controlnet_rotary_emb = self._prepare_controlnet_inputs(
            hidden_states=hidden_states,
            render_latent=render_latent,
            render_mask=render_mask,
            camera_embedding=camera_embedding,
            concat_hidden_prefix=True,
        )

        rope_tmp = torch.cat([hidden_states, hidden_states], dim=-1)
        tmp_rotary_emb = self.rope(rope_tmp)

        # Reshape rotary embeddings from (b, l, n, c) to (b, f, h, w, n, c)
        tmp_rotary_emb_0 = einops.rearrange(tmp_rotary_emb[0], "b (f h w) n c -> b f h w n c", f=post_patch_num_frames, h=post_patch_height, w=2 * post_patch_width)
        tmp_rotary_emb_1 = einops.rearrange(tmp_rotary_emb[1], "b (f h w) n c -> b f h w n c", f=post_patch_num_frames, h=post_patch_height, w=2 * post_patch_width)

        rotary_emb = [
            einops.rearrange(tmp_rotary_emb_0[:, :, :, :post_patch_width], "b f h w n c -> b (f h w) n c"),
            einops.rearrange(tmp_rotary_emb_1[:, :, :, :post_patch_width], "b f h w n c -> b (f h w) n c")
        ]

        # Use ref_index to select specific frames from ref_rotary_emb
        ref_rotary_emb_0 = tmp_rotary_emb_0[:, :, :, post_patch_width:]  # b f h w n c
        ref_rotary_emb_1 = tmp_rotary_emb_1[:, :, :, post_patch_width:]  # b f h w n c
        ref_rotary_emb = [
            einops.rearrange(ref_rotary_emb_0[:, ref_index + 1], "b f h w n c -> b (f h w) n c"),  # ref_index + 1, 0~19-->1~20
            einops.rearrange(ref_rotary_emb_1[:, ref_index + 1], "b f h w n c -> b (f h w) n c")
        ]

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if reference_latent is not None:
            reference_latent = self.patch_embedding(reference_latent)
            reference_latent = reference_latent.flatten(2).transpose(1, 2)

        # Add camera embedding if enabled
        if self.controlnet_cfg.get("main_camera_embedding", False):
            if camera_qt is not None:
                # camera_qt: [b, f, cam_emb_dim] -> [b, f, inner_dim]
                cam_emb = self.camera_embedding(camera_qt)  # [b, f, inner_dim]
                # Expand to match hidden_states shape: [b, f*h*w, inner_dim]
                cam_emb = cam_emb.unsqueeze(2).unsqueeze(2)  # [b, f, 1, 1, inner_dim]
                cam_emb = cam_emb.expand(-1, -1, post_patch_height, post_patch_width, -1)  # [b, f, h, w, inner_dim]
                cam_emb = einops.rearrange(cam_emb, "b f h w c -> b (f h w) c")
                hidden_states = hidden_states + cam_emb

            if camera_qt_ref is not None:
                # camera_qt_ref: [b, f_ref, cam_emb_dim] -> [b, f_ref, inner_dim]
                ref_cam_emb = self.camera_embedding(camera_qt_ref)  # [b, f_ref, inner_dim]
                # Expand to match reference_latent shape
                ref_cam_emb = ref_cam_emb.unsqueeze(2).unsqueeze(2)  # [b, f_ref, 1, 1, inner_dim]
                ref_cam_emb = ref_cam_emb.expand(-1, -1, post_patch_height, post_patch_width, -1)  # [b, f_ref, h, w, inner_dim]
                ref_cam_emb = einops.rearrange(ref_cam_emb, "b f h w c -> b (f h w) c")
                reference_latent = reference_latent + ref_cam_emb

        temb, timestep_proj, encoder_hidden_states, ts_seq_len = self._embed_timestep_and_conditions(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )

        if controlnet_inputs is not None:
            ### additional infos ###
            add_infos = self._make_controlnet_add_infos(
                extrinsics=None,
                intrinsics=None,
                post_patch_width=post_patch_width,
                post_patch_height=post_patch_height,
                image_width=image_width,
                image_height=image_height,
            )

            controlnet_states = self._run_controlnet(
                controlnet_inputs=controlnet_inputs,
                controlnet_rotary_emb=controlnet_rotary_emb,
                temb=temb,
                add_infos=add_infos,
                parallel_dims=parallel_dims,
                use_5b_last_temb=False,
            )
            ### controlnet encoding over ###
        else:
            controlnet_states = []

        ### sp
        if self.sp_size > 1:
            assert hidden_states.shape[1] % self.sp_size == 0
            hidden_states = torch.chunk(hidden_states, self.sp_size, dim=1)[parallel_dims.sp_rank]
            rotary_emb = (torch.chunk(rotary_emb[0], self.sp_size, dim=1)[parallel_dims.sp_rank],
                          torch.chunk(rotary_emb[1], self.sp_size, dim=1)[parallel_dims.sp_rank])
            if ts_seq_len is not None:  # 5B
                timestep_proj = torch.chunk(timestep_proj, self.sp_size, dim=1)[parallel_dims.sp_rank]

            if reference_latent is not None:
                assert reference_latent.shape[1] % self.sp_size == 0
                reference_latent = torch.chunk(reference_latent, self.sp_size, dim=1)[parallel_dims.sp_rank]
                ref_rotary_emb = (torch.chunk(ref_rotary_emb[0], self.sp_size, dim=1)[parallel_dims.sp_rank],
                                  torch.chunk(ref_rotary_emb[1], self.sp_size, dim=1)[parallel_dims.sp_rank])

        # 4. Transformer blocks
        ref_states = reference_latent
        for i, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, ref_states = self._gradient_checkpointing_func(
                    block, hidden_states,
                    ref_states if self.controlnet_cfg.update_ref else reference_latent,
                    encoder_hidden_states, timestep_proj, rotary_emb, ref_rotary_emb, post_patch_num_frames, post_patch_height, post_patch_width, ref_index
                )
            else:
                hidden_states, ref_states = block(hidden_states,
                                                  ref_states if self.controlnet_cfg.update_ref else reference_latent,
                                                  encoder_hidden_states, timestep_proj, rotary_emb, ref_rotary_emb, post_patch_num_frames, post_patch_height, post_patch_width, ref_index)
            # adding control features
            if i < len(controlnet_states):
                hidden_states += controlnet_states[i]

        output = self._apply_output_projection(
            hidden_states=hidden_states,
            temb=temb,
            parallel_dims=parallel_dims,
            batch_size=batch_size,
            post_patch_num_frames=post_patch_num_frames,
            post_patch_height=post_patch_height,
            post_patch_width=post_patch_width,
            p_t=p_t,
            p_h=p_h,
            p_w=p_w,
        )
        self._unscale_lora_if_needed(lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)