# Copyright Alibaba Inc. All Rights Reserved.

import copy
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from typing import Optional

from ..fusion.layer.block import IRGBlock, PartialVGGTBlock
from openworldlib.base_models.diffusion_model.diffsynth import WanVideoPipeline, ModelManager
from openworldlib.base_models.diffusion_model.diffsynth.models.camera_control import CameraConditionModel
from openworldlib.base_models.diffusion_model.diffsynth.models.wan_video_dit import (
    build_freqs_3d_with_extra_cis,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)
from openworldlib.base_models.three_dimensions.point_clouds.vggt.fantasy_world_vggt.models.vggt import VGGT


class FantasyWorldFusionModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        start_index: int = 16,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        cross_attention_list: list = [0],
        dit_path=None,
        vggt_cfg: dict | None = None,
        camera_control: bool = False,
        camera_cfg: dict | None = None,
        drop_ratio: float = 0.17,
    ):
        super().__init__()

        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(
            dit_path,
            torch_dtype=torch.bfloat16,
            # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8
            # quantization.
        )

        self.pipe = WanVideoPipeline.from_model_manager(
            model_manager, device='cpu')

        self.vggt = VGGT(**vggt_cfg)
        self.vggt.to(torch.bfloat16)
        self.camera_control = camera_control
        if self.camera_control:
            self.camera_condition = CameraConditionModel(
                self.pipe.dit, **camera_cfg).to("cuda")

        self.start_index = start_index
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.cross_attention_list = cross_attention_list
        self.device = "cuda"

        irg_blocks = nn.ModuleList()
        self.bicross_dim = 1152
        self.bicross_num_heads = 12
        head_dim = self.bicross_dim // self.bicross_num_heads

        self.freqs_bicross = precompute_freqs_cis_3d(head_dim)

        for idx in self.cross_attention_list:
            src_dit_blk = self.pipe.dit.blocks[idx + self.start_index]
            src_agg_blk = self.vggt.aggregator.global_blocks[idx]
            dit_blk_copy = copy.deepcopy(src_dit_blk)
            agg_blk_copy = PartialVGGTBlock(copy.deepcopy(src_agg_blk))
            self.pipe.dit.blocks[idx + self.start_index] = nn.Identity()
            self.vggt.aggregator.global_blocks[idx] = nn.Identity()
            irg_blocks.append(
                IRGBlock(
                    x_agg_block=agg_blk_copy,
                    x_dit_block=dit_blk_copy,
                    m1_dim=self.pipe.dit.dim,
                    m2_dim=self.vggt.embed_dim,
                    hidden_size=self.bicross_dim,
                    num_heads=self.bicross_num_heads,
                    drop_path=None,
                )
            )
        self.IRGBlock = irg_blocks

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]
        self.use_info = camera_cfg['use_info']
        self.drop_ratio = drop_ratio
        self.to(torch.bfloat16)

    def joint_forward(self,
                      x: torch.Tensor,
                      timestep: torch.Tensor,
                      context: torch.Tensor,
                      clip_feature: Optional[torch.Tensor] = None,
                      y: Optional[torch.Tensor] = None,
                      use_gradient_checkpointing: bool = True,
                      camera_token=None,
                      plucker_fea: Optional[torch.Tensor] = None,
                      plucker_context_lens: Optional[torch.Tensor] = None,
                      uncond=False,
                      return_prediction=False,
                      **kwargs,):
        f = x.shape[2]

        t = self.pipe.dit.time_embedding(
            sinusoidal_embedding_1d(self.pipe.dit.freq_dim, timestep))
        t_mod = self.pipe.dit.time_projection(
            t).unflatten(1, (6, self.pipe.dit.dim))
        context = self.pipe.dit.text_embedding(context)

        if self.pipe.dit.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = self.pipe.dit.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.pipe.dit.patchify(x)

        freqs = torch.cat([
            self.pipe.dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.pipe.dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.pipe.dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        freqs_bi_dit = torch.cat([
            self.freqs_bicross[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs_bicross[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs_bicross[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        freqs_bi_agg = build_freqs_3d_with_extra_cis(self.freqs_bicross,
                                                     f, h, w,
                                                     n_extra=5 * f,
                                                     device=x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward
        kwargs = dict(
            plucker_fea=plucker_fea,
            plucker_context_lens=plucker_context_lens)

        for i, block in enumerate(self.pipe.dit.blocks):
            if use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False, **kwargs
                )
                if i == self.start_index - 1:
                    break
            else:
                x = block(x, context, t_mod, freqs, **kwargs)
                if i == self.start_index - 1:
                    break
        vggt_x = x.reshape(x.shape[0], f, h, w, 5120)
        vggt_x = vggt_x.permute(0, 4, 1, 2, 3)
        patch_token, camera_token, e0 = self.vggt._process_wan_input(
            patch_token=vggt_x, camera_token=camera_token, t=timestep)
        tokens, pos = self.vggt.aggregator._process_aggregator_input(
            patch_token, camera_token)

        B, T, _, _, C = patch_token.shape
        S = T
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for i in range(len(self.pipe.dit.blocks) - self.start_index):
            tokens, frame_idx, frame_intermediates = self.vggt.aggregator._process_frame_attention(
                tokens, B, S, P, C, frame_idx, pos=pos, e0=e0, )
            if i in self.cross_attention_list:
                x, tokens, global_intermediates = self.IRGBlock[i](
                    x_dit=x, x_agg=tokens, context=context, t_mod=t_mod,
                    freqs=freqs, freqs_dit=freqs_bi_dit, freqs_agg=freqs_bi_agg,
                    pos=pos, e0=e0, uncond=uncond, **kwargs,
                )
                global_idx += 1
            else:
                if self.pipe.dit.training and use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.pipe.dit.blocks[i + self.start_index]),
                        x, context, t_mod, freqs,
                        use_reentrant=False, **kwargs,
                    )
                else:
                    x = self.pipe.dit.blocks[i +
                                             self.start_index](x, context, t_mod, freqs, **kwargs)

                tokens, global_idx, global_intermediates = self.vggt.aggregator._process_global_attention(
                    tokens, B, S, P, C, global_idx, pos=pos, e0=e0)
            for index in range(len(frame_intermediates)):

                concat_inter = torch.cat(
                    [frame_intermediates[index], global_intermediates[index]], dim=-1)
                output_list.append(concat_inter)

        x = self.pipe.dit.head(x, t)
        x = self.pipe.dit.unpatchify(x, (f, h, w))

        if return_prediction:
            prediction = {}
            prediction = self.vggt._head_predction(
                patch_token, self.vggt.aggregator.patch_start_idx, output_list)

            return x, prediction
        else:
            return x, None

    @torch.no_grad()
    def generate_video(
        self,
        context_pos: torch.Tensor,
        context_neg: Optional[torch.Tensor] = None,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        camera_token=None,
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=50,
        cfg_scale=5.0,
        seed=None,
        device="cuda",
        plucker_embedding=None,
        geo_prior=None,
        **kwargs
    ):
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
        self.pipe.scheduler.set_timesteps(num_inference_steps)

        if seed is not None:
            torch.manual_seed(1024)

        noise = self.pipe.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed, device=device, dtype=torch.float32
        )
        noise = noise.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        latents = noise
        if self.camera_control:
            if self.use_info == 'rgb_conf':
                video_guidence = geo_prior
            else:
                if self.use_info == 'all':
                    video_guidence = torch.cat(
                        [geo_prior, plucker_embedding], dim=-1)

                elif self.use_info == 'plucker':
                    video_guidence = plucker_embedding
                else:
                    raise NotImplementedError
            plucker_fea = self.camera_condition.get_pose_fea(video_guidence)
            plucker_context_lens = torch.ones(
                video_guidence.shape[1] // 4 + 1,
                dtype=torch.long,
                device=plucker_fea.device)
            plucker_context_lens[1:] = 4
        else:
            plucker_fea = None
            plucker_context_lens = None

        image_emb = {}
        if clip_feature is not None:
            image_emb["clip_feature"] = clip_feature.to(self.pipe.device)
        if y is not None:
            image_emb["y"] = y.to(self.pipe.device)

        extra_input = self.pipe.prepare_extra_input(latents)
        self.pipe.load_models_to_device(["dit"])
        for progress_id, timestep in enumerate(
                tqdm(range(num_inference_steps))):

            t = self.pipe.scheduler.timesteps[progress_id].unsqueeze(
                0).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

            noise_pred_posi, final_prediction = self.joint_forward(
                latents, timestep=t,
                context=context_pos,
                clip_feature=image_emb.get("clip_feature"),
                y=image_emb.get("y"),
                use_gradient_checkpointing=use_gradient_checkpointing,
                camera_token=camera_token, plucker_fea=plucker_fea,
                plucker_context_lens=plucker_context_lens,
                return_prediction=True if progress_id == num_inference_steps - 1 else False,
                **extra_input
            )

            if cfg_scale != 1.0 and context_neg is not None:
                noise_pred_nega, _ = self.joint_forward(
                    latents, timestep=t,
                    context=context_neg,
                    clip_feature=image_emb.get("clip_feature"),
                    y=image_emb.get("y"),
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    camera_token=camera_token, plucker_fea=plucker_fea,
                    plucker_context_lens=plucker_context_lens,
                    **extra_input
                )
                noise_pred = noise_pred_nega + cfg_scale * \
                    (noise_pred_posi - noise_pred_nega)
            latents = latents.to('cuda')
            latents = self.pipe.scheduler.step(
                noise_pred, self.pipe.scheduler.timesteps[progress_id], latents)

        return latents, final_prediction
