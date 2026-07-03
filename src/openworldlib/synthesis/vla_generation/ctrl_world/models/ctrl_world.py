# from diffusers import StableVideoDiffusionPipeline
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

import numpy as np
import torch
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import json
from decord import VideoReader, cpu
import wandb
import swanlab
import mediapy


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Action_encoder2(nn.Module):
    def __init__(self, action_dim, action_num, hidden_size, text_cond=True):
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        self.hidden_size = hidden_size
        self.text_cond = text_cond

        input_dim = int(action_dim)
        self.action_encode = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024)
        )
        # kaiming initialization
        nn.init.kaiming_normal_(self.action_encode[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.action_encode[2].weight, mode='fan_in', nonlinearity='relu')

    def forward(self, action,  texts=None, text_tokinizer=None, text_encoder=None, frame_level_cond=True,):
        # action: (B, action_num, action_dim)
        B,T,D = action.shape
        if not frame_level_cond:
            action = einops.rearrange(action, 'b t d -> b 1 (t d)')
        action = self.action_encode(action)

        if texts is not None and self.text_cond:
            # with 50% probability, add text condition
            with torch.no_grad():
                inputs = text_tokinizer(texts, padding='max_length', return_tensors="pt", truncation=True).to(text_encoder.device)
                outputs = text_encoder(**inputs)
                hidden_text = outputs.text_embeds # (B, 512)
                hidden_text = einops.repeat(hidden_text, 'b c -> b 1 (n c)', n=2) # (B, 1, 1024)

            action = action + hidden_text # (B, T, hidden_size)
        return action # (B, 1, hidden_size) or (B, T, hidden_size) if frame_level_cond


class CrtlWorld(nn.Module):
    def __init__(self, args):
        super(CrtlWorld, self).__init__()

        self.args = args

        # load from pretrained stable video diffusion
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(args.svd_model_path)
        # repalce the unet to support frame_level pose condition
        print("replace the unet to support action condition and frame_level pose!")
        unet = UNetSpatioTemporalConditionModel()
        unet.load_state_dict(self.pipeline.unet.state_dict(), strict=False)
        self.pipeline.unet = unet

        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.image_encoder = self.pipeline.image_encoder
        self.scheduler = self.pipeline.scheduler

        # freeze vae, image_encoder, enable unet gradient ckpt
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)
        self.unet.enable_gradient_checkpointing()

        # SVD is a img2video model, load a clip text encoder
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(args.clip_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.clip_model_path,use_fast=False)
        self.text_encoder.requires_grad_(False)

        # initialize an action projector
        self.action_encoder = Action_encoder2(action_dim=args.action_dim, action_num=int(args.num_history+args.num_frames), hidden_size=1024, text_cond=args.text_cond)



    def forward(self, batch):
        latents = batch['latent'] # (B, 16, 4, 32, 32)
        texts = batch['text']
        dtype = self.unet.dtype
        device = self.unet.device
        P_mean=0.7
        P_std=1.6
        noise_aug_strength = 0.0

        num_history  = self.args.num_history
        latents = latents.to(device) #[B, num_history + num_frames]

        # current img as condition image to stack at channel wise, add random noise to current image, noise strength 0.0~0.2
        current_img = latents[:,num_history:(num_history+1)] # (B, 1, 4, 32, 32)
        bsz,num_frames = latents.shape[:2]
        current_img = current_img[:,0] # (B, 4, 32, 32)
        sigma = torch.rand([bsz, 1, 1, 1], device=device) * 0.2
        c_in = 1 / (sigma**2 + 1) ** 0.5
        current_img = c_in*(current_img + torch.randn_like(current_img) * sigma)
        condition_latent = einops.repeat(current_img, 'b c h w -> b f c h w', f=num_frames) # (8, 16,12, 32,32)
        if self.args.his_cond_zero:
            condition_latent[:, :num_history] = 0.0 # (B, num_history+num_frames, 4, 32, 32)


        # action condition
        action = batch['action'] # (B, f, 7)
        action = action.to(device)
        action_hidden = self.action_encoder(action, texts, self.tokenizer, self.text_encoder, frame_level_cond=self.args.frame_level_cond) # (B, f, 1024)

        # for classifier-free guidance, with 5% probability, set action_hidden to 0
        uncond_hidden_states = torch.zeros_like(action_hidden)
        text_mask = (torch.rand(action_hidden.shape[0], device=device)>0.05).unsqueeze(1).unsqueeze(2)
        action_hidden = action_hidden*text_mask+uncond_hidden_states*(~text_mask)

        # diffusion forward process on future latent
        rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        c_skip = 1 / (sigma**2 + 1)
        c_out =  -sigma / (sigma**2 + 1) ** 0.5
        c_in = 1 / (sigma**2 + 1) ** 0.5
        c_noise = (sigma.log() / 4).reshape([bsz])
        loss_weight = (sigma ** 2 + 1) / sigma ** 2
        noisy_latents = (latents + torch.randn_like(latents) * sigma)

        # add 0~0.3 noise to history, history as condition
        sigma_h = torch.randn([bsz, num_history, 1, 1, 1], device=device) * 0.3
        history = latents[:,:num_history] # (B, num_history, 4, 32, 32)
        noisy_history = 1/(sigma_h**2+1)**0.5 *(history + sigma_h * torch.randn_like(history)) # (B, num_history, 4, 32, 32)
        input_latents = torch.cat([noisy_history, c_in*noisy_latents[:,num_history:]], dim=1) # (B, num_history+num_frames, 4, 32, 32)

        # svd stack a img at channel wise
        input_latents = torch.cat([input_latents, condition_latent/self.vae.config.scaling_factor], dim=2)
        motion_bucket_id = self.args.motion_bucket_id
        fps = self.args.fps
        added_time_ids = self.pipeline._get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, action_hidden.dtype, bsz, 1, False)
        added_time_ids = added_time_ids.to(device)

        # forward unet
        loss = 0
        model_pred = self.unet(input_latents, c_noise, encoder_hidden_states=action_hidden, added_time_ids=added_time_ids,frame_level_cond=self.args.frame_level_cond).sample
        predict_x0 = c_out * model_pred + c_skip * noisy_latents

        # only calculate loss on future frames
        loss += ((predict_x0[:,num_history:] - latents[:,num_history:])**2 * loss_weight).mean()

        return loss, torch.tensor(0.0, device=device,dtype=dtype)
