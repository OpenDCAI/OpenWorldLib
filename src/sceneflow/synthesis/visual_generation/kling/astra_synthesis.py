import torch
import torch.nn as nn
from huggingface_hub import snapshot_download, hf_hub_download
from ...base_synthesis import BaseSynthesis

from .astra.pipelines.wan_video_astra import WanVideoAstraPipeline
from .astra.models.model_manager import ModelManager

import os
import numpy as np

from .astra.astra_utils import (
    replace_dit_model_in_manager, add_framepack_components, add_moe_components
)

class AstraSynthesis(BaseSynthesis):
    def __init__(self, pipe):
        self.pipe = pipe
        self.device = pipe.device
        self.tiler_kwargs = {"tiled": True, "tile_size": (34, 34), "tile_stride": (18, 16)}

    @staticmethod
    def _resolve_path(path_or_id):
        """
        如果路径存在，直接返回；
        如果不存在，尝试作为 HuggingFace Repo ID 下载。
        """
        if os.path.exists(path_or_id):
            return path_or_id
        
        print(f"Path '{path_or_id}' not found locally, attempting to download from HuggingFace...")
        try:
            resolved_path = snapshot_download(repo_id=path_or_id)
            return resolved_path
            
        except Exception as e:
            print(f"Error downloading from HF: {e}")
            return path_or_id

    @classmethod
    def from_pretrained(cls, config, device="cuda"): # the config is the AstraConfig from the pipeline
        print("Resolving model paths...")
        resolved_wan_path = cls._resolve_path(config.wan_model_path)
        resolved_astra_path = cls._resolve_path(config.astra_path)
        # 1. Model initialization & Patching
        replace_dit_model_in_manager()
        
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([
            os.path.join(resolved_wan_path, "diffusion_pytorch_model.safetensors"),
            os.path.join(resolved_wan_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(resolved_wan_path, "Wan2.1_VAE.pth"),
        ])
        pipe = WanVideoAstraPipeline.from_model_manager(model_manager, device=device)

        # 2. Add components
        dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in pipe.dit.blocks:
            block.cam_encoder = nn.Linear(13, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))
        
        add_framepack_components(pipe.dit)
        
        moe_config = {
            "num_experts": config.moe_num_experts,
            "top_k": config.moe_top_k,
            "hidden_dim": config.moe_hidden_dim or dim * 2,
            "sekai_input_dim": 13,
            "nuscenes_input_dim": 8,
            "openx_input_dim": 13
        }
        add_moe_components(pipe.dit, moe_config)
        
        # 3. Load weights
        print(f"Loading DiT weights from: {resolved_astra_path} ...")
        astra_dit_path = os.path.join(resolved_astra_path,
                                      "models/Astra/checkpoints/diffusion_pytorch_model.ckpt")
        dit_state_dict = torch.load(astra_dit_path, map_location="cpu")
        pipe.dit.load_state_dict(dit_state_dict, strict=False)
        pipe = pipe.to(device)
        
        model_dtype = next(pipe.dit.parameters()).dtype
        if hasattr(pipe.dit, 'clean_x_embedder'):
            pipe.dit.clean_x_embedder = pipe.dit.clean_x_embedder.to(dtype=model_dtype)
        
        pipe.scheduler.set_timesteps(50)
        
        return cls(pipe)
    
    def encode_frames(self, frames):
        """
        Input: frames [C, T, H, W] (from Operator)
        Output: latents
        """
        frames = frames.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            latents = self.pipe.encode_video(frames, **self.tiler_kwargs)[0]
        if latents.dim() == 5 and latents.shape[0] == 1:
            latents = latents.squeeze(0)
        return latents.cpu()

    def load_encoded_video_from_pth(self, pth_path, start_frame=0, num_frames=10):
        """Helper to load pre-encoded .pth"""
        print(f"Loading encoded video from {pth_path}")
        encoded_data = torch.load(pth_path, weights_only=False, map_location="cpu")
        full_latents = encoded_data['latents']
        if start_frame + num_frames > full_latents.shape[1]:
            raise ValueError(f"Not enough frames: requested {start_frame + num_frames}, available {full_latents.shape[1]}")
        condition_latents = full_latents[:, start_frame:start_frame + num_frames, :, :]
        return condition_latents, encoded_data

    @torch.no_grad()
    def predict(self, framepack_data, current_generation, prompt_emb_pos, prompt_emb_neg, args, camera_embedding_uncond=None):
        """
        Runs the diffusion denoising loop.
        """
        device = self.device
        model_dtype = next(self.pipe.dit.parameters()).dtype
        
        # Prepare input tensors
        clean_latents = framepack_data['clean_latents'].unsqueeze(0)
        clean_latents_2x = framepack_data['clean_latents_2x'].unsqueeze(0)
        clean_latents_4x = framepack_data['clean_latents_4x'].unsqueeze(0)
        camera_embedding = framepack_data['camera_embedding'].unsqueeze(0)
        modality_type = framepack_data['modality_type']
        
        modality_inputs = {modality_type: camera_embedding}
        
        modality_inputs_uncond = None
        camera_embedding_uncond_batch = None
        if args.use_camera_cfg and camera_embedding_uncond is not None:
            camera_embedding_uncond_batch = camera_embedding_uncond[:camera_embedding.shape[1], :].unsqueeze(0)
            modality_inputs_uncond = {modality_type: camera_embedding_uncond_batch}
            
        latent_indices = framepack_data['latent_indices'].unsqueeze(0).cpu()
        clean_latent_indices = framepack_data['clean_latent_indices'].unsqueeze(0).cpu()
        clean_latent_2x_indices = framepack_data['clean_latent_2x_indices'].unsqueeze(0).cpu()
        clean_latent_4x_indices = framepack_data['clean_latent_4x_indices'].unsqueeze(0).cpu()

        C = clean_latents.shape[1] 
        H = clean_latents.shape[3]
        W = clean_latents.shape[4]
        
        new_latents = torch.randn(1, C, current_generation, H, W, device=device, dtype=model_dtype)
        extra_input = self.pipe.prepare_extra_input(new_latents)
        timesteps = self.pipe.scheduler.timesteps
        
        for i, timestep in enumerate(timesteps):
            if i % 10 == 0: print(f"  Denoising step {i+1}/{len(timesteps)}")
            timestep_tensor = timestep.unsqueeze(0).to(device, dtype=model_dtype)
            
            # --- CFG LOGIC ---
            if args.use_camera_cfg and args.camera_guidance_scale > 1.0:
                 noise_pred_cond, _ = self.pipe.dit(new_latents, timestep=timestep_tensor, cam_emb=camera_embedding, modality_inputs=modality_inputs, latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices, **prompt_emb_pos, **extra_input)
                 noise_pred_uncond, _ = self.pipe.dit(new_latents, timestep=timestep_tensor, cam_emb=camera_embedding_uncond_batch, modality_inputs=modality_inputs_uncond, latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices, **(prompt_emb_neg if prompt_emb_neg else prompt_emb_pos), **extra_input)
                 noise_pred = noise_pred_uncond + args.camera_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                 
                 if args.text_guidance_scale > 1.0 and prompt_emb_neg:
                     noise_pred_text_uncond, _ = self.pipe.dit(new_latents, timestep=timestep_tensor, cam_emb=camera_embedding, modality_inputs=modality_inputs, latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices, **prompt_emb_neg, **extra_input)
                     noise_pred = noise_pred_text_uncond + args.text_guidance_scale * (noise_pred - noise_pred_text_uncond)
            elif args.text_guidance_scale > 1.0 and prompt_emb_neg:
                 noise_pred_cond, _ = self.pipe.dit(new_latents, timestep=timestep_tensor, cam_emb=camera_embedding, modality_inputs=modality_inputs, latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices, **prompt_emb_pos, **extra_input)
                 noise_pred_uncond, _ = self.pipe.dit(new_latents, timestep=timestep_tensor, cam_emb=camera_embedding, modality_inputs=modality_inputs, latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices, **prompt_emb_neg, **extra_input)
                 noise_pred = noise_pred_uncond + args.text_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                 noise_pred, _ = self.pipe.dit(new_latents, timestep=timestep_tensor, cam_emb=camera_embedding, modality_inputs=modality_inputs, latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices, **prompt_emb_pos, **extra_input)
            
            new_latents = self.pipe.scheduler.step(noise_pred, timestep, new_latents)
        
        return new_latents

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        """Decode latents to video frames"""
        decoded_video = self.pipe.decode_video(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_np = decoded_video[0].to(torch.float32).permute(1, 2, 3, 0).cpu().numpy()
        video_np = (video_np * 0.5 + 0.5).clip(0, 1)
        video_np = (video_np * 255).astype(np.uint8)
        return video_np