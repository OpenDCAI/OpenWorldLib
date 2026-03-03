import os
import sys
import math
import random
import types
from contextlib import contextmanager
from functools import partial
from huggingface_hub import snapshot_download

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from einops import rearrange

from ...base_synthesis import BaseSynthesis

from .lingbot_world.configs import WAN_CONFIGS
from .lingbot_world.modules.model import WanModel
from .lingbot_world.modules.t5 import T5EncoderModel
from .lingbot_world.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .lingbot_world.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .lingbot_world.distributed.sequence_parallel import sp_dit_forward
from ....base_models.diffusion_model.video.wan_2p2.modules.vae2_1 import Wan2_1_VAE
from ....base_models.diffusion_model.video.wan_2p2.distributed.fsdp import shard_model
from ....base_models.diffusion_model.video.wan_2p2.distributed.sequence_parallel import sp_attn_forward
from ....base_models.diffusion_model.video.wan_2p2.distributed.util import get_world_size

class LingBotSynthesis(BaseSynthesis):
    def __init__(self, 
                 config,
                 text_encoder,
                 vae,
                 low_noise_model,
                 high_noise_model,
                 device="cuda",
                 rank=0,
                 t5_cpu=False,
                 use_sp=False,
                 sp_size=1,
                 offload_model=True):
        super(LingBotSynthesis, self).__init__()
        
        self.config = config
        self.device = device
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.use_sp = use_sp
        self.sp_size = sp_size
        self.offload_model = offload_model
        
        # Core Components
        self.text_encoder = text_encoder
        self.vae = vae
        self.low_noise_model = low_noise_model
        self.high_noise_model = high_noise_model
        
        # Config params
        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.sample_neg_prompt = config.sample_neg_prompt

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_path,
                        task="i2v-A14B",
                        device="cuda",
                        rank=0,
                        t5_fsdp=False,
                        dit_fsdp=False,
                        ulysses_size=1,
                        t5_cpu=False,
                        offload_model=True,
                        **kwargs):

        if not os.path.isdir(pretrained_model_path):
            print(f"[Rank {rank}] '{pretrained_model_path}' is not a local directory. Attempting download from HuggingFace Hub...")
            try:
                pretrained_model_path = snapshot_download(pretrained_model_path)
                print(f"[Rank {rank}] Model downloaded to: {pretrained_model_path}")
            except Exception as e:
                raise ValueError(f"Failed to load model. '{pretrained_model_path}' is neither a local directory nor a valid HuggingFace repo ID. Error: {e}")

        if task not in WAN_CONFIGS:
            raise ValueError(f"Unsupported task: {task}")
        config = WAN_CONFIGS[task]
        
        device_obj = torch.device(device)
        
        # --- Distributed Strategy Logic ---
        use_sp = (ulysses_size > 1)
        init_on_cpu = True
        
        # If using FSDP or Sequence Parallel, we cannot init strictly on CPU 
        # (or logic changes for sharding)
        if t5_fsdp or dit_fsdp or use_sp:
            init_on_cpu = False
            
        # Helper for FSDP
        # Extract device_id from string "cuda:X"
        try:
            device_id = int(device.split(":")[-1])
        except:
            device_id = 0
            
        shard_fn = partial(shard_model, device_id=device_id) if (t5_fsdp or dit_fsdp) else None

        # --- Load T5 Encoder ---
        print(f"[Rank {rank}] Loading T5 Encoder...")
        text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'), 
            checkpoint_path=os.path.join(pretrained_model_path, config.t5_checkpoint),
            tokenizer_path=os.path.join(pretrained_model_path, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        # --- Load VAE ---
        print(f"[Rank {rank}] Loading VAE...")
        vae = Wan2_1_VAE(
            vae_pth=os.path.join(pretrained_model_path, config.vae_checkpoint),
            device=device_obj
        )

        # --- Load WanModels (DiT) ---
        print(f"[Rank {rank}] Loading WanModels (Low/High Noise)...")
        
        def _configure_model(model, use_sp, dit_fsdp, shard_fn):
            # 1. Set eval
            model.eval().requires_grad_(False)
            
            # 2. Patch for Sequence Parallel (Ulysses)
            if use_sp:
                for block in model.blocks:
                    block.self_attn.forward = types.MethodType(sp_attn_forward, block.self_attn)
                model.forward = types.MethodType(sp_dit_forward, model)
            
            # 3. Barrier before sharding
            if dist.is_initialized():
                dist.barrier()
            
            # 4. FSDP Sharding or Dtype Conversion
            if dit_fsdp and shard_fn is not None:
                model = shard_fn(model)
            else:
                model.to(config.param_dtype)
                if not init_on_cpu:
                    model.to(device_obj)
            return model

        low_noise_model = WanModel.from_pretrained(
            pretrained_model_path, subfolder=config.low_noise_checkpoint, torch_dtype=torch.bfloat16
        )
        low_noise_model = _configure_model(low_noise_model, use_sp, dit_fsdp, shard_fn)

        high_noise_model = WanModel.from_pretrained(
            pretrained_model_path, subfolder=config.high_noise_checkpoint, torch_dtype=torch.bfloat16
        )
        high_noise_model = _configure_model(high_noise_model, use_sp, dit_fsdp, shard_fn)

        return cls(
            config=config,
            text_encoder=text_encoder,
            vae=vae,
            low_noise_model=low_noise_model,
            high_noise_model=high_noise_model,
            device=device,
            rank=rank,
            t5_cpu=t5_cpu,
            use_sp=use_sp,
            sp_size=ulysses_size,
            offload_model=offload_model
        )

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        if t.item() >= boundary:
            required_model = self.high_noise_model
            offload_target = self.low_noise_model
        else:
            required_model = self.low_noise_model
            offload_target = self.high_noise_model
            
        if offload_model:
            # If offload target is on CUDA, move to CPU
            if next(offload_target.parameters()).device.type == 'cuda':
                offload_target.to('cpu')
            # If required model is on CPU, move to CUDA
            if next(required_model.parameters()).device.type == 'cpu':
                required_model.to(self.device)
                
        return required_model

    @torch.no_grad()
    def predict(self, image_tensor, prompt, camera_data=None, num_output_frames=81, 
                height=480, width=832, shift=None, sample_solver='unipc',
                sampling_steps=None, guide_scale=None, n_prompt="", seed=-1, **kwargs):
        
        # Override offload_model from kwargs if present, else use self
        offload_model = kwargs.get('offload_model', self.offload_model)

        if shift is None:
            shift = getattr(self.config, 'sample_shift', 5.0)
        if sampling_steps is None:
            sampling_steps = getattr(self.config, 'sample_steps', 40)
        if guide_scale is None:
            guide_scale = getattr(self.config, 'sample_guide_scale', 5.0)
        if n_prompt == "":
            n_prompt = getattr(self.config, 'sample_neg_prompt', "")

        F, h, w = num_output_frames, height, width
        
        # Latent dimensions calculation
        aspect_ratio = h / w
        lat_h = round(h / self.vae_stride[1] / self.patch_size[1] * self.patch_size[1])
        lat_w = round(w / self.vae_stride[2] / self.patch_size[2] * self.patch_size[2])
        h, w = lat_h * self.vae_stride[1], lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        max_seq_len = lat_f * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device).manual_seed(seed)
        
        # 1. Init Noise
        noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, 
                            generator=seed_g, device=self.device)

        # 2. Mask
        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]

        # 3. Encode Text
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model: self.text_encoder.model.cpu()
        else:
            context = [t.to(self.device) for t in self.text_encoder([prompt], torch.device('cpu'))]
            context_null = [t.to(self.device) for t in self.text_encoder([n_prompt], torch.device('cpu'))]

        # 4. Encode Image (VAE)
        input_video_zeros = torch.zeros(3, F - 1, h, w).to(self.device)
        img_input = image_tensor.unsqueeze(1).to(self.device)
        vae_input = torch.concat([img_input, input_video_zeros], dim=1).unsqueeze(0)
        y = torch.concat([msk, self.vae.encode([vae_input[0]])[0]])

        # 5. Camera Condition
        dit_cond_dict = None
        if camera_data is not None:
            c2ws_plucker_emb = camera_data["c2ws_plucker_emb"]
            
            # Ensure Tensor is on correct device
            if not isinstance(c2ws_plucker_emb, torch.Tensor):
                c2ws_plucker_emb = torch.from_numpy(c2ws_plucker_emb)
            c2ws_plucker_emb = c2ws_plucker_emb.to(self.device)

            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            
            # Add batch dimension and adjust to channel first
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb, 
                'b (f h w) c -> b c f h w', 
                f=lat_f, 
                h=lat_h, 
                w=lat_w
            )
            
            c2ws_plucker_emb = c2ws_plucker_emb.to(self.param_dtype)
            
            dit_cond_dict = {
                "c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0),
            }

        # 6. Diffusion Sampling
        guide_scale_tuple = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale
        if isinstance(guide_scale, (list, tuple)) and len(guide_scale) == 2:
            guide_scale_tuple = guide_scale

        @contextmanager
        def noop_no_sync(): yield

        no_sync_low = getattr(self.low_noise_model, 'no_sync', noop_no_sync)
        no_sync_high = getattr(self.high_noise_model, 'no_sync', noop_no_sync)

        with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad(), no_sync_low(), no_sync_high():
            boundary_step = int(self.boundary * self.num_train_timesteps)
            if sample_solver == 'unipc':
                scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
                scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = scheduler.timesteps
            else:
                scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
                sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(scheduler, device=self.device, sigmas=sigmas)

            latent = noise
            arg_c = {'context': [context[0]], 'seq_len': max_seq_len, 'y': [y], 'dit_cond_dict': dit_cond_dict}
            arg_null = {'context': context_null, 'seq_len': max_seq_len, 'y': [y], 'dit_cond_dict': dit_cond_dict}

            if offload_model: torch.cuda.empty_cache()

            iterator = tqdm(timesteps, desc="Sampling", disable=(self.rank!=0))
            for _, t in enumerate(iterator):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)
                model = self._prepare_model_for_timestep(t, boundary_step, offload_model)
                curr_guide_scale = guide_scale_tuple[1] if t.item() >= boundary_step else guide_scale_tuple[0]

                noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
                if offload_model: torch.cuda.empty_cache()
                noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                if offload_model: torch.cuda.empty_cache()

                noise_pred = noise_pred_uncond + curr_guide_scale * (noise_pred_cond - noise_pred_uncond)
                latent = scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g)[0].squeeze(0)

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            videos = self.vae.decode([latent]) if self.rank == 0 else None

        if dist.is_initialized(): dist.barrier()
        if self.rank != 0: return None

        video = ((videos[0].permute(1, 2, 3, 0) + 1.0) / 2.0).clamp(0.0, 1.0)
        return video.float().cpu().numpy()