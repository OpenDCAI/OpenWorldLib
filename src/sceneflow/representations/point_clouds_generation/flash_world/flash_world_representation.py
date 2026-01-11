import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from huggingface_hub import snapshot_download

from ...base_representation import BaseRepresentation

# Import FlashWorld models
from .flash_world.autoencoder_kl_wan import AutoencoderKLWan
from .flash_world.transformer_wan import WanTransformer3DModel
from .flash_world.reconstruction_model import WANDecoderPixelAligned3DGSReconstructionModel
from .flash_world.utils import (
    create_raymaps,
    normalize_cameras,
    sample_from_dense_cameras,
)

from transformers import T5TokenizerFast, UMT5EncoderModel
from diffusers import FlowMatchEulerDiscreteScheduler
import einops


@contextmanager
def onload_model(model, device, onload=False):
    """Context manager for moving model to GPU and back to CPU."""
    if onload and device != "cpu":
        model.to(device) 
        try:
            yield model
        finally:
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    else:
        yield model


class MyFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """Custom scheduler for FlashWorld."""
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        return torch.argmin(
            (timestep - schedule_timesteps.to(timestep.device)).abs(), dim=0).item()


class FlashWorldRepresentation(BaseRepresentation):
    """Representation for FlashWorld 3D scene generation."""
    
    def __init__(self, model: Optional[nn.Module] = None, device: Optional[str] = None):
        """
        Initialize FlashWorldRepresentation.
        
        Args:
            model: Pre-loaded GenerationSystem model (optional)
            device: Device to run on ('cuda' or 'cpu')
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        
        if self.model is not None:
            self.model = self.model.to(self.device).eval()
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> 'FlashWorldRepresentation':
        """
        Create representation instance from pretrained model.
        
        Args:
            pretrained_model_path: HuggingFace repo ID (e.g., "Wan-AI/FlashWorld")
            device: Device to run on
            **kwargs: Additional arguments
            
        Returns:
            FlashWorldRepresentation instance
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Download from HuggingFace if needed
        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            print(f"Downloading weights from HuggingFace repo: {pretrained_model_path}")
            model_root = snapshot_download(pretrained_model_path)
            print(f"Model downloaded to: {model_root}")
        
        # Load model
        model = cls._load_generation_system(model_root, device, **kwargs)
        
        return cls(model=model, device=device)
    
    @staticmethod
    def _load_generation_system(model_root: str, device: str, **kwargs) -> nn.Module:
        """Load GenerationSystem model."""
        # Find checkpoint file
        ckpt_path = kwargs.get('ckpt_path', None)
        if ckpt_path is None:
            # Try to find checkpoint in model_root
            ckpt_files = list(Path(model_root).glob("*.pt")) + list(Path(model_root).glob("*.pth")) + list(Path(model_root).glob("*.ckpt"))
            if ckpt_files:
                ckpt_path = str(ckpt_files[0])
            else:
                # Try common checkpoint names
                for name in ["model.ckpt", "checkpoint.pt", "model.pt", "flash_world.pt"]:
                    potential_path = os.path.join(model_root, name)
                    if os.path.exists(potential_path):
                        ckpt_path = potential_path
                        break
        
        offload_t5 = kwargs.get('offload_t5', False)
        offload_vae = kwargs.get('offload_vae', False)
        offload_transformer_during_vae = kwargs.get('offload_transformer_during_vae', False)
        
        # Initialize GenerationSystem
        generation_system = GenerationSystem(
            ckpt_path=ckpt_path,
            device=device,
            offload_t5=offload_t5,
            offload_vae=offload_vae,
            offload_transformer_during_vae=offload_transformer_during_vae,
        )
        
        return generation_system
    
    def api_init(self, api_key, endpoint):
        """Initialize API connection if needed."""
        pass
    
    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get 3D scene representation from input data.
        
        Args:
            data: Dictionary containing:
                - 'text_prompt': str, text description
                - 'cameras': torch.Tensor, camera parameters (N, 11)
                - 'image': PIL.Image or None, reference image (optional)
                - 'image_index': int, frame index for reference image (default: 0)
                - 'image_height': int, output image height (default: 480)
                - 'image_width': int, output image width (default: 704)
                - 'num_frames': int, number of frames (default: 16)
                - 'video_fps': int, fps for video rendering (default: 15)
                
        Returns:
            Dictionary containing:
                - 'scene_params': torch.Tensor, 3D Gaussian Splatting parameters
                - 'ref_w2c': torch.Tensor, reference world-to-camera transform
                - 'T_norm': torch.Tensor, normalization transform
                - 'video_frames': List[PIL.Image], rendered video frames (if requested)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() first.")
        
        text_prompt = data.get('text_prompt', "")
        cameras = data['cameras']  # Required
        image = data.get('image', None)
        image_index = data.get('image_index', 0)
        image_height = data.get('image_height', 480)
        image_width = data.get('image_width', 704)
        num_frames = data.get('num_frames', 16)
        video_fps = data.get('video_fps', 15)
        return_video = data.get('return_video', False)
        
        # Convert PIL Image to tensor if provided
        if image is not None:
            if isinstance(image, Image.Image):
                # Resize and center crop image to match target dimensions (same as FlashWorld/cli.py)
                image = image.convert('RGB')
                w, h = image.size
                
                # Calculate scale factor to maintain aspect ratio
                if image_height / h > image_width / w:
                    scale = image_height / h
                else:
                    scale = image_width / w
                
                # Calculate new dimensions for center crop
                new_h = int(image_height / scale)
                new_w = int(image_width / scale)
                
                # Center crop and resize to target dimensions
                image = image.crop((
                    (w - new_w) // 2, 
                    (h - new_h) // 2, 
                    new_w + (w - new_w) // 2, 
                    new_h + (h - new_h) // 2
                )).resize((image_width, image_height), Image.Resampling.BICUBIC)
                
                # Convert to tensor: (C, H, W) in range [-1, 1]
                image_array = np.array(image)
                image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1) / 255.0 * 2 - 1
            else:
                image_tensor = image
        else:
            image_tensor = None
        
        # Generate scene
        video_path = None
        if return_video:
            import tempfile
            video_path = tempfile.mktemp(suffix='.mp4')
        
        scene_params, ref_w2c, T_norm = self.model.generate(
            cameras=cameras,
            n_frame=num_frames,
            image=image_tensor,
            text=text_prompt,
            image_index=image_index,
            image_height=image_height,
            image_width=image_width,
            video_path=video_path,
            video_fps=video_fps,
        )
        
        result = {
            'scene_params': scene_params,
            'ref_w2c': ref_w2c,
            'T_norm': T_norm,
        }
        
        # Load video frames if generated
        if return_video and video_path and os.path.exists(video_path):
            import imageio
            video_frames = []
            reader = imageio.get_reader(video_path)
            for frame in reader:
                video_frames.append(Image.fromarray(frame))
            reader.close()
            os.remove(video_path)  # Clean up temp file
            result['video_frames'] = video_frames
        
        return result


class GenerationSystem(nn.Module):
    """FlashWorld Generation System."""
    
    def __init__(self, ckpt_path=None, device="cuda", offload_t5=False, offload_vae=False, offload_transformer_during_vae=False):
        super().__init__()
        # Convert device string to torch.device, defaulting to cuda:0 if just "cuda" is provided
        if isinstance(device, str):
            if device == "cuda":
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = device
        self.offload_t5 = offload_t5
        self.offload_vae = offload_vae
        self.offload_transformer_during_vae = offload_transformer_during_vae

        self.latent_dim = 48
        self.temporal_downsample_factor = 4
        self.spatial_downsample_factor = 16

        self.feat_dim = 1024

        self.latent_patch_size = 2

        self.denoising_steps = [0, 250, 500, 750]

        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

        self.vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float).eval()

        from .flash_world.autoencoder_kl_wan import WanCausalConv3d
        with torch.no_grad():
            for name, module in self.vae.named_modules():
                if isinstance(module, WanCausalConv3d):
                    time_pad = module._padding[4]
                    module.padding = (0, module._padding[2], module._padding[0])
                    module._padding = (0, 0, 0, 0, 0, 0)
                    module.weight = nn.Parameter(module.weight[:, :, time_pad:].clone())

        self.vae.requires_grad_(False)

        self.register_buffer('latents_mean', torch.tensor(self.vae.config.latents_mean).float().view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device))
        self.register_buffer('latents_std', torch.tensor(self.vae.config.latents_std).float().view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device))

        self.tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer")

        self.text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float32).eval().requires_grad_(False).to(self.device if not self.offload_t5 else "cpu")

        self.transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float32).train().requires_grad_(False)
        
        self.transformer.patch_embedding.weight = nn.Parameter(torch.nn.functional.pad(self.transformer.patch_embedding.weight, (0, 0, 0, 0, 0, 0, 0, 6 + self.latent_dim)))
        
        weight = self.transformer.proj_out.weight.reshape(self.latent_patch_size ** 2, self.latent_dim, self.transformer.proj_out.weight.shape[1])
        bias = self.transformer.proj_out.bias.reshape(self.latent_patch_size ** 2, self.latent_dim)

        extra_weight = torch.randn(self.latent_patch_size ** 2, self.feat_dim, self.transformer.proj_out.weight.shape[1]) * 0.02
        extra_bias = torch.zeros(self.latent_patch_size ** 2, self.feat_dim)
 
        self.transformer.proj_out.weight = nn.Parameter(torch.cat([weight, extra_weight], dim=1).flatten(0, 1).detach().clone())
        self.transformer.proj_out.bias = nn.Parameter(torch.cat([bias, extra_bias], dim=1).flatten(0, 1).detach().clone())

        self.recon_decoder = WANDecoderPixelAligned3DGSReconstructionModel(self.vae, self.feat_dim, use_render_checkpointing=True, use_network_checkpointing=False).train().requires_grad_(False)

        self.scheduler = MyFlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", shift=3)

        self.register_buffer('timesteps', self.scheduler.timesteps.clone().to(self.device))

        self.transformer.disable_gradient_checkpointing()
        self.transformer.gradient_checkpointing = False

        self.add_feedback_for_transformer()

        if ckpt_path is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.transformer.load_state_dict(state_dict["transformer"])
            self.recon_decoder.load_state_dict(state_dict["recon_decoder"])
            print(f"Loaded {ckpt_path}.")

        try:
            from quant import FluxFp8GeMMProcessor
            FluxFp8GeMMProcessor(self.transformer)
        except (ImportError, ModuleNotFoundError):
            pass

        del self.vae.post_quant_conv, self.vae.decoder
        self.vae.to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16)
        self.recon_decoder.to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16)

        self.transformer.to(self.device if not self.offload_transformer_during_vae else "cpu")

    def latent_scale_fn(self, x):
        return (x - self.latents_mean) / self.latents_std

    def latent_unscale_fn(self, x):
        return x * self.latents_std + self.latents_mean

    def add_feedback_for_transformer(self):
        self.use_feedback = True
        self.transformer.patch_embedding.weight = nn.Parameter(torch.nn.functional.pad(self.transformer.patch_embedding.weight, (0, 0, 0, 0, 0, 0, 0, self.feat_dim + self.latent_dim)))
    
    def encode_text(self, texts):
        max_sequence_length = 512

        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if getattr(self, "offload_t5", False):
            text_input_ids = text_inputs.input_ids.to("cpu")
            mask = text_inputs.attention_mask.to("cpu")
        else:
            text_input_ids = text_inputs.input_ids.to(self.device)
            mask = text_inputs.attention_mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        if getattr(self, "offload_t5", False):
            with torch.no_grad():
                text_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state.to(self.device)
        else:
            text_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state
        text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in text_embeds], dim=0
        )
        return text_embeds.float()

    def forward_generator(self, noisy_latents, raymaps, condition_latents, t, text_embeds, cameras, render_cameras, image_height, image_width, need_3d_mode=True):
        with onload_model(self.transformer, self.device, onload=self.offload_transformer_during_vae):
            out = self.transformer(
                hidden_states=torch.cat([noisy_latents, raymaps, condition_latents], dim=1),
                timestep=t,
                encoder_hidden_states=text_embeds,
                return_dict=False,
            )[0]

        v_pred, feats = out.split([self.latent_dim, self.feat_dim], dim=1)
               
        sigma = torch.stack([self.scheduler.sigmas[self.scheduler.index_for_timestep(_t)] for _t in t.unbind(0)], dim=0).to(self.device)
        latents_pred_2d = noisy_latents - sigma * v_pred

        if need_3d_mode:
            scene_params = self.recon_decoder(
                                einops.rearrange(feats, 'B C T H W -> (B T) C H W').unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                                einops.rearrange(self.latent_unscale_fn(latents_pred_2d.detach()), 'B C T H W -> (B T) C H W').unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                                cameras.to(self.device if not self.offload_vae else "cpu").float()
                            ).flatten(1, -2).to(self.device).float()

            images_pred, _ = self.recon_decoder.render(scene_params.unbind(0), render_cameras, image_height, image_width, bg_mode="white")

            latents_pred_3d = einops.rearrange(self.latent_scale_fn(self.vae.encode(
                            einops.rearrange(images_pred, 'B T C H W -> (B T) C H W', T=images_pred.shape[1]).unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                        ).latent_dist.sample().to(self.device)).squeeze(2), '(B T) C H W -> B C T H W', T=images_pred.shape[1]).to(noisy_latents.dtype)

        return {
            '2d': latents_pred_2d,
            '3d': latents_pred_3d if need_3d_mode else None,
            'rgb_3d': images_pred if need_3d_mode else None,
            'scene': scene_params if need_3d_mode else None,
            'feat': feats
        }

    @torch.no_grad()
    def generate(self, cameras, n_frame, image=None, text="", image_index=0, image_height=480, image_width=704, video_path=None, video_fps=15):  
        # Use the device set during initialization, don't override it
        if not hasattr(self, 'device') or self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device)  
        self.vae.to(self.device)
        self.text_encoder.to(self.device if not self.offload_t5 else "cpu")
        self.transformer.to(self.device)
        self.recon_decoder.to(self.device)
        self.timesteps = self.timesteps.to(self.device)
        self.latents_mean = self.latents_mean.to(self.device)
        self.latents_std = self.latents_std.to(self.device)

        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            batch_size = 1
            
            cameras = cameras.to(self.device).unsqueeze(0)

            if cameras.shape[1] != n_frame:
                cameras = sample_from_dense_cameras(cameras.squeeze(0), torch.linspace(0, 1, n_frame, device=self.device)).unsqueeze(0)

            if video_path is not None:
                render_cameras = sample_from_dense_cameras(cameras.squeeze(0), torch.linspace(0, 1, (n_frame - 1) * video_fps + 1, device=self.device)).unsqueeze(0)
            else:
                render_cameras = None
            
            cameras, ref_w2c, T_norm = normalize_cameras(cameras, return_meta=True, n_frame=None)

            render_cameras = normalize_cameras(render_cameras, ref_w2c=ref_w2c, T_norm=T_norm, n_frame=None) if render_cameras is not None else None

            text = "[Static] " + text

            text_embeds = self.encode_text([text])

            masks = torch.zeros(batch_size, n_frame, device=self.device)

            condition_latents = torch.zeros(batch_size, self.latent_dim, n_frame, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor, device=self.device)

            if image is not None:
                image = image.to(self.device)

                latent = self.latent_scale_fn(self.vae.encode(
                        image.unsqueeze(0).unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16)
                    ).latent_dist.sample().to(self.device)).squeeze(2)

                masks[:, image_index] = 1
                condition_latents[:, :, image_index] = latent

            raymaps = create_raymaps(cameras, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor)
            raymaps = einops.rearrange(raymaps, 'B T H W C -> B C T H W', T=n_frame)
            
            noise = torch.randn(batch_size, self.latent_dim, n_frame, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor, device=self.device)

            noisy_latents = noise 

            if self.use_feedback:
                prev_latents_pred = torch.zeros(batch_size, self.latent_dim, n_frame, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor, device=self.device)

                prev_feats = torch.zeros(batch_size, self.feat_dim, n_frame, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor, device=self.device)

            for i in range(len(self.denoising_steps)):
                t_ids = torch.full((noisy_latents.shape[0],), self.denoising_steps[i], device=self.device)

                t = self.timesteps[t_ids]

                if self.use_feedback:
                    _condition_latents = torch.cat([condition_latents, prev_feats, prev_latents_pred], dim=1)
                else:
                    _condition_latents = condition_latents

                if i < len(self.denoising_steps) - 1:
                    out = self.forward_generator(noisy_latents, raymaps, _condition_latents, t, text_embeds, cameras, cameras, image_height, image_width, need_3d_mode=True)

                    latents_pred = out["3d"]

                    if self.use_feedback:
                        prev_latents_pred = latents_pred
                        prev_feats = out['feat']
                   
                    noisy_latents = self.scheduler.scale_noise(latents_pred, self.timesteps[torch.full((noisy_latents.shape[0],), self.denoising_steps[i + 1], device=self.device)], torch.randn_like(noise))
                    
                else:
                    with onload_model(self.transformer, self.device, onload=self.offload_transformer_during_vae):
                        out = self.transformer(
                            hidden_states=torch.cat([noisy_latents, raymaps, _condition_latents], dim=1),
                            timestep=t,
                            encoder_hidden_states=text_embeds,
                            return_dict=False,
                        )[0]

                    v_pred, feats = out.split([self.latent_dim, self.feat_dim], dim=1)
                        
                    sigma = torch.stack([self.scheduler.sigmas[self.scheduler.index_for_timestep(_t)] for _t in t.unbind(0)], dim=0).to(self.device)
                    latents_pred = noisy_latents - sigma * v_pred

                    scene_params = self.recon_decoder(
                                        einops.rearrange(feats, 'B C T H W -> (B T) C H W').unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                                        einops.rearrange(self.latent_unscale_fn(latents_pred.detach()), 'B C T H W -> (B T) C H W').unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                                        cameras.to(self.device if not self.offload_vae else "cpu").float()
                                    ).flatten(1, -2).to(self.device).float()

            if video_path is not None:
                interpolated_images_pred, _ = self.recon_decoder.render(scene_params.unbind(0), render_cameras, image_height, image_width, bg_mode="white")

                interpolated_images_pred = einops.rearrange(interpolated_images_pred[0].clamp(-1, 1).add(1).div(2), 'T C H W -> T H W C')

                interpolated_images_pred = [img.detach().cpu().mul(255).numpy().astype(np.uint8) for img in interpolated_images_pred.unbind(0)]

                import imageio
                imageio.mimwrite(video_path, interpolated_images_pred, fps=video_fps, quality=8, macro_block_size=1) 

        scene_params = scene_params[0]

        return scene_params, ref_w2c, T_norm

