import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
import torchvision.transforms.functional as F
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from huggingface_hub import snapshot_download
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from ....base_models.diffusion_model.video.wan_2p1.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ....base_models.diffusion_model.video.wan_2p1.modules.vae import WanVAE
from ...base_synthesis import BaseSynthesis
from .cosmos2p5.utils.utils import load_official_weights, get_cosmos_2b_config
from .cosmos2p5.utils.registry import COSMOS_2P5_TASKS, COSMOS_2P5_REGISTRY
from .cosmos2p5.modules import (
    Cosmos25ControlNet3DModel, Cosmos25MultiControlNet3DModel, Cosmos25Transformer3DModel, Reason1TextEncoder
)


PipelineImageInput = Union[
    Image.Image, 
    torch.Tensor, 
    List[Image.Image], 
    List[torch.Tensor]
]


class CosmosPredict2p5Synthesis(BaseSynthesis):

    def __init__(
        self,
        mode: str = "img2world",
        text_encoder: Any = None,
        transformer: Any = None,
        vae: Any = None,
        scheduler: Any = None,
    ):
        self.mode = mode 

        # Initialize components
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler

        # Hard coded Wan VAE scaling factors (refactor later if needed)
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8
        self.latent_channels = 16

        self.video_processor = VideoProcessor(vae_scale_factor=8)
        self._guidance_scale = 1.0
        self._execution_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        mode: str,
        transformer_model_path: str,
        text_encoder_model_path: str,
        vae_model_path: str,
        token: Optional[str] = None,
        device: Optional[torch.device] = None,
        weight_dtype: Optional[torch.dtype] = torch.float32,
    ) -> "Cosmos2p5PredictSynthesis":
        """
        Cosmos-Predict2.5 supports two modes: ['img2world', 'video2world'], currently we only support img2world.
        """
        # Check mode
        if mode not in COSMOS_2P5_TASKS:
            raise ValueError(f"Unsupported mode: {mode}")

        # Download models from Hugging Face if local paths are not provided
        transformer_path_obj = Path(transformer_model_path)
        if not transformer_path_obj.is_dir() and not transformer_path_obj.is_file():
            if token is None:
                raise ValueError(
                    "Hugging Face token must be provided to download Cosmos checkpoints "
                )
            local_dir = Path.cwd() / transformer_model_path.split("/")[-1]
            local_dir.mkdir(parents=True, exist_ok=True)
            transformer_path_obj = Path(snapshot_download(
                repo_id=transformer_model_path,
                allow_patterns="base/post_trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                token=token
            ))
        if transformer_path_obj.is_dir():
            transformer_ckpt = transformer_path_obj / "base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
        else:
            transformer_ckpt = transformer_path_obj
        if not transformer_ckpt.exists():
            raise FileNotFoundError(f"Transformer ckpt not found: {transformer_ckpt}")

        text_encoder_path = Path(text_encoder_model_path)
        if not  text_encoder_path.exists():
            local_dir = Path.cwd() / text_encoder_model_path.split("/")[-1]
            local_dir.mkdir(parents=True, exist_ok=True)
            text_encoder_path = Path(snapshot_download(
                repo_id=text_encoder_model_path,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            ))

        vae_path_obj = Path(vae_model_path)
        if not vae_path_obj.is_dir() and not vae_path_obj.is_file():
            local_dir = Path.cwd() / vae_model_path.split("/")[-1]
            vae_path_obj = Path(snapshot_download(
                repo_id=vae_model_path,
                allow_patterns="Wan2.1_VAE.pth",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            ))
        if vae_path_obj.is_dir():
            vae_ckpt = vae_path_obj / "Wan2.1_VAE.pth"
        else:
            vae_ckpt = vae_path_obj
        if not vae_ckpt.exists():
            raise FileNotFoundError(f"VAE ckpt not found: {vae_ckpt}")

        #  Load cosmos models
        config_args = get_cosmos_2b_config(mode='base')
        transformer = Cosmos25Transformer3DModel(**config_args)
        load_official_weights(transformer, str(transformer_ckpt))
        text_encoder = Reason1TextEncoder(str(text_encoder_path))
        vae = WanVAE(vae_pth=str(vae_ckpt), dtype=weight_dtype, device=device)  # Vae device & dtype specify in initalization
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        
        # Move to device & change dtype
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transformer = transformer.to(device, weight_dtype)
        text_encoder = text_encoder.to(device, weight_dtype)
        
        instance = cls(
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mode=mode
        )
        instance._execution_device = device
        return instance

    def api_init(self, api_key, endpoint):
        raise NotImplementedError

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        """Encodes the text prompt(s) into embeddings."""
        if prompt_embeds is None:
            if isinstance(prompt, str):
                prompt = [prompt]
            prompt_embeds = self.text_encoder.encode_prompts(prompt)
        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = [''] * prompt_embeds.shape[0]
                elif isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]
                negative_prompt_embeds = self.text_encoder.encode_prompts(negative_prompt)
        return prompt_embeds, negative_prompt_embeds

    def encode(self, video):
        """Encodes a video tensor into the latent space using the VAE."""
        video = video.to(self.vae.dtype)
        latents = self.vae.encode(video)[0]  # List[torch.Tensor]
        return latents

    def decode(self, latents):
        """Decodes a latent tensor back into a video using the VAE."""
        latents = latents.to(self.vae.dtype)
        video = self.vae.decode(latents)[0]  # List[torch.Tensor]
        return video

    def prepare_cond_latents(
        self,
        image: PipelineImageInput,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        pad_mode: str,
    ):
        """Prepares conditional latents from an initial image for image-to-video generation."""
        if image is None:
            images = []
        elif isinstance(image, list) or isinstance(image, torch.Tensor):
            images = image
        else:
            images = [image]

        if len(images) == 0:  # Text-to-video case
            latents_shape = [
                1,
                self.latent_channels,
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            ]
            cond_latents = torch.zeros(latents_shape, device=device, dtype=dtype)
            cond_masks = torch.zeros((1, 1, cond_latents.shape[2], 1, 1), device=device, dtype=dtype)
        else:  # Image-to-video case
            num_images = len(images)
            assert num_images <= num_frames and (num_images - 1) % self.vae_scale_factor_temporal == 0
            num_cond_frames = 1 + (num_images - 1) // self.vae_scale_factor_temporal

            # Preprocess and normalize images
            if isinstance(images, torch.Tensor):
                cond_images = 2.0 * images - 1.0
            else:
                cond_images = [np.array(image) for image in images]
                cond_images = np.stack(cond_images, axis=0)
                cond_images = rearrange(cond_images, 't h w c -> t c h w')
                cond_images = cond_images / 127.5 - 1.0
                cond_images = torch.from_numpy(cond_images)
            cond_images = cond_images.to(device, dtype)
            cond_images = F.resize(
                cond_images,
                size=(height, width),  # type: ignore
                interpolation=F.InterpolationMode.BICUBIC,
                antialias=True,
            )

            # Pad the conditional frames to the full length
            if pad_mode == 'repeat':
                last_image = cond_images[-1:]
            elif pad_mode == 'zero':
                last_image = torch.zeros_like(cond_images[-1:])
            else:
                raise ValueError(f'Invalid pad_mode: {pad_mode}')
            last_images = last_image.repeat(num_frames - cond_images.shape[0], 1, 1, 1)
            cond_images = torch.cat([cond_images, last_images], dim=0)
            cond_images = rearrange(cond_images, 't c h w -> 1 c t h w')

            # Encode images to latents and create the mask
            cond_latents = self.encode(cond_images).to(dtype)
            # Reshape to desire tensor
            if cond_latents.ndim == 4:
                cond_latents = cond_latents.unsqueeze(0)
            expected_t = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            if cond_latents.shape[2] != expected_t and cond_latents.shape[-1] == expected_t:
                cond_latents = cond_latents.unsqueeze(0).permute(0, 1, 4, 2, 3)
            # Compute cond mask
            cond_masks = torch.zeros((1, 1, cond_latents.shape[2], 1, 1), device=device, dtype=dtype)
            cond_masks[:, :, :num_cond_frames] = 1
        return cond_latents, cond_masks

    def prepare_control_video(self, control_video, control_scale, num_frames, height, width):
        """Prepares control video(s) by processing and encoding them into latents."""
        if isinstance(control_video, dict):  # Multi-ControlNet case
            control_latents = dict()
            for key, video in control_video.items():
                assert len(video) == num_frames
                video = self.video_processor.preprocess_video(video, height, width).to(self.vae.device)
                control_latent = self.encode(video)
                if self.do_classifier_free_guidance:
                    control_latent = torch.cat([control_latent, control_latent])
                control_latents[key] = control_latent

            if isinstance(control_scale, dict):
                control_scales = control_scale
                assert len(control_latents) == len(control_scales)
                for key in control_latents:
                    assert key in control_scales
            elif isinstance(control_scale, float):
                control_scales = {key: control_scale for key in control_latents}
            else:
                raise TypeError('control_scale must be a float or a dictionary of floats.')
        else:  # Single ControlNet case
            assert len(control_video) == num_frames and isinstance(control_scale, float)
            control_video = self.video_processor.preprocess_video(control_video, height, width).to(self.vae.device)
            control_latents = self.encode(control_video)
            if self.do_classifier_free_guidance:
                control_latents = torch.cat([control_latents, control_latents])
            control_scales = control_scale
        return control_latents, control_scales

    @property
    def guidance_scale(self):
        """The guidance scale for classifier-free guidance."""
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        """Whether to perform classifier-free guidance."""
        return self._guidance_scale > 1.0

    @torch.no_grad()
    def predict(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image: Optional[PipelineImageInput] = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 35,
        fps: int = 28,
        num_frames: int = 93,
        height: int = 704,
        width: int = 1280,
        action_latents: Optional[torch.Tensor] = None,
        control_video: Optional[Union[List[Image.Image], Dict[str, List[Image.Image]]]] = None,
        control_scale: Optional[Union[float, Dict[str, float]]] = 1.0,
        cond_timestep: float = 0,
        timestep_scale: float = 0.001,
        seed: int = -1,
        use_kerras_sigma: bool = True,
        pad_mode: str = 'repeat',
        output_type: Optional[str] = 'pil',
    ):
        """The main call function to generate a video.

        Args:
            prompt (`str`): The text prompt to guide generation.
            negative_prompt (`str`, *optional*): A prompt to steer generation away from.
            image (`PipelineImageInput`, *optional*): An initial image for image-to-video generation.
            guidance_scale (`float`, *optional*, defaults to 7.0): Scale for classifier-free guidance.
            num_inference_steps (`int`, *optional*, defaults to 35): Number of denoising steps.
            fps (`int`, *optional*, defaults to 28): Frames per second of the output video.
            num_frames (`int`, *optional*, defaults to 93): Number of frames in the output video.
            height (`int`, *optional*, defaults to 704): Height of the output video.
            width (`int`, *optional*, defaults to 1280): Width of the output video.
            action_latents (`torch.Tensor`, *optional*): Pre-computed action latents for conditioning.
            control_video (`Union[List[Image.Image], Dict[str, List[Image.Image]]]`, *optional*): Control video(s).
            control_scale (`Union[float, Dict[str, float]]`, *optional*, defaults to 1.0): ControlNet scaling factor(s).
            cond_timestep (`float`, *optional*, defaults to 0): Timestep for applying image conditioning.
            timestep_scale (`float`, *optional*, defaults to 0.001): A scaling factor for the timestep.
            seed (`int`, *optional*, defaults to -1): A seed for reproducible generation.
            use_kerras_sigma (`bool`, *optional*, defaults to True): Whether to use Kerras sigmas for the scheduler.
            pad_mode (`str`, *optional*, defaults to 'repeat'): Padding mode for image-to-video.
            output_type (`str`, *optional*, defaults to 'pil'): The desired output format ('pil', 'latent', etc.).

        Returns:
            The generated video in the format specified by `output_type`.
        """
        self._guidance_scale = guidance_scale
        batch_size = 1
        device = self._execution_device
        dtype = self.transformer.dtype

        # Setup random generator for reproducibility
        generator = None
        if seed > 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        # Encode prompts
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt=negative_prompt)
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_embeds = prompt_embeds.to(device, dtype)

        # Prepare scheduler and timesteps
        self.scheduler.set_timesteps(
            num_inference_steps,
            device=device,
            shift=5.0,
        )
        timesteps = self.scheduler.timesteps

        # Prepare initial random latents
        shape = (
            batch_size,
            self.latent_channels,
            ((num_frames - 1) // self.vae_scale_factor_temporal + 1),
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # Prepare conditional latents for image-to-video
        cond_latents, cond_masks = self.prepare_cond_latents(
            image,
            num_frames=num_frames,
            height=height,
            width=width,
            device=device,
            dtype=torch.float32,
            pad_mode=pad_mode,
        )
        masks_input = cond_masks.repeat(batch_size, 1, 1, cond_latents.shape[-2], cond_latents.shape[-1])
        if self.do_classifier_free_guidance:
            masks_input = torch.cat([masks_input] * 2)
        masks_input = masks_input.to(dtype)

        # # Prepare action latents if provided
        # if action_latents is not None:
        #     action_latents = action_latents.to(device=device, dtype=dtype)
        #     if self.do_classifier_free_guidance:
        #         action_latents = torch.cat([action_latents] * 2)

        # # Prepare control video latents if provided
        # if control_video is not None:
        #     control_latents, control_scales = self.prepare_control_video(
        #         control_video,
        #         control_scale,
        #         num_frames=num_frames,
        #         height=height,
        #         width=width,
        #     )

        # Prepare padding mask
        padding_mask = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
        if self.do_classifier_free_guidance:
            padding_mask = torch.cat([padding_mask, padding_mask], dim=0)

        # Denoising loop
        noise = latents
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Apply conditioning from the initial image
                latent_model_input = latents
                latent_model_input = cond_latents * cond_masks + latent_model_input * (1 - cond_masks)
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latent_model_input] * 2)
                latent_model_input = latent_model_input.to(dtype)

                # Prepare timestep tensor, applying conditioning if specified
                timestep = torch.stack([t]).unsqueeze(0)
                if cond_timestep > 0:
                    cond_timestep_masks = cond_masks[:, 0, :, 0, 0]
                    timestep = cond_timestep * cond_timestep_masks + timestep * (1 - cond_timestep_masks)
                if self.do_classifier_free_guidance:
                    timestep = torch.cat([timestep] * 2)
                timestep = timestep * timestep_scale

                # Prepare arguments for the transformer model
                kwargs = dict(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    fps=fps,
                    condition_mask=masks_input,
                    padding_mask=padding_mask,
                )
                # # Add ControlNet conditioning
                # if control_video is not None:
                #     control_samples = self.controlnet(
                #         control_cond=control_latents,
                #         control_scale=control_scales,
                #         **kwargs,
                #     )
                #     kwargs['control_hidden_states'] = control_samples
                # # Add action conditioning
                # if action_latents is not None:
                #     kwargs['action'] = action_latents

                # Predict the noise
                noise_pred = self.transformer(**kwargs)

                # Perform classifier-free guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_text + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                noise_pred = noise_pred.float()
                # Re-apply image conditioning on the predicted noise
                cond_latents_velocity = noise - cond_latents
                noise_pred = cond_latents_velocity * cond_masks + noise_pred * (1 - cond_masks)

                # Scheduler step
                latents = self.scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0]
                latents = latents.squeeze(0)

                # Update progress bar
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        # Decode latents to video
        if not output_type == 'latent':
            video = self.decode(latents).unsqueeze(0)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        return video