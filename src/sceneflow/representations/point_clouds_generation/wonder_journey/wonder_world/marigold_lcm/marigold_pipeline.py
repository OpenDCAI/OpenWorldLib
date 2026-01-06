# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from .marigold_process import MarigoldImageProcessor

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import resize, pil_to_tensor
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.utils.torch_utils import randn_tensor
from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depths
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)
import torch.nn.functional as F

def l2_loss(input, target):
    return (input - target) ** 2

def l1_loss(input, target):
    return (input - target).abs()

class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        seed: Union[int, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        depth_conditioning: bool = False,
        target_depth: torch.Tensor = None,
        mask_align: torch.Tensor = None,
        mask_farther: torch.Tensor = None,
        guidance_steps: int = 8,
        logger=None,
    ):
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            processing_res (`int`, *optional*, defaults to `768`):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            denoising_steps (`int`, *optional*, defaults to `10`):
                Number of diffusion denoising steps (DDIM) during inference.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            seed (`int`, *optional*, defaults to `None`)
                Reproducibility seed.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        assert processing_res >= 0
        assert ensemble_size >= 1

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image.squeeze()
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            3 == rgb.dim() and 3 == input_size[0]
        ), f"Wrong input shape {input_size}, expected [rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            depth_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                seed=seed,
                depth_conditioning=depth_conditioning,
                target_depth=target_depth,
                mask_align=mask_align,
                mask_farther=mask_farther,
                guidance_steps=guidance_steps,
                logger=logger,
            )
            depth_pred_ls.append(depth_pred_raw.detach())
        depth_preds = torch.concat(depth_pred_ls, dim=0).squeeze()
        # torch.cuda.empty_cache()  # clear vram cache for ensembling  # This takes 0.24s

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depths(
                depth_preds, **(ensemble_kwargs or {})
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None

        # ----------------- Post processing -----------------
        # Scale prediction to [0, 1]
        # min_d = torch.min(depth_pred)
        # max_d = torch.max(depth_pred)
        # depth_pred = (depth_pred - min_d) / (max_d - min_d)

        # Resize back to original resolution
        if match_input_res:
            depth_pred = resize(
                depth_pred.unsqueeze(0),
                input_size[1:],
                interpolation=resample_method,
                antialias=True,
            ).squeeze()

        # Convert to numpy
        depth_pred = depth_pred.clamp(0, 1)

        return depth_pred

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        seed: Union[int, None],
        show_pbar: bool,
        depth_conditioning: bool, target_depth: None, mask_align: None,
        mask_farther: torch.Tensor = None,
        guidance_steps: int = 8,
        logger=None,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # Initial depth map (noise)
        if seed is None:
            rand_num_generator = None
        else:
            rand_num_generator = torch.Generator(device=device)
            rand_num_generator.manual_seed(seed)
        depth_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=rand_num_generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        # if depth_conditioning and target_depth is not None and mask_align is not None and torch.any(mask_align):
        #     target_depth_ = (target_depth - 0.001) * 200 * 2 - 1
        #     target_depth_[~mask_align] = torch.randn_like(target_depth_[~mask_align]).clip(-1, 1)
        #     target_depth_ = target_depth_.expand(-1, 3, -1, -1)
        #     depth_latent = self.encode_rgb(target_depth_)

        for i, t in iterable:
            if depth_conditioning and target_depth is not None and mask_align is not None and torch.any(mask_align):
                if i >= len(timesteps) - guidance_steps:
                # if False:
                    with torch.set_grad_enabled(depth_conditioning):
                        depth_latent.requires_grad_(depth_conditioning)
                    
                        unet_input = torch.cat(
                            [rgb_latent, depth_latent], dim=1
                        )  # this order is important

                        # predict the noise residual
                        noise_pred = self.unet(
                            unet_input, t, encoder_hidden_states=batch_empty_text_embed
                        ).sample  # [B, 4, h, w]

                        # compute the previous noisy sample x_t -> x_t-1
                        _step_index = self.scheduler._step_index
                        depth_latent_orig = self.scheduler.step(noise_pred, t, depth_latent).prev_sample
                        self.scheduler._step_index = _step_index  # reset step index; it automatically +=1 in step function

                        # depth conditiong 
                        depth_orig = self.decode_depth(depth_latent_orig)
                        depth_orig = torch.clip(depth_orig, -1.0, 1.0)
                        depth_orig = (depth_orig + 1.0) / 2.0
                        depth_orig = depth_orig / 200   # scale
                        depth_orig = depth_orig + 0.001 # shift  
                        loss = l2_loss(target_depth.detach(), depth_orig)
                        mask_align = mask_align.detach()
                        loss = loss[mask_align].mean()
                        if mask_farther is not None:
                            margin = 0.
                            hinge_loss = (target_depth.detach() + margin - depth_orig).clamp(min=0)
                            hinge_loss = l2_loss(hinge_loss, torch.zeros_like(hinge_loss))
                            hinge_loss = hinge_loss[mask_farther]
                            if torch.any(hinge_loss > 0):
                                hinge_loss = hinge_loss[hinge_loss > 0].mean()
                                loss += hinge_loss
                        dg_grad_wt = 3e9
                        dc_grad = torch.autograd.grad(loss*dg_grad_wt, depth_latent)[0]
                        noise_pred_mag = (noise_pred**2).sum().sqrt().detach()
                        dc_grad_mag = (dc_grad**2).sum().sqrt().detach()
                        n_buffer_steps = 1
                        max_guidance_ratio = 2
                        minimal_guidance_ratio = 0.1
                        ratios = torch.cat([torch.linspace(max_guidance_ratio, minimal_guidance_ratio, guidance_steps-n_buffer_steps), minimal_guidance_ratio*torch.ones(n_buffer_steps)], dim=0)
                        ratio = ratios[i-(len(timesteps) - guidance_steps)]
                        dc_grad = dc_grad / dc_grad_mag * noise_pred_mag * ratio
                        # dc_grad_mag = (dc_grad**2).sum().sqrt().detach()
                        noise_pred = noise_pred + dc_grad
                        # print('Guidance_grad / Noise magnitude', (dc_grad_mag/noise_pred_mag).item(), '\n')
                else:
                    unet_input = torch.cat(
                        [rgb_latent, depth_latent], dim=1
                    )  # this order is important

                    # predict the noise residual
                    noise_pred = self.unet(
                        unet_input, t, encoder_hidden_states=batch_empty_text_embed
                    ).sample  # [B, 4, h, w]

                # compute the previous noisy sample x_t -> x_t-1
                depth_out = self.scheduler.step(
                    noise_pred, t, depth_latent, generator=rand_num_generator
                )
                depth_latent = depth_out.prev_sample.detach()

                # ## Visualize the final prediction at each step
                # pred_original_sample = depth_out.pred_original_sample.detach().to(torch.bfloat16)
                # depth = self.decode_depth(pred_original_sample)

                # # clip prediction
                # depth = torch.clip(depth, -1.0, 1.0)
                # # shift to [0, 1]
                # depth = (depth + 1.0) / 2.0
                # from util.utils import save_depth_map
                # # save_depth_map(depth[0, 0].to(torch.float32).cpu().numpy() / 200 + 0.001, f"tmp/viz_depth_progression/{i:02d}.png", vmax=0.006, vmin=0)
                # if target_depth is not None:
                #     depth_orig = depth / 200   # scale
                #     depth_orig = depth_orig + 0.001 # shift  
                #     loss = l1_loss(target_depth.detach(), depth_orig)
                #     mask_align = mask_align.detach()
                #     loss = loss[mask_align].mean()
                #     if mask_farther is None:
                #         print('-- Alignment loss {}'.format(loss.item()))
                #     else:
                #         margin = 0.0001
                #         hinge_loss = (target_depth.detach() + margin - depth_orig).clamp(min=0)
                #         hinge_loss = l2_loss(hinge_loss, torch.zeros_like(hinge_loss))
                #         hinge_loss = hinge_loss[mask_farther].mean()
                #         hinge_loss *= 5
                #         print('[Depth] -- Alignment loss {} hinge_loss {}'.format(loss.item(), hinge_loss.item()))

            else:
                unet_input = torch.cat(
                    [rgb_latent, depth_latent], dim=1
                )  # this order is important

                # predict the noise residual
                noise_pred = self.unet(
                    unet_input, t, encoder_hidden_states=batch_empty_text_embed
                ).sample  # [B, 4, h, w]

                # compute the previous noisy sample x_t -> x_t-1
                depth_latent = self.scheduler.step(
                    noise_pred, t, depth_latent, generator=rand_num_generator
                ).prev_sample

        depth = self.decode_depth(depth_latent)

        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        if target_depth is not None:
            with torch.no_grad():
                depth_orig = depth / 200   # scale
                depth_orig = depth_orig + 0.001 # shift  
                loss = l1_loss(target_depth.detach(), depth_orig)
                mask_align = mask_align.detach()
                loss = loss[mask_align].mean()
                if mask_farther is None:
                    print('[Depth] -- Alignment loss {}'.format(loss.item()))
                else:
                    margin = 0.0001
                    hinge_loss = (target_depth.detach() + margin - depth_orig).clamp(min=0)
                    hinge_loss = l2_loss(hinge_loss, torch.zeros_like(hinge_loss))
                    hinge_loss = hinge_loss[mask_farther].mean()
                    hinge_loss *= 5
                    print('[Depth] -- Alignment loss {} hinge_loss {}'.format(loss.item(), hinge_loss.item()))

        return depth

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean




class MarigoldPipelineNormal(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        seed: Union[int, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        depth_conditioning: bool = False,
        target_depth: torch.Tensor = None,
        mask_align: torch.Tensor = None,
        mask_farther: torch.Tensor = None,
        logger=None,
    ):
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            processing_res (`int`, *optional*, defaults to `768`):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            denoising_steps (`int`, *optional*, defaults to `10`):
                Number of diffusion denoising steps (DDIM) during inference.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            seed (`int`, *optional*, defaults to `None`)
                Reproducibility seed.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        assert processing_res >= 0
        assert ensemble_size >= 1

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        rgb = input_image.squeeze()
        input_size = rgb.shape

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb * 2.0 - 1.0  #  [0, 1] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        batched_img = rgb_norm.unsqueeze(0)
        depth_pred = self.single_infer(
            rgb_in=batched_img,
            num_inference_steps=denoising_steps,
            show_pbar=show_progress_bar,
            seed=seed,
            depth_conditioning=depth_conditioning,
            target_depth=target_depth,
            mask_align=mask_align,
            mask_farther=mask_farther,
            logger=logger,
        )  # [1, 3, H, W], ranging  [-1, 1]
        
        # Resize back to original resolution
        if match_input_res:
            depth_pred = resize(
                depth_pred,
                input_size[1:],
                interpolation=resample_method,
                antialias=True,
            ).squeeze()

        return depth_pred

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        seed: Union[int, None],
        show_pbar: bool,
        depth_conditioning: bool, target_depth: None, mask_align: None,
        mask_farther: torch.Tensor = None,
        logger=None,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # Initial depth map (noise)
        if seed is None:
            rand_num_generator = None
        else:
            rand_num_generator = torch.Generator(device=device)
            rand_num_generator.manual_seed(seed)
        depth_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=rand_num_generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, depth_latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(
                noise_pred, t, depth_latent, generator=rand_num_generator
            ).prev_sample

        decoded = self.decode_depth(depth_latent)

        # clip prediction
        decoded_clipped = torch.clip(decoded, -1.0, 1.0)  # [1, 3, H, W]
        decoded_clipped[:, -1].clamp_(0, 1)
        normal_renormed = F.normalize(decoded_clipped, p=2, dim=1)  # [1, 3, H, W], ranging  [-1, 1]

        return normal_renormed

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        decoded = self.vae.decoder(z)
        return decoded


def normalize_normals(normals: torch.FloatTensor, eps: float = 1e-6) -> torch.FloatTensor:
    assert normals.dim() == 4

    norm = torch.norm(normals, dim=1, keepdim=True)
    normals /= norm.clamp(min=eps)

    return normals


def ensemble_normals(
    normals: torch.FloatTensor, output_uncertainty: bool, reduction: str = "closest"
) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
    assert normals.dim() == 4
    assert reduction in ("closest", "mean")

    E, C, H, W = normals.shape
    assert C == 3

    mean_normals = normals.mean(dim=0, keepdim=True)  # [1,3,H,W]
    mean_normals = normalize_normals(mean_normals)  # [1,3,H,W]

    sim_cos = (mean_normals * normals).sum(dim=1, keepdim=True)  # [E,1,H,W]

    uncertainty = None
    if output_uncertainty:
        uncertainty = sim_cos.arccos()  # [E,1,H,W]
        uncertainty = uncertainty.mean(dim=0, keepdim=True) / np.pi  # [1,1,H,W]

    if reduction == "mean":
        return mean_normals, uncertainty  # [1,3,H,W], [1,1,H,W]

    closest_indices = sim_cos.argmax(dim=0, keepdim=True)  # [1,1,H,W]
    closest_indices = closest_indices.repeat(1, 3, 1, 1)  # [1,3,H,W]
    closest_normals = torch.gather(normals, 0, closest_indices)

    return closest_normals, uncertainty  # [1,3,H,W], [1,1,H,W]


@dataclass
class MarigoldNormalsOutput(BaseOutput):
    """
    Output class for Marigold monocular normals prediction pipeline.

    Args:
        prediction (`np.ndarray`, `torch.FloatTensor`):
            Predicted normals, with values in the range [-1, 1]. For types `np.ndarray` or `torch.FloatTensor`, the
            shape is always $numimages \times 3 \times height \times width$.
        visualization (`None` or List[PIL.Image.Image]):
            Colorized predictions for visualization.
        uncertainty (`None`, `np.ndarray`, `torch.FloatTensor`):
            Uncertainty maps computed from the ensemble. The shape is $numimages \times 1 \times height \times width$.
        latent (`None`, `torch.FloatTensor`):
            Latent features corresponding to the predictions. The shape is $numimages * numensemble \times 4 \times
            latentheight \times latentwidth$.
    """

    prediction: Union[np.ndarray, torch.FloatTensor]
    visualization: Union[None, Image.Image, List[Image.Image]]
    uncertainty: Union[None, np.ndarray, torch.FloatTensor]
    latent: Union[None, torch.FloatTensor]


class MarigoldNormalsPipeline(DiffusionPipeline):
    """
    Pipeline for monocular normals estimation using the Marigold method: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the normals latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and predictions to and from latent
            representations.
        scheduler (`DDIMScheduler` or `LCMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    model_cpu_offload_seq = "text_encoder->vae.encoder->unet->vae.decoder"

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
        use_full_z_range: Optional[bool] = None,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
            use_full_z_range=use_full_z_range,
        )

        self.vae_scale_factor = 8
        self.latent_space_size = self.vae.config.latent_channels
        self.latent_scaling_factor = self.vae.config.scaling_factor
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution
        self.use_full_z_range = use_full_z_range

        self.empty_text_embedding = None

        self.image_processor = MarigoldImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @torch.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, torch.FloatTensor],
        num_inference_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_resolution: Optional[int] = None,
        match_input_resolution: bool = True,
        resample_method_input: str = "bilinear",
        resample_method_output: str = "bilinear",
        batch_size: int = 1,
        check_input: bool = True,
        ensembling_kwargs: Optional[Dict[str, Any]] = None,
        latents: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_prediction_format: str = "np",
        output_visualization: bool = True,
        output_visualization_kwargs: Optional[Dict[str, Any]] = None,
        output_uncertainty: bool = True,
        output_latent: bool = False,
        **kwargs,
    ) -> MarigoldNormalsOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`):
                Input image or stacked images.
            num_inference_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, defaults to `1`):
                Number of ensemble predictions. Recommended values are 5 and higher for better precision, or 1 for
                faster inference.
            processing_resolution (`int`, *optional*, defaults to None):
                Effective processing resolution. When set to `0`, matches the larger input image dimension. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_resolution (`bool`, *optional*, defaults to `True`):
                When enabled, the output prediction is resized to match the input dimensions. When disabled, the longer
                side of the output will equal to `processing_resolution`.
            resample_method_input: (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize input images to `processing_resolution`. The accepted values are:
                `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            resample_method_output: (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize output predictions to match the input resolution. The accepted values
                are `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            batch_size (`int`, *optional*, defaults to `1`):
                Batch size; only matters when setting `ensemble_size` or passing a tensor of images.
            check_input (`bool`, defaults to `False`):
                Extra steps to validate compatibility of the inputs with the model.
            ensembling_kwargs (`dict`, *optional*, defaults to `None`)
                Extra dictionary with arguments for precise ensembling control. The following options are available:
                - reduction (`str`, *optional*, defaults to `"closest"`): Defines the ensembling function applied in
                  every pixel location, can be either `"closest"` or `"mean"`.
            latents (`torch.Tensor`, *optional*, defaults to `None`):
                Latent noise tensors to replace the random initialization. These can be taken from the previous
                function call's output.
            generator (`torch.Generator`, or `List[torch.Generator]`, *optional*, defaults to `None`):
                Random number generator object to ensure reproducibility.
            output_prediction_format (`str`, *optional*, defaults to `"np"`):
                Preferred format of the output's `prediction` and the optional `uncertainty` fields. The accepted
                values are: `"np"` (numpy array) or `"pt"` (torch tensor).
            output_visualization (`bool`, *optional*, defaults to `True`):
                When enabled, the output's `visualization` field contains a PIL.Image that can be used for visual
                quality inspection.
            output_visualization_kwargs (`dict`, *optional*, defaults to `None`):
                Extra dictionary with arguments for precise visualization control. Flipping axes leads to a different
                color scheme. The following options are available:
                - flip_x (`bool`, *optional*, defaults to `False`): Flips the X axis of the normals frame of reference.
                  Default direction is right.
                - flip_y (`bool`, *optional*, defaults to `False`): Flips the Y axis of the normals frame of reference.
                  Default direction is top.
                - flip_z (`bool`, *optional*, defaults to `False`): Flips the Z axis of the normals frame of reference.
                  Default direction is facing the observer.
            output_uncertainty (`bool`, *optional*, defaults to `True`):
                When enabled, the output's `uncertainty` field contains the predictive uncertainty map, provided that
                the `ensemble_size` argument is set to a value above 2.
            output_latent (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `latent` field contains the latent codes corresponding to the predictions
                within the ensemble. These codes can be saved, modified, and used for subsequent calls with the
                `latents` argument.

        Examples:

        Returns:
            `MarigoldNormalsOutput`: Output class instance for Marigold monocular normals prediction pipeline.
        """

        t0 = time.time()
        # 0. Resolving variables
        device = self._execution_device
        dtype = self.dtype

        num_images = 1
        if (isinstance(image, np.ndarray) or torch.is_tensor(image)) and image.ndim == 4:
            num_images = image.shape[0]

        if num_inference_steps is None:
            num_inference_steps = self.default_denoising_steps
        if processing_resolution is None:
            processing_resolution = self.default_processing_resolution

        # 2. Prepare empty text conditioning. Model invocation: self.tokenizer, self.text_encoder
        if self.empty_text_embedding is None:
            self.encode_empty_text()

        # 3. Preprocessing input image
        image, original_resolution = self.image_processor.preprocess(
            image, processing_resolution, resample_method_input, check_input, device, dtype
        )  # [N,3,PPH,PPW], ranging [-1, 1]

        # 4. Encode input image into latent space. Model invocation: self.vae.encoder
        image_latent, pred_latent = self.prepare_latent(
            image, latents, generator, ensemble_size, batch_size
        )  # [N*E,4,h,w], [N*E,4,h,w]

        batch_empty_text_embedding = self.empty_text_embedding.to(device=device, dtype=dtype).repeat(
            batch_size, 1, 1
        )  # [B,1024,2]

        # 5. Denoising loop. Model invocation: self.unet
        clean_latent = []

        with self.progress_bar(total=num_images * ensemble_size * num_inference_steps) as progress_bar:
            for i in range(0, num_images * ensemble_size, batch_size):
                batch_image_latent = image_latent[i : i + batch_size]  # [B,4,h,w]
                batch_pred_latent = pred_latent[i : i + batch_size]  # [B,4,h,w]
                B = batch_image_latent.shape[0]

                batch_text_embedding = batch_empty_text_embedding[:B]  # [B,2,1024]

                self.scheduler.set_timesteps(num_inference_steps, device=device)

                for t in self.scheduler.timesteps:
                    batch_latent = torch.cat([batch_image_latent, batch_pred_latent], dim=1)  # [B,8,h,w]
                    noise = self.unet(batch_latent, t, encoder_hidden_states=batch_text_embedding).sample  # [B,4,h,w]
                    batch_pred_latent = self.scheduler.step(
                        noise, t, batch_pred_latent, generator=generator
                    ).prev_sample  # [B,4,h,w]
                    progress_bar.update(B)

                clean_latent.append(batch_pred_latent)

                del batch_image_latent, batch_pred_latent, batch_text_embedding, batch_latent, noise

        pred_latent = torch.cat(clean_latent, dim=0)  # [N*E,4,h,w]

        # 6. Decode prediction from latent into pixel space. Model invocation: self.vae.decoder
        prediction = torch.cat(
            [
                self.decode_prediction(pred_latent[i : i + batch_size])
                for i in range(0, pred_latent.shape[0], batch_size)
            ],
            dim=0,
        )  # [N*E,3,PPH,PPW]

        if not output_latent:
            pred_latent = None

        # 7. Postprocess predictions
        # prediction = self.image_processor.unpad_image(prediction, padding)  # [N*E,3,PH,PW]

        uncertainty = None
        if ensemble_size > 1:
            prediction = prediction.reshape(num_images, ensemble_size, *prediction.shape[1:])  # [N,E,3,PH,PW]
            prediction = [
                ensemble_normals(prediction[i], output_uncertainty, **(ensembling_kwargs or {}))
                for i in range(num_images)
            ]  # [ [[1,3,PH,PW], [1,1,PH,PW]], ... ]
            prediction, uncertainty = zip(*prediction)  # [[1,3,PH,PW], ... ], [[1,1,PH,PW], ... ]
            prediction = torch.cat(prediction, dim=0)  # [N,3,PH,PW]
            uncertainty = torch.cat(uncertainty, dim=0)  # [N,1,PH,PW]

        if match_input_resolution:
            prediction = self.image_processor.resize_antialias(
                prediction, original_resolution, resample_method_output, is_aa=False
            )  # [N,3,H,W]
            prediction = normalize_normals(prediction)  # [N,3,H,W]
            if uncertainty is not None and output_uncertainty:
                uncertainty = self.image_processor.resize_antialias(
                    uncertainty, original_resolution, resample_method_output, is_aa=False
                )  # [N,1,H,W]

        return prediction

    def prepare_latent(
        self,
        image: torch.FloatTensor,
        latents: Optional[torch.FloatTensor],
        generator: Optional[torch.Generator],
        ensemble_size: int,
        batch_size: int,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        image_latent = torch.cat(
            [self.encode_image(image[i : i + batch_size]) for i in range(0, image.shape[0], batch_size)], dim=0
        )  # [N,4,h,w]
        image_latent = image_latent.repeat_interleave(ensemble_size, dim=0)  # [N*E,4,h,w]

        pred_latent = latents
        if pred_latent is None:
            pred_latent = randn_tensor(
                image_latent.shape,
                generator=generator,
                device=image_latent.device,
                dtype=image_latent.dtype,
            )  # [N*E,4,h,w]

        return image_latent, pred_latent

    def decode_prediction(self, pred_latent: torch.FloatTensor) -> torch.FloatTensor:
        assert pred_latent.dim() == 4 and pred_latent.shape[1] == self.latent_space_size  # [B,4,h,w]

        prediction = self.vae.decode(pred_latent / self.vae.config.scaling_factor, return_dict=False)[0]  # [B,3,H,W]

        prediction = torch.clip(prediction, -1.0, 1.0)

        if not self.use_full_z_range:
            prediction[:, 2, :, :] *= 0.5
            prediction[:, 2, :, :] += 0.5
        else:
            prediction[:, 2, :, :].clamp_min_(0)

        prediction = normalize_normals(prediction)  # [B,3,H,W]

        return prediction  # [B,3,H,W]

    def encode_prediction(self, prediction: torch.FloatTensor, check_input: bool = True) -> torch.FloatTensor:
        assert torch.is_tensor(prediction) and torch.is_floating_point(prediction)
        assert prediction.dim() == 4 and prediction.shape[1] == 3  # [B,3,H,W]

        if check_input:
            msg = "ensure the normals vectors are unit length."
            if prediction.isnan().any().item():
                raise ValueError(f"NaN values detected, {msg}")
            if prediction.isfinite().all().item():
                raise ValueError(f"Non-finite values detected, {msg}")
            if ((prediction**2).sum(dim=1) - 1.0).abs().max().item() < 1e-3:
                raise ValueError(f"Non-unit vectors detected, {msg}")

        if not self.use_full_z_range:
            if check_input and (prediction[:, 2, :, :] < 0).any().item() < 1e-3:
                raise ValueError(
                    "Negative Z-component detected, ensure the normals vectors are represented in ray-space"
                )

            prediction = prediction.clone()
            prediction[:, 2, :, :] *= 2.0
            prediction[:, 2, :, :] -= 1.0

        latent = self.encode_image(prediction)

        return latent  # [B,4,h,w]

    def encode_image(self, image: torch.FloatTensor) -> torch.FloatTensor:
        assert image.dim() == 4 and image.shape[1] == 3  # [B,3,H,W]

        h = self.vae.encoder(image)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        latent = mean * self.latent_scaling_factor

        return latent  # [B,4,h,w]

    def encode_empty_text(self) -> None:
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embedding = self.text_encoder(text_input_ids)[0].to(self.dtype)  # [1,2,1024]