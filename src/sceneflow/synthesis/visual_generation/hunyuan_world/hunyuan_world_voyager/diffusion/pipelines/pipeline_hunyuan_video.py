# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import torch.distributed as dist

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils.torch_utils import randn_tensor

from ...constants import PRECISION_TO_TYPE
from .......base_models.diffusion_model.video.hunyuan_video.vae.autoencoder_kl_causal_3d import (
    AutoencoderKLCausal3D
)
from .......base_models.diffusion_model.video.hunyuan_video.diffusion.pipelines.pipeline_hunyuan_video import (
    rescale_noise_cfg,
    retrieve_timesteps,
    HunyuanVideoPipelineOutput,
    HunyuanVideoPipeline as BaseHunyuanVideoPipeline,
)
from ...text_encoder import TextEncoder
from ...utils.data_utils import black_image

logger = logging.get_logger(__name__)


class HunyuanVideoPipeline(BaseHunyuanVideoPipeline):
    """
    Extended Pipeline for image-to-video generation using HunyuanVideo.
    
    Extends the base HunyuanVideoPipeline with support for:
    - Image-to-video (I2V) generation with multiple condition types
    - Semantic image conditioning
    - RGB + Depth output processing
    """

    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
        semantic_images=None,
    ):
        """
        Encodes the prompt into text encoder hidden states.
        Extended to support semantic_images for conditioning.
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(text_encoder.model, lora_scale)
            else:
                scale_lora_layers(text_encoder.model, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)

            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

            if clip_skip is None:
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, 
                    semantic_images=semantic_images, device=device
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs, output_hidden_states=True, data_type=data_type,
                    semantic_images=semantic_images, device=device,
                )
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(prompt_embeds)

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(bs_embed * num_videos_per_prompt, seq_len)

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, "
                    f"but got {type(negative_prompt)} != {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, "
                    f"but `prompt`: {prompt} has batch size {batch_size}."
                )
            else:
                uncond_tokens = negative_prompt

            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, text_encoder.tokenizer)

            uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)

            # Create black images for unconditional generation
            if semantic_images is not None:
                uncond_image = [black_image(img.size[0], img.size[1]) for img in semantic_images]
            else:
                uncond_image = None

            negative_prompt_outputs = text_encoder.encode(
                uncond_input, data_type=data_type, 
                semantic_images=uncond_image, device=device
            )
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(1, num_videos_per_prompt)
                negative_attention_mask = negative_attention_mask.view(
                    batch_size * num_videos_per_prompt, seq_len
                )

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt)
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, -1
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )

        if text_encoder is not None:
            if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(text_encoder.model, lora_scale)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
        img_latents=None,
        i2v_mode=False,
        i2v_condition_type=None,
        i2v_stability=True,
    ):
        """Prepare latents for denoising, extended with I2V support."""
        num_channels_latents = img_latents.shape[1]
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, "
                f"but requested an effective batch size of {batch_size}."
            )

        if i2v_mode and i2v_stability:
            if img_latents.shape[2] == 1:
                img_latents = img_latents.repeat(1, 1, video_length, 1, 1)
            x0 = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            x1 = img_latents
            t = torch.tensor([0.999]).to(device=device)
            latents = x0 * t + x1 * (1 - t)
            latents = latents.to(dtype=dtype)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
            
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        freqs_cis_cond: Tuple[torch.Tensor, torch.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        i2v_mode: bool = False,
        i2v_condition_type: str = None,
        i2v_stability: bool = True,
        img_latents: Optional[torch.Tensor] = None,
        semantic_images=None,
        partial_cond=None,
        partial_mask=None,
        **kwargs,
    ):
        """Extended call function with I2V support and depth output processing."""
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate("callback", "1.0.0",
                     "Passing `callback` as an input argument to `__call__` is deprecated, "
                     "consider using `callback_on_step_end`")
        if callback_steps is not None:
            deprecate("callback_steps", "1.0.0",
                     "Passing `callback_steps` as an input argument to `__call__` is deprecated, "
                     "consider using `callback_on_step_end`")

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs
        self.check_inputs(
            prompt, height, width, video_length, callback_steps,
            negative_prompt, prompt_embeds, negative_prompt_embeds,
            callback_on_step_end_tensor_inputs, vae_ver=vae_ver,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = (torch.device(f"cuda:{dist.get_rank()}") 
                  if dist.is_initialized() else self._execution_device)

        # 3. Encode input prompt
        lora_scale = (self.cross_attention_kwargs.get("scale", None)
                      if self.cross_attention_kwargs is not None else None)

        prompt_embeds, negative_prompt_embeds, prompt_mask, negative_prompt_mask = self.encode_prompt(
            prompt, device, num_videos_per_prompt, self.do_classifier_free_guidance,
            negative_prompt, prompt_embeds=prompt_embeds, attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds, negative_attention_mask=negative_attention_mask,
            lora_scale=lora_scale, clip_skip=self.clip_skip, data_type=data_type,
            semantic_images=semantic_images,
        )
        
        if self.text_encoder_2 is not None:
            prompt_embeds_2, negative_prompt_embeds_2, prompt_mask_2, negative_prompt_mask_2 = self.encode_prompt(
                prompt, device, num_videos_per_prompt, self.do_classifier_free_guidance,
                negative_prompt, prompt_embeds=None, attention_mask=None,
                negative_prompt_embeds=None, negative_attention_mask=None,
                lora_scale=lora_scale, clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2, data_type=data_type,
            )
        else:
            prompt_embeds_2 = negative_prompt_embeds_2 = prompt_mask_2 = negative_prompt_mask_2 = None

        # Concatenate for CFG
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        # 4. Prepare timesteps
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas,
            **extra_set_timesteps_kwargs,
        )

        if "884" in vae_ver:
            video_length = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            video_length = (video_length - 1) // 8 + 1

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt, num_channels_latents,
            height, width, video_length, prompt_embeds.dtype, device, generator,
            latents, img_latents=img_latents, i2v_mode=i2v_mode,
            i2v_condition_type=i2v_condition_type, i2v_stability=i2v_stability,
        )

        # Prepare I2V concat inputs
        if i2v_mode and i2v_condition_type == "latent_concat":
            if img_latents.shape[2] == 1:
                img_latents_concat = img_latents.repeat(1, 1, video_length, 1, 1)
            else:
                img_latents_concat = img_latents
            img_latents_concat[:, :, 1:, ...] = 0

            mask_concat = torch.ones(
                img_latents_concat.shape[0], 1, img_latents_concat.shape[2],
                img_latents_concat.shape[3], img_latents_concat.shape[4]
            ).to(device=img_latents.device)
            mask_concat[:, :, 1:, ...] = 0

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator, "eta": eta},
        )

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = target_dtype != torch.float32 and not self.args.disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = vae_dtype != torch.float32 and not self.args.disable_autocast

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if i2v_mode and i2v_condition_type == "token_replace":
                    latents = torch.concat([img_latents, latents[:, :, 1:, :, :]], dim=2)

                # Build model input
                if i2v_mode and i2v_condition_type == "latent_concat":
                    latent_model_input = torch.concat(
                        [latents, img_latents_concat, mask_concat, partial_cond, partial_mask], dim=1
                    )
                else:
                    latent_model_input = latents

                latent_model_input = (
                    torch.cat([latent_model_input] * 2)
                    if self.do_classifier_free_guidance else latent_model_input
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32, device=device,
                    ).to(target_dtype) * 1000.0
                    if embedded_guidance_scale is not None else None
                )

                # Predict noise
                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    noise_pred = self.transformer(
                        latent_model_input, t_expand,
                        text_states=prompt_embeds, text_mask=prompt_mask,
                        text_states_2=prompt_embeds_2,
                        freqs_cos=freqs_cis[0], freqs_sin=freqs_cis[1],
                        freqs_cos_cond=freqs_cis_cond[0], freqs_sin_cond=freqs_cis_cond[1],
                        guidance=guidance_expand, return_dict=True,
                    )["x"]

                # Perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale
                    )

                # Compute previous sample
                if i2v_mode and i2v_condition_type == "token_replace":
                    latents = self.scheduler.step(
                        noise_pred[:, :, 1:, :, :], t, latents[:, :, 1:, :, :],
                        **extra_step_kwargs, return_dict=False
                    )[0]
                    latents = torch.concat([img_latents, latents], dim=2)
                else:
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # Decode latents
        if not output_type == "latent":
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) != 5:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), "
                    f"but got {latents.shape}."
                )

            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            else:
                latents = latents / self.vae.config.scaling_factor

            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
                if enable_tiling:
                    self.vae.enable_tiling()
                image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)
        else:
            image = latents

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float()

        # I2V post-processing
        if i2v_mode and i2v_condition_type == "latent_concat":
            image = image[:, :, 4:, :, :]

        # Split RGB and depth, process depth output separately
        half_height = (height - 16) // 2
        rgb = image[..., :half_height, :]
        depth = image[..., -half_height:, :]
        depth = depth[:, 0] * 0.299 + depth[:, 1] * 0.587 + depth[:, 2] * 0.114
        depth = depth.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        image = torch.cat([rgb, depth], dim=-2)

        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return HunyuanVideoPipelineOutput(videos=image)
