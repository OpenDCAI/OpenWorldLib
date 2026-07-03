# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    QwenImageEditPlusPipeline,
    PipelineImageInput,
    QwenImagePipelineOutput,
    calculate_shift,
    retrieve_timesteps,
    calculate_dimensions,
)
from diffusers.utils import is_torch_xla_available, logging

try:
    from diffusers.models.autoencoders.vae import DecoderOutput
except ImportError:
    from diffusers.models.vae import DecoderOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

# Target pixel counts for the two preprocessing passes.
# Condition images fed to the text encoder use a small thumbnail (~384×384),
# while images fed to the VAE use a higher-resolution crop (~1024×1024).
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


class PanoDiffusionPipeline(QwenImageEditPlusPipeline):
    """Panorama-aware diffusion pipeline built on top of QwenImageEditPlusPipeline."""

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        # ── Output size ────────────────────────────────────────────────
        # Derive height/width from the input image's aspect ratio if not
        # specified, targeting a fixed pixel budget (1024²).
        image_size = image.size
        calculated_width, calculated_height = calculate_dimensions(
            1024 * 1024, image_size[0] / image_size[1]
        )
        height = height or calculated_height
        width = width or calculated_width

        # Snap to the nearest multiple of (vae_scale_factor × 2) so that the
        # VAE spatial downsampling and the transformer's 2× patch embedding both
        # divide evenly into the output resolution.
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 1. Check inputs
        self.check_inputs(
            prompt, height, width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Preprocess image
        # The input image is resized to two different resolutions:
        #   • condition_image  (~384²)  — low-res thumbnail fed to the
        #     multimodal text encoder (cross-attention keys/values).
        #   • vae_image        (~1024²) — high-res crop encoded by the VAE
        #     into the latent space and concatenated with the noisy latent.
        if image is not None and not (
            isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels
        ):
            image_width, image_height = image.size
            condition_width, condition_height = calculate_dimensions(
                CONDITION_IMAGE_SIZE, image_width / image_height
            )
            vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
            condition_image = self.image_processor.resize(image, condition_height, condition_width)
            # unsqueeze(2) adds a temporal dimension expected by the 3-D VAE.
            vae_image = self.image_processor.preprocess(image, vae_height, vae_width).unsqueeze(2)

        # ── True CFG eligibility check ─────────────────────────────────
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is "
                f"not enabled since no negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                "negative_prompt is passed but classifier-free guidance is not enabled since "
                "true_cfg_scale <= 1"
            )

        # True CFG is active only when a negative prompt is given AND the scale
        # is above 1.  Otherwise a single forward pass per step is performed.
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # Encode the conditional prompt (text + condition thumbnail).
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=[condition_image], prompt=prompt,
            prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask,
            device=device, num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            # Encode the unconditional (negative) prompt separately.
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=[condition_image], prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device, num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            [vae_image],
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height, width,
            prompt_embeds.dtype,
            device, generator, latents,
        )

        # img_shapes tells the transformer the spatial extent of each sequence
        # segment so it can apply the correct positional embeddings.
        # Layout per sample: [output_shape, cond_shape]
        # where each shape is (1, H // vae_scale_factor // 2,
        #                         W // vae_scale_factor // 2).
        # The division by 2 accounts for the transformer's 2× patch embedding.
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2),
            ]
        ] * batch_size

        # 5. Prepare timesteps
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None else sigmas
        )
        image_seq_len = latents.shape[1]
        # calculate_shift adapts the flow-matching time shift μ based on the
        # sequence length so that longer sequences denoise more slowly.
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Handle guidance
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
            logger.warning(
                f"guidance_scale is passed as {guidance_scale}, but ignored since the model "
                f"is not guidance-distilled."
            )
            guidance = None
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        # Pre-compute per-sample text sequence lengths to avoid recomputing
        # them inside the loop; used by the transformer's attention mask logic.
        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        )
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if negative_prompt_embeds_mask is not None else None
        )

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                # Concatenate the noisy output latent with the VAE-encoded
                # condition latents along the sequence dimension.
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # ── Conditional forward pass ───────────────────────────
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    # Slice off the output-latent portion; the transformer
                    # outputs predictions for all packed tokens (output + cond).
                    noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    # ── Unconditional forward pass ─────────────────────
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]

                    # ── Norm-rescaled true CFG ─────────────────────────
                    # Standard CFG blend: move from unconditional toward
                    # conditional by true_cfg_scale.
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    # Rescale the combined prediction so its L2 norm matches
                    # the conditional prediction's norm.  This prevents the
                    # CFG blend from amplifying the overall signal magnitude,
                    # which would otherwise cause over-saturation at high scales.
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # Compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # 7. Decode latents
        if output_type == "latent":
            image = latents
        else:
            # _unpack_latents converts the packed flat-sequence representation
            # back into a spatial (B, C, T, H, W) tensor.
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            # Denormalize: the VAE encoder stored latents as
            #   z_norm = (z - mean) * std_inv
            # so we invert: z = z_norm / std_inv + mean.
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents.to(self.vae.dtype)
            latents = latents / latents_std + latents_mean
            # Decode; [:, :, 0] selects the single temporal frame.
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
