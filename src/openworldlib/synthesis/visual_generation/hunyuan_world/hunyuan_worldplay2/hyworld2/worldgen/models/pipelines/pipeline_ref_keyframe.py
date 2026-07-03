# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

try:
    from ._pipeline_common import KeyframePipelineMixin
except ImportError:
    from models.pipelines._pipeline_common import KeyframePipelineMixin

try:
    from ...models.worldstereo import WorldStereoRefSModel
    from ...src.vae_utils import keyframe_vae_encode, keyframe_vae_decode
except ImportError:
    from models.worldstereo import WorldStereoRefSModel
    from src.vae_utils import keyframe_vae_encode, keyframe_vae_decode

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class KFPCDControllerRefPipeline(KeyframePipelineMixin, DiffusionPipeline, WanLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
            self,
            tokenizer: AutoTokenizer,
            text_encoder: UMT5EncoderModel,
            image_encoder: CLIPVisionModel,
            image_processor: CLIPImageProcessor,
            transformer: WorldStereoRefSModel,
            vae: AutoencoderKLWan,
            scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self._init_keyframe_pipeline_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            image_processor=image_processor,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

    @torch.no_grad()
    def __call__(
            self,
            image: PipelineImageInput,
            render_video: torch.Tensor,
            render_mask: torch.Tensor,
            camera_embedding: torch.Tensor = None,
            extrinsics: torch.Tensor = None,
            intrinsics: torch.Tensor = None,
            prompt: Union[str, List[str]] = None,
            negative_prompt: Union[str, List[str]] = None,
            # new params for reference
            reference_video=None,
            ref_index=None,
            camera_qt=None,
            camera_qt_ref=None,
            latent_cond_mode="full_vae",
            # new params end
            mode: str = "train",
            height: int = 480,
            width: int = 768,
            num_frames: int = 81,
            num_inference_steps: int = 40,
            guidance_scale: float = 5.0,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            image_embeds: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "np",
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 512,
            **kwargs
    ):

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        if image_embeds is None:
            image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        latents, condition = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            latent_cond_mode,
        )

        ### 5.5 Prepare keyframe render_latent ###
        # Explicitly cast to VAE weight dtype to avoid input/bias dtype mismatch under autocast
        render_video = render_video.to(dtype=self.vae.dtype)
        render_latent = keyframe_vae_encode(self.vae, render_video, rescale=True)

        # Prepare reference_latent
        if reference_video is not None:
            reference_video = reference_video.to(dtype=self.vae.dtype)
            reference_latent = keyframe_vae_encode(self.vae, reference_video, rescale=True)
            # Build reference_latent structure: [latent(16) + mask(4) + latent(16)] = 36 channels
            # This is consistent with training: reference_latent = torch.cat([reference_latent, reference_latent_mask, reference_latent], dim=1)
            reference_latent_mask = torch.ones_like(reference_latent[:, :4]).to(reference_latent.dtype).to(reference_latent.device)
            reference_latent = torch.cat([reference_latent, reference_latent_mask, reference_latent], dim=1)
        else:
            reference_latent = None
            ref_noise = None

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    render_latent=render_latent,
                    reference_latent=reference_latent,
                    render_mask=render_mask,
                    camera_embedding=camera_embedding,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    ref_index=ref_index,
                    camera_qt=camera_qt,
                    camera_qt_ref=camera_qt_ref,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        render_latent=render_latent,
                        reference_latent=reference_latent,
                        render_mask=render_mask,
                        camera_embedding=camera_embedding,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        ref_index=ref_index,
                        camera_qt=camera_qt,
                        camera_qt_ref=camera_qt_ref,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            video = keyframe_vae_decode(self.vae, latents, rescale=True)  # (b, c, f, h, w)
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)