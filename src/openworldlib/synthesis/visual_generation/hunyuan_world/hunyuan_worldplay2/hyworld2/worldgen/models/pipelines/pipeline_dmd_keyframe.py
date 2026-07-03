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
from diffusers.utils import is_torch_xla_available, logging
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

try:
    from ._pipeline_common import KeyframePipelineMixin, retrieve_latents
except ImportError:
    from models.pipelines._pipeline_common import KeyframePipelineMixin, retrieve_latents

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


class RefKFDMDGeneratorPipeline(KeyframePipelineMixin, DiffusionPipeline, WanLoraLoaderMixin):
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
            scheduler,
            device,
            vae_compile: bool = False,
            vae_compile_mode: str = "max-autotune"
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
        # self.transformer = transformer
        self.device_ = device

        # VAE compile settings
        self.vae_compile = vae_compile
        self.vae_compile_mode = vae_compile_mode

    def _execution_device(self):
        return self.device_

    # @torch.no_grad()
    # we need to update generator through the full generator inference
    # NOTE: generator does not need CFG
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

        # 1. Check inputs. Raise error if not correct 0%
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

        self._guidance_scale = 1.0
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device()

        # 2. Define call parameters 0%
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt 2-3% 0.3-0.4s
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=False,
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
            with torch.no_grad():
                image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps 0%
        if mode == "train":  # random selection during training, return all steps during inference
            timesteps = self.scheduler.gen_train_timesteps()
        else:
            timesteps = self.scheduler.gen_test_timesteps()

        # 5. Prepare latent variables 31% 4.5-4.8s -> 0.5s -> 0.1s (next version)
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
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
                latent_cond_mode
            )

        ### 5.5 Prepare keyframe render_latent ###
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # Explicitly cast to VAE weight dtype to avoid input/bias dtype mismatch under autocast
            render_video = render_video.to(dtype=self.vae.dtype)
            if render_video.shape[2] == num_frames // 4 + 1:  # process with divided 21 frames
                render_latent = keyframe_vae_encode(self.vae, render_video, rescale=True,
                                                    use_compile=self.vae_compile, compile_mode=self.vae_compile_mode)
            else:
                render_latent = retrieve_latents(self.vae.encode(render_video), sample_mode="argmax")
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(render_latent.device, render_latent.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                    render_latent.device, render_latent.dtype
                )
                render_latent = (render_latent - latents_mean) * latents_std

        # Prepare reference_latent
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            if reference_video is not None:
                reference_latent = keyframe_vae_encode(self.vae, reference_video, rescale=True,
                                                       use_compile=self.vae_compile, compile_mode=self.vae_compile_mode)
                reference_latent_mask = torch.ones_like(reference_latent[:, :4]).to(reference_latent.dtype).to(reference_latent.device)
                reference_latent = torch.cat([reference_latent, reference_latent_mask, reference_latent], dim=1)
            else:
                reference_latent = None

        # 6. Denoising loop 55-60% 8-9s
        num_warmup_steps = 0
        self._num_timesteps = len(timesteps)

        # Pre-convert condition to transformer dtype (avoid repeated conversion in loop)
        condition = condition.to(transformer_dtype)

        # Pre-expand all timesteps (avoid repeated expand in loop)
        expanded_timesteps = [t.expand(latents.shape[0]).to(device) for t in timesteps]

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # Use pre-converted condition and concatenate with latents
                latent_model_input = torch.cat([latents.to(transformer_dtype), condition], dim=1)
                timestep = expanded_timesteps[i]

                # Only allow gradient backprop on the last step in train mode
                if mode != "train" or (mode == "train" and i < len(timesteps) - 1):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            render_latent=render_latent,
                            point_map_latent=None,
                            point_map_ref_latent=None,
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
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            render_latent=render_latent,
                            point_map_latent=None,
                            point_map_ref_latent=None,
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

                # compute the previous noisy sample x_t -> x_t-1, if t is the last step of generator, pred x_0
                latents = self.scheduler.step(noise_pred, latents, timesteps, i)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) > num_warmup_steps:
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # 7. decode latent 2.5% 0.38s
        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                video = keyframe_vae_decode(self.vae, latents, rescale=True)  # (b, c, f, h, w)
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        # self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)

    # only used for training
    def fast_infer(
            self,
            random_noise_latent,
            condition_latent,
            render_latent,
            render_mask,
            prompt_embeds,
            image_embeds,
            camera_embedding=None,
            # new params for reference
            reference_latent=None,
            camera_qt=None,
            camera_qt_ref=None,
            ref_index=None
    ):

        timesteps = self.scheduler.gen_train_timesteps()
        latents = random_noise_latent
        device = self._execution_device()

        # print("[DEBUG] FAST INFER Timesteps:", timesteps)

        pred_latents = None

        for i, t in enumerate(timesteps):

            latent_model_input = torch.cat([latents, condition_latent], dim=1)
            timestep = t.expand(latents.shape[0]).to(device)

            # Only the last step allows gradient backpropagation
            if i < len(timesteps) - 1:
                with torch.no_grad():
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        render_latent=render_latent,
                        point_map_latent=None,
                        point_map_ref_latent=None,
                        reference_latent=reference_latent,
                        render_mask=render_mask,
                        camera_embedding=camera_embedding,
                        camera_qt=camera_qt,
                        camera_qt_ref=camera_qt_ref,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        return_dict=False,
                        ref_index=ref_index,
                    )[0]
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, latents, timesteps, i)
            else:
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    render_latent=render_latent,
                    point_map_latent=None,
                    point_map_ref_latent=None,
                    reference_latent=reference_latent,
                    render_mask=render_mask,
                    camera_embedding=camera_embedding,
                    camera_qt=camera_qt,
                    camera_qt_ref=camera_qt_ref,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    return_dict=False,
                    ref_index=ref_index,
                )[0]
                # compute the previous noisy sample x_t -> x_t-1, if t is the last step of generator, pred x_0
                pred_latents = self.scheduler.step(noise_pred, latents, timesteps, i)

        return pred_latents