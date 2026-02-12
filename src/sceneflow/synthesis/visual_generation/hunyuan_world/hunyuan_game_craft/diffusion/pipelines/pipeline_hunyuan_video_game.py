import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
from packaging import version
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from ...constants import PRECISION_TO_TYPE
from ...text_encoder import TextEncoder
from einops import rearrange
from ...modules import HYVideoDiffusionTransformer


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
from .......base_models.diffusion_model.video.hunyuan_video.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from .......base_models.diffusion_model.video.hunyuan_video.diffusion.pipelines.pipeline_hunyuan_video import (
    HunyuanVideoPipeline,
    HunyuanVideoPipelineOutput,
    rescale_noise_cfg,
    retrieve_timesteps
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """"""

class HunyuanVideoGamePipeline(HunyuanVideoPipeline):
    r"""
    Pipeline for text-to-video generation using HunyuanVideo (Game Version).

    Inherits from HunyuanVideoPipeline to reuse encoding and decoding logic.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["text_encoder_2"]
    _exclude_from_cpu_offload = ["transformer"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HYVideoDiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        progress_bar_config: Dict[str, Any] = None,
        args=None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
            progress_bar_config=progress_bar_config,
            args=args
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
    def check_inputs(
        self,
        prompt,
        height,
        width,
        frame,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        vae_ver='88-4c-sd'
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # if frame is not None:
        #     if '884' in vae_ver:
        #         if frame!=1 and (frame-1)%4!=0:
        #             raise ValueError(f'`frame` has to be 1 or a multiple of 4 but is {frame}.')
        #     elif '888' in vae_ver:
        #         if frame!=1 and (frame-1)%8!=0:
        #             raise ValueError(f'`frame` has to be 1 or a multiple of 8 but is {frame}.')

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in \
                    {self._callback_tensor_inputs}, but found \
                        {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps.to(device), num_inference_steps - t_start
    
    def prepare_latents(self, batch_size, num_channels_latents, num_inference_steps,
                        height, width, frame, dtype, device, timesteps, generator, 
                        latents=None, gt_latents=None, denoise_strength=1.0,):
        shape = (
            batch_size,
            num_channels_latents,
            frame,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, denoise_strength, device)

        if gt_latents.shape[2] == 1:
            gt_latents = gt_latents.repeat(1, 1, frame, 1, 1)

        # TODO: correct
        x0 = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # print("!!!!!!!!!!!!!! RANDOM NOISE !!!!!!!!!!!!!!!!!!")
        # x0 = randn_tensor(shape, device=device, dtype=dtype)
        x1 = gt_latents

        t = torch.tensor([0.999]).to(device=device)
        latents = x0 * t + x1 * (1 - t)
        latents = torch.randn_like(x1)
        # print("!!!randn_like", latents.shape)
        latents = latents.to(dtype=dtype)
        
        if latents is None:
            latents = noise 
            original_latents = None
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        return latents, timesteps

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
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            pixel_value_llava (`torch.Tensor`, *optional*):
                The image tensor for llava. 
            uncond_pixel_value_llava (`torch.Tensor`, *optional*):
                The image tensor for llava.  Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
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
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)
            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

            if clip_skip is None:
                prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(text_inputs, output_hidden_states=True, data_type=data_type)
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
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
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, text_encoder.tokenizer)            
            uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)

            negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type)
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(1, num_videos_per_prompt)
                negative_attention_mask = negative_attention_mask.view(batch_size * num_videos_per_prompt, seq_len)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, -1)
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        if text_encoder is not None:
            if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(text_encoder.model, lora_scale)

        return prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        cam_latents: Union[torch.Tensor],               # cam_latents
        last_latents: Union[torch.Tensor],
        uncond_cam_latents: Union[torch.Tensor],
        gt_latents: Union[torch.Tensor],
        height: int,
        width: int,
        video_length: int, # frame is called video_len in hunyuan_multimodal/dev_video 
        data_type: str='video',
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        ref_latents: Optional[torch.Tensor] = None,
        uncond_ref_latents: Optional[torch.Tensor] = None,
        ip_cfg_scale: float = 0.0,
        use_deepcache: int = 1,
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
        vae_ver: str='88-4c-sd',
        enable_tiling: bool=False,
        n_tokens: Optional[int] = None,
        video_val_flag: bool=False,
        denoise_strength: float = 1.0,
        mask = None,
        cpu_offload: bool=False,
        use_sage: bool=False,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            video_length (`int`):
                The number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the DDIM paper.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A torch.Generator to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a `HunyuanVideoPipelineOutput`.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary passed along to the `AttentionProcessor`.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                Callback function called at the end of each step.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the callback function.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned,
                otherwise a `tuple` is returned.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to \
                    `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to \
                `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to transformer
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            video_length,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            vae_ver=vae_ver
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

        # device = self._execution_device
        device = torch.device("cuda")

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds, prompt_mask, negative_prompt_mask = \
            self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                attention_mask=attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_attention_mask=negative_attention_mask,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                data_type=data_type
            )
        
        if self.text_encoder_2 is not None:
            prompt_embeds_2, negative_prompt_embeds_2, prompt_mask_2, negative_prompt_mask_2 = \
                self.encode_prompt(
                    prompt,
                    device,
                    num_videos_per_prompt,
                    self.do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=None,
                    attention_mask=None,
                    negative_prompt_embeds=None,
                    negative_attention_mask=None,
                    lora_scale=lora_scale,
                    clip_skip=self.clip_skip,
                    text_encoder=self.text_encoder_2,
                )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        if self.do_classifier_free_guidance:
            if ref_latents is not None:
                ref_latents = torch.cat([ref_latents, ref_latents], dim=0)
                if prompt_mask[0].sum() > 575:
                    prompt_mask[0] = torch.cat(
                        [torch.ones((1, prompt_mask[0].sum() - 575)).to(prompt_mask), 
                         torch.zeros((1, prompt_mask.shape[1] - prompt_mask[0].sum() + 575)).to(prompt_mask)], dim=1)

        if ip_cfg_scale>0:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds[1:]])
            prompt_embeds_2 = torch.cat([prompt_embeds_2, prompt_embeds_2[1:]])
            prompt_mask = torch.cat([prompt_mask, prompt_mask[1:]], dim=0)
            ref_latents = torch.cat([uncond_ref_latents, uncond_ref_latents, ref_latents[1:]], dim=0)

        # 4. Prepare timesteps
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas, **extra_set_timesteps_kwargs,
        )

            
        if '884' in vae_ver:
            frame_length = (video_length - 2) // 4 + 2
        elif '888' in vae_ver:
            frame_length = (video_length - 1) // 8 + 1
        else:
            frame_length = video_length

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents, timesteps  = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_inference_steps,
            height,
            width,
            frame_length,
            prompt_embeds.dtype,
            device,
            timesteps,
            generator,
            latents,
            gt_latents,
            denoise_strength,
        )
        
        gt_latents = gt_latents.repeat(1, 1, frame_length, 1, 1)
        gt_latents_concat = gt_latents.clone()
        
        if frame_length == 10:
            gt_latents_concat[:,:,1:,:,:] = 0.0
            mask_concat = torch.ones(gt_latents.shape[0], 
                                     1, 
                                     gt_latents.shape[2], 
                                     gt_latents.shape[3], 
                                     gt_latents.shape[4]).to(device=gt_latents.device)
            mask_concat[:, :, 1:,...] = 0.0             
        else:
            gt_latents_concat[:,:,gt_latents_concat.shape[2]//2:,:,:] = 0.0
            mask_zeros = torch.zeros(gt_latents.shape[0], 
                                     1, 
                                     gt_latents.shape[2]//2, 
                                     gt_latents.shape[3], 
                                     gt_latents.shape[4])
            mask_ones = torch.ones(gt_latents.shape[0], 
                                   1, 
                                   gt_latents.shape[2]//2, 
                                   gt_latents.shape[3], 
                                   gt_latents.shape[4])
            mask_concat = torch.cat([mask_ones, mask_zeros], dim=2).to(device=gt_latents.device)

        # 6. Prepare extra step kwargs. 
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator, "eta": eta},
        )

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (target_dtype != torch.float32) and not self.args.val_disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (vae_dtype != torch.float32) and not self.args.val_disable_autocast

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        start_scale = ip_cfg_scale  #  3.0
        end_scale = 1.0
        step_scale = (start_scale - end_scale) / (self._num_timesteps - 1 + 1e-3)
        if cpu_offload: torch.cuda.empty_cache()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                if last_latents.shape[2] == 1:
                    latents[:,:,0,:,:] = last_latents[:,:,-1,:,:]
                else:
                    latents[:,:,:latents.shape[2]//2,:,:] = last_latents
                    gt_latents_concat[:,:,:latents.shape[2]//2,:,:] = last_latents
                
                # expand the latents if we are doing classifier free guidance
                latents_concat = torch.concat([latents, gt_latents_concat, mask_concat], dim=1)
                latent_model_input = torch.cat([latents_concat] * 2) \
                    if self.do_classifier_free_guidance else latents_concat

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                t_expand = t.repeat(latent_model_input.shape[0])

                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = None
                
                cam_latents_ = torch.cat([uncond_cam_latents, cam_latents], dim=0) \
                    if self.do_classifier_free_guidance else cam_latents
                
                # predict the noise residual
                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    is_cache = False
                    if use_deepcache and num_inference_steps==50:
                        
                        no_cache_steps =  list(range(0, 10)) + list(range(10, 40, 2)) + list(range(40, 50))
                        if i in no_cache_steps:
                            is_cache = False
                        else:
                            is_cache = True
                    if latent_model_input.shape[-1]*latent_model_input.shape[-2]>64*112 and cpu_offload:
                        if i==0:
                            print(f'cpu_offload={cpu_offload} and \
                                {latent_model_input.shape[-2:]} is large, split infer noise-pred')
                        noise_pred_uncond = self.transformer(latent_model_input[:1], 
                                                             t_expand[:1], 
                                                             text_states=prompt_embeds[:1],
                                                             text_mask=prompt_mask[:1],  
                                                             text_states_2=prompt_embeds_2[:1], 
                                                             freqs_cos=freqs_cis[0],
                                                             freqs_sin=freqs_cis[1], 
                                                             guidance=guidance_expand,
                                                             return_dict=True,
                                                             is_cache=is_cache, 
                                                             cam_latents=cam_latents_[:1])['x']
                        torch.cuda.empty_cache()
                        noise_pred_text = self.transformer(latent_model_input[1:], 
                                                           t_expand[1:], 
                                                           text_states=prompt_embeds[1:],
                                                           text_mask=prompt_mask[1:],
                                                           text_states_2=prompt_embeds_2[1:],
                                                           freqs_cos=freqs_cis[0],
                                                           freqs_sin=freqs_cis[1],
                                                           guidance=guidance_expand,
                                                           return_dict=True,
                                                           is_cache=is_cache,
                                                           cam_latents=cam_latents_[1:])['x']
                        noise_pred = torch.cat([noise_pred_uncond, noise_pred_text], dim=0)
                        torch.cuda.empty_cache()
                    else:
                        noise_pred = self.transformer(             # For an input image (1, 256, 256)
                            latent_model_input,             # [2, 16, 1, 32, 32] #
                            t_expand,                       # [2]
                            text_states=prompt_embeds,      # [2, 256, 4096]
                            text_mask=prompt_mask,          # [2, 256]
                            text_states_2=prompt_embeds_2,  # [2, 768]
                            freqs_cos=freqs_cis[0],         # [seqlen, head_dim]
                            freqs_sin=freqs_cis[1],         # [seqlen, head_dim]
                            guidance=guidance_expand,
                            return_dict=True,
                            is_cache=is_cache,
                            cam_latents=cam_latents_,
                            use_sage=use_sage,
                        )['x']

                # perform guidance
                if self.do_classifier_free_guidance and ip_cfg_scale < 0.1:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if ip_cfg_scale > 0:
                    noise_pred_uncond, noise_pred_text, noise_pred_ip = noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond + self.guidance_scale * \
                        (noise_pred_text - noise_pred_uncond) + start_scale * (noise_pred_ip-noise_pred_text)
                    start_scale -= step_scale
                    if i==0:
                        print(f'i={i}, noise_pred shape={noise_pred.shape}')

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, 
                                                   noise_pred_text, 
                                                   guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if last_latents.shape[2] == 1:
                    latents[:,:,1:,:,:] = self.scheduler.step(noise_pred[:,:,1:,:,:],
                                                              t, 
                                                              latents[:,:,1:,:,:], 
                                                              **extra_step_kwargs, 
                                                              return_dict=False)[0]
                else:
                    latents[:,:,noise_pred.shape[2]//2:,:,:] = self.scheduler.step(
                                                                noise_pred[:,:,noise_pred.shape[2]//2:,:,:], 
                                                                t, 
                                                                latents[:,:,latents.shape[2]//2:,:,:], 
                                                                **extra_step_kwargs, return_dict=False)[0]


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
                    if progress_bar is not None:
                        progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if cpu_offload: torch.cuda.empty_cache()
        # if mask_latents is not None:
        #     latents = mask_latents * latents + (1 - mask_latents) * original_latents
        if last_latents.shape[2] == 1:
            latents = latents[:,:,1:,:,:]
        
        if not output_type == "latent":
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}.")

            if not last_latents.shape[2] == 1:
                last_latents = latents[:,:,latents.shape[2]//2:,:,:]
            else:
                last_latents = latents
            latent_decode = last_latents.clone()
            latent_decode = latent_decode / self.vae.config.scaling_factor

            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
                if enable_tiling:
                    self.vae.enable_tiling()
                    if cpu_offload:
                        self.vae.post_quant_conv.to('cuda')
                        self.vae.decoder.to('cuda')
                    image = self.vae.decode(latent_decode, return_dict=False, generator=generator)[0]
                    self.vae.disable_tiling()
                    if cpu_offload:
                        self.vae.post_quant_conv.to('cpu')
                        self.vae.decoder.to('cpu')
                        torch.cuda.empty_cache()
                else:
                    image = self.vae.decode(latent_decode, return_dict=False, generator=generator)[0]
            # if image is None:
            #     return (None, )

            # if expand_temporal_dim or (not video_val_flag and image.shape[2] == 1):
            #     image = image.squeeze(2)
            
            if image is not None and (expand_temporal_dim or (not video_val_flag and image.shape[2] == 1)):
                image = image.squeeze(2)

        if image is not None:
            image = (image / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            image = image.cpu().float()

        # Offload all models
        self.maybe_free_model_hooks()
        
        if cpu_offload: torch.cuda.empty_cache()
        if not return_dict:
            return image
        
        return_latents = kwargs.get("return_latents", False)

        if return_latents:
            return HunyuanVideoPipelineOutput(videos=image), \
                latents, timesteps, last_latents, last_latents[:,:,-1:, ...]
        
        return HunyuanVideoPipelineOutput(videos=image)