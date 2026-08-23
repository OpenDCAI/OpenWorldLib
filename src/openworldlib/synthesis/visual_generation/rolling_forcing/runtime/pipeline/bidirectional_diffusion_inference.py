from tqdm import tqdm
from typing import List
import torch

from ..wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from ..wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BidirectionalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        wan_model_path = getattr(args, "wan_model_path", None)
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=False, model_path=wan_model_path) if generator is None else generator
        self.text_encoder = WanTextEncoder(model_path=wan_model_path, device=device) if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper(model_path=wan_model_path) if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = 8.0

        self.args = args

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents=False
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        latents = noise

        sample_scheduler = self._initialize_sample_scheduler(noise)
        for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
            latent_model_input = latents
            timestep = t * torch.ones([latents.shape[0], 21], device=noise.device, dtype=torch.float32)

            flow_pred_cond, _ = self.generator(latent_model_input, conditional_dict, timestep)
            flow_pred_uncond, _ = self.generator(latent_model_input, unconditional_dict, timestep)

            flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                flow_pred_cond - flow_pred_uncond)

            temp_x0 = sample_scheduler.step(
                flow_pred.unsqueeze(0),
                t,
                latents.unsqueeze(0),
                return_dict=False)[0]
            latents = temp_x0.squeeze(0)

        x0 = latents
        video = self.vae.decode_to_pixel(x0)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        del sample_scheduler

        if return_latents:
            return video, latents
        else:
            return video

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
