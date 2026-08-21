import copy
from ..pipeline import RollingForcingTrainingPipeline
import torch.nn.functional as F
from typing import Tuple
import torch

from .base import RollingForcingModel


class GAN(RollingForcingModel):
    def __init__(self, args, device):
        """
        Initialize the GAN module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.concat_time_embeddings = getattr(args, "concat_time_embeddings", False)
        self.num_class = args.num_class
        self.relativistic_discriminator = getattr(args, "relativistic_discriminator", False)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.fake_score.adding_cls_branch(
            atten_dim=1536, num_class=args.num_class, time_embed_dim=1536 if self.concat_time_embeddings else 0)
        self.fake_score.model.requires_grad_(True)

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: RollingForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.critic_timestep_shift = getattr(args, "critic_timestep_shift", self.timestep_shift)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        self.gan_g_weight = getattr(args, "gan_g_weight", 1e-2)
        self.gan_d_weight = getattr(args, "gan_d_weight", 1e-2)
        self.r1_weight = getattr(args, "r1_weight", 0.0)
        self.r2_weight = getattr(args, "r2_weight", 0.0)
        self.r1_sigma = getattr(args, "r1_sigma", 0.01)
        self.r2_sigma = getattr(args, "r2_sigma", 0.01)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _run_cls_pred_branch(self,
                             noisy_image_or_video: torch.Tensor,
                             conditional_dict: dict,
                             timestep: torch.Tensor) -> torch.Tensor:
        """
            Run the classifier prediction branch on the generated image or video.
            Input:
                - image_or_video: a tensor with shape [B, F, C, H, W].
            Output:
                - cls_pred: a tensor with shape [B, 1, 1, 1, 1] representing the feature map for classification.
        """
        _, _, noisy_logit = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            classify_mode=True,
            concat_time_embeddings=self.concat_time_embeddings
        )

        return noisy_logit

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        # Step 2: Get timestep and add noise to generated/real latents
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.critic_timestep_shift > 1:
            critic_timestep = self.critic_timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.critic_timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(pred_image)
        noisy_fake_latent = self.scheduler.add_noise(
            pred_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        # Step 4: Compute the real GAN discriminator loss
        real_image_or_video = clean_latent.clone()
        critic_noise = torch.randn_like(real_image_or_video)
        noisy_real_latent = self.scheduler.add_noise(
            real_image_or_video.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        conditional_dict["prompt_embeds"] = torch.concatenate(
            (conditional_dict["prompt_embeds"], conditional_dict["prompt_embeds"]), dim=0)
        critic_timestep = torch.concatenate((critic_timestep, critic_timestep), dim=0)
        noisy_latent = torch.concatenate((noisy_fake_latent, noisy_real_latent), dim=0)
        _, _, noisy_logit = self.fake_score(
            noisy_image_or_video=noisy_latent,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
            classify_mode=True,
            concat_time_embeddings=self.concat_time_embeddings
        )
        noisy_fake_logit, noisy_real_logit = noisy_logit.chunk(2, dim=0)

        if not self.relativistic_discriminator:
            gan_G_loss = F.softplus(-noisy_fake_logit.float()).mean() * self.gan_g_weight
        else:
            relative_fake_logit = noisy_fake_logit - noisy_real_logit
            gan_G_loss = F.softplus(-relative_fake_logit.float()).mean() * self.gan_g_weight

        return gan_G_loss

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        real_image_or_video: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to, num_sim_steps = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
            )

        # Step 2: Get timestep and add noise to generated/real latents
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.critic_timestep_shift > 1:
            critic_timestep = self.critic_timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.critic_timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_fake_latent = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        # Step 4: Compute the real GAN discriminator loss
        noisy_real_latent = self.scheduler.add_noise(
            real_image_or_video.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        conditional_dict_cloned = copy.deepcopy(conditional_dict)
        conditional_dict_cloned["prompt_embeds"] = torch.concatenate(
            (conditional_dict_cloned["prompt_embeds"], conditional_dict_cloned["prompt_embeds"]), dim=0)
        _, _, noisy_logit = self.fake_score(
            noisy_image_or_video=torch.concatenate((noisy_fake_latent, noisy_real_latent), dim=0),
            conditional_dict=conditional_dict_cloned,
            timestep=torch.concatenate((critic_timestep, critic_timestep), dim=0),
            classify_mode=True,
            concat_time_embeddings=self.concat_time_embeddings
        )
        noisy_fake_logit, noisy_real_logit = noisy_logit.chunk(2, dim=0)

        if not self.relativistic_discriminator:
            gan_D_loss = F.softplus(-noisy_real_logit.float()).mean() + F.softplus(noisy_fake_logit.float()).mean()
        else:
            relative_real_logit = noisy_real_logit - noisy_fake_logit
            gan_D_loss = F.softplus(-relative_real_logit.float()).mean()
        gan_D_loss = gan_D_loss * self.gan_d_weight

        # R1 regularization
        if self.r1_weight > 0.:
            noisy_real_latent_perturbed = noisy_real_latent.clone()
            epison_real = self.r1_sigma * torch.randn_like(noisy_real_latent_perturbed)
            noisy_real_latent_perturbed = noisy_real_latent_perturbed + epison_real
            noisy_real_logit_perturbed = self._run_cls_pred_branch(
                noisy_image_or_video=noisy_real_latent_perturbed,
                conditional_dict=conditional_dict,
                timestep=critic_timestep
            )

            r1_grad = (noisy_real_logit_perturbed - noisy_real_logit) / self.r1_sigma
            r1_loss = self.r1_weight * torch.mean((r1_grad)**2)
        else:
            r1_loss = torch.zeros_like(gan_D_loss)

        # R2 regularization
        if self.r2_weight > 0.:
            noisy_fake_latent_perturbed = noisy_fake_latent.clone()
            epison_generated = self.r2_sigma * torch.randn_like(noisy_fake_latent_perturbed)
            noisy_fake_latent_perturbed = noisy_fake_latent_perturbed + epison_generated
            noisy_fake_logit_perturbed = self._run_cls_pred_branch(
                noisy_image_or_video=noisy_fake_latent_perturbed,
                conditional_dict=conditional_dict,
                timestep=critic_timestep
            )

            r2_grad = (noisy_fake_logit_perturbed - noisy_fake_logit) / self.r2_sigma
            r2_loss = self.r2_weight * torch.mean((r2_grad)**2)
        else:
            r2_loss = torch.zeros_like(r2_loss)

        critic_log_dict = {
            "critic_timestep": critic_timestep.detach(),
            'noisy_real_logit': noisy_real_logit.detach(),
            'noisy_fake_logit': noisy_fake_logit.detach(),
        }

        return (gan_D_loss, r1_loss, r2_loss), critic_log_dict
