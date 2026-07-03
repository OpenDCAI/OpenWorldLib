# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright the Self-Forcing authors (Guande He et al., NeurIPS 2025).
# SPDX-License-Identifier: Apache-2.0
#


#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#


from abc import ABC, abstractmethod

import torch


class SchedulerInterface(ABC):


    alphas_cumprod: torch.Tensor
    timesteps: torch.Tensor

    @abstractmethod
    def add_noise(self, clean_latent: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:

        pass

    def convert_x0_to_noise(self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:


        original_dtype = x0.dtype
        x0, xt, alphas_cumprod = map(lambda x: x.double().to(x0.device), [x0, xt, self.alphas_cumprod])

        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        noise_pred = (xt - alpha_prod_t ** (0.5) * x0) / beta_prod_t ** (0.5)
        return noise_pred.to(original_dtype)

    def convert_noise_to_x0(self, noise: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:


        original_dtype = noise.dtype
        noise, xt, alphas_cumprod = map(lambda x: x.double().to(noise.device), [noise, xt, self.alphas_cumprod])
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (xt - beta_prod_t ** (0.5) * noise) / alpha_prod_t ** (0.5)
        return x0_pred.to(original_dtype)

    def convert_velocity_to_x0(self, velocity: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:


        original_dtype = velocity.dtype
        velocity, xt, alphas_cumprod = map(
            lambda x: x.double().to(velocity.device),
            [velocity, xt, self.alphas_cumprod],
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (alpha_prod_t**0.5) * xt - (beta_prod_t**0.5) * velocity
        return x0_pred.to(original_dtype)


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    def step(self, model_output, timestep, sample, to_final=False):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples, noise, timestep):

        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):

        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
