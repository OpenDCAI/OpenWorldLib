# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from einops import rearrange

from gamma_world._src.imaginaire.utils import misc
from gamma_world._src.gamma_world.inference.inference_i2v import (
    IS_PREPROCESSED_KEY,
    _DEFAULT_NEGATIVE_PROMPT,
    I2VInference,
    to_with_skip_tensor,
)


class BidirectionalInference(I2VInference):


    @torch.no_grad()
    def generate_single_shot(
        self,
        data_batch: dict,
        guidance: float | None = None,
        seed: int = 1,
        num_steps: int | None = None,
        shift: float | None = None,
        use_negative_prompt: bool = True,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
    ) -> torch.Tensor:
        model = self.model
        guidance = guidance if guidance is not None else self.guidance
        num_steps = num_steps if num_steps is not None else self.num_sampling_steps
        shift = shift if shift is not None else self.shift

        if "video" in data_batch:
            data_batch["video"] = data_batch["video"].float()
            if not data_batch.get(IS_PREPROCESSED_KEY, False):
                data_batch["video"] = data_batch["video"] / 127.5 - 1.0
            data_batch["video"] = torch.clamp(data_batch["video"], -1, 1)
        data_batch[IS_PREPROCESSED_KEY] = True
        data_batch = to_with_skip_tensor(data_batch, **model.tensor_kwargs)

        self.inplace_compute_text_embeddings_online(
            data_batch, use_negative_prompt=use_negative_prompt, negative_prompt=negative_prompt
        )

        n_views = int(data_batch["sample_n_views"].cpu().item()) if "sample_n_views" in data_batch else 1
        data_batch = model.get_data_batch_with_latent_view_indices(data_batch)
        _, x0, _ = model.get_data_and_condition(data_batch)
        velocity_fn = model.get_velocity_fn_from_batch(
            data_batch, n_views, guidance=guidance, is_negative_prompt=use_negative_prompt
        )

        noise = misc.arch_invariant_rand(
            (x0.shape[0],) + tuple(x0.shape[1:]), torch.float32, model.tensor_kwargs["device"], seed
        )
        seed_generator = torch.Generator(device=model.tensor_kwargs["device"])
        seed_generator.manual_seed(seed)

        scheduler = model.sample_scheduler
        scheduler.config.shift = shift
        scheduler.set_timesteps(num_steps, device=model.tensor_kwargs["device"], shift=shift)

        latents = noise
        batch_size, num_latent_frames = noise.shape[0], noise.shape[2]
        for current_timestep in scheduler.timesteps:
            timestep_B_T = torch.ones([batch_size, num_latent_frames], device=noise.device, dtype=torch.int64) * current_timestep
            velocity_pred = velocity_fn(latents, timestep_B_T, noise=noise)
            latents = scheduler.step(
                velocity_pred.unsqueeze(0), current_timestep, latents.unsqueeze(0), return_dict=False, generator=seed_generator
            )[0].squeeze(0)

        video = model.decode(latents)
        n_views = int(data_batch["sample_n_views"].item()) if "sample_n_views" in data_batch else 1
        if n_views > 1:
            video = rearrange(video, "B C (V T) H W -> B C T H (V W)", V=n_views)
        return video
