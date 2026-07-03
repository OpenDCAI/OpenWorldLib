# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright the Self-Forcing authors (Guande He et al., NeurIPS 2025).
# SPDX-License-Identifier: Apache-2.0
#


#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#


from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

from gamma_world._src.gamma_world.third_party.self_forcing.scheduler import FlowMatchScheduler


def print_rank0(*msgs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*msgs)
    else:
        print(*msgs)


class SelfForcingTrainingPipeline:
    def __init__(
        self,
        denoising_step_list: List[int] | List[List[int]],
        scheduler: FlowMatchScheduler,
        generator: Callable,
        num_frame_per_block=3,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        num_heads: int = 12,
        num_transformer_blocks: int = 30,
        local_attn_size: int = -1,
        sink_size: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list

        if isinstance(self.denoising_step_list, list):
            self.independent_denoising_step_list = True
        else:
            self.independent_denoising_step_list = False
            if self.denoising_step_list[-1] == 0:
                self.denoising_step_list = self.denoising_step_list[:-1]


        self.num_transformer_blocks = num_transformer_blocks
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        self.num_heads = num_heads

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device) -> List[int]:
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:

            indices = torch.randint(low=0, high=num_denoising_steps, size=(num_blocks,), device=device)
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices, src=0)
        return indices.tolist()

    def inference_with_trajectory(
        self, noise: torch.Tensor, initial_latent: Optional[torch.Tensor] = None, **conditional_dict
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):


            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:

            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )


        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)


        current_start_frame = 0
        if initial_latent is not None:
            current_end_frame = current_start_frame + 1
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0

            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    current_end=current_end_frame * self.frame_seq_length,
                    start_frame_for_rope=current_start_frame,
                )
            current_start_frame = current_end_frame


        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        exit_flags = []
        if self.independent_denoising_step_list:
            for block_index in range(len(all_num_frames)):
                exit_flag = self.generate_and_sync_list(
                    1, len(self.denoising_step_list[block_index]), device=noise.device
                )
                exit_flags.append(exit_flag[0])
        else:
            num_denoising_steps = len(self.denoising_step_list)
            exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)


        start_gradient_frame_index = num_output_frames - 21


        for block_index, current_num_frames in enumerate(all_num_frames):
            current_end_frame = current_start_frame + current_num_frames
            noisy_input = noise[
                :,
                current_start_frame - num_input_frames : current_end_frame - num_input_frames,
            ]


            denoising_step_list = (
                self.denoising_step_list[block_index]
                if self.independent_denoising_step_list
                else self.denoising_step_list
            )
            for index, current_timestep in enumerate(denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = index == exit_flags[0]
                else:
                    exit_flag = (
                        index == exit_flags[block_index]
                    )
                timestep = (
                    torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64,
                    )
                    * current_timestep
                )

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            current_end=current_end_frame * self.frame_seq_length,
                            start_frame_for_rope=current_start_frame,
                        )
                        next_timestep = denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])

                else:

                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                current_end=current_end_frame * self.frame_seq_length,
                                start_frame_for_rope=current_start_frame,
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            current_end=current_end_frame * self.frame_seq_length,
                            start_frame_for_rope=current_start_frame,
                        )
                    break


            output[:, current_start_frame:current_end_frame] = denoised_pred


            context_timestep = torch.ones_like(timestep) * self.context_noise

            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep
                * torch.ones(
                    [batch_size * current_num_frames],
                    device=noise.device,
                    dtype=torch.long,
                ),
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    current_end=current_end_frame * self.frame_seq_length,
                    start_frame_for_rope=current_start_frame,
                )


            current_start_frame = current_end_frame


        denoised_timestep_from, denoised_timestep_to = None, None

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):

        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, self.kv_cache_size, self.num_heads, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, self.kv_cache_size, self.num_heads, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):

        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, 512, self.num_heads, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, self.num_heads, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache
