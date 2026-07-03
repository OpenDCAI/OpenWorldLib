from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin


class FlowGeneratorScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            start_timesteps: int = 250,
            num_train_timesteps: int = 1000,
            shift: Optional[float] = 1.0,
            use_timestep_transform=False,  # whether to apply shift offset during training
            t_min=0.0,
            t_max=1.0,
            dmd_steps=4,
            rank=0
    ):
        self.num_timesteps = num_train_timesteps
        self.shift = shift
        self.use_timestep_transform = use_timestep_transform
        self.t_min = t_min
        self.t_max = t_max
        self.dmd_steps = dmd_steps
        timesteps = np.linspace(start_timesteps, num_train_timesteps, self.dmd_steps, dtype=np.float32)[::-1].copy()
        timesteps = torch.tensor(timesteps, dtype=torch.float32)
        timesteps = timesteps / num_train_timesteps
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        self.timesteps = timesteps * num_train_timesteps
        if rank == 0:
            print("Generator timesteps:", self.timesteps)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        sigma = timesteps.float() / self.num_timesteps
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        sigma = sigma.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])
        return (1 - sigma) * original_samples + sigma * noise

    def step(self, model_output, sample, timesteps, idx):
        sigma = timesteps[idx].float() / self.num_timesteps
        if idx < len(timesteps) - 1:
            sigma_next = timesteps[idx + 1].float() / self.num_timesteps
        else:
            sigma_next = 0.0
        dt = sigma_next - sigma

        prev_sample = sample + dt * model_output

        return prev_sample

    def gen_train_timesteps(self):
        # Generate random indices
        indices = torch.randint(
            low=0,
            high=self.dmd_steps,
            size=(1,),
        )

        if dist.is_initialized():
            dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks

        train_timesteps = self.timesteps[:indices.item() + 1]

        return train_timesteps

    def gen_test_timesteps(self):
        return self.timesteps


if __name__ == '__main__':
    scheduler = FlowGeneratorScheduler(shift=5.0, use_timestep_transform=True)
    for _ in range(10):
        print(scheduler.gen_train_timesteps())
