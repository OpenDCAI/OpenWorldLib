import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional

class CameraPoseEncoder(nn.Module):

    def __init__(self,
                 context_dim: int = 2048,
                 dim: int = 5120,
                 patch_size: Tuple[int, int, int] = [1, 2, 2],
                 in_channels: int = 6,
                 downscale_coef: int = 8,
                 pose_inject_method='adaln',
                 **kwargs):
        super(CameraPoseEncoder, self).__init__()
        start_channels = in_channels * (downscale_coef ** 2)
        input_channels = [start_channels, start_channels, start_channels * 2]
        self.pose_inject_method = pose_inject_method
        self.unshuffle = nn.PixelUnshuffle(downscale_coef)

        self.controlnet_encode_first = nn.Sequential(
            nn.Conv2d(input_channels[0], input_channels[1], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(2, input_channels[1]),
            nn.Conv2d(input_channels[1], input_channels[1], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(2, input_channels[1]),
            nn.ReLU(),
        )

        self.controlnet_encode_second = nn.Sequential(
            nn.Conv2d(input_channels[1], input_channels[2], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(2, input_channels[2]),
            nn.ReLU(),
        )

        self.patch_embedding = nn.Conv3d(
            input_channels[2], dim, kernel_size=patch_size, stride=patch_size)


        if pose_inject_method=='adaln' or pose_inject_method=='latent_split':
            self.fc = nn.Sequential(
                nn.Linear(dim, dim//2),
                nn.LayerNorm(dim//2),
                nn.GELU(),
                nn.Linear(dim//2, context_dim),
                nn.LayerNorm(context_dim)
            )



    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def compress_time(self, x, num_frames):
        x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
        batch_size, frames, channels, height, width = x.shape
        x = rearrange(x, 'b f c h w -> (b h w) c f')

        if x.shape[-1] % 2 == 1:
            x_first, x_rest = x[..., 0], x[..., 1:]
            if x_rest.shape[-1] > 0:
                x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

            x = torch.cat([x_first[..., None], x_rest], dim=-1)
        else:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = rearrange(x, '(b h w) c f -> (b f) c h w', b=batch_size, h=height, w=width)
        return x

    def patchify(self, x: torch.Tensor):
        #x: (b, c, f, h, w)
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def forward(self, x):

        batch_size = x.shape[0]
        num_frames = x.shape[1]

        # 0. Controlnet encoder
        x = rearrange(x, "b f h w c -> (b f) c h w")
        x = self.unshuffle(x)
        x = self.controlnet_encode_first(x)
        x = self.compress_time(x, num_frames=num_frames)
        num_frames = x.shape[0] // batch_size

        x = self.controlnet_encode_second(x)
        x = self.compress_time(x, num_frames=num_frames)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=batch_size)

        ## train patchfy
        x, (f, h, w) = self.patchify(x)

        if self.pose_inject_method=='latent_split':
            x = self.fc(x)  # shape: (bs, f, 2)
        elif self.pose_inject_method=='latent_overall':
            pass
        elif self.pose_inject_method=='adaln':
            x = self.fc(x)  # shape: (bs, f, 2)

        return x # plucker_fea
