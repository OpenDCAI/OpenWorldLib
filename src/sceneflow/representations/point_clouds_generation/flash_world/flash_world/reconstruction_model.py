import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from .utils import zero_init, EMANorm, create_rays

import einops

from .render import gaussian_render

from .utils import quaternion_to_matrix

def inverse_sigmoid(x):
    if type(x) == torch.Tensor:
        return torch.log(x/(1-x))
    else:
        return math.log(x/(1-x))

def inverse_softplus(x, beta=1):
    if type(x) == torch.Tensor:
        return (torch.exp(beta * x) - 1).log() / beta
    else:
        return math.log((math.exp(beta * x) - 1)) / beta

import copy

import math
import torch
import torch.nn as nn
import numpy as np

from .autoencoder_kl_wan import WanCausalConv3d, WanRMS_norm, unpatchify


class WANDecoderPixelAligned3DGSReconstructionModel(nn.Module):
    def __init__(self, 
                 vae_model, 
                 feat_dim,
                #  num_remove_decoder_up_blocks=0,
                #  num_points_per_pixel=4,
                 use_network_checkpointing=True,
                 use_render_checkpointing=True
        ):
        super().__init__()

        self.decoder = copy.deepcopy(vae_model.decoder).requires_grad_(True)
        self.post_quant_conv = copy.deepcopy(vae_model.post_quant_conv).requires_grad_(True)

        self.extra_conv_in = WanCausalConv3d(feat_dim, self.decoder.conv_in.weight.shape[0], 3, padding=1)

        time_pad = self.extra_conv_in._padding[4]
        self.extra_conv_in.padding = (0, self.extra_conv_in._padding[2], self.extra_conv_in._padding[0])
        self.extra_conv_in._padding = (0, 0, 0, 0, 0, 0)
        self.extra_conv_in.weight = torch.nn.Parameter(self.extra_conv_in.weight[:, :, time_pad:].clone())

        with torch.no_grad():
            self.extra_conv_in.weight.data.zero_()
            self.extra_conv_in.bias.data.zero_()

        # remove one block
        # self.decoder.up_blocks = self.decoder.up_blocks[:-1]
        dims = [self.decoder.dim * u for u in [self.decoder.dim_mult[-1]] + self.decoder.dim_mult[::-1]]
        # self.decoder.up_blocks[-1].upsampler.mode = None
        # self.decoder.up_blocks[-1].upsampler.resample = nn.Identity()
        # self.decoder.up_blocks[-1].avg_shortcut = None

        self.decoder.norm_out = WanRMS_norm(dims[-1], images=False, bias=False)
        self.decoder.conv_out = nn.Identity()

        # add ema_norm for vae
        # for i_level in reversed(range(len(self.decoder.up_blocks))):
        #     if self.decoder.up_blocks[i_level].upsampler is not None:
        #         self.decoder.up_blocks[i_level].upsampler.resample = nn.Sequential(
        #             self.decoder.up_blocks[i_level].upsampler.resample,
        #         )

        self.patch_size = vae_model.config.patch_size
        # assert dims[-1] % 4 == 0
        self.gs_head = PixelAligned3DGS(dims[-1], num_points_per_pixel=2)

        del self.decoder.up_blocks[0].upsampler.time_conv
        del self.decoder.up_blocks[1].upsampler.time_conv

        self.decoder.conv_out = nn.Identity()

        self.network_checkpointing = use_network_checkpointing
        self.render_checkpointing = use_render_checkpointing
    
    def decode(self, feats, z):
        ## conv1
        x = self.decoder.conv_in(self.post_quant_conv(z)) + self.extra_conv_in(feats)

        ## middle
        if self.network_checkpointing and torch.is_grad_enabled():
            x = torch.utils.checkpoint.checkpoint(self.decoder.mid_block, x, None, [0], use_reentrant=False)
        else:
            x = self.decoder.mid_block(x, None, [0])

        ## upsamples
        for i, up_block in enumerate(self.decoder.up_blocks):
            if self.network_checkpointing and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(up_block, x, None, [0], True, use_reentrant=False)
            else:
                x = up_block(x, None, [0], first_chunk=True)

        # head
        x = self.decoder.norm_out(x)
        x = self.decoder.nonlinearity(x)
        x = self.decoder.conv_out(x)

        # if self.patch_size is not None:
        #     x = unpatchify(x, patch_size=self.patch_size)

        return x

    def forward(self, feats, z, cameras):

        x = self.decode(feats, z).squeeze(2)

        gaussian_params = self.gs_head(x, cameras.flatten(0, 1)).unflatten(0, (cameras.shape[0], cameras.shape[1]))
        
        return gaussian_params
    
    # def forward(self, images, cameras, scene_chunk_lens):

    #     x, z, feats = self.encode(images)

    #     return self.reconstruct(x, z, feats, cameras, scene_chunk_lens)
    
    @torch.amp.autocast(device_type='cuda', enabled=False)
    def render(self, gaussian_params, camerass, height, width, bg_mode='random'):

        camerass = camerass.to(torch.float32)

        test_c2ws = torch.eye(4, device=camerass.device)[None][None].repeat(camerass.shape[0], camerass.shape[1], 1, 1).float()
        test_c2ws[:, :, :3, :3] = quaternion_to_matrix(camerass[:, :, :4])
        test_c2ws[:, :, :3, 3] = camerass[:, :, 4:7]

        test_intr = torch.eye(3, device=camerass.device)[None, None].repeat(camerass.shape[0], camerass.shape[1], 1, 1).float()
        fx, fy, cx, cy = camerass[:, :, 7:11].split([1, 1, 1, 1], dim=-1)

        test_intr = torch.cat([fx * width, fy * height, cx * width, cy * height], dim=-1)

        return gaussian_render(gaussian_params, test_c2ws, test_intr, width, height, use_checkpoint=self.render_checkpointing, sh_degree=self.gs_head.sh_degree, bg_mode=bg_mode)

from torch.autograd import Function

class _trunc_exp(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-10, 10))

trunc_exp = _trunc_exp.apply

class PixelAligned3DGS(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            sh_degree=2,
            use_mask=False, 
            scale_range=(0, 16), # related to pixel size
            num_points_per_pixel=1,
        ):
        super().__init__()

        self.sh_degree = sh_degree

        # sh, uv_offset, depth, opacity, scales, rotations
        # TODO: handle different sh_degree
        self.gaussian_channels = [3 * (self.sh_degree + 1) ** 2, 2, 1, 1, 3, 4, (1 if use_mask else 0)]

        self.gs_proj = nn.Conv2d(embed_dim, num_points_per_pixel * sum(self.gaussian_channels), 3, 1, 1)
        self.register_buffer("lrs_mul", torch.Tensor(
                [1] * 3 + # sh 0
                [0.5] * 3 * ((self.sh_degree + 1) ** 2 - 1) + # other sh
                [0.01] * 2 + # uv_offset
                [1] * 1 + # depth
                [1] * 1 + # opacity
                [1] * 3 + # scales
                [1] * 4 + # rotations
                [0.1] * (1 if use_mask else 0) #  mask
            ).repeat(num_points_per_pixel), persistent=True)

        self.lrs_mul = self.lrs_mul / self.lrs_mul.max()

        self.use_mask = use_mask

        self.scale_range = scale_range

        with torch.no_grad():
            self.gs_proj.weight.data.zero_()
            self.gs_proj.bias = nn.Parameter(torch.Tensor(
                [0.0] * 3 * (self.sh_degree + 1) ** 2 + # sh
                [0.0] * 2 + # uv_offset
                [math.log(1)] * 1 + # depth
                # [inverse_softplus(1)] * 1 + # depth
                [inverse_sigmoid(0.1)] * 1 + # opacity
                [inverse_sigmoid((1 - scale_range[0]) / (scale_range[1] - scale_range[0]))] * 3 + # scales (default: 1 hence the gaussian scale is equal to pixel size)
                # [inverse_softplus(0.005)] * 3 + # scales (default: 1 hence the gaussian scale is equal to pixel size)
                [1., 0, 0, 0] + # rotations
                [inverse_sigmoid(0.9)] * (1 if use_mask else 0) #  mask (default: 0.9)
            ).repeat(num_points_per_pixel) / self.lrs_mul)

        self.num_points_per_pixel = num_points_per_pixel

    @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(self, x, cameras):

        BN, _, h, w = x.shape

        local_gaussian_params = F.conv2d(x, self.gs_proj.weight * self.lrs_mul[:, None, None, None].to(x.dtype), self.gs_proj.bias * self.lrs_mul.to(x.dtype), stride=1, padding=1).unflatten(1, (self.num_points_per_pixel, -1))

        local_gaussian_params = local_gaussian_params.to(torch.float32)
        cameras = cameras.to(torch.float32)
        # local_gaussian_params = F.conv2d(x, self.gs_proj.weight, self.gs_proj.bias, stride=1, padding=1).unflatten(1, (self.num_points_per_pixel, -1))

        # batch * n_frame, num_points_per_pixel, c, h, w -> batch * n_frame, num_points_per_pixel, h, w, c
        local_gaussian_params = local_gaussian_params.permute(0, 1, 3, 4, 2)

        features, uv_offset, depth, opacity, scales, rotations, mask = local_gaussian_params.split(self.gaussian_channels, dim=-1)

        rays_o, rays_d = create_rays(cameras[:, None].repeat(1, self.num_points_per_pixel, 1), uv_offset=uv_offset, h=h, w=w)

        depth = trunc_exp(depth)
        # depth = F.softplus(depth, beta=1)
        xyz = (rays_o + depth * rays_d)

        # features = features.unflatten(-1, (-1, 3))

        opacity = torch.sigmoid(opacity)
        if self.use_mask:
            if torch.is_grad_enabled():
                mask = torch.sigmoid(mask)
                hard_mask = (mask > torch.rand_like(mask)).float()
                opacity = opacity * (mask + (hard_mask - mask).detach())
            else:
                mask = torch.sigmoid(mask)
                hard_mask = (mask > torch.rand_like(mask)).float()
                opacity = opacity * hard_mask

        fx, fy = cameras[:, 7:9].split([1, 1], dim=-1)
        fx, fy = fx / w, fy / h
        pixel_size = torch.sqrt(fx.pow(2) + fy.pow(2))[:, None, None, None] * depth
        scales = (torch.sigmoid(scales) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]) * pixel_size
        # scales = F.softplus(scales, beta=1)

        # It’s not required to be normalized for gspalt rasterization?
        rotations = torch.nn.functional.normalize(rotations, dim=-1)

        gaussian_params = torch.cat([xyz, opacity, scales, rotations, features], dim=-1)
    
        return gaussian_params