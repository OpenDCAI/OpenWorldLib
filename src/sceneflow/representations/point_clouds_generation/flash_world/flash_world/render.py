import os
import time
from xml.dom.minidom import Notation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat import rasterization

# torch.backends.cuda.preferred_linalg_library(backend="magma")

""""
modified from https://github.com/arthurhero/Long-LRM/blob/main/model/llrm.py
"""
class GaussianRendererWithCheckpoint(torch.autograd.Function):
    @staticmethod
    def render(xyz, feature, scale, rotation, opacity, test_c2w, test_intr, 
               W, H, sh_degree, near_plane, far_plane, backgrounds):
        test_w2c = test_c2w.float().inverse().unsqueeze(0) # (1, 4, 4)
        test_intr_i = torch.zeros(3, 3).to(test_intr.device)
        test_intr_i[0, 0] = test_intr[0]
        test_intr_i[1, 1] = test_intr[1]
        test_intr_i[0, 2] = test_intr[2]
        test_intr_i[1, 2] = test_intr[3]
        test_intr_i[2, 2] = 1
        test_intr_i = test_intr_i.unsqueeze(0) # (1, 3, 3)
        rendering, alpha, _ = rasterization(xyz, rotation, scale, opacity, feature,
                                        test_w2c, test_intr_i, W, H, sh_degree=sh_degree, 
                                        near_plane=near_plane, far_plane=far_plane,
                                        render_mode="RGB+D",
                                        backgrounds=backgrounds[None],
                                        rasterize_mode='classic') # (1, H, W, 4) 
        # rendering[..., 3:] = rendering[..., 3:] + far_plane * (1 - alpha)
        return rendering

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr,
                W, H, sh_degree, near_plane, far_plane, backgrounds):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, backgrounds)
        ctx.W = W
        ctx.H = H
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        with torch.no_grad():
            V, _ = test_intr.shape
            renderings = torch.zeros(V, H, W, 4).to(xyz.device)
            alphas = torch.rand(V, device=xyz.device)
            for iv in range(V):
                rendering = GaussianRendererWithCheckpoint.render(xyz, feature, scale, rotation, opacity, 
                                                                      test_c2ws[iv], test_intr[iv], W, H, sh_degree, near_plane, far_plane, backgrounds[iv])
                renderings[iv:iv+1] = rendering

        renderings = renderings.requires_grad_()
        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, backgrounds = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        sh_degree = ctx.sh_degree
        near_plane = ctx.near_plane
        far_plane = ctx.far_plane
        with torch.enable_grad():
            V, _ = test_intr.shape
            for iv in range(V):
                rendering = GaussianRendererWithCheckpoint.render(xyz, feature, scale, rotation, opacity, 
                                                        test_c2ws[iv], test_intr[iv], W, H, sh_degree, near_plane, far_plane, backgrounds[iv])
                rendering.backward(grad_output[iv:iv+1])

        return xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad, None, None, None, None, None, None, None, None

def gaussian_render(gaussian_params, test_c2ws, test_intr, W, H, near_plane=0.01, far_plane=1000, use_checkpoint=False, sh_degree=0, bg_mode='random'):

    if not torch.is_grad_enabled():
        use_checkpoint = False

     # opengl2colmap, see https://github.com/imlixinyang/Director3D/blob/main/modules/renderers/gaussians_renderer.py
    test_c2ws[:, :, :3, 1:3] *= -1

    device = test_intr.device
    B, V, _ = test_intr.shape
    
    renderings = []

    for ib in range(B):
        if bg_mode == 'random':
            backgrounds = torch.rand(V, 3).to(device)
        elif bg_mode == 'white':
            backgrounds = torch.ones(V, 3).to(device)
        elif bg_mode == 'black':
            backgrounds = torch.zeros(V, 3).to(device)
        else:
            raise ValueError(f"Invalid background mode: {bg_mode}")
            
        xyz_i, opacity_i, scale_i, rotation_i, feature_i = gaussian_params[ib].float().split([3, 1, 3, 4, (sh_degree + 1)**2 * 3], dim=-1)

        opacity_i = opacity_i.squeeze(-1)
        feature_i = feature_i.reshape(-1, (sh_degree + 1)**2, 3)

        if use_checkpoint:
            
            renderings.append(GaussianRendererWithCheckpoint.apply(xyz_i, feature_i, scale_i, rotation_i, opacity_i, test_c2ws[ib], test_intr[ib], W, H, sh_degree, near_plane, far_plane, backgrounds))

        else:
            rendering = torch.zeros(V, H, W, 4).to(device)
            for iv in range(V):
                rendering[iv:iv+1] = GaussianRendererWithCheckpoint.render(xyz_i, feature_i, scale_i, rotation_i, opacity_i, 
                                                                      test_c2ws[ib][iv], test_intr[ib][iv], W, H, sh_degree, near_plane, far_plane, backgrounds[iv])

            # test_w2c_i = test_c2ws[ib].float().inverse() # (V, 4, 4)
            # test_intr_i = torch.zeros(V, 3, 3).to(device)
            # test_intr_i[:, 0, 0] = test_intr[ib, :, 0]
            # test_intr_i[:, 1, 1] = test_intr[ib, :, 1]
            # test_intr_i[:, 0, 2] = test_intr[ib, :, 2]
            # test_intr_i[:, 1, 2] = test_intr[ib, :, 3]
            # test_intr_i[:, 2, 2] = 1

            # # print(backgrounds.shape)
            # rendering, _, _ = rasterization(xyz_i, rotation_i, scale_i, opacity_i, feature_i,
            #                                     test_w2c_i, test_intr_i, W, H, sh_degree=sh_degree, 
            #                                     near_plane=near_plane, far_plane=far_plane,
            #                                     render_mode="RGB+D",
            #                                     backgrounds=backgrounds,
            #                                     rasterize_mode='classic') # (V, H, W, 3) 
            renderings.append(rendering)
    
    renderings = torch.stack(renderings, dim=0).permute(0, 1, 4, 2, 3).contiguous() # (B, 3, V, H, W)
    rgb = renderings[:, :, :3].mul_(2).add_(-1).clamp(-1, 1)
    depth = renderings[:, :, 3:]
    return rgb, depth