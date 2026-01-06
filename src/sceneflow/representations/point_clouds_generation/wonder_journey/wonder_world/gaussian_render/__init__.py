#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from depth_diff_gaussian_rasterization_wonderworld import GaussianRasterizationSettings, GaussianRasterizer
from ..scene.gaussian_model import WonderWorldGaussianModel as GaussianModel


def render(viewpoint_camera, pc: GaussianModel, opt, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None, render_visible=False, exclude_sky=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros_like(pc.get_xyz_all, dtype=pc.get_xyz_all.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=opt.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means3D = pc.get_xyz_all
    means2D = screenspace_points
    # opacity = pc.get_opacity_with_3D_filter
    opacity = pc.get_opacity_all
    # opacity = pc.get_opacity_with_3D_filter_all

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if opt.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance_all(scaling_modifier)
    else:
        scales = pc.get_scaling_all
        rotations = pc.get_rotation_all

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if opt.convert_SHs_python:
            shs_view = pc.get_features_all.transpose(1, 2).view(-1, 3)
            colors_precomp = pc.color_activation(shs_view)
        else:
            # shs = pc.get_features
            shs = pc.get_features_all
    else:
        colors_precomp = override_color

    if render_visible:
        visibility_filter_all = pc.visibility_filter_all & ~pc.delete_mask_all  # Seen in screen
    else:
        visibility_filter_all = ~pc.delete_mask_all

    if exclude_sky:
        visibility_filter_all = visibility_filter_all & ~pc.is_sky_filter

    if len(visibility_filter_all) == len(means3D):
        means3D = means3D[visibility_filter_all]
        means2D = means2D[visibility_filter_all]
        shs = None if shs is None else shs[visibility_filter_all]
        colors_precomp = None if colors_precomp is None else colors_precomp[visibility_filter_all]
        opacity = opacity[visibility_filter_all]
        scales = scales[visibility_filter_all]
        rotations = rotations[visibility_filter_all]
        cov3D_precomp = None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all]
    else:
        means3D = means3D
        means2D = means2D
        shs = None if shs is None else shs
        colors_precomp = None if colors_precomp is None else colors_precomp
        opacity = opacity
        scales = scales
        rotations = rotations
        cov3D_precomp = None if cov3D_precomp is None else cov3D_precomp

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, median_depth, final_opacity = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "final_opacity": final_opacity,
            "depth": depth,
            "median_depth": median_depth,}