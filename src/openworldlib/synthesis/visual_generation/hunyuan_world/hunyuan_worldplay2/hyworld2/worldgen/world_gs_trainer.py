import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from fused_ssim import fused_ssim
from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from nerfview import CameraState, RenderTabState, apply_float_colormap
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from gs.gsplat_viewer import GsplatViewer, GsplatRenderTabState
from gs.utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed, Depth2Normal, depth_to_normal
from gs.opencv import Dataset, Parser
from gs.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = ""
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "results/debug"
    # Every N images there is a test image
    test_every: int = 32
    # Train only with panorama (and optionally polar) images, excluding video frames
    pano_only: bool = False
    # Train only with video frames, excluding panorama and polar images (reverse of pano_only)
    video_only: bool = False
    # Repeat panorama/polar images N times in the training set to increase their sampling weight
    pano_repeat: int = 1
    # Azimuth subsampling interval for panorama views (1=all, 4=original density)
    pano_azimuth_interval: int = 1
    # Azimuth subsampling interval for polar upper views (1=all, 4=original density)
    polar_up_azimuth_interval: int = 1
    # Include extra elevation layers (65/78 deg) for polar upper
    polar_up_extra_layers: bool = True
    # Azimuth subsampling interval for polar lower views (1=all, 2=original density)
    polar_down_azimuth_interval: int = 1
    # Only keep video frames whose name starts with this prefix (e.g. "view_000-traj0"). None means no filter.
    video_prefix_filter: Optional[str] = None
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    no_normalize: bool = False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 443

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 5_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [4_000, 5_000, 6_000, 8_000, 10_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Whether to convert to spz
    convert_to_spz: bool = False
    # Whether to convert to spx
    convert_to_spx: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [4_000, 5_000, 6_000, 8_000, 10_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 1_000_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 0
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1500
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    lpips_lambda1: float = 0.2
    lpips_lambda2: float = 0.1

    preload_gs_path: str = None

    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    use_scale_regularization: bool = False
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    max_gauss_ratio: float = 5.0
    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    apply_mask: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # background colors
    background_color: str = None

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda1: float = 0.2
    depth_lambda2: float = 0.01

    # Enable normal loss. (experimental)
    normal_loss: bool = False
    # Weight for normal loss
    normal_lambda1: float = 0.1
    normal_lambda2: float = 0.5

    # Sky depth smoothness loss: TV regularization on rendered depth in sky regions
    sky_depth_smooth: bool = False
    sky_depth_smooth_lambda1: float = 1e-2
    sky_depth_smooth_lambda2: float = 1e-4
    # L2 norm threshold on GT normal to classify a pixel as sky
    sky_normal_threshold: float = 0.1
    # Erode sky mask by this many pixels to avoid boundary artifacts
    sky_erode_pixels: int = 3

    # Sky depth from point cloud: pre-compute sky depth by rendering the initial sky
    # point cloud, merge into GT depth for sky pixels (triple-condition: depth==0 AND
    # normal-based sky AND sky rendering coverage). Requires depth_loss=True.
    sky_depth_from_pcd: bool = False
    # Save debug visualizations of sky depth, sky mask, and merged depth to disk
    sky_depth_from_pcd_debug: bool = False

    # Distortion loss: depth variance proxy for 2DGS distortion regularization
    dist_loss: bool = False
    # Weight for distortion loss
    dist_lambda: float = 1e-2
    # Iteration to start distortion loss regularization
    dist_start_iter: int = 3000

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "vgg"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    # downsampling
    downsample_pts_num: int = 1_000_000
    # downsampling mode: "random" (default) or "geometry_aware" (curvature-aware voxel thinning)
    downsample_mode: str = "geometry_aware"
    # Dynamic pruning, used only while saving.
    do_prune: bool = False
    prune_opacity_threshold: float = 0.01

    # mesh params
    export_mesh: bool = False
    voxel_size: float = 0.05  # Fine: 0.01, coarse: 0.05.
    sdf_trunc: float = 0.3  # Usually 4x the voxel size.
    downsample_perc: float = 0.1

    # ====== MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks ======
    # Enable MaskGaussian probabilistic mask for Gaussian pruning
    use_mask_gaussian: bool = False
    # Learning rate for mask_score parameter
    mask_lr: float = 0.01
    # Sparsity regularization coefficient λ for mask loss
    mask_lambda: float = 0.003
    # Iteration to start applying mask loss
    mask_from_iter: int = 300
    # Iteration to stop applying mask loss
    mask_until_iter: int = 1_500
    # Perform mask-based pruning every this many steps (used after densification ends)
    mask_prune_iter: int = 1_000
    # Number of stochastic samples for pruning: remove Gaussians never sampled alive.
    mask_prune_sample_times: int = 4
    # Initial value for mask_score[:, 0] (bias towards keeping)
    mask_init_value: float = 4.0
    # Keep-probability threshold used when filtering Gaussians during PLY export.
    mask_export_prob_threshold: float = 0.20
    # Use stochastic sampling (Bernoulli draw per Gaussian) instead of hard
    # threshold when filtering Gaussians during PLY export.
    mask_export_stochastic: bool = True
    # MaskGaussian mode: "from_scratch" or "post_training"
    # from_scratch: train mask jointly with all parameters from the beginning
    # post_training: load a pretrained checkpoint, freeze other params, train mask only first, then finetune
    mask_mode: Literal["from_scratch", "post_training"] = "from_scratch"
    # For post_training mode: path to pretrained checkpoint (.pt file)
    mask_pretrained_ckpt: Optional[str] = None
    # For post_training mode: number of steps to train mask only (other params frozen)
    mask_only_steps: int = 5_000
    # For post_training mode: number of finetune steps after pruning (all params unfrozen)
    mask_finetune_steps: int = 5_000
    # Gumbel softmax temperature (fixed, following MaskGaussian paper)
    mask_temperature: float = 0.7

    # ====== Anchor Protection: protect sky/anchor points from pruning ======
    # Enable anchor protection (sky points are automatically marked as anchors)
    use_anchor_protection: bool = False
    # Protect anchor points from split / duplicate during densification
    anchor_no_densify: bool = False
    # Protect anchor opacity from being reset during densification
    anchor_opa_reset_protection: bool = True
    # Protect anchor points from MaskGaussian periodic / post-training pruning
    mask_prune_anchor_protection: bool = True
    # Protect anchor points from MaskGaussian filtering during PLY export
    mask_export_anchor_protection: bool = True
    # Freeze anchor point positions (zero out means gradient for anchors)
    anchor_freeze_means: bool = False

    # ====== Align Protection: protect aligned_pcd points independently ======
    # Enable align protection (aligned_pcd points are marked separately from anchors)
    use_align_protection: bool = False
    # Protect align points from split / duplicate during densification
    align_no_densify: bool = True
    # Protect align points from pruning during densification
    align_no_prune: bool = True
    # Protect align opacity from being reset during densification
    align_opa_reset_protection: bool = True
    # Protect align points from MaskGaussian periodic / post-training pruning
    mask_prune_align_protection: bool = True
    # Protect align points from MaskGaussian filtering during PLY export
    mask_export_align_protection: bool = True
    # Freeze align point positions (zero out means gradient for align points)
    align_freeze_means: bool = False

    # ====== Align+Sky Only Init: skip base points.ply, use only sky_points.ply + align_points.ply ======
    init_align_sky_only: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
        parser: Parser,
        init_type: str = "sfm",
        init_num_pts: int = 100_000,
        init_extent: float = 3.0,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
        preload_gs_path: str = None,
        means_lr: float = 1.6e-4,
        scales_lr: float = 5e-3,
        opacities_lr: float = 5e-2,
        quats_lr: float = 1e-3,
        sh0_lr: float = 2.5e-3,
        shN_lr: float = 2.5e-3,
        scene_scale: float = 1.0,
        sh_degree: int = 3,
        sparse_grad: bool = False,
        visible_adam: bool = False,
        batch_size: int = 1,
        feature_dim: Optional[int] = None,
        device: str = "cuda",
        world_rank: int = 0,
        world_size: int = 1,
        # MaskGaussian parameters
        use_mask_gaussian: bool = False,
        mask_lr: float = 0.01,
        mask_init_value: float = 10.0,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        valid_mask = torch.isfinite(points).all(dim=-1)
        if not valid_mask.all():
            num_invalid = (~valid_mask).sum().item()
            print(f"Warning: filtered {num_invalid} invalid (NaN/Inf) points out of {points.shape[0]}")
            points = points[valid_mask]
            rgbs = rgbs[valid_mask]
        if points.shape[0] == 0:
            print("Warning: no valid SfM points, falling back to random initialization")
            points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
            rgbs = torch.rand((init_num_pts, 3))
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ["means", torch.nn.Parameter(points), means_lr * scene_scale],
        ["scales", torch.nn.Parameter(scales), scales_lr],
        ["quats", torch.nn.Parameter(quats), quats_lr],
        ["opacities", torch.nn.Parameter(opacities), opacities_lr],
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(["sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr])
        params.append(["shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr])
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(["features", torch.nn.Parameter(features), sh0_lr])
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(["colors", torch.nn.Parameter(colors), sh0_lr])

    if preload_gs_path is not None:
        print("Loading GS from {}".format(preload_gs_path))
        preload_gs = torch.load(preload_gs_path, weights_only=False)
        for i in range(len(params)):
            key = params[i][0]
            key = "means3d" if key == "means" else key
            if key == "shN":
                pad_shape = params[i][1].shape[1] - preload_gs["splats"][key].shape[1]
                if pad_shape > 0:
                    pad_shape = torch.zeros((preload_gs["splats"][key].shape[0], pad_shape, 3), dtype=torch.float32)
                    preload_gs["splats"][key] = torch.cat([preload_gs["splats"][key], pad_shape], dim=1)
            params[i][1] = torch.cat((params[i][1], torch.nn.Parameter(preload_gs["splats"][key])), dim=0)
        print("GS Loading over...")

    # MaskGaussian: add mask_score parameter [N, 2]
    if use_mask_gaussian:
        N_final = params[0][1].shape[0]  # get current N after potential preloading
        # Initialize mask_score: column 0 = mask_init_value (bias to keep), column 1 = 1.0
        mask_scores = torch.zeros((N_final, 2))
        mask_scores[:, 0] = mask_init_value
        mask_scores[:, 1] = 1.0
        params.append(["mask_score", torch.nn.Parameter(mask_scores), mask_lr])

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


# ====== MaskGaussian Helper Functions ======
def compute_mask_gate(
        mask_score: torch.Tensor,
        training: bool = True,
        temperature: float = 1.0,
) -> torch.Tensor:
    """Compute the mask gate from mask_score using Gumbel-Softmax.

    Following the MaskGaussian paper (arXiv:2412.20522) and official repo:
    - mask_score: [N, 2] learnable logits (2-class Categorical)
    - Training: Gumbel-Softmax sampling with straight-through gradient.
      The CUDA-level mask rasterization handles gradient flow independently:
      dL/d(mask) = dL/d(ma) * alpha, which is non-zero even when mask=0,
      so no soft residual trick is needed.
    - Inference: deterministic argmax

    Args:
        mask_score: [N, 2] tensor of mask logits
        training: whether in training mode
        temperature: Gumbel-Softmax temperature (lower = more discrete)

    Returns:
        gate: [N] tensor of gate values (training: 0/1 with STE gradient; inference: exact 0/1)
    """
    if training:
        log_prob = torch.nn.functional.log_softmax(mask_score, dim=-1)
        # Gumbel-Softmax: hard=True gives 0/1 forward, soft gradient backward
        gate = torch.nn.functional.gumbel_softmax(log_prob, tau=temperature, hard=True)[:, 0]  # [N]
        # No soft residual trick needed: CUDA-level mask rasterization computes
        # dL/d(mask) = dL/d(ma) * alpha independently from dL/d(alpha) = dL/d(ma) * mask.
        # When mask=0: dL/d(alpha)=0 but dL/d(mask) = v_ma * alpha ≠ 0.
        # The gradient flows back to mask_score through the Gumbel-Softmax STE.
    else:
        gate = (mask_score[:, 0] > mask_score[:, 1]).float()  # [N]
    return gate


def compute_mask_keep_prob(mask_score: torch.Tensor) -> torch.Tensor:
    """Return deterministic keep probability for each Gaussian.

    We explicitly materialize a contiguous tensor because downstream distributed
    collectives such as ``dist.gather`` require contiguous inputs.
    """
    return torch.softmax(mask_score, dim=-1)[:, 0].contiguous()


def compute_mask_prune_mask(
        mask_score: torch.Tensor,
        sample_times: int = 10,
) -> Tuple[torch.Tensor, Dict[str, Union[str, Optional[torch.Tensor]]]]:
    """Build a pruning mask from stochastic samples.

    Following the MaskGaussian paper: sample each Gaussian `sample_times` times
    and mark those that are never sampled alive for removal.
    """
    keep_prob = compute_mask_keep_prob(mask_score)
    prune_mask = torch.zeros_like(keep_prob, dtype=torch.bool)
    alive_count = None

    if sample_times > 0:
        log_prob = torch.nn.functional.log_softmax(mask_score, dim=-1)
        batched = log_prob.unsqueeze(0).expand(sample_times, -1, -1)
        sampled = torch.nn.functional.gumbel_softmax(batched, hard=True, dim=-1)[:, :, 0]
        alive_count = sampled.sum(0)
        prune_mask |= alive_count == 0

    criterion = f"alive_count==0/{sample_times}" if sample_times > 0 else "disabled"
    stats: Dict[str, Union[str, Optional[torch.Tensor]]] = {
        "criterion": criterion,
        "keep_prob": keep_prob,
        "alive_count": alive_count,
    }
    return prune_mask, stats


def compute_camera_obb(
        camera_positions: np.ndarray,
        up_direction: np.ndarray,
        padding: float = 0.1,
) -> list:
    """Compute a 2D OBB on the horizontal plane from camera positions.

    Returns [cx, cy, cz, halfW, halfH, angle] following the Camera-OBB protocol.
    """
    up = up_direction / np.linalg.norm(up_direction)

    if abs(up[0]) < 0.9:
        fallback = np.array([1.0, 0.0, 0.0])
    else:
        fallback = np.array([0.0, 1.0, 0.0])
    ref_right = np.cross(fallback, up)
    ref_right /= np.linalg.norm(ref_right)
    ref_forward = np.cross(up, ref_right)
    ref_forward /= np.linalg.norm(ref_forward)

    proj_u = camera_positions @ ref_right
    proj_v = camera_positions @ ref_forward

    coords_2d = np.stack([proj_u, proj_v], axis=1)
    mean_2d = coords_2d.mean(axis=0)
    centered = coords_2d - mean_2d
    cov = (centered.T @ centered) / len(centered)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    principal = eigenvectors[:, idx[0]]

    angle = np.arctan2(principal[1], principal[0])

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    axis0_2d = np.array([cos_a, sin_a])
    axis1_2d = np.array([-sin_a, cos_a])
    proj_axis0 = centered @ axis0_2d
    proj_axis1 = centered @ axis1_2d

    halfW = (proj_axis0.max() - proj_axis0.min()) / 2 * (1 + padding)
    halfH = (proj_axis1.max() - proj_axis1.min()) / 2 * (1 + padding)

    center_3d = mean_2d[0] * ref_right + mean_2d[1] * ref_forward
    up_component = (camera_positions @ up).mean()
    center_3d += up_component * up

    return [
        float(center_3d[0]), float(center_3d[1]), float(center_3d[2]),
        float(halfW), float(halfH), float(angle),
    ]


class Runner:
    """Engine for training and testing."""

    def __init__(
            self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=False if cfg.no_normalize else True,
            test_every=cfg.test_every,
            downsample_pts_num=cfg.downsample_pts_num,
            downsample_mode=cfg.downsample_mode,
            detect_anchor_candidates=cfg.use_anchor_protection,
            world_rank=self.world_rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
            align_sky_only=cfg.init_align_sky_only,
        )

        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss or cfg.sky_depth_from_pcd,
            load_normals=cfg.normal_loss or cfg.sky_depth_smooth or cfg.sky_depth_from_pcd,
            pano_only=cfg.pano_only,
            video_only=cfg.video_only,
            pano_repeat=cfg.pano_repeat,
            pano_azimuth_interval=cfg.pano_azimuth_interval,
            polar_up_azimuth_interval=cfg.polar_up_azimuth_interval,
            polar_up_extra_layers=cfg.polar_up_extra_layers,
            polar_down_azimuth_interval=cfg.polar_down_azimuth_interval,
            video_prefix_filter=cfg.video_prefix_filter,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            preload_gs_path=cfg.preload_gs_path,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            # MaskGaussian
            use_mask_gaussian=cfg.use_mask_gaussian,
            mask_lr=cfg.mask_lr,
            mask_init_value=cfg.mask_init_value,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Anchor protection: mark outlier/sparse initial points as anchors
        N_init = len(self.splats["means"])
        if cfg.use_anchor_protection:
            is_outlier_full = torch.from_numpy(self.parser.points_is_outlier).bool()
            # Distribute to current rank (same as points[world_rank::world_size])
            is_outlier = is_outlier_full[self.world_rank::self.world_size]
            # If preload_gs_path is used, the splats are concatenated with preloaded ones.
            # Preloaded points are NOT marked as anchors.
            if is_outlier.shape[0] < N_init:
                is_outlier = torch.cat([
                    is_outlier,
                    torch.zeros(N_init - is_outlier.shape[0], dtype=torch.bool)
                ])
            self.strategy_state["is_anchor"] = is_outlier.to(self.device)
            n_anchor = self.strategy_state["is_anchor"].sum().item()
            print(f"[Anchor Protection] Protected anchor points: {n_anchor} / {N_init}")
        else:
            self.strategy_state["is_anchor"] = None
        self.strategy_state["anchor_no_densify"] = cfg.anchor_no_densify
        self.strategy_state["anchor_opa_reset_protection"] = cfg.anchor_opa_reset_protection

        # Align protection: mark aligned_pcd points
        if cfg.use_align_protection:
            is_align_full = torch.from_numpy(self.parser.points_is_align).bool()
            is_align = is_align_full[self.world_rank::self.world_size]
            if is_align.shape[0] < N_init:
                is_align = torch.cat([
                    is_align,
                    torch.zeros(N_init - is_align.shape[0], dtype=torch.bool)
                ])
            self.strategy_state["is_align"] = is_align.to(self.device)
            n_align = self.strategy_state["is_align"].sum().item()
            print(f"[Align Protection] Protected align points: {n_align} / {N_init}")
        else:
            self.strategy_state["is_align"] = None
        self.strategy_state["align_no_densify"] = cfg.align_no_densify
        self.strategy_state["align_no_prune"] = cfg.align_no_prune
        self.strategy_state["align_opa_reset_protection"] = cfg.align_opa_reset_protection

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        # Depth to normal module
        if cfg.normal_loss:
            self.depth2normal_layer = Depth2Normal()
        else:
            self.depth2normal_layer = None

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Sky depth from point cloud: pre-compute merged sky depth maps
        self.sky_depth_maps = {}
        if cfg.sky_depth_from_pcd:
            from gs.sky_depth import precompute_sky_depth_maps, render_sky_depth_gsplat
            self.sky_depth_maps = precompute_sky_depth_maps(
                render_fn=render_sky_depth_gsplat,
                parser=self.parser,
                trainset=self.trainset,
                sky_normal_threshold=cfg.sky_normal_threshold,
                result_dir=cfg.result_dir,
                device=self.device,
                world_rank=self.world_rank,
                world_size=self.world_size,
                debug=cfg.sky_depth_from_pcd_debug,
            )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.server.initial_camera.position = self.parser.center_point
            self.server.initial_camera.look_at = self.parser.facing_direction
            self.server.initial_camera.up = self.parser.up_direction
            self.server.scene.set_up_direction(self.parser.up_direction)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        # Training state flag (used by MaskGaussian to switch between stochastic/deterministic gate)
        self.is_training = False

    def rasterize_splats(
            self,
            camtoworlds: Tensor,
            Ks: Tensor,
            width: int,
            height: int,
            masks: Optional[Tensor] = None,
            rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
            camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
            mask_gate: Optional[Tensor] = None,
            **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        # MaskGaussian: compute per-Gaussian mask gate for CUDA-level mask rasterization
        # Instead of multiplying mask into opacities (which kills gradient flow),
        # we pass the mask as a separate tensor to the CUDA rasterizer.
        gauss_masks_tensor = None
        if self.cfg.use_mask_gaussian and "mask_score" in self.splats:
            if mask_gate is None:
                # No external gate provided (eval/viewer); compute it internally.
                mask_gate = compute_mask_gate(
                    self.splats["mask_score"],
                    training=self.is_training,
                    temperature=self.cfg.mask_temperature,
                )
            gauss_masks_tensor = mask_gate  # [N] — pass to CUDA as gauss_masks

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        if self.cfg.use_mask_gaussian:
            kwargs["gauss_masks"] = gauss_masks_tensor
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            if self.cfg.sky_depth_from_pcd and render_colors.shape[-1] > 3:
                render_colors[..., :3][~masks] = 0
            else:
                render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        self.is_training = True

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        save_steps = {i - 1 for i in cfg.save_steps}
        ply_steps = {i - 1 for i in cfg.ply_steps}
        eval_steps = {i - 1 for i in cfg.eval_steps}
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            loss_decay = (max_steps - step) / max_steps
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                    pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if cfg.apply_mask and "mask" in data else None  # [1, H, W]

            opt_depth = False
            original_depth_mask = None
            if cfg.depth_loss:
                depths_gt = data["depths"].to(device)  # [1, M]
                if depths_gt.sum() > 0:
                    opt_depth = True
                if cfg.sky_depth_from_pcd:
                    original_depth_mask = (depths_gt > 1e-4)  # before sky merge
                    image_id_val = data["image_id"].item() if isinstance(data["image_id"], torch.Tensor) else data["image_id"]
                    if image_id_val in self.sky_depth_maps and self.sky_depth_maps[image_id_val] is not None:
                        depths_gt = self.sky_depth_maps[image_id_val].unsqueeze(0).to(device)
                        opt_depth = True
            else:
                depths_gt = None

            opt_normal = False
            if cfg.normal_loss:
                normals_gt = data["normals"].to(device)  # [1, M, 3]
                if normals_gt.abs().sum() > 0:
                    normal_mask_gt = ((normals_gt ** 2).sum(dim=1) > 0.1)
                    normals_gt = F.normalize(normals_gt, dim=1)  # Normalize GT normals along the 3-channel dimension.
                    opt_normal = True
            else:
                normals_gt = None
                normal_mask_gt = None

            if masks is not None:
                pixels[~masks] = 0

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            # sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            sh_degree_to_use = cfg.sh_degree

            # MaskGaussian: compute mask gate once per step, shared between render and loss
            cur_mask_gate = None
            if cfg.use_mask_gaussian and "mask_score" in self.splats:
                cur_mask_gate = compute_mask_gate(
                    self.splats["mask_score"],
                    training=True,
                    temperature=cfg.mask_temperature,
                )

            # forward
            need_depth = opt_depth or opt_normal or cfg.sky_depth_smooth or cfg.dist_loss
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if need_depth else "RGB",
                masks=masks,
                mask_gate=cur_mask_gate,
                distloss=cfg.dist_loss,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)
            elif cfg.background_color is not None:
                background_color = cfg.background_color.split()
                background_color = [int(b) for b in background_color]
                bkgd = torch.tensor(background_color, device=device, dtype=torch.float32).reshape(1, 3) / 255
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if cfg.lpips_lambda1 > 0:
                pixels_p = torch.clip(pixels.permute(0, 3, 1, 2), 0, 1)  # [1, 3, H, W]
                colors_p = torch.clip(colors.permute(0, 3, 1, 2), 0, 1)  # [1, 3, H, W]
                lpipsloss = self.lpips(colors_p, pixels_p)
                if cfg.lpips_lambda2 > 0:
                    lpips_weight = cfg.lpips_lambda2 + loss_decay * (cfg.lpips_lambda1 - cfg.lpips_lambda2)
                else:
                    lpips_weight = cfg.lpips_lambda1
                loss = loss + lpips_weight * lpipsloss
            else:
                lpipsloss = torch.tensor(0.0).to(self.device)

            if cfg.use_scale_regularization and step % 10 == 0:
                scale_exp = torch.exp(self.splats["scales"])
                scale_reg = (
                        torch.maximum(
                            scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                            scale_exp.new_tensor(cfg.max_gauss_ratio),
                        )
                        - cfg.max_gauss_ratio
                )
                scale_reg = 0.1 * scale_reg.mean()
                loss = loss + scale_reg
            else:
                scale_reg = torch.tensor(0.0).to(self.device)

            if opt_depth:
                # calculate loss in disparity space
                if depths_gt.ndim == 3:
                    depths_gt = depths_gt.unsqueeze(-1)
                depth_mask = (depths > 0.0) & (depths_gt > 1e-4)
                if depth_mask.any():
                    disp = 1.0 / depths[depth_mask]
                    disp_gt = 1.0 / depths_gt[depth_mask]
                    depthloss = F.l1_loss(disp, disp_gt, reduction="mean") * self.scene_scale
                else:
                    depthloss = torch.tensor(0.0, dtype=torch.float32, device=device)
                depth_loss_weight = cfg.depth_lambda2 + loss_decay * (cfg.depth_lambda1 - cfg.depth_lambda2)
                loss += depthloss * depth_loss_weight
            else:
                depthloss = torch.tensor(0.0, dtype=torch.float32, device=device)

            if opt_normal:
                depths = depths.permute(0, 3, 1, 2)  # [B, 1, H, W]
                intrinsics = Ks.repeat(depths.shape[0], 1, 1)
                if cfg.sky_depth_from_pcd and original_depth_mask is not None:
                    normal_mask = original_depth_mask.unsqueeze(1) if original_depth_mask.ndim == 3 else original_depth_mask
                    normal_mask = normal_mask & (depths > 0)
                else:
                    normal_mask = (depths > 0)
                normals_pred, normals_pred_mask = self.depth2normal_layer(
                    depth=depths,
                    intrinsics=intrinsics,
                    masks=normal_mask,
                    scale=1.0,
                )  # overwrite pred_normal

                normals_gt = normals_gt * normals_pred_mask
                normals_pred = normals_pred * normals_pred_mask

                dot = (normals_gt * normals_pred).sum(dim=1)  # [B, H, W] per-pixel dot product
                valid_mask = normals_pred_mask.squeeze(1) & normal_mask_gt  # [B, H, W]
                normalloss = (1 - dot[valid_mask]).mean() if valid_mask.any() else torch.tensor(0.0, device=dot.device)
                normal_loss_weight = cfg.normal_lambda2 + loss_decay * (cfg.normal_lambda1 - cfg.normal_lambda2)
                loss += normalloss * normal_loss_weight
            else:
                normalloss = torch.tensor(0.0, dtype=torch.float32, device=device)

            # Sky depth smoothness: TV loss on rendered depth in sky regions
            sky_depth_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if cfg.sky_depth_smooth and depths is not None:
                normals_for_sky = data["normals"].to(device)  # [1, 3, H, W]
                sky_mask = (normals_for_sky.norm(dim=1) < cfg.sky_normal_threshold)  # [1, H, W]

                # Erode sky mask to avoid boundary artifacts
                if cfg.sky_erode_pixels > 0 and sky_mask.any():
                    non_sky = (~sky_mask).float().unsqueeze(1)  # [1, 1, H, W]
                    k = 2 * cfg.sky_erode_pixels + 1
                    non_sky_dilated = F.max_pool2d(non_sky, kernel_size=k, stride=1, padding=cfg.sky_erode_pixels)
                    sky_mask = (non_sky_dilated < 0.5).squeeze(1)  # [1, H, W]

                if sky_mask.sum() > 100:
                    depth_hw = depths.reshape(depths.shape[0], height, width)  # [B, H, W]
                    # Exclude unrendered (depth~0) pixels to avoid 1/0 and huge disparity spikes.
                    rendered_mask = (depth_hw > 1e-3)
                    sky_rendered_mask = sky_mask & rendered_mask
                    disp_hw = torch.where(rendered_mask, 1.0 / depth_hw.clamp(min=1e-6), torch.zeros_like(depth_hw))
                    dx = (disp_hw[:, :, 1:] - disp_hw[:, :, :-1]).abs()
                    dy = (disp_hw[:, 1:, :] - disp_hw[:, :-1, :]).abs()
                    mask_dx = sky_rendered_mask[:, :, 1:] & sky_rendered_mask[:, :, :-1]
                    mask_dy = sky_rendered_mask[:, 1:, :] & sky_rendered_mask[:, :-1, :]

                    smooth_x = dx[mask_dx].mean() if mask_dx.any() else torch.tensor(0.0, device=device)
                    smooth_y = dy[mask_dy].mean() if mask_dy.any() else torch.tensor(0.0, device=device)
                    sky_depth_loss = (smooth_x + smooth_y) * self.scene_scale

                    sdl_weight = cfg.sky_depth_smooth_lambda2 + loss_decay * (cfg.sky_depth_smooth_lambda1 - cfg.sky_depth_smooth_lambda2)
                    loss += sky_depth_loss * sdl_weight

            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            # Distortion loss (depth variance proxy for 2DGS distortion)
            dist_loss_val = torch.tensor(0.0, dtype=torch.float32, device=device)
            if cfg.dist_loss and step > cfg.dist_start_iter and "render_distort" in info:
                dist_loss_val = info["render_distort"].mean()
                loss += dist_loss_val * cfg.dist_lambda

            # MaskGaussian: mask sparsity regularization
            # Uses the SAME mask_gate sampled above (shared with rendering) to match
            # the original MaskGaussian repo: mask_loss = (torch.mean(mask))**2
            mask_loss_val = torch.tensor(0.0, device=device)
            if cfg.use_mask_gaussian and cur_mask_gate is not None:
                mask_loss_val = (cur_mask_gate.mean()) ** 2
                lambda_mask = cfg.mask_lambda if (cfg.mask_from_iter <= step <= cfg.mask_until_iter) else 0.0
                loss = loss + lambda_mask * mask_loss_val

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth={depthloss.item():.3f}| "
            if cfg.normal_loss:
                desc += f"normal={normalloss.item():.3f}| "
            if cfg.sky_depth_smooth:
                desc += f"sky={sky_depth_loss.item():.3f}| "
            if cfg.dist_loss:
                desc += f"dist_loss={dist_loss_val.item():.3f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose_err={pose_err.item():.4f}| "
            if cfg.use_mask_gaussian and "mask_score" in self.splats:
                with torch.no_grad():
                    _mp = torch.softmax(self.splats["mask_score"], dim=-1)[:, 0]
                    _n_active = int((_mp > 0.5).sum().item())
                desc += f"act_mask={_n_active}/{len(_mp)}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/lpipsloss", lpipsloss.item(), step)
                self.writer.add_scalar("train/scale_reg", scale_reg.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.sky_depth_smooth:
                    self.writer.add_scalar("train/sky_depth_loss", sky_depth_loss.item(), step)
                if cfg.dist_loss:
                    self.writer.add_scalar("train/dist_loss", dist_loss_val.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                # MaskGaussian tensorboard logs
                if cfg.use_mask_gaussian and "mask_score" in self.splats:
                    with torch.no_grad():
                        _mask_prob = torch.softmax(self.splats["mask_score"], dim=-1)[:, 0]
                    self.writer.add_scalar("train/mask_prob_mean", _mask_prob.mean().item(), step)
                    self.writer.add_scalar("train/mask_active_ratio", (_mask_prob > 0.5).float().mean().item(), step)
                    self.writer.add_scalar("train/mask_loss", mask_loss_val.item(), step)
                    self.writer.add_scalar("train/mask_num_active", int((_mask_prob > 0.5).sum().item()), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in save_steps or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                        f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                        "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()

                # save the normalization params
                data["transform"] = self.parser.transform
                # save initial camera params
                data["up_direction"] = self.parser.up_direction
                data["facing_direction"] = self.parser.facing_direction
                data["center_point"] = self.parser.center_point

                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (step in ply_steps or step == max_steps - 1) and cfg.save_ply:
                # load scene-level meta info
                if os.path.exists(f"{'/'.join(cfg.data_dir.split('/')[:-1])}/meta_info.json"):
                    with open(f"{'/'.join(cfg.data_dir.split('/')[:-1])}/meta_info.json") as f:
                        scene_type = json.load(f)["scene_type"]
                elif os.path.exists(f"{cfg.data_dir}/meta_info.json"):
                    with open(f"{cfg.data_dir}/meta_info.json") as f:
                        scene_type = json.load(f)["scene_type"]
                else:
                    scene_type = None

                if self.cfg.app_opt:
                    # eval at origin to bake the appearance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                mask_prob = None
                if cfg.use_mask_gaussian and "mask_score" in self.splats:
                    mask_prob = compute_mask_keep_prob(self.splats["mask_score"])
                is_anchor_export = None
                if cfg.mask_export_anchor_protection:
                    _anchor = self.strategy_state.get("is_anchor", None)
                    if _anchor is not None:
                        is_anchor_export = _anchor.clone()

                is_align_export = None
                if cfg.mask_export_align_protection:
                    _align = self.strategy_state.get("is_align", None)
                    if _align is not None:
                        is_align_export = _align.clone()

                # gather all results
                if self.world_size > 1:
                    import torch.distributed as dist

                    # Step 1: Synchronize per-rank Gaussian counts (they diverge
                    # after densify/prune, especially with MaskGaussian pruning).
                    local_n = torch.tensor([means.shape[0]], dtype=torch.long, device=means.device)
                    all_n = [torch.zeros(1, dtype=torch.long, device=means.device) for _ in range(self.world_size)]
                    dist.all_gather(all_n, local_n)
                    all_n = [int(n.item()) for n in all_n]
                    max_n = max(all_n)
                    total_n = sum(all_n)
                    if self.world_rank == 0:
                        print(f"[Gather] Per-rank GS counts: {all_n}, total={total_n}, max={max_n}")

                    # Step 2: Helper – pad tensor's first dim to target_n with zeros.
                    def _pad_to(tensor, target_n):
                        if tensor.shape[0] >= target_n:
                            return tensor
                        pad = torch.zeros(
                            target_n - tensor.shape[0], *tensor.shape[1:],
                            dtype=tensor.dtype, device=tensor.device,
                        )
                        return torch.cat([tensor, pad], dim=0)

                    # Step 3: Gather variable-size tensors across ranks.
                    # Each rank pads to max_n, gathers, then rank 0 trims and concatenates.
                    def _gather_variable_size(tensor, all_n, max_n):
                        tensor_padded = _pad_to(tensor, max_n)
                        if self.world_rank == 0:
                            gathered = [
                                torch.zeros(max_n, *tensor.shape[1:],
                                            dtype=tensor.dtype, device=self.device)
                                for _ in range(self.world_size)
                            ]
                        else:
                            gathered = None
                        dist.gather(tensor_padded, gathered, dst=0)
                        if self.world_rank == 0:
                            return torch.cat([g[:n] for g, n in zip(gathered, all_n)], dim=0)
                        return None

                    means = _gather_variable_size(means, all_n, max_n)
                    scales = _gather_variable_size(scales, all_n, max_n)
                    quats = _gather_variable_size(quats, all_n, max_n)
                    opacities = _gather_variable_size(opacities, all_n, max_n)
                    sh0 = _gather_variable_size(sh0, all_n, max_n)
                    shN = _gather_variable_size(shN, all_n, max_n)
                    if mask_prob is not None:
                        mask_prob = _gather_variable_size(mask_prob, all_n, max_n)
                    if is_anchor_export is not None:
                        is_anchor_export = _gather_variable_size(
                            is_anchor_export.float(), all_n, max_n
                        )
                        if is_anchor_export is not None:
                            is_anchor_export = is_anchor_export.bool()
                    if is_align_export is not None:
                        is_align_export = _gather_variable_size(
                            is_align_export.float(), all_n, max_n
                        )
                        if is_align_export is not None:
                            is_align_export = is_align_export.bool()

                if self.world_rank == 0:

                    if self.cfg.do_prune:
                        print("Pruning by opacities...")
                        opacities_sigmoid = torch.sigmoid(opacities)
                        opacity_median = torch.median(opacities_sigmoid).item()
                        prune_mask = opacities_sigmoid > min(self.cfg.prune_opacity_threshold, opacity_median)
                        # Anchor protection: keep anchors even if opacity is low
                        if is_anchor_export is not None:
                            n_anchor_opa = (~prune_mask & is_anchor_export).sum().item()
                            prune_mask = prune_mask | is_anchor_export
                            if n_anchor_opa > 0:
                                print(f"[Anchor Protection] PLY opacity prune: saved {n_anchor_opa} anchor points.")
                        # Align protection: keep align points even if opacity is low
                        if is_align_export is not None:
                            n_align_opa = (~prune_mask & is_align_export).sum().item()
                            prune_mask = prune_mask | is_align_export
                            if n_align_opa > 0:
                                print(f"[Align Protection] PLY opacity prune: saved {n_align_opa} align points.")
                        print(f"Pruned {opacities_sigmoid.shape[0]} splats, {prune_mask.sum()} splats left.")
                        means = means[prune_mask]
                        scales = scales[prune_mask]
                        quats = quats[prune_mask]
                        opacities = opacities[prune_mask]
                        sh0 = sh0[prune_mask]
                        shN = shN[prune_mask]
                        if mask_prob is not None:
                            mask_prob = mask_prob[prune_mask]
                        if is_anchor_export is not None:
                            is_anchor_export = is_anchor_export[prune_mask]
                        if is_align_export is not None:
                            is_align_export = is_align_export[prune_mask]

                    # MaskGaussian: filter out low-probability Gaussians before PLY export
                    if mask_prob is not None:
                        n_before = len(mask_prob)
                        if cfg.mask_export_stochastic:
                            mask_keep = torch.bernoulli(mask_prob).bool()
                            filter_desc = "stochastic Bernoulli sampling"
                        else:
                            mask_keep = mask_prob > cfg.mask_export_prob_threshold
                            filter_desc = f"keep_prob>{cfg.mask_export_prob_threshold}"
                        # Anchor protection: keep anchors regardless of mask_prob
                        n_anchor_mask = 0
                        if is_anchor_export is not None:
                            n_anchor_mask = (~mask_keep & is_anchor_export).sum().item()
                            mask_keep = mask_keep | is_anchor_export
                        # Align protection: keep align points regardless of mask_prob
                        n_align_mask = 0
                        if is_align_export is not None:
                            n_align_mask = (~mask_keep & is_align_export).sum().item()
                            mask_keep = mask_keep | is_align_export
                        means = means[mask_keep]
                        scales = scales[mask_keep]
                        quats = quats[mask_keep]
                        opacities = opacities[mask_keep]
                        sh0 = sh0[mask_keep]
                        shN = shN[mask_keep]
                        prot_parts = []
                        if n_anchor_mask > 0:
                            prot_parts.append(f"anchor: {n_anchor_mask}")
                        if n_align_mask > 0:
                            prot_parts.append(f"align: {n_align_mask}")
                        prot_msg = f" (protected: {', '.join(prot_parts)})" if prot_parts else ""
                        print(
                            f"[MaskGaussian] PLY export: kept {mask_keep.sum().item()}/{n_before} "
                            f"Gaussians by {filter_desc}{prot_msg}."
                        )

                    # Normalize quaternions before exporting to PLY / SPZ.
                    # Rendering path can normalize internally, but exported files should store unit quaternions.
                    export_quats = F.normalize(quats, p=2, dim=-1)
                    export_splats(
                        means=means,
                        scales=scales,
                        quats=export_quats,
                        opacities=opacities,
                        sh0=sh0,
                        shN=shN,
                        format="ply",
                        save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                    )

                    # saving position info
                    x_min, x_max = means[:, 0].min().item(), means[:, 0].max().item()
                    y_min, y_max = means[:, 1].min().item(), means[:, 1].max().item()
                    z_min, z_max = means[:, 2].min().item(), means[:, 2].max().item()

                    camtoworlds_np = np.array(self.parser.camtoworlds)
                    train_cam_positions = camtoworlds_np[self.trainset.indices, :3, 3]
                    up_dir = np.array(self.parser.up_direction)
                    camera_obb = compute_camera_obb(train_cam_positions, up_dir)

                    air_wall_bbox = [x_min, x_max, y_min, y_max, z_min, z_max]
                    if camera_obb and isinstance(camera_obb, list):
                        air_wall_bbox.extend(camera_obb)

                    pos_meta_info = {
                        "up_direction": self.parser.up_direction.tolist(),
                        "facing_direction": self.parser.facing_direction.tolist(),
                        "center_point": self.parser.center_point.tolist(),
                        "scale": self.parser.rescale,
                        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "z_min": z_min, "z_max": z_max,
                        "scene_type": scene_type,
                        "bbox": [],
                        "human_scale": 1.0,
                        "kwargs": {
                            "camera_obb": camera_obb,
                        },
                        "air_wall": {
                            "scene_type": scene_type or "",
                            "bbox": air_wall_bbox,
                        },
                    }

                    with open(f"{self.ply_dir}/position_meta_info.json", "w") as f:
                        json.dump(pos_meta_info, f)

                    if cfg.convert_to_spz:
                        import spz
                        print("Converting to SPZ...")
                        unpack_options = spz.UnpackOptions()
                        unpack_options.to_coord = spz.CoordinateSystem.RUB
                        # Load the PLY file
                        cloud = spz.load_splat_from_ply(f"{self.ply_dir}/point_cloud_{step}.ply", unpack_options)
                        if cfg.antialiased:
                            cloud.antialiased = True
                        # Save as compressed SPZ format

                        pack_options = spz.PackOptions()
                        if hasattr(pack_options, "version"):
                            pack_options.version = 3
                        pack_options.from_coord = spz.CoordinateSystem.RDF  # from RDF to RUB
                        spz.save_spz(cloud, pack_options, f"{self.ply_dir}/point_cloud_{step}.spz")

                    if cfg.convert_to_spx:
                        # convert to spx
                        try:
                            os.system(f"gsbox p2x -i {self.ply_dir}/point_cloud_{step}.ply -o {self.ply_dir}/point_cloud_{step}.spx")
                        except:
                            print("Failed to convert to spx, please install gsbox at first.")

                # training over, saving mesh
                if cfg.export_mesh:
                    import open3d as o3d
                    from gs.extract_mesh import estimate_bounding_sphere, extract_mesh_bounded, post_process_mesh
                    if self.world_rank == 0:
                        print("Exporting mesh...")

                    camtoworlds_mesh = self.parser.camtoworlds  # (N, 4, 4)
                    Ks_mesh = [self.parser.Ks_dict[cid].copy() for cid in self.parser.camera_ids]
                    widths = [self.parser.imsize_dict[cid][0] for cid in self.parser.camera_ids]
                    heights = [self.parser.imsize_dict[cid][1] for cid in self.parser.camera_ids]

                    rgbmaps, depthmaps = [], []
                    for j in tqdm.tqdm(range(camtoworlds_mesh.shape[0]), desc="Rendering RGB and depth maps", disable=self.world_rank != 0):
                        c2w = camtoworlds_mesh[j].copy()[None]
                        K = Ks_mesh[j].copy()[None]

                        if isinstance(c2w, np.ndarray):
                            c2w = torch.from_numpy(c2w).float().to(self.device)
                        if isinstance(K, np.ndarray):
                            K = torch.from_numpy(K).float().to(self.device)

                        with torch.no_grad():
                            renders, _, _ = self.rasterize_splats(
                                camtoworlds=c2w,
                                Ks=K,
                                width=widths[j],
                                height=heights[j],
                                sh_degree=cfg.sh_degree,
                                near_plane=cfg.near_plane,
                                far_plane=cfg.far_plane,
                                masks=None,
                                render_mode="RGB+ED",
                            )  # [1, H, W, 4]
                        rgbmap = torch.clamp(renders[0, :, :, 0:3].permute(2, 0, 1), 0.0, 1.0)  # [3, H, W]
                        depthmap = renders[0, :, :, 3:4].permute(2, 0, 1).cpu()  # [1, H, W]
                        rgbmaps.append(rgbmap.cpu())
                        depthmaps.append(depthmap.cpu())

                    if self.world_rank == 0:
                        # ---- Estimate bounding sphere ----
                        center, radius = estimate_bounding_sphere(
                            means, camtoworlds_mesh,
                            method="camera",
                            gs_percentile=99,
                            gs_scale=1.1,
                        )

                        num_cluster = 1  # Use the same mesh hyperparameter indoors and outdoors.
                        cfg_depth_trunc = 5.0

                        print("Extracting bounded mesh ...")
                        depth_trunc = (radius * 2.0) if cfg_depth_trunc < 0 else cfg_depth_trunc
                        voxel_size = (depth_trunc / 128) if cfg.voxel_size < 0 else cfg.voxel_size
                        sdf_trunc = (4.0 * voxel_size) if cfg.sdf_trunc < 0 else cfg.sdf_trunc

                        mesh = extract_mesh_bounded(
                            rgbmaps, depthmaps, camtoworlds_mesh, Ks_mesh, widths, heights,
                            voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc,
                        )

                        print("Post-processing mesh ...")
                        mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
                        o3d.io.write_triangle_mesh(f"{self.ply_dir}/fuse_post.ply", mesh_post)

                        if cfg.downsample_perc < 1.0:
                            try:
                                import pymeshlab
                                ms = pymeshlab.MeshSet()
                                ms.load_new_mesh(f"{self.ply_dir}/fuse_post.ply")
                                ms.apply_filter(
                                    'meshing_decimation_quadric_edge_collapse',
                                    targetperc=cfg.downsample_perc,  # Target face ratio, e.g. 0.25 keeps 25%.
                                    preservenormal=True  # Preserve normal directions as much as possible.
                                )
                                ms.save_current_mesh(f"{self.ply_dir}/fuse_simplified.ply")
                            except (ImportError, OSError) as e:
                                print(f"[WARNING] pymeshlab unavailable ({e}), "
                                      f"falling back to Open3D mesh simplification.")
                                mesh_to_simplify = o3d.io.read_triangle_mesh(
                                    f"{self.ply_dir}/fuse_post.ply")
                                target_faces = int(
                                    len(mesh_to_simplify.triangles) * cfg.downsample_perc)
                                mesh_simplified = mesh_to_simplify.simplify_quadric_decimation(
                                    target_number_of_triangles=target_faces)
                                o3d.io.write_triangle_mesh(
                                    f"{self.ply_dir}/fuse_simplified.ply", mesh_simplified)

                    dist.barrier()

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # Freeze anchor positions: zero out means grad so optimizer produces no update
            if cfg.anchor_freeze_means and self.strategy_state.get("is_anchor") is not None:
                anchor_mask = self.strategy_state["is_anchor"]
                if self.splats["means"].grad is not None:
                    self.splats["means"].grad[anchor_mask] = 0.0

            # Freeze align positions
            if cfg.align_freeze_means and self.strategy_state.get("is_align") is not None:
                align_mask = self.strategy_state["is_align"]
                if self.splats["means"].grad is not None:
                    self.splats["means"].grad[align_mask] = 0.0

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # MaskGaussian: periodic mask-based pruning
            # Following the paper: prune at every densification step during
            # densification, and every mask_prune_iter steps after densification.
            if cfg.use_mask_gaussian and "mask_score" in self.splats:
                strat = self.cfg.strategy
                is_densify_step = False
                densify_done = True
                if isinstance(strat, (DefaultStrategy, MCMCStrategy)):
                    densify_done = step >= strat.refine_stop_iter
                    is_densify_step = (
                            step >= strat.refine_start_iter
                            and step < strat.refine_stop_iter
                            and step % strat.refine_every == 0
                    )

                should_prune = (
                        step > 0
                        and cfg.mask_from_iter <= step <= cfg.mask_until_iter
                        and (
                                is_densify_step
                                or (densify_done and step % cfg.mask_prune_iter == 0)
                        )
                )

                if should_prune:
                    from gsplat.strategy.ops import remove as gsplat_remove
                    with torch.no_grad():
                        prune_mask, prune_stats = compute_mask_prune_mask(
                            self.splats["mask_score"],
                            sample_times=cfg.mask_prune_sample_times,
                        )
                        # Anchor protection: never mask-prune anchor points
                        n_anchor_saved = 0
                        if cfg.mask_prune_anchor_protection:
                            is_anchor = self.strategy_state.get("is_anchor", None)
                            if is_anchor is not None:
                                n_anchor_saved = (prune_mask & is_anchor).sum().item()
                                prune_mask = prune_mask & ~is_anchor

                        # Align protection: never mask-prune align points
                        n_align_saved = 0
                        if cfg.mask_prune_align_protection:
                            is_align = self.strategy_state.get("is_align", None)
                            if is_align is not None:
                                n_align_saved = (prune_mask & is_align).sum().item()
                                prune_mask = prune_mask & ~is_align

                        n_prune = prune_mask.sum().item()
                        if n_prune > 0:
                            gsplat_remove(
                                params=self.splats,
                                optimizers=self.optimizers,
                                state=self.strategy_state,
                                mask=prune_mask,
                            )
                            if world_rank == 0:
                                keep_prob = prune_stats["keep_prob"]
                                prot_parts = []
                                if n_anchor_saved > 0:
                                    prot_parts.append(f"anchor: {n_anchor_saved}")
                                if n_align_saved > 0:
                                    prot_parts.append(f"align: {n_align_saved}")
                                prot_msg = f" (protected: {', '.join(prot_parts)})" if prot_parts else ""
                                print(
                                    f"[MaskGaussian] Step {step}: pruned {n_prune} Gaussians "
                                    f"with criterion [{prune_stats['criterion']}]{prot_msg}. "
                                    f"keep_prob mean={keep_prob.mean().item():.4f}, "
                                    f"min={keep_prob.min().item():.4f}. "
                                    f"Now having {len(self.splats['means'])} Gaussians."
                                )

            # eval the full set
            if step in eval_steps:
                self.eval(step)
                self.render_traj(step)
                self.is_training = True  # restore training mode after eval

            # run compression
            if cfg.compression is not None and step in eval_steps:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                        num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        self.is_training = False
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=1)

        ellipse_time = 0
        metrics = defaultdict(list)
        # from src.utils import perturb_eval_cameras
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)

            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2)[0].cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                if canvas.ndim == 2:
                    canvas = canvas[:, :, np.newaxis].repeat(3, axis=2)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_bilateral_grid:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int, save_name="traj"):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        is_writer_rank = self.world_rank == 0
        if is_writer_rank:
            print("Running trajectory rendering...")
        self.is_training = False
        cfg = self.cfg
        device = self.device
        dist = None
        if self.world_size > 1:
            import torch.distributed as dist

        camtoworlds_all = self.parser.camtoworlds[5:-23]
        if cfg.render_traj_path == "interp":
            if camtoworlds_all.shape[0] > 100:
                rdv = np.linspace(0, camtoworlds_all.shape[0], num=100, endpoint=False, dtype=int)
                camtoworlds_all = camtoworlds_all[rdv]
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        if is_writer_rank:
            os.makedirs(video_dir, exist_ok=True)

        # All ranks must participate in distributed rasterization.
        # Only rank0 stores the rendered frames and writes the final video.
        frames = [] if is_writer_rank else None
        for i in tqdm.trange(
                len(camtoworlds_all),
                desc="Rendering trajectory",
                disable=not is_writer_rank,
        ):
            camtoworlds = camtoworlds_all[i: i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            if not is_writer_rank:
                continue

            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            canvas = torch.cat(canvas_list, dim=2)[0].cpu().numpy()  # [H, W*2, 3]
            canvas = (canvas * 255).astype(np.uint8)
            # ensure canvas is 3D: (H, W, 3)
            if canvas.ndim == 2:
                canvas = canvas[:, :, np.newaxis].repeat(3, axis=2)
            assert canvas.ndim == 3 and canvas.shape[2] == 3, \
                f"Unexpected canvas shape: {canvas.shape}, expected (H, W, 3)"
            frames.append(canvas)

        if dist is not None and dist.is_available() and dist.is_initialized():
            dist.barrier()

        if not is_writer_rank:
            return

        # try multiple approaches to save video
        video_path = f"{video_dir}/{save_name}_{step}.mp4"
        saved = False

        # Approach 1: try imageio.mimwrite (simplest, lets imageio pick codec)
        if not saved:
            try:
                imageio.mimwrite(video_path, frames, fps=30)
                saved = True
                print(f"Video saved to {video_path}")
            except Exception as e:
                print(f"imageio.mimwrite mp4 failed: {e}")

        # Approach 2: try with explicit codecs via get_writer
        if not saved:
            for codec in ["libx264", "h264", "mpeg4", "libx265", "vp9"]:
                try:
                    writer = imageio.get_writer(video_path, fps=30, codec=codec)
                    for frame in frames:
                        writer.append_data(frame)
                    writer.close()
                    saved = True
                    print(f"Video saved to {video_path} (codec={codec})")
                    break
                except Exception as e:
                    print(f"codec {codec} failed: {e}")

        # Approach 3: fallback to saving individual frames as images
        if not saved:
            print("All video codecs failed. Saving individual frames as images.")
            frames_dir = f"{video_dir}/{save_name}_{step}_frames"
            os.makedirs(frames_dir, exist_ok=True)
            for idx, frame in enumerate(frames):
                imageio.imwrite(f"{frames_dir}/{idx:04d}.png", frame)
            print(f"Frames saved to {frames_dir}/")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
            self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        self.is_training = False  # viewer always uses deterministic gate
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
            "normal": "ED"
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
                        / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        elif render_tab_state.render_mode == "normal":
            depth = render_colors[0, ..., 0]  # (H, W)
            normal_vis = depth_to_normal(depth, K)  # (H, W, 3)
            renders = normal_vis.cpu().numpy()
        return renders

    def train_post_training_mask(self):
        """MaskGaussian post_training mode:
        1. Load a pretrained checkpoint
        2. Add mask_score parameter and optimizer
        3. Phase 1: Train mask only (other params frozen) for mask_only_steps
        4. Prune based on mask probabilities
        5. Phase 2: Finetune all params for mask_finetune_steps
        """
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        assert cfg.mask_pretrained_ckpt is not None, \
            "post_training mode requires mask_pretrained_ckpt to be set."

        # 1. Load pretrained checkpoint
        print(f"[MaskGaussian post_training] Loading pretrained checkpoint: {cfg.mask_pretrained_ckpt}")
        ckpt = torch.load(cfg.mask_pretrained_ckpt, map_location=device, weights_only=False)

        # Load all params from checkpoint (except mask_score which we re-init)
        # Use list() to avoid RuntimeError from mutating dict during iteration
        for k in list(self.splats.keys()):
            if k == "mask_score":
                continue
            if k in ckpt["splats"]:
                ckpt_val = ckpt["splats"][k].to(device)
                self.splats[k] = torch.nn.Parameter(ckpt_val, requires_grad=True)

        # Re-initialize mask_score for the loaded model's size
        N = self.splats["means"].shape[0]
        mask_scores = torch.zeros((N, 2), device=device)
        mask_scores[:, 0] = cfg.mask_init_value
        mask_scores[:, 1] = 1.0
        self.splats["mask_score"] = torch.nn.Parameter(mask_scores, requires_grad=True)

        # Re-create ALL optimizers for the new parameter sizes
        BS = cfg.batch_size * self.world_size
        lr_map = {
            "means": cfg.means_lr * self.scene_scale,
            "scales": cfg.scales_lr,
            "quats": cfg.quats_lr,
            "opacities": cfg.opacities_lr,
            "sh0": cfg.sh0_lr,
            "shN": cfg.shN_lr,
            "mask_score": cfg.mask_lr,
        }
        optimizer_class = torch.optim.Adam
        if cfg.sparse_grad:
            optimizer_class = torch.optim.SparseAdam
        elif cfg.visible_adam:
            optimizer_class = SelectiveAdam
        self.optimizers = {}
        for name in self.splats.keys():
            lr = lr_map.get(name, cfg.sh0_lr)  # default to sh0_lr for unknown keys
            self.optimizers[name] = optimizer_class(
                [{"params": self.splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )

        print(f"[MaskGaussian post_training] Loaded {N} Gaussians from checkpoint.")

        # Re-check strategy sanity
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        # Data loader
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # ====== Phase 1: Train mask only (freeze other params) ======
        print(f"[MaskGaussian post_training] Phase 1: Training mask only for {cfg.mask_only_steps} steps...")
        self.is_training = True

        # Freeze all params except mask_score
        frozen_params = {}
        for name in self.splats.keys():
            if name != "mask_score":
                frozen_params[name] = self.splats[name].requires_grad
                self.splats[name].requires_grad_(False)

        # ExponentialLR for mask optimizer (following prune_finetune.py: gamma=0.95)
        mask_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["mask_score"], gamma=0.95
        )

        pbar = tqdm.tqdm(range(cfg.mask_only_steps), desc="MaskGaussian Phase1: Mask-only")
        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if (cfg.apply_mask and "mask" in data) else None
            if masks is not None:
                pixels[~masks] = 0
            height, width = pixels.shape[1:3]

            # Compute mask gate once, shared between render and loss
            cur_mask_gate = compute_mask_gate(
                self.splats["mask_score"], training=True, temperature=cfg.mask_temperature,
            )

            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds, Ks=Ks, width=width, height=height,
                sh_degree=cfg.sh_degree, near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                image_ids=image_ids, masks=masks,
                mask_gate=cur_mask_gate,
            )
            colors = renders[..., 0:3]

            # Loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            # Mask sparsity loss: use the same sampled mask_gate
            mask_loss_val = (cur_mask_gate.mean()) ** 2
            loss = loss + cfg.mask_lambda * mask_loss_val

            loss.backward()

            # Only step mask optimizer
            self.optimizers["mask_score"].step()
            self.optimizers["mask_score"].zero_grad(set_to_none=True)
            # Zero grad for others just in case
            for name, opt in self.optimizers.items():
                if name != "mask_score":
                    opt.zero_grad(set_to_none=True)

            if step % 400 == 0 and step > 0:
                mask_scheduler.step()

            with torch.no_grad():
                _mp = torch.softmax(self.splats["mask_score"], dim=-1)[:, 0]
                _n_active = int((_mp > 0.5).sum().item())
            pbar.set_description(
                f"Phase1 loss={loss.item():.4f} mask_active={_n_active}/{len(_mp)}"
            )

            # Tensorboard
            if world_rank == 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar("post_training/phase1_loss", loss.item(), step)
                self.writer.add_scalar("post_training/phase1_mask_prob_mean", _mp.mean().item(), step)
                self.writer.add_scalar("post_training/phase1_mask_active", _n_active, step)
                self.writer.flush()

        # ====== Prune after Phase 1 ======
        print("[MaskGaussian post_training] Pruning based on mask probabilities...")
        from gsplat.strategy.ops import remove as gsplat_remove

        # Unfreeze params for pruning (remove needs requires_grad info)
        for name in frozen_params:
            self.splats[name].requires_grad_(frozen_params[name])

        with torch.no_grad():
            prune_mask, prune_stats = compute_mask_prune_mask(
                self.splats["mask_score"],
                sample_times=cfg.mask_prune_sample_times,
            )
            # Anchor protection: never mask-prune anchor points
            n_anchor_saved = 0
            if cfg.mask_prune_anchor_protection:
                is_anchor = self.strategy_state.get("is_anchor", None)
                if is_anchor is not None:
                    n_anchor_saved = (prune_mask & is_anchor).sum().item()
                    prune_mask = prune_mask & ~is_anchor

            # Align protection: never mask-prune align points
            n_align_saved = 0
            if cfg.mask_prune_align_protection:
                is_align = self.strategy_state.get("is_align", None)
                if is_align is not None:
                    n_align_saved = (prune_mask & is_align).sum().item()
                    prune_mask = prune_mask & ~is_align

            n_before = len(self.splats["means"])
            n_prune = prune_mask.sum().item()

            if n_prune > 0:
                gsplat_remove(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    mask=prune_mask,
                )

            n_after = len(self.splats["means"])
            keep_prob = prune_stats["keep_prob"]
            prot_parts = []
            if n_anchor_saved > 0:
                prot_parts.append(f"anchor: {n_anchor_saved}")
            if n_align_saved > 0:
                prot_parts.append(f"align: {n_align_saved}")
            prot_msg = f" (protected: {', '.join(prot_parts)})" if prot_parts else ""
            print(
                f"[MaskGaussian post_training] Pruned {n_prune} Gaussians with criterion "
                f"[{prune_stats['criterion']}]{prot_msg}: {n_before} -> {n_after}. "
                f"keep_prob mean={keep_prob.mean().item():.4f}, min={keep_prob.min().item():.4f}"
            )

        # ====== Phase 2: Finetune all params ======
        print(f"[MaskGaussian post_training] Phase 2: Finetuning all params for {cfg.mask_finetune_steps} steps...")

        # Reset mask_score to favor keeping all remaining Gaussians
        with torch.no_grad():
            N_new = self.splats["means"].shape[0]
            self.splats["mask_score"].data[:, 0] = cfg.mask_init_value
            self.splats["mask_score"].data[:, 1] = 1.0

        # Re-initialize all optimizers for finetuning
        BS = cfg.batch_size * self.world_size
        for name in self.splats.keys():
            if name in self.optimizers:
                old_lr = self.optimizers[name].param_groups[0]["lr"]
                self.optimizers[name] = torch.optim.Adam(
                    [{"params": self.splats[name], "lr": old_lr, "name": name}],
                    eps=1e-15 / math.sqrt(BS),
                    betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
                )

        finetune_schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max(cfg.mask_finetune_steps, 1))
            ),
        ]

        trainloader_iter = iter(trainloader)
        pbar = tqdm.tqdm(range(cfg.mask_finetune_steps), desc="MaskGaussian Phase2: Finetune")
        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if (cfg.apply_mask and "mask" in data) else None
            if masks is not None:
                pixels[~masks] = 0
            height, width = pixels.shape[1:3]

            # Compute mask gate once, shared between render and loss
            cur_mask_gate = compute_mask_gate(
                self.splats["mask_score"], training=True, temperature=cfg.mask_temperature,
            )

            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds, Ks=Ks, width=width, height=height,
                sh_degree=cfg.sh_degree, near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                image_ids=image_ids, masks=masks,
                mask_gate=cur_mask_gate,
            )
            colors = renders[..., 0:3]

            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            # Continue mask loss during finetuning (use sampled gate)
            mask_loss_val = (cur_mask_gate.mean()) ** 2
            loss = loss + cfg.mask_lambda * mask_loss_val

            loss.backward()

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in finetune_schedulers:
                scheduler.step()

            pbar.set_description(f"Phase2 loss={loss.item():.4f}")

            if world_rank == 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar("post_training/phase2_loss", loss.item(), step)
                self.writer.flush()

        # Final eval
        self.is_training = False
        print("[MaskGaussian post_training] Training complete. Running final evaluation...")
        self.eval(step=cfg.mask_only_steps + cfg.mask_finetune_steps, stage="post_training")

        # Save checkpoint
        final_step = cfg.mask_only_steps + cfg.mask_finetune_steps
        data = {"step": final_step, "splats": self.splats.state_dict()}
        data["transform"] = self.parser.transform
        data["up_direction"] = self.parser.up_direction
        data["facing_direction"] = self.parser.facing_direction
        data["center_point"] = self.parser.center_point
        torch.save(
            data, f"{self.ckpt_dir}/ckpt_post_training_{final_step}_rank{self.world_rank}.pt"
        )
        print(f"[MaskGaussian post_training] Checkpoint saved. Final GS count: {len(self.splats['means'])}")


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=False)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            if k in ckpts[0]["splats"]:
                runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step, stage="val")
        # runner.render_traj(step=step, save_name="eval")
        if cfg.compression is not None:
            runner.run_compression(step=step)
    elif cfg.use_mask_gaussian and cfg.mask_mode == "post_training":
        runner.train_post_training_mask()
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
        "prune_only": (
            "Gaussian splatting training using prune only strategy.",
            Config(
                strategy=DefaultStrategy(verbose=True, prune_opa=0.005, grow_grad2d=9999, grow_scale3d=9999, grow_scale2d=9999, prune_scale3d=0.1, prune_scale2d=0.15),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
