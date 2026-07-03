"""
Mesh extraction script for gsplat-based Gaussian Splatting models.

Adapts the TSDF fusion mesh extraction from gs2d_mesh_extraction.py to work with
show_gs.py's checkpoint format (.pt files with 'splats' dict) and gsplat rasterization.

Usage:
    # Bounded mesh extraction (default)
    python extract_mesh.py --ckpt results/xxx/ckpts/ckpt_5000.pt --data_dir data/scene

    # Unbounded mesh extraction
    python extract_mesh.py --ckpt results/xxx/ckpts/ckpt_5000.pt --data_dir data/scene --unbounded

    # Custom resolution and truncation
    python extract_mesh.py --ckpt results/xxx/ckpts/ckpt_5000.pt --data_dir data/scene \
        --mesh_res 1024 --depth_trunc 6.0 --voxel_size 0.01

    # Multi-rank checkpoints (three equivalent ways):
    #   1. Quote the glob pattern to prevent shell expansion:
    python extract_mesh.py --ckpt 'results/xxx/ckpts/ckpt_rank*.pt' --data_dir data/scene
    #   2. Let the shell expand (no quotes needed):
    python extract_mesh.py --ckpt results/xxx/ckpts/ckpt_rank*.pt --data_dir data/scene
    #   3. List files explicitly:
    python extract_mesh.py --ckpt ckpt_rank0.pt ckpt_rank1.pt --data_dir data/scene
"""

import argparse
import math
import os

import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
import torch.nn.functional as F
from glob import glob
from tqdm.auto import tqdm
from skimage import measure
import trimesh

from gsplat.rendering import rasterization
from .opencv import Parser
from concurrent.futures import ThreadPoolExecutor
import threading


# ============ Checkpoint Loading (same as show_gs.py) ============

def load_checkpoint(ckpt_paths, device="cuda"):
    """
    Load Gaussian splats from checkpoint file(s), supporting single and multi-rank.

    Args:
        ckpt_paths: str or list of str. Can be:
            - A single file path: "results/xxx/ckpt.pt"
            - A glob pattern string: "results/xxx/ckpt_rank*.pt"
            - A list of file paths (from shell expansion): ["ckpt_rank0.pt", "ckpt_rank1.pt"]
        device: torch device string

    Returns: means, quats, scales, opacities, colors, sh_degree, metadata
    """
    # Normalize input to a list of file paths
    if isinstance(ckpt_paths, str):
        # Single string: could be a glob pattern or a single file
        if "*" in ckpt_paths or "?" in ckpt_paths:
            ckpt_files = sorted(glob(ckpt_paths))
            if len(ckpt_files) == 0:
                raise FileNotFoundError(f"No checkpoint files matched: {ckpt_paths}")
        else:
            ckpt_files = [ckpt_paths]
    elif isinstance(ckpt_paths, (list, tuple)):
        # Already a list (e.g. from shell glob expansion via nargs="+")
        ckpt_files = sorted(ckpt_paths)
    else:
        raise TypeError(f"ckpt_paths must be str or list, got {type(ckpt_paths)}")

    if len(ckpt_files) > 1:
        print(f"Loading {len(ckpt_files)} checkpoint files...")
        means_all, quats_all, scales_all, opacities_all = [], [], [], []
        sh0_all, shN_all = [], []
        metadata = {}

        for ckpt_file in tqdm(ckpt_files, desc="Loading ckpts"):
            ckpt_all = torch.load(ckpt_file, map_location=device, weights_only=False)
            for key in ["up_direction", "facing_direction", "center_point"]:
                if key in ckpt_all:
                    metadata[key] = ckpt_all[key]
            ckpt = ckpt_all["splats"]
            means_all.append(ckpt["means"])
            quats_all.append(F.normalize(ckpt["quats"], p=2, dim=-1))
            scales_all.append(torch.exp(ckpt["scales"]))
            opacities_all.append(torch.sigmoid(ckpt["opacities"]))
            sh0_all.append(ckpt["sh0"])
            shN_all.append(ckpt["shN"])

        means = torch.cat(means_all, dim=0)
        quats = torch.cat(quats_all, dim=0)
        scales = torch.cat(scales_all, dim=0)
        opacities = torch.cat(opacities_all, dim=0)
        sh0 = torch.cat(sh0_all, dim=0)
        shN = torch.cat(shN_all, dim=0)
    else:
        ckpt_file = ckpt_files[0]
        print(f"Loading checkpoint: {ckpt_file}")

        if ckpt_file.endswith(".pt"):
            ckpt_all = torch.load(ckpt_file, map_location=device, weights_only=False)
            metadata = {}
            for key in ["up_direction", "facing_direction", "center_point"]:
                if key in ckpt_all:
                    metadata[key] = ckpt_all[key]
            ckpt = ckpt_all["splats"]
            means = ckpt["means"]
            quats = F.normalize(ckpt["quats"], p=2, dim=-1)
            scales = torch.exp(ckpt["scales"])
            opacities = torch.sigmoid(ckpt["opacities"])
            sh0 = ckpt["sh0"]
            shN = ckpt["shN"]
        elif ckpt_file.endswith(".ply"):
            from plyfile import PlyData
            import json
            meta_info = json.load(open("/".join(ckpt_file.split("/")[:-1]) + "/position_meta_info.json"))
            up_direction = np.array(meta_info["up_direction"])
            facing_direction = np.array(meta_info["facing_direction"])
            center_point = np.array(meta_info["center_point"])
            metadata = {
                "up_direction": up_direction,
                "facing_direction": facing_direction,
                "center_point": center_point
            }

            print(f"[load_ply] Reading {ckpt_file} ...")
            plydata = PlyData.read(ckpt_file)
            vertex = plydata['vertex']
            n_points = len(vertex.data)
            print(f"[load_ply] Number of points: {n_points}")

            # 1. 位置 means (N, 3)
            means = torch.tensor(np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1), dtype=torch.float32, device=device)

            # 2. 四元数 quats (N, 4) — PLY 中存储顺序为 rot_0~rot_3 (wxyz)
            quats = torch.tensor(np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=-1), dtype=torch.float32, device=device)
            quats = F.normalize(quats, p=2, dim=-1)

            # 3. 尺度 scales (N, 3) — PLY 中为 log space
            scales = torch.tensor(np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1), dtype=torch.float32, device=device)
            scales = torch.exp(scales)

            # 4. 不透明度 opacities (N,) — PLY 中为 logit space (sigmoid 前)
            opacities = torch.tensor(np.array(vertex['opacity']), dtype=torch.float32, device=device)
            opacities = torch.sigmoid(opacities)

            # 5. SH 系数: DC (sh0) + rest (shN)
            # sh0: f_dc_0, f_dc_1, f_dc_2 → (N, 1, 3)
            sh_dc = torch.tensor(np.stack([
                vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']
            ], axis=-1), dtype=torch.float32, device=device)  # (N, 3)
            sh0 = sh_dc.unsqueeze(1)  # (N, 1, 3)

            # shN: f_rest_* → (N, C, 3)
            rest_names = sorted(
                [p.name for p in vertex.properties if p.name.startswith('f_rest_')],
                key=lambda x: int(x.split('_')[-1])
            )

            if len(rest_names) > 0:
                sh_rest_flat = torch.tensor(np.stack(
                    [vertex[name] for name in rest_names], axis=-1
                ), dtype=torch.float32, device=device)  # (N, num_rest)

                # 标准 3DGS: f_rest 按 (C * 3) 展开，需要 reshape 为 (N, C, 3)
                num_rest_coeffs = len(rest_names)
                assert num_rest_coeffs % 3 == 0, \
                    f"f_rest 数量 {num_rest_coeffs} 不是3的倍数，PLY格式异常"
                num_sh_rest = num_rest_coeffs // 3
                shN = sh_rest_flat.reshape(n_points, num_sh_rest, 3)  # (N, C, 3)
            else:
                shN = torch.zeros(n_points, 0, 3, dtype=torch.float32, device=device)
        else:
            raise NotImplementedError

    colors = torch.cat([sh0, shN], dim=-2)  # [N, K, 3]
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    print(f"Loaded {len(means)} Gaussians, SH degree: {sh_degree}")
    return means, quats, scales, opacities, colors, sh_degree, metadata


# ============ Rendering with gsplat ============

@torch.no_grad()
def render_views_gsplat(
    means, quats, scales, opacities, colors, sh_degree,
    camtoworlds, Ks, widths, heights, device="cuda",
):
    """
    Render RGB and depth maps for all views using gsplat rasterization.

    Args:
        camtoworlds: list of (4,4) numpy arrays or tensor, camera-to-world transforms
        Ks: list of (3,3) numpy arrays or tensor, intrinsic matrices
        widths: list of int
        heights: list of int

    Returns:
        rgbmaps: list of (3, H, W) tensors (on cpu)
        depthmaps: list of (1, H, W) tensors (on cpu)
    """
    rgbmaps = []
    depthmaps = []

    # Use diffuse only (SH degree 0) for mesh texturing
    diffuse_colors = colors[:, :1, :]  # only the DC term [N, 1, 3]
    diffuse_sh_degree = 0

    for i in tqdm(range(len(camtoworlds)), desc="Rendering RGB and depth maps"):
        c2w = camtoworlds[i]
        K = Ks[i]
        W = widths[i]
        H = heights[i]

        if isinstance(c2w, np.ndarray):
            c2w = torch.from_numpy(c2w).float().to(device)
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K).float().to(device)

        viewmat = torch.linalg.inv(c2w)

        # Render RGB
        renders, _, _ = rasterization(
            means, quats, scales, opacities,
            diffuse_colors,
            viewmat[None], K[None],
            W, H,
            sh_degree=diffuse_sh_degree,
            render_mode="RGB+ED",
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=3,
        )
        rgb = renders[0, ..., :3].permute(2, 0, 1).clamp(0, 1)  # (3, H, W)
        depth = renders[0, ..., 3:4].permute(2, 0, 1)  # (1, H, W)

        rgbmaps.append(rgb.cpu())
        depthmaps.append(depth.cpu())

    return rgbmaps, depthmaps


# ============ Bounding Sphere Estimation ============

def estimate_bounding_sphere_from_cameras(camtoworlds_np):
    """
    Estimate the bounding sphere from camera pose distribution.
    Center is the focus point of all cameras, radius is the min distance from center to any camera.
    Returns: center (torch.Tensor on cuda), radius (float)
    """
    c2ws = camtoworlds_np
    poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])

    # Focus point calculation
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]

    center = focus_pt
    radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
    center = torch.from_numpy(center).float().cuda()
    print(f"[camera] Estimated bounding sphere: radius={radius:.2f}, center={center}")
    print(f"[camera] Use at least {2.0 * radius:.2f} for depth_trunc")

    return center, radius


def estimate_bounding_sphere_from_gaussians(means, percentile=99.0, scale_factor=1.1):
    """
    Estimate the bounding sphere from the Gaussian point cloud (xyz) distribution.
    Uses a percentile-based approach to be robust against outlier Gaussians.

    Args:
        means: (N, 3) tensor of Gaussian centers
        percentile: percentile of distances to keep (filters outlier splats), default 99
        scale_factor: multiply the radius by this factor for margin, default 1.1

    Returns: center (torch.Tensor on cuda), radius (float)
    """
    xyz = means.detach().cpu().numpy()  # (N, 3)

    # Robust center: use median to avoid outlier influence
    center = np.median(xyz, axis=0)

    # Compute distances from center
    dists = np.linalg.norm(xyz - center, axis=-1)

    # Use percentile to filter outliers
    radius = float(np.percentile(dists, percentile)) * scale_factor

    center_t = torch.from_numpy(center).float().cuda()
    print(f"[gaussians] Estimated bounding sphere: radius={radius:.2f}, center={center_t}")
    print(f"[gaussians] Based on {len(xyz)} Gaussians, {percentile}th percentile, scale={scale_factor}")
    print(f"[gaussians] Use at least {2.0 * radius:.2f} for depth_trunc")

    return center_t, radius


def estimate_bounding_sphere(means, camtoworlds_np, method="camera", gs_percentile=99.0, gs_scale=1.1):
    """
    Unified bounding sphere estimation.

    Args:
        means: (N, 3) tensor of Gaussian centers
        camtoworlds_np: (M, 4, 4) numpy array of camera-to-world transforms
        method: "camera" (from camera distribution), "gaussians" (from GS xyz),
                or "both" (use intersection — tighter bound)
        gs_percentile: percentile for gaussian-based estimation
        gs_scale: scale factor for gaussian-based estimation

    Returns: center (torch.Tensor on cuda), radius (float)
    """
    if method == "camera":
        return estimate_bounding_sphere_from_cameras(camtoworlds_np)
    elif method == "gaussians":
        return estimate_bounding_sphere_from_gaussians(means, percentile=gs_percentile, scale_factor=gs_scale)
    elif method == "both":
        center_cam, radius_cam = estimate_bounding_sphere_from_cameras(camtoworlds_np)
        center_gs, radius_gs = estimate_bounding_sphere_from_gaussians(means, percentile=gs_percentile, scale_factor=gs_scale)
        # Use the tighter (smaller) radius with its corresponding center
        if radius_gs < radius_cam:
            print(f"[both] Using Gaussian-based bound (tighter): radius={radius_gs:.2f}")
            return center_gs, radius_gs
        else:
            print(f"[both] Using camera-based bound (tighter): radius={radius_cam:.2f}")
            return center_cam, radius_cam
    else:
        raise ValueError(f"Unknown bound method: {method}. Choose from: camera, gaussians, both")


# ============ Camera to Open3D format ============

def to_cam_open3d(camtoworlds_np, Ks, widths, heights):
    """
    Convert camera params to Open3D PinholeCameraParameters.
    """
    camera_traj = []
    for i in range(len(camtoworlds_np)):
        K = Ks[i]
        W = widths[i]
        H = heights[i]
        c2w = camtoworlds_np[i]
        w2c = np.linalg.inv(c2w)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=W, height=H,
            fx=K[0, 0], fy=K[1, 1],
            cx=K[0, 2], cy=K[1, 2],
        )
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = w2c
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


# ============ Bounded TSDF Fusion ============

@torch.no_grad()
def extract_mesh_bounded(
    rgbmaps, depthmaps, camtoworlds_np, Ks, widths, heights,
    voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3.0,
    num_workers=16,
):
    """预处理并行，integrate 顺序执行"""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    cam_o3ds = to_cam_open3d(camtoworlds_np, Ks, widths, heights)
    n_frames = len(cam_o3ds)

    # ---- 并行预处理：RGB/Depth → Open3D RGBDImage ----
    def prepare_rgbd(i):
        rgb = rgbmaps[i]
        depth = depthmaps[i]
        rgb_np = np.asarray(
            np.clip(rgb.permute(1, 2, 0).numpy(), 0.0, 1.0) * 255,
            order="C", dtype=np.uint8,
        )
        depth_np = np.asarray(depth.permute(1, 2, 0).numpy(), order="C")
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_np),
            o3d.geometry.Image(depth_np),
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        return rgbd

    # 并行预处理所有帧
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        rgbd_list = list(tqdm(
            pool.map(prepare_rgbd, range(n_frames)),
            total=n_frames, desc="Preparing RGBD"
        ))

    # ---- 顺序集成 ----
    for i in tqdm(range(n_frames), desc="TSDF integration"):
        volume.integrate(rgbd_list[i], cam_o3ds[i].intrinsic, cam_o3ds[i].extrinsic)

    return volume.extract_triangle_mesh()

@torch.no_grad()
def extract_mesh_bounded_gpu(
    rgbmaps, depthmaps, camtoworlds_np, Ks, widths, heights,
    voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3.0,
    device_str="CUDA:0",
):
    import time

    device = o3c.Device(device_str)
    cpu = o3c.Device("CPU:0")  # ★ 相机参数必须在 CPU

    # ---- 预热 CUDA JIT（把慢的部分提前，避免误导计时） ----
    t0 = time.time()
    _warmup = o3c.Tensor(np.zeros(1, dtype=np.float32), device=device)
    del _warmup
    print(f"  CUDA warmup: {time.time() - t0:.2f}s")

    # ---- 估算 block_count ----
    positions = camtoworlds_np[:, :3, 3]
    scene_min = positions.min(axis=0) - depth_trunc
    scene_max = positions.max(axis=0) + depth_trunc
    scene_extent = scene_max - scene_min
    block_size = voxel_size * 16

    estimated_blocks = int(np.prod(np.ceil(scene_extent / block_size)) * 0.1)
    estimated_blocks = max(1000, min(estimated_blocks, 50000))
    print(f"  Scene extent: {scene_extent}")
    print(f"  Estimated blocks: {estimated_blocks}")

    trunc_voxel_multiplier = sdf_trunc / voxel_size

    t0 = time.time()
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=estimated_blocks,
        device=device,
    )
    print(f"  VBG init: {time.time() - t0:.2f}s")

    cam_o3ds = to_cam_open3d(camtoworlds_np, Ks, widths, heights)

    for i in tqdm(range(len(cam_o3ds)), desc="GPU TSDF integration"):
        rgb = rgbmaps[i]
        depth = depthmaps[i]

        # ★ color: float32, 范围 [0, 1]
        rgb_np = np.asarray(
            np.clip(rgb.permute(1, 2, 0).numpy(), 0.0, 1.0),
            order="C", dtype=np.float32,
        )
        # depth: float32
        depth_np = np.asarray(
            depth.permute(1, 2, 0).numpy(),
            order="C", dtype=np.float32,
        )

        # 图像 → GPU
        rgb_t = o3d.t.geometry.Image(o3c.Tensor(rgb_np, device=device))
        depth_t = o3d.t.geometry.Image(o3c.Tensor(depth_np, device=device))

        # 相机参数 → CPU
        intrinsic_t = o3c.Tensor(Ks[i].astype(np.float64), device=cpu)
        extrinsic_t = o3c.Tensor(
            np.linalg.inv(camtoworlds_np[i]).astype(np.float64), device=cpu
        )

        frustum_hash = vbg.compute_unique_block_coordinates(
            depth_t, intrinsic_t, extrinsic_t,
            depth_scale=1.0, depth_max=depth_trunc,
        )

        vbg.integrate(
            frustum_hash, depth_t, rgb_t,
            intrinsic_t, extrinsic_t,
            depth_scale=1.0,
            depth_max=depth_trunc,
            trunc_voxel_multiplier=trunc_voxel_multiplier,
        )

    mesh = vbg.extract_triangle_mesh()
    return mesh.to_legacy()

# ============ Unbounded TSDF Fusion ============

@torch.no_grad()
def extract_mesh_unbounded(
    rgbmaps, depthmaps, camtoworlds_np, Ks, widths, heights,
    means, center, radius, resolution=1024,
):
    """
    Mesh extraction for unbounded scenes using space contraction.

    Returns: o3d.geometry.TriangleMesh
    """

    def contract(x):
        mag = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
        return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

    def uncontract(y):
        mag = torch.linalg.norm(y, ord=2, dim=-1, keepdim=True)
        return torch.where(mag < 1, y, (1 / (2 - mag)) * (y / mag))

    def normalize(x):
        return (x - center) / radius

    def unnormalize(x):
        return (x * radius) + center

    def inv_contraction(x):
        return unnormalize(uncontract(x))

    def compute_sdf_perframe(i, points, depthmap, rgbmap, c2w, K, W, H):
        """Compute per-frame SDF."""
        w2c = torch.linalg.inv(c2w)
        # Project points to camera
        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        cam_points = (w2c @ points_h.T).T  # (N, 4)
        z = cam_points[:, 2:3]

        # Project to pixel coordinates
        proj = (K @ cam_points[:, :3].T).T  # (N, 3)
        pix_x = proj[:, 0:1] / proj[:, 2:3]
        pix_y = proj[:, 1:2] / proj[:, 2:3]

        # Normalize to [-1, 1] for grid_sample
        pix_x_norm = 2.0 * pix_x / W - 1.0
        pix_y_norm = 2.0 * pix_y / H - 1.0
        pix_coords = torch.cat([pix_x_norm, pix_y_norm], dim=-1)  # (N, 2)

        mask_proj = (
            (pix_coords > -1.0) & (pix_coords < 1.0) & (z > 0)
        ).all(dim=-1)

        # Sample depth and rgb
        sampled_depth = F.grid_sample(
            depthmap.cuda()[None],
            pix_coords[None, None],
            mode='bilinear', padding_mode='border', align_corners=True,
        ).reshape(-1, 1)

        sampled_rgb = F.grid_sample(
            rgbmap.cuda()[None],
            pix_coords[None, None],
            mode='bilinear', padding_mode='border', align_corners=True,
        ).reshape(3, -1).T

        sdf = sampled_depth - z
        return sdf, sampled_rgb, mask_proj

    def compute_unbounded_tsdf(samples, use_inv_contraction, voxel_size, return_rgb=False):
        """Fuse all frames with adaptive SDF truncation in contracted space."""
        if use_inv_contraction:
            mask = torch.linalg.norm(samples, dim=-1) > 1
            sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
            sdf_trunc[mask] *= 1 / (2 - torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
            world_samples = inv_contraction(samples)
        else:
            sdf_trunc = 5 * voxel_size
            world_samples = samples

        tsdfs = torch.ones_like(samples[:, 0])
        rgbs = torch.zeros((samples.shape[0], 3)).cuda()
        weights = torch.ones_like(samples[:, 0])

        for i in tqdm(range(len(camtoworlds_np)), desc="TSDF integration"):
            c2w = torch.from_numpy(camtoworlds_np[i]).float().cuda()
            K_i = torch.from_numpy(Ks[i]).float().cuda()
            W_i, H_i = widths[i], heights[i]

            sdf, rgb, mask_proj = compute_sdf_perframe(
                i, world_samples,
                depthmap=depthmaps[i],
                rgbmap=rgbmaps[i],
                c2w=c2w, K=K_i, W=W_i, H=H_i,
            )

            sdf = sdf.flatten()
            mask_proj = mask_proj & (sdf > -sdf_trunc)
            sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
            w = weights[mask_proj]
            wp = w + 1
            tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
            rgbs[mask_proj] = (rgbs[mask_proj] * w[:, None] + rgb[mask_proj]) / wp[:, None]
            weights[mask_proj] = wp

        if return_rgb:
            return tsdfs, rgbs
        return tsdfs

    # Main logic
    N = resolution
    voxel_size = radius * 2 / N
    print(f"Computing SDF grid resolution {N} x {N} x {N}")
    print(f"Voxel size: {voxel_size}")

    def sdf_function(x):
        return compute_unbounded_tsdf(x, use_inv_contraction=True, voxel_size=voxel_size)

    R = contract(normalize(means)).norm(dim=-1).cpu().numpy()
    R = np.quantile(R, q=0.95)
    R = min(R + 0.01, 1.9)

    mesh = marching_cubes_with_contraction(
        sdf=sdf_function,
        bounding_box_min=(-R, -R, -R),
        bounding_box_max=(R, R, R),
        level=0,
        resolution=N,
        inv_contraction=inv_contraction,
    )

    # Texturing the mesh
    torch.cuda.empty_cache()
    mesh = mesh.as_open3d
    print("Texturing mesh ...")
    _, rgbs = compute_unbounded_tsdf(
        torch.tensor(np.asarray(mesh.vertices)).float().cuda(),
        use_inv_contraction=False, voxel_size=voxel_size, return_rgb=True,
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
    return mesh


def marching_cubes_with_contraction(
    sdf,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    level=0,
    inv_contraction=None,
    max_range=32.0,
):
    """Run marching cubes with optional inverse contraction."""
    assert resolution % 512 == 0

    resN = resolution
    cropN = 512
    N = resN // cropN

    xs = np.linspace(bounding_box_min[0], bounding_box_max[0], N + 1)
    ys = np.linspace(bounding_box_min[1], bounding_box_max[1], N + 1)
    zs = np.linspace(bounding_box_min[2], bounding_box_max[2], N + 1)

    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(f"Marching cubes block ({i}, {j}, {k})")
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = torch.linspace(x_min, x_max, cropN).cuda()
                y = torch.linspace(y_min, y_max, cropN).cuda()
                z = torch.linspace(z_min, z_max, cropN).cuda()

                xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
                points = torch.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T.float().cuda()

                # Evaluate SDF in chunks
                z_vals = []
                for pnts in torch.split(points, 256**3, dim=0):
                    z_vals.append(sdf(pnts))
                z_np = torch.cat(z_vals, dim=0).detach().cpu().numpy()

                if not (np.min(z_np) > level or np.max(z_np) < level):
                    z_np = z_np.astype(np.float32)
                    verts, faces, normals, _ = measure.marching_cubes(
                        volume=z_np.reshape(cropN, cropN, cropN),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                    )
                    verts = verts + np.array([x_min, y_min, z_min])
                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    meshes.append(meshcrop)

                print("Finished block")

    combined = trimesh.util.concatenate(meshes)
    combined.merge_vertices(digits_vertex=6)

    # Inverse contraction and clip
    if inv_contraction is not None:
        combined.vertices = inv_contraction(
            torch.from_numpy(combined.vertices).float().cuda()
        ).cpu().numpy()
        combined.vertices = np.clip(combined.vertices, -max_range, max_range)

    return combined


# ============ Post-processing ============

def post_process_mesh(mesh, cluster_to_keep=50):
    """
    Post-process mesh: remove small disconnected components.
    """
    import copy
    print(f"Post-processing: keeping top {cluster_to_keep} clusters")
    mesh_0 = copy.deepcopy(mesh)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_0.cluster_connected_triangles()
        )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()

    print(f"Vertices: {len(mesh.vertices)} -> {len(mesh_0.vertices)}")
    return mesh_0