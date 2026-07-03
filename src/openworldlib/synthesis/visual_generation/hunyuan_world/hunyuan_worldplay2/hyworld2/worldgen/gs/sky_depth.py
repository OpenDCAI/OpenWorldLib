"""Sky depth pre-computation module.

Pre-renders depth maps from a sky point cloud and merges them with GT depth
using a triple-condition mask (no GT depth AND sky-like normal AND sky
rendering coverage).

To switch the rendering backend, pass a different ``render_fn`` to
:func:`precompute_sky_depth_maps`.  The default is
:func:`render_sky_depth_gsplat`.
"""

from __future__ import annotations

import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import imageio
import numpy as np
import torch
import tqdm
from .utils import knn, load_16bit_png_depth
from torch import Tensor

# ---------------------------------------------------------------------------
# Camera specification passed to render functions
# ---------------------------------------------------------------------------
CameraSpec = namedtuple("CameraSpec", ["item_idx", "viewmat", "K", "width", "height"])

# Type alias for a render function:
#   (points, cameras, device, batch_size) -> {item_idx: depth_hw}
RenderFn = Callable[[np.ndarray, List[CameraSpec], torch.device, int], Dict[int, Tensor]]


# ===================================================================
# Render backends  (swap by passing a different function)
# ===================================================================

def render_sky_depth_gsplat(
    points: np.ndarray,
    cameras: List[CameraSpec],
    device: torch.device,
    batch_size: int = 32,
) -> Dict[int, Tensor]:
    """Render sky depth maps via gsplat Gaussian rasterization.

    Builds temporary isotropic Gaussians from *points* (KNN-based scale),
    groups *cameras* by resolution, and batch-rasterises depth-only.

    Returns ``{item_idx: depth_hw}`` where each ``depth_hw`` is a
    ``[H, W]`` CPU float tensor.
    """
    from gsplat.rendering import rasterization

    sky_pts = torch.from_numpy(points).float()
    n_sky = sky_pts.shape[0]

    dist2_avg = (knn(sky_pts, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    sky_scales = torch.exp(torch.log(dist_avg * 1.0).unsqueeze(-1).repeat(1, 3))

    _means = sky_pts.to(device)
    _scales = sky_scales.to(device)
    _quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(n_sky, 1)
    _opacities = torch.ones(n_sky, device=device)
    _dummy_colors = torch.zeros(n_sky, 3, device=device)

    size_groups: Dict[tuple, List[CameraSpec]] = {}
    for cam in cameras:
        key = (cam.width, cam.height)
        size_groups.setdefault(key, []).append(cam)

    result: Dict[int, Tensor] = {}

    for (W_g, H_g), group in size_groups.items():
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            viewmats = torch.stack([c.viewmat for c in batch]).to(device)
            Ks = torch.stack([c.K for c in batch]).to(device)

            with torch.no_grad():
                sky_render, _, _ = rasterization(
                    means=_means,
                    quats=_quats,
                    scales=_scales,
                    opacities=_opacities,
                    colors=_dummy_colors,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=W_g,
                    height=H_g,
                    render_mode="ED",
                    sh_degree=None,
                )

            for i, cam in enumerate(batch):
                result[cam.item_idx] = sky_render[i, :, :, 0].cpu()

    del _means, _scales, _quats, _opacities, _dummy_colors
    torch.cuda.empty_cache()

    return result


# ===================================================================
# Orchestrator  (shared logic, backend-agnostic)
# ===================================================================

def precompute_sky_depth_maps(
    render_fn: RenderFn,
    parser,
    trainset,
    sky_normal_threshold: float,
    result_dir: str,
    device: torch.device,
    world_rank: int = 0,
    world_size: int = 1,
    debug: bool = False,
    batch_size: int = 32,
) -> Dict[int, Optional[Tensor]]:
    """Pre-compute merged sky depth maps for all training views.

    1. Extract sky points from *parser*.
    2. Build :class:`CameraSpec` list and split across ranks.
    3. Call *render_fn* to obtain raw sky depth maps.
    4. Load GT depth + normals, apply triple-condition merge.
    5. Cache partial results per rank, barrier, then load & merge.

    Parameters
    ----------
    render_fn : callable
        ``(points, cameras, device, batch_size) -> {item_idx: depth_hw}``.
        Default: :func:`render_sky_depth_gsplat`.
    """
    sky_mask_pcd = parser.sky_mask
    has_sky = sky_mask_pcd is not None and sky_mask_pcd.any()

    if not has_sky:
        if world_rank == 0:
            print("[Sky Depth] WARNING: sky_depth_from_pcd=True but no sky points found")
        return {}

    sky_pts = parser.points[sky_mask_pcd]

    # --- build camera list for this rank ---
    train_indices = trainset.indices
    all_cameras: List[CameraSpec] = []
    for item_idx in range(len(train_indices)):
        index = train_indices[item_idx]
        camera_id = parser.camera_ids[index]
        c2w = torch.from_numpy(parser.camtoworlds[index]).float()
        K = torch.from_numpy(parser.Ks_dict[camera_id].copy()).float()
        W, H = parser.imsize_dict[camera_id]
        all_cameras.append(CameraSpec(
            item_idx=item_idx,
            viewmat=torch.linalg.inv(c2w),
            K=K,
            width=W,
            height=H,
        ))

    # multi-GPU: each rank takes its own slice
    my_cameras = all_cameras[world_rank::world_size]

    print(f"[Sky Depth] Rank {world_rank}: rendering {len(my_cameras)}/{len(all_cameras)} views "
          f"from {len(sky_pts)} sky points")

    # --- render ---
    raw_depths = render_fn(sky_pts, my_cameras, device, batch_size)

    # --- merge with GT depth ---
    from PIL import Image as PILImage

    debug_dir = None
    if debug and world_rank == 0:
        debug_dir = f"{result_dir}/sky_depth_debug"
        os.makedirs(debug_dir, exist_ok=True)

    def _load_depth_normal(camera_id):
        depth_path = parser.depth_dict.get(camera_id)
        normal_path = parser.normal_dict.get(camera_id)
        gt_depth = None
        if depth_path is not None:
            gt_depth = torch.from_numpy(load_16bit_png_depth(depth_path)).float()
            if parser.rescale is not None:
                gt_depth = gt_depth * parser.rescale
        normals_out = None
        if normal_path is not None:
            normal_img = np.array(PILImage.open(normal_path)) / 255.0
            normal_arr = (normal_img.astype(np.float32) * 2.0 - 1.0).transpose(2, 0, 1)
            normals_out = torch.from_numpy(normal_arr).float()
        return gt_depth, normals_out

    my_results: Dict[int, Optional[Tensor]] = {}
    io_pool = ThreadPoolExecutor(max_workers=8)
    futures = {}
    for cam in my_cameras:
        index = train_indices[cam.item_idx]
        camera_id = parser.camera_ids[index]
        futures[cam.item_idx] = io_pool.submit(_load_depth_normal, camera_id)

    pbar = tqdm.tqdm(my_cameras, desc=f"[Rank {world_rank}] Merging sky depth")
    for cam in pbar:
        sky_depth_hw = raw_depths[cam.item_idx]
        gt_depth, normals = futures[cam.item_idx].result()

        if gt_depth is not None and normals is not None:
            no_gt = gt_depth < 1e-4
            is_sky_normal = normals.norm(dim=0) < sky_normal_threshold
            has_sky_coverage = sky_depth_hw > 0
            fill_mask = no_gt & is_sky_normal & has_sky_coverage

            merged = gt_depth.clone()
            merged[fill_mask] = sky_depth_hw[fill_mask]
            my_results[cam.item_idx] = merged

            if debug_dir is not None:
                index = train_indices[cam.item_idx]
                image_name = parser.image_names[index]
                save_sky_depth_debug(
                    debug_dir, image_name,
                    gt_depth, sky_depth_hw, fill_mask, merged,
                )
        else:
            my_results[cam.item_idx] = gt_depth

    io_pool.shutdown(wait=False)

    # --- cache & DDP sync ---
    if world_size > 1:
        import torch.distributed as dist

        cache_path = f"{result_dir}/sky_depth_cache_rank{world_rank}.pt"
        torch.save(my_results, cache_path)
        dist.barrier()

        # each rank loads all partial caches and merges
        merged_maps: Dict[int, Optional[Tensor]] = {}
        for r in range(world_size):
            rpath = f"{result_dir}/sky_depth_cache_rank{r}.pt"
            partial = torch.load(rpath, map_location="cpu", weights_only=False)
            merged_maps.update(partial)
    else:
        merged_maps = my_results
        cache_path = f"{result_dir}/sky_depth_cache.pt"
        torch.save(merged_maps, cache_path)

    n_filled = sum(1 for v in merged_maps.values() if v is not None)
    print(f"[Sky Depth] Rank {world_rank}: {n_filled}/{len(all_cameras)} views ready")

    return merged_maps


# ===================================================================
# Debug visualization helper
# ===================================================================

def save_sky_depth_debug(debug_dir, image_name, gt_depth, sky_depth, fill_mask, merged_depth):
    """Save a side-by-side debug image: GT | sky | mask | merged."""
    import matplotlib.cm as cm

    valid_depths = merged_depth[merged_depth > 0]
    if valid_depths.numel() == 0:
        return
    vmin, vmax = valid_depths.min().item(), valid_depths.max().item()
    if vmax - vmin < 1e-6:
        return

    def to_colormap(depth_map):
        d = depth_map.numpy().copy()
        mask = d > 0
        d_norm = np.zeros_like(d)
        if mask.any():
            d_norm[mask] = (d[mask] - vmin) / (vmax - vmin + 1e-8)
        colored = cm.turbo(d_norm)[..., :3]
        colored[~mask] = 0
        return (colored * 255).astype(np.uint8)

    gt_img = to_colormap(gt_depth)
    sky_img = to_colormap(sky_depth)
    mask_img = (fill_mask.numpy() * 255).astype(np.uint8)
    mask_img = np.stack([mask_img, mask_img, mask_img], axis=-1)
    merged_img = to_colormap(merged_depth)

    canvas = np.concatenate([gt_img, sky_img, mask_img, merged_img], axis=1)
    imageio.imwrite(f"{debug_dir}/{image_name}_sky_depth_debug.png", canvas)
