import json
import os
from glob import glob
from typing import Any, Dict, List, Optional

import cv2
import imagesize
import numpy as np
import torch
import torch.distributed as dist
import trimesh
from PIL import Image
from tqdm import tqdm
from typing_extensions import assert_never
import imageio.v2 as imageio

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

from .utils import load_16bit_png_depth

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def _broadcast_pcd_arrays(
    points: Optional[np.ndarray],
    points_rgb: Optional[np.ndarray],
    points_is_outlier: Optional[np.ndarray],
    points_is_align: Optional[np.ndarray],
    world_rank: int,
    local_rank: int,
    world_size: int,
    sky_mask: Optional[np.ndarray] = None,
) -> tuple:
    """Broadcast point-cloud arrays from rank 0 to all other ranks.

    On rank 0 the arrays must be valid numpy arrays.
    On other ranks they may be *None*; they will be allocated after the
    shape broadcast and filled in.

    All broadcast ops use **CUDA tensors** (nccl requirement).

    Returns:
        (points, points_rgb, points_is_outlier, points_is_align, sky_mask) as numpy arrays.
    """
    device = torch.device(f"cuda:{local_rank}")

    if world_rank == 0:
        N = points.shape[0]
        n_tensor = torch.tensor([N], dtype=torch.int64, device=device)
    else:
        n_tensor = torch.zeros(1, dtype=torch.int64, device=device)
    dist.broadcast(n_tensor, src=0)
    N = int(n_tensor.item())

    def _bcast(arr, shape, dtype, name=""):
        if world_rank == 0:
            t = torch.from_numpy(arr.astype(np.float32 if dtype != bool else np.float32)).to(device)
        else:
            t = torch.zeros(shape, dtype=torch.float32, device=device)
        dist.broadcast(t, src=0)
        out = t.cpu().numpy()
        if dtype == bool:
            out = out > 0.5
        return out

    points     = _bcast(points,           (N, 3), np.float32, name="points")
    points_rgb = _bcast(points_rgb,       (N, 3), np.float32, name="points_rgb")
    points_is_outlier = _bcast(points_is_outlier, (N,), bool, name="points_is_outlier")
    points_is_align = _bcast(points_is_align, (N,), bool, name="points_is_align")
    sky_mask = _bcast(sky_mask, (N,), bool, name="sky_mask")

    return points, points_rgb, points_is_outlier, points_is_align, sky_mask


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


def _voxel_subsample(points: np.ndarray, voxel_size: float, K: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Voxel grid downsampling: keep at most K random points per voxel.

    Args:
        points: (N, 3) positions.
        voxel_size: side length of each voxel cube.
        K: max points to keep per voxel.
        rng: numpy random generator.

    Returns:
        selected_indices: (M,) indices into `points`.
    """
    bbox_min = points.min(axis=0)
    voxel_coords = ((points - bbox_min) / voxel_size).astype(np.int64)  # (N, 3)

    cx, cy, cz = voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]
    cx = cx - cx.min()
    cy = cy - cy.min()
    cz = cz - cz.min()
    max_y = cy.max() + 1
    max_z = cz.max() + 1
    voxel_hash = cx * (max_y * max_z) + cy * max_z + cz  # (N,)

    if K == 1:
        # Vectorized fast path: shuffle then take first occurrence per voxel
        shuffle_idx = rng.permutation(len(voxel_hash))
        shuffled_hash = voxel_hash[shuffle_idx]
        _, first_occurrence = np.unique(shuffled_hash, return_index=True)
        return shuffle_idx[first_occurrence]

    # General path for K > 1: sort-based grouping with Python loop
    sort_order = np.argsort(voxel_hash)
    sorted_hash = voxel_hash[sort_order]

    boundaries = np.where(np.diff(sorted_hash) != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [len(sorted_hash)]])

    selected = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        voxel_indices = sort_order[start:end]
        if len(voxel_indices) <= K:
            selected.append(voxel_indices)
        else:
            chosen = rng.choice(voxel_indices, size=K, replace=False)
            selected.append(chosen)

    return np.concatenate(selected)


def geometry_aware_downsample(
    points: np.ndarray,
    target_num: int,
    K_per_voxel: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Uniform voxel-based downsampling.

    Strategy:
      1. Binary-search the voxel size so that keeping K_per_voxel points per
         voxel yields exactly target_num total.
      2. Within each voxel, randomly pick K_per_voxel points (uniform, no bias).

    Args:
        points: (N, 3) positions.
        target_num: exact number of points to return.
        K_per_voxel: fixed number of points to keep per voxel (default 1).
        seed: random seed.

    Returns:
        indices: (target_num,) int64 array of selected point indices.
    """
    rng = np.random.default_rng(seed=seed)
    N = points.shape[0]
    if N <= target_num:
        return np.arange(N, dtype=np.int64)

    all_indices = np.arange(N, dtype=np.int64)
    remaining_budget = target_num
    inlier_points = points  # (N, 3)
    inlier_indices = all_indices
    outlier_indices = np.array([], dtype=np.int64)

    # --- Step 2: binary search for voxel size ---
    # Goal: find voxel_size such that _voxel_subsample returns ~remaining_budget points
    bbox_min = inlier_points.min(axis=0)
    bbox_max = inlier_points.max(axis=0)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)

    # Search bounds: very small voxel → keep almost all points; very large → keep very few
    lo, hi = bbox_diag * 1e-6, bbox_diag * 0.5
    best_voxel_size = (lo + hi) / 2.0

    # Pre-compute shifted coords to avoid repeated allocation in the loop
    _shifted_pts = inlier_points - bbox_min

    tol = remaining_budget * 0.15  # 5% tolerance; trim/pad handles the gap
    for iteration in range(10):
        mid = (lo + hi) / 2.0
        count = _voxel_count(inlier_points, mid, K_per_voxel, bbox_min, _shifted_pts=_shifted_pts)
        if abs(count - remaining_budget) <= tol:
            best_voxel_size = mid
            break
        elif count > remaining_budget:
            lo = mid
        else:
            hi = mid
        best_voxel_size = mid

    # Do the actual selection with the best voxel size
    selected_inlier_local = _voxel_subsample(inlier_points, best_voxel_size, K_per_voxel, rng)
    num_selected = len(selected_inlier_local)

    # --- Step 3: adjust to exact target_num ---
    selected_inlier_global = inlier_indices[selected_inlier_local]

    if num_selected > remaining_budget:
        # Too many: randomly drop excess
        keep = rng.choice(num_selected, size=remaining_budget, replace=False)
        selected_inlier_global = selected_inlier_global[keep]
    elif num_selected < remaining_budget:
        # Too few: add random unselected inlier points
        selected_set = set(selected_inlier_global.tolist())
        unselected_inlier = np.array(
            [idx for idx in inlier_indices if idx not in selected_set], dtype=np.int64
        )
        need = remaining_budget - num_selected
        if len(unselected_inlier) >= need:
            extra = rng.choice(unselected_inlier, size=need, replace=False)
        else:
            extra = unselected_inlier
        selected_inlier_global = np.concatenate([selected_inlier_global, extra])

    # Combine
    all_selected = np.concatenate([outlier_indices, selected_inlier_global])

    # Final trim/pad (should be very rare after binary search)
    if len(all_selected) > target_num:
        all_selected = rng.choice(all_selected, size=target_num, replace=False)
    elif len(all_selected) < target_num:
        remaining_pool = np.array(
            [i for i in range(N) if i not in set(all_selected.tolist())], dtype=np.int64
        )
        need = target_num - len(all_selected)
        extra = rng.choice(remaining_pool, size=min(need, len(remaining_pool)), replace=False)
        all_selected = np.concatenate([all_selected, extra])

    assert len(all_selected) == target_num, \
        f"Expected {target_num} points, got {len(all_selected)}"

    return all_selected.astype(np.int64)


def geometry_aware_downsample_o3d(
    points: np.ndarray,
    target_num: int,
    seed: int = 42,
    tol_ratio: float = 0.15,
) -> np.ndarray:
    """Voxel-based downsampling using Open3D's C++ backend, returning original indices.

    Uses Open3D for fast binary-search counting and voxel downsampling,
    then maps centroids back to nearest original points via KD-tree.

    Args:
        points: (N, 3) positions.
        target_num: desired number of output points.
        seed: random seed for trim/pad.
        tol_ratio: tolerance ratio for early stopping in binary search.

    Returns:
        indices: (target_num,) int64 array of selected point indices.
    """
    assert HAS_OPEN3D, (
        "open3d is required for downsample_mode='geometry_aware_o3d'. "
        "Install with: pip install open3d"
    )
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed=seed)
    N = points.shape[0]
    if N <= target_num:
        return np.arange(N, dtype=np.int64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    bbox_diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    lo, hi = bbox_diag * 1e-6, bbox_diag * 0.5

    tol = target_num * tol_ratio
    best_voxel_size = (lo + hi) / 2.0
    for iteration in range(10):
        mid = (lo + hi) / 2.0
        count = len(pcd.voxel_down_sample(voxel_size=mid).points)
        if abs(count - target_num) <= tol:
            best_voxel_size = mid
            break
        elif count > target_num:
            lo = mid
        else:
            hi = mid
        best_voxel_size = mid

    pcd_down = pcd.voxel_down_sample(voxel_size=best_voxel_size)
    down_pts = np.asarray(pcd_down.points)
    num_down = len(down_pts)

    tree = cKDTree(points)
    _, nn_indices = tree.query(down_pts, k=1)
    selected = np.unique(nn_indices)

    if len(selected) > target_num:
        selected = rng.choice(selected, size=target_num, replace=False)
    elif len(selected) < target_num:
        unselected = np.setdiff1d(np.arange(N), selected)
        need = target_num - len(selected)
        extra = rng.choice(unselected, size=min(need, len(unselected)), replace=False)
        selected = np.concatenate([selected, extra])

    return selected.astype(np.int64)


def _voxel_count(points: np.ndarray, voxel_size: float, K: int,
                 bbox_min: np.ndarray, _shifted_pts: np.ndarray = None) -> int:
    """Fast count of how many points would be kept by voxel subsampling with K per voxel.
    (No actual selection, just counting — used for binary search.)

    If _shifted_pts is provided, it should be (points - bbox_min) pre-computed
    to avoid repeated allocation in the binary-search loop.
    """
    if _shifted_pts is None:
        _shifted_pts = points - bbox_min
    voxel_coords = (_shifted_pts / voxel_size).astype(np.int64)
    cx, cy, cz = voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]
    cx = cx - cx.min()
    cy = cy - cy.min()
    cz = cz - cz.min()
    max_y = int(cy.max()) + 1
    max_z = int(cz.max()) + 1
    voxel_hash = cx * (max_y * max_z) + cy * max_z + cz

    if K == 1:
        # Fast path: just count unique voxels
        return int(np.unique(voxel_hash).shape[0])

    unique_hashes, counts = np.unique(voxel_hash, return_counts=True)
    # Each voxel keeps min(count, K) points
    return int(np.minimum(counts, K).sum())


def _open3d_voxel_downsample(
    base_points: np.ndarray,
    base_rgb: np.ndarray,
    sky_points: Optional[np.ndarray],
    sky_rgb: Optional[np.ndarray],
    target_num: int,
    seed: int = 42,
    align_points: Optional[np.ndarray] = None,
    align_rgb: Optional[np.ndarray] = None,
) -> tuple:
    """Voxel downsampling using Open3D's C++ backend (much faster than NumPy).

    Base, sky, and align points are downsampled separately to preserve their
    identity.  Binary search finds a voxel_size that yields approximately
    target_num total points; final random trim/pad adjusts to the exact count.

    Args:
        base_points: (N_base, 3) non-sky, non-align positions.
        base_rgb: (N_base, 3) non-sky, non-align colors (uint8 or float).
        sky_points: (N_sky, 3) sky positions, or None.
        sky_rgb: (N_sky, 3) sky colors, or None.
        target_num: desired total number of output points.
        seed: random seed for trim/pad.
        align_points: (N_align, 3) align positions, or None.
        align_rgb: (N_align, 3) align colors, or None.

    Returns:
        (points, points_rgb, sky_mask, align_mask) as numpy arrays.
    """
    assert HAS_OPEN3D, (
        "open3d is required for downsample_mode='open3d_voxel'. "
        "Install with: pip install open3d"
    )
    rng = np.random.default_rng(seed=seed)

    def _to_o3d_pcd(pts, rgb):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        rgb_f = rgb.astype(np.float64)
        if rgb_f.max() > 1.0:
            rgb_f = rgb_f / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb_f)
        return pcd

    total_orig = base_points.shape[0]
    n_sky = sky_points.shape[0] if sky_points is not None and len(sky_points) > 0 else 0
    n_align = align_points.shape[0] if align_points is not None and len(align_points) > 0 else 0
    total_orig += n_sky + n_align

    base_pcd = _to_o3d_pcd(base_points, base_rgb)

    # Downsample sky points once with a reasonable voxel size
    sky_down_pts = None
    sky_down_rgb = None
    if n_sky > 0:
        sky_pcd = _to_o3d_pcd(sky_points, sky_rgb)
        bbox_sky = np.linalg.norm(sky_points.max(axis=0) - sky_points.min(axis=0))
        sky_ratio = n_sky / total_orig
        sky_target = max(int(target_num * sky_ratio), 1)
        lo_s, hi_s = bbox_sky * 1e-6, bbox_sky * 0.5
        for _ in range(10):
            mid_s = (lo_s + hi_s) / 2.0
            cnt = len(sky_pcd.voxel_down_sample(voxel_size=mid_s).points)
            if cnt > sky_target:
                lo_s = mid_s
            else:
                hi_s = mid_s
        sky_pcd_down = sky_pcd.voxel_down_sample(voxel_size=(lo_s + hi_s) / 2.0)
        sky_down_pts = np.asarray(sky_pcd_down.points).astype(np.float32)
        sky_down_rgb = (np.asarray(sky_pcd_down.colors) * 255).astype(np.uint8)

    # Downsample align points with proportional budget
    align_down_pts = None
    align_down_rgb = None
    if n_align > 0:
        align_pcd = _to_o3d_pcd(align_points, align_rgb)
        bbox_align = np.linalg.norm(align_points.max(axis=0) - align_points.min(axis=0))
        align_ratio = n_align / total_orig
        align_target = max(int(target_num * align_ratio), 1)
        lo_a, hi_a = bbox_align * 1e-6, bbox_align * 0.5
        for _ in range(25):
            mid_a = (lo_a + hi_a) / 2.0
            cnt = len(align_pcd.voxel_down_sample(voxel_size=mid_a).points)
            if cnt > align_target:
                lo_a = mid_a
            else:
                hi_a = mid_a
        align_pcd_down = align_pcd.voxel_down_sample(voxel_size=(lo_a + hi_a) / 2.0)
        align_down_pts = np.asarray(align_pcd_down.points).astype(np.float32)
        align_down_rgb = (np.asarray(align_pcd_down.colors) * 255).astype(np.uint8)

    n_sky_down = sky_down_pts.shape[0] if sky_down_pts is not None else 0
    n_align_down = align_down_pts.shape[0] if align_down_pts is not None else 0
    base_budget = target_num - n_sky_down - n_align_down

    # Binary search for base voxel size
    bbox_base = np.linalg.norm(base_points.max(axis=0) - base_points.min(axis=0))
    lo, hi = bbox_base * 1e-6, bbox_base * 0.5
    best_voxel_size = (lo + hi) / 2.0

    for iteration in range(30):
        mid = (lo + hi) / 2.0
        cnt = len(base_pcd.voxel_down_sample(voxel_size=mid).points)
        if cnt == base_budget:
            best_voxel_size = mid
            break
        elif cnt > base_budget:
            lo = mid
        else:
            hi = mid
        best_voxel_size = mid

    base_pcd_down = base_pcd.voxel_down_sample(voxel_size=best_voxel_size)
    base_down_pts = np.asarray(base_pcd_down.points).astype(np.float32)
    base_down_rgb = (np.asarray(base_pcd_down.colors) * 255).astype(np.uint8)
    n_base_down = base_down_pts.shape[0]

    # Adjust base to exact budget via random trim/pad
    if n_base_down > base_budget:
        keep = rng.choice(n_base_down, size=base_budget, replace=False)
        base_down_pts = base_down_pts[keep]
        base_down_rgb = base_down_rgb[keep]
    elif n_base_down < base_budget:
        need = base_budget - n_base_down
        extra_idx = rng.choice(base_points.shape[0], size=need, replace=False)
        base_down_pts = np.concatenate([base_down_pts, base_points[extra_idx].astype(np.float32)])
        base_down_rgb = np.concatenate([base_down_rgb, base_rgb[extra_idx]])

    # Concatenate base + sky + align
    parts_pts = [base_down_pts]
    parts_rgb = [base_down_rgb]
    n_base_final = len(base_down_pts)

    sky_mask_parts = [np.zeros(n_base_final, dtype=bool)]
    align_mask_parts = [np.zeros(n_base_final, dtype=bool)]

    if sky_down_pts is not None and len(sky_down_pts) > 0:
        parts_pts.append(sky_down_pts)
        parts_rgb.append(sky_down_rgb)
        sky_mask_parts.append(np.ones(len(sky_down_pts), dtype=bool))
        align_mask_parts.append(np.zeros(len(sky_down_pts), dtype=bool))

    if align_down_pts is not None and len(align_down_pts) > 0:
        parts_pts.append(align_down_pts)
        parts_rgb.append(align_down_rgb)
        sky_mask_parts.append(np.zeros(len(align_down_pts), dtype=bool))
        align_mask_parts.append(np.ones(len(align_down_pts), dtype=bool))

    out_pts = np.concatenate(parts_pts)
    out_rgb = np.concatenate(parts_rgb)
    out_sky_mask = np.concatenate(sky_mask_parts)
    out_align_mask = np.concatenate(align_mask_parts)

    return out_pts, out_rgb, out_sky_mask, out_align_mask


class Parser:
    """OPENCV parser."""

    def __init__(
            self,
            data_dir: str,
            factor: int = 1,
            normalize: bool = False,
            test_every: int = 8,
            downsample_pts_num: int = None,
            downsample_mode: str = "random",
            load_pcd: bool = True,
            detect_anchor_candidates: bool = True,
            world_rank: int = 0,
            world_size: int = 1,
            local_rank: int = 0,
            align_sky_only: bool = False,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.detect_anchor_candidates = detect_anchor_candidates
        self.world_rank = world_rank
        self.world_size = world_size
        self.local_rank = local_rank
        self._device = f"cuda:{local_rank}"

        image_names = glob(f"{data_dir}/images/*.png")
        image_names = [os.path.basename(im).split(".")[0] for im in image_names]

        camera_path = f"{data_dir}/cameras.json"

        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        depth_dict = dict()
        normal_dict = dict()

        cam_info = json.load(open(camera_path))

        depths_dir = f"{data_dir}/depths"
        normals_dir = f"{data_dir}/normals"
        depth_files = set(os.listdir(depths_dir)) if os.path.isdir(depths_dir) else set()
        normal_files = set(os.listdir(normals_dir)) if os.path.isdir(normals_dir) else set()

        for image_name in image_names:
            camera_id = image_name
            w2c = np.array(cam_info[camera_id]["extrinsic"])
            w2c_mats.append(w2c)

            camera_ids.append(camera_id)
            K = np.array(cam_info[camera_id]["intrinsic"])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"

            params_dict[camera_id] = params
            width, height = imagesize.get(f"{data_dir}/images/{camera_id}.png")

            imsize_dict[camera_id] = (width // factor, height // factor)
            # imsize_dict[camera_id] = (cam_info["width"] // factor, cam_info["height"] // factor)
            fname = f"{camera_id}.png"
            depth_dict[camera_id] = f"{depths_dir}/{fname}" if fname in depth_files else None
            normal_dict[camera_id] = f"{normals_dir}/{fname}" if fname in normal_files else None

        print(f"[Parser Rank{self.world_rank}] {len(image_names)} images, taken by {len(set(camera_ids))} cameras.")

        if len(image_names) == 0:
            raise ValueError("No images found in World GS Path.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        image_paths = []
        for image_name in image_names:
            image_paths.append(os.path.join(image_dir, f"{image_name}.png"))

        # 3D points and {image_name -> [point_idx]}
        # ── Only rank 0 does the heavy lifting; others receive via broadcast ──
        _is_multi_gpu = (self.world_size > 1) and dist.is_initialized()

        if self.world_rank == 0:
            # --- Rank 0: load, downsample, anchor kNN ---
            if load_pcd and align_sky_only:
                # ── align_sky_only mode: skip base points.ply, use only sky + align ──
                points_parts = []
                rgb_parts = []
                sky_counts = []
                align_counts = []

                sky_pcd_path = f"{data_dir}/sky_points.ply"
                if os.path.exists(sky_pcd_path):
                    if HAS_OPEN3D:
                        _o3d_sky = o3d.io.read_point_cloud(sky_pcd_path)
                        sky_pts = np.asarray(_o3d_sky.points)
                        sky_rgb_arr = (np.asarray(_o3d_sky.colors) * 255).astype(np.uint8) if _o3d_sky.has_colors() else np.zeros((len(_o3d_sky.points), 3), dtype=np.uint8)
                    else:
                        sky_pcd = trimesh.load(sky_pcd_path)
                        sky_pts = sky_pcd.vertices
                        sky_rgb_arr = sky_pcd.colors[:, :3]
                    points_parts.append(sky_pts)
                    rgb_parts.append(sky_rgb_arr)
                    sky_counts.append(sky_pts.shape[0])
                    print(f"[Parser align_sky_only] Loaded {sky_pts.shape[0]} sky points from sky_points.ply")
                else:
                    print("[Parser align_sky_only] WARNING: sky_points.ply not found")

                align_pcd_path = f"{data_dir}/align_points.ply"
                if os.path.exists(align_pcd_path):
                    if HAS_OPEN3D:
                        _o3d_align = o3d.io.read_point_cloud(align_pcd_path)
                        align_pts = np.asarray(_o3d_align.points)
                        align_rgb_arr = (np.asarray(_o3d_align.colors) * 255).astype(np.uint8) if _o3d_align.has_colors() else np.zeros((len(_o3d_align.points), 3), dtype=np.uint8)
                    else:
                        align_pcd_loaded = trimesh.load(align_pcd_path)
                        align_pts = align_pcd_loaded.vertices
                        align_rgb_arr = align_pcd_loaded.colors[:, :3]
                    points_parts.append(align_pts)
                    rgb_parts.append(align_rgb_arr)
                    align_counts.append(align_pts.shape[0])
                    print(f"[Parser align_sky_only] Loaded {align_pts.shape[0]} align points from align_points.ply")
                else:
                    print("[Parser align_sky_only] WARNING: align_points.ply not found")

                if len(points_parts) == 0:
                    raise RuntimeError(
                        "align_sky_only=True but neither sky_points.ply nor align_points.ply found in "
                        f"{data_dir}"
                    )

                points = np.concatenate(points_parts, axis=0).astype(np.float32)
                points_rgb = np.concatenate(rgb_parts, axis=0).astype(np.uint8)
                n_sky_total = sum(sky_counts)
                n_align_total = sum(align_counts)
                sky_mask = np.concatenate([
                    np.ones(n_sky_total, dtype=bool),
                    np.zeros(n_align_total, dtype=bool),
                ]) if (n_sky_total + n_align_total) > 0 else np.zeros(0, dtype=bool)
                align_mask = np.concatenate([
                    np.zeros(n_sky_total, dtype=bool),
                    np.ones(n_align_total, dtype=bool),
                ]) if (n_sky_total + n_align_total) > 0 else np.zeros(0, dtype=bool)
                print(f"[Parser align_sky_only] Total points: {points.shape[0]} "
                      f"(sky={n_sky_total}, align={n_align_total}), base points.ply skipped")

            elif load_pcd:
                if HAS_OPEN3D:
                    _o3d_pcd = o3d.io.read_point_cloud(f"{data_dir}/points.ply")
                    points = np.asarray(_o3d_pcd.points)
                    points_rgb = (np.asarray(_o3d_pcd.colors) * 255).astype(np.uint8) if _o3d_pcd.has_colors() else np.zeros((len(_o3d_pcd.points), 3), dtype=np.uint8)
                else:
                    pcd = trimesh.load(f"{data_dir}/points.ply")
                    points = pcd.vertices
                    points_rgb = pcd.colors[:, :3]
                num_base_points = points.shape[0]

                sky_pcd_path = f"{data_dir}/sky_points.ply"
                if os.path.exists(sky_pcd_path):
                    if HAS_OPEN3D:
                        _o3d_sky = o3d.io.read_point_cloud(sky_pcd_path)
                        sky_pts = np.asarray(_o3d_sky.points)
                        sky_rgb = (np.asarray(_o3d_sky.colors) * 255).astype(np.uint8) if _o3d_sky.has_colors() else np.zeros((len(_o3d_sky.points), 3), dtype=np.uint8)
                    else:
                        sky_pcd = trimesh.load(sky_pcd_path)
                        sky_pts = sky_pcd.vertices
                        sky_rgb = sky_pcd.colors[:, :3]
                    points = np.concatenate([points, sky_pts])
                    points_rgb = np.concatenate([points_rgb, sky_rgb])
                    sky_mask = np.concatenate([
                        np.zeros(num_base_points, dtype=bool),
                        np.ones(sky_pts.shape[0], dtype=bool),
                    ])
                else:
                    sky_mask = np.zeros(num_base_points, dtype=bool)

                # --- Load align points (before downsampling, participates in downsampling) ---
                align_pcd_path = f"{data_dir}/align_points.ply"
                if os.path.exists(align_pcd_path):
                    if HAS_OPEN3D:
                        _o3d_align = o3d.io.read_point_cloud(align_pcd_path)
                        align_pts = np.asarray(_o3d_align.points)
                        align_rgb = (np.asarray(_o3d_align.colors) * 255).astype(np.uint8) if _o3d_align.has_colors() else np.zeros((len(_o3d_align.points), 3), dtype=np.uint8)
                    else:
                        align_pcd_loaded = trimesh.load(align_pcd_path)
                        align_pts = align_pcd_loaded.vertices
                        align_rgb = align_pcd_loaded.colors[:, :3]
                    num_before_align = points.shape[0]
                    points = np.concatenate([points, align_pts])
                    points_rgb = np.concatenate([points_rgb, align_rgb])
                    sky_mask = np.concatenate([sky_mask, np.zeros(align_pts.shape[0], dtype=bool)])
                    align_mask = np.concatenate([
                        np.zeros(num_before_align, dtype=bool),
                        np.ones(align_pts.shape[0], dtype=bool),
                    ])
                    print(f"[Parser] Loaded {align_pts.shape[0]} align points from align_points.ply")
                else:
                    align_mask = np.zeros(points.shape[0], dtype=bool)
            else:
                points = np.zeros((10_000, 3), dtype=np.float32)
                points_rgb = np.zeros((10_000, 3), dtype=np.uint8)
                sky_mask = np.zeros(points.shape[0], dtype=bool)
                align_mask = np.zeros(points.shape[0], dtype=bool)

            if load_pcd and downsample_pts_num is not None and points.shape[0] > downsample_pts_num:
                if downsample_mode == "open3d_voxel":
                    base_mask = ~sky_mask & ~align_mask
                    base_pts = points[base_mask]
                    base_rgb = points_rgb[base_mask]
                    sky_pts_ds = points[sky_mask] if sky_mask.any() else None
                    s_rgb_ds = points_rgb[sky_mask] if sky_mask.any() else None
                    align_pts_ds = points[align_mask] if align_mask.any() else None
                    a_rgb_ds = points_rgb[align_mask] if align_mask.any() else None
                    points, points_rgb, sky_mask, align_mask = _open3d_voxel_downsample(
                        base_pts, base_rgb, sky_pts_ds, s_rgb_ds,
                        target_num=downsample_pts_num, seed=42,
                        align_points=align_pts_ds, align_rgb=a_rgb_ds,
                    )
                elif downsample_mode == "geometry_aware_o3d":
                    rdv_indices = geometry_aware_downsample_o3d(
                        points, target_num=downsample_pts_num, seed=42,
                    )
                    points = points[rdv_indices]
                    points_rgb = points_rgb[rdv_indices]
                    sky_mask = sky_mask[rdv_indices]
                    align_mask = align_mask[rdv_indices]
                elif downsample_mode == "geometry_aware":
                    rdv_indices = geometry_aware_downsample(
                        points, target_num=downsample_pts_num, seed=42,
                    )
                    points = points[rdv_indices]
                    points_rgb = points_rgb[rdv_indices]
                    sky_mask = sky_mask[rdv_indices]
                    align_mask = align_mask[rdv_indices]
                else:
                    rng = np.random.default_rng(seed=42)
                    rdv_indices = rng.choice(points.shape[0], downsample_pts_num, replace=False)
                    points = points[rdv_indices]
                    points_rgb = points_rgb[rdv_indices]
                    sky_mask = sky_mask[rdv_indices]
                    align_mask = align_mask[rdv_indices]

            # --- Anchor candidates: optionally derive them from sky points ---
            if self.detect_anchor_candidates:
                points_is_outlier = sky_mask
            else:
                points_is_outlier = np.zeros(points.shape[0], dtype=bool)
            points_is_align = align_mask
        else:
            # Other ranks: placeholders — will be filled by broadcast
            points = None
            points_rgb = None
            points_is_outlier = None
            points_is_align = None
            sky_mask = None

        # ── Broadcast from rank 0 to all other ranks ──
        if _is_multi_gpu:
            points, points_rgb, points_is_outlier, points_is_align, sky_mask = _broadcast_pcd_arrays(
                points, points_rgb, points_is_outlier, points_is_align,
                world_rank=self.world_rank,
                local_rank=self.local_rank,
                world_size=self.world_size,
                sky_mask=sky_mask,
            )
            dist.barrier()
            print(f"[Parser Rank{self.world_rank}] Received {points.shape[0]} points via broadcast.")

        self.points_is_outlier = points_is_outlier
        self.points_is_align = points_is_align
        self.sky_mask = sky_mask
        point_indices = dict()

        # Normalize the world space.
        self.rescale = None
        self.up_direction = np.array([0, 0, 1], dtype=np.float32)
        self.facing_direction = np.array([-1, 0, 0], dtype=np.float32)
        self.center_point = np.array([0, 0, 0], dtype=np.float32)
        if normalize:
            T1, scale = similarity_from_cameras(camtoworlds)
            self.rescale = scale
            self.center_point = self.center_point.reshape(1, 3)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)
            self.center_point = transform_points(T1, self.center_point)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)
            self.center_point = transform_points(T2, self.center_point)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
                self.center_point = transform_points(T3, self.center_point)

            self.up_direction = self.up_direction @ transform[:3, :3].T
            self.up_direction /= np.linalg.norm(self.up_direction)
            self.facing_direction = self.facing_direction @ transform[:3, :3].T
            self.facing_direction /= np.linalg.norm(self.facing_direction)
            self.center_point = self.center_point.reshape(3)
        else:
            transform = np.eye(4)

        print(f"[Rank{self.world_rank}] up_direction:", self.up_direction)
        print(f"[Rank{self.world_rank}] facing_direction:", self.facing_direction)
        print(f"[Rank{self.world_rank}] center_point:", self.center_point)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.depth_dict = depth_dict  # Dict of camera_id -> depth
        self.normal_dict = normal_dict  # Dict of camera_id -> normal
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                    camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1 ** 2 + y1 ** 2)
                r = (
                        1.0
                        + params[0] * theta ** 2
                        + params[1] * theta ** 4
                        + params[2] * theta ** 6
                        + params[3] * theta ** 8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


import re

_STRUCTURED_PANO_RE = re.compile(r"^panorama_L(\d+)_A(\d+)$")
_STRUCTURED_POLAR_RE = re.compile(r"^polar_(up|down)_L(\d+)_A(\d+)$")


def _parse_pano_polar_name(name: str):
    """Parse structured panorama/polar names.

    Returns (source_type, layer_idx, azimuth_idx) or None for old-format / non-pano names.
    source_type is one of 'panorama', 'polar_up', 'polar_down'.
    """
    m = _STRUCTURED_PANO_RE.match(name)
    if m:
        return "panorama", int(m.group(1)), int(m.group(2))
    m = _STRUCTURED_POLAR_RE.match(name)
    if m:
        return f"polar_{m.group(1)}", int(m.group(2)), int(m.group(3))
    return None


# Original polar upper layers are at index 0 (45 deg) and 3 (84 deg) in the 4-layer scheme
_POLAR_UP_ORIGINAL_LAYERS = {0, 3}


class Dataset:
    """A simple dataset class."""

    def __init__(
            self,
            parser: Parser,
            split: str = "train",
            patch_size: Optional[int] = None,
            load_depths: bool = False,
            load_normals: bool = False,
            pano_only: bool = False,
            video_only: bool = False,
            pano_repeat: int = 1,
            pano_azimuth_interval: int = 1,
            polar_up_azimuth_interval: int = 1,
            polar_up_extra_layers: bool = True,
            polar_down_azimuth_interval: int = 1,
            video_prefix_filter: Optional[str] = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.load_normals = load_normals
        indices = np.arange(len(self.parser.image_names))
        self.indices = []
        pano_indices = []
        if split == "train":
            for idx in indices:
                name = self.parser.image_names[idx]
                is_pano = name.startswith("panorama_") or name.startswith("polar_")

                if is_pano and not self._passes_interval_filter(
                    name, pano_azimuth_interval, polar_up_azimuth_interval,
                    polar_up_extra_layers, polar_down_azimuth_interval,
                ):
                    continue

                if pano_only:
                    if is_pano:
                        self.indices.append(idx)
                elif video_only:
                    if not is_pano and idx % self.parser.test_every != 0:
                        if video_prefix_filter is None or name.startswith(video_prefix_filter):
                            self.indices.append(idx)
                else:
                    if idx % self.parser.test_every != 0 or is_pano:
                        self.indices.append(idx)
                    if is_pano:
                        pano_indices.append(idx)
            if pano_repeat > 1 and not pano_only:
                for _ in range(pano_repeat - 1):
                    self.indices.extend(pano_indices)
        else:
            for idx in indices:
                name = self.parser.image_names[idx]
                if idx % self.parser.test_every == 0 and ("panorama_" not in name and "polar_" not in name):
                    self.indices.append(idx)

        # Log dataset composition by source
        names = [self.parser.image_names[i] for i in self.indices]
        n_panorama = sum(1 for n in names if n.startswith("panorama_"))
        n_polar_up = sum(1 for n in names if n.startswith("polar_up_"))
        n_polar_down = sum(1 for n in names if n.startswith("polar_down_"))
        n_polar_old = sum(1 for n in names if n.startswith("polar_") and not n.startswith("polar_up_") and not n.startswith("polar_down_"))
        n_polar = n_polar_up + n_polar_down + n_polar_old
        n_video = len(names) - n_panorama - n_polar
        print(f"[Dataset-{split}] total={len(names)}, video_frames={n_video}, "
              f"panorama={n_panorama}, polar_up={n_polar_up}, polar_down={n_polar_down}"
              + (f", polar_old={n_polar_old}" if n_polar_old > 0 else "")
              + (f", pano_repeat={pano_repeat}" if pano_repeat > 1 else ""))

    @staticmethod
    def _passes_interval_filter(
        name: str,
        pano_azimuth_interval: int,
        polar_up_azimuth_interval: int,
        polar_up_extra_layers: bool,
        polar_down_azimuth_interval: int,
    ) -> bool:
        """Return True if this pano/polar view passes the interval-based subsampling filter."""
        parsed = _parse_pano_polar_name(name)
        if parsed is None:
            return True
        source_type, layer_idx, az_idx = parsed
        if source_type == "panorama":
            return az_idx % pano_azimuth_interval == 0
        elif source_type == "polar_up":
            if not polar_up_extra_layers and layer_idx not in _POLAR_UP_ORIGINAL_LAYERS:
                return False
            return az_idx % polar_up_azimuth_interval == 0
        elif source_type == "polar_down":
            return az_idx % polar_down_azimuth_interval == 0
        return True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y: y + h, x: x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y: y + self.patch_size, x: x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        image_name = self.parser.image_names[index]
        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "image_name": image_name,
        }

        if self.load_depths:
            depth_path = self.parser.depth_dict[camera_id]
            if depth_path is not None:
                depth = load_16bit_png_depth(depth_path)
                data["depths"] = torch.from_numpy(depth).float()
            else:
                data["depths"] = torch.zeros((image.shape[0], image.shape[1]), dtype=torch.float32)
            if self.parser.rescale is not None:
                data["depths"] *= self.parser.rescale

            mask = data["depths"] > 0
            data["mask"] = mask.bool()

        if self.load_normals:
            normal_path = self.parser.normal_dict[camera_id]
            if normal_path is not None:
                normal = Image.open(normal_path)
                normal = np.array(normal) / 255
                assert normal.ndim == 3 and normal.shape[2] == 3
                normal = normal.astype(np.float32) * 2. - 1.
                normal = normal.transpose(2, 0, 1)
                data["normals"] = torch.from_numpy(normal).float()
            else:
                data["normals"] = torch.zeros((3, image.shape[0], image.shape[1]), dtype=torch.float32)

        return data
