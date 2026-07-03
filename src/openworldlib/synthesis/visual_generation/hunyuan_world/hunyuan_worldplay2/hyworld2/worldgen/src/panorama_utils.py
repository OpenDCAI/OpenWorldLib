from collections import defaultdict
from typing import Optional, Literal

import cupy as cp
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
import utils3d
from PIL import Image
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse.linalg import lsmr as cp_lsmr
from moge.utils.panorama import (
    List, convolve, vstack, grad_equation, poisson_equation
)
from scipy.sparse import csr_array
from tqdm import tqdm


def subdivide_icosahedron(subdivisions: int = 1) -> np.ndarray:
    """
    Subdivide an icosahedron to generate denser spherical sample points.

    Args:
        subdivisions: Number of subdivisions.
            - 0: 12 vertices
            - 1: 42 vertices
            - 2: 162 vertices
            - 3: 642 vertices
            Formula: V = 10 * 4^n + 2

    Returns:
        vertices: (N, 3) Subdivided vertex coordinates on the unit sphere.
    """
    vertices, faces = utils3d.numpy.icosahedron()

    # Convert to a list so new vertices can be appended dynamically.
    vertices_list = [v for v in vertices]

    for _ in range(subdivisions):
        edge_midpoint_cache = {}  # Cache edge midpoint indices to avoid duplicates.
        new_faces = []

        def get_or_create_midpoint(idx1: int, idx2: int) -> int:
            """
            Get the midpoint index for the edge between two vertices.
            Create it if missing, otherwise return the cached index.
            """
            # Use the sorted tuple as the key so (a, b) and (b, a) are the same edge.
            edge_key = (min(idx1, idx2), max(idx1, idx2))

            if edge_key in edge_midpoint_cache:
                return edge_midpoint_cache[edge_key]

            # Create a new midpoint.
            v1 = vertices_list[idx1]
            v2 = vertices_list[idx2]
            midpoint = (v1 + v2) / 2.0

            # Project onto the unit sphere.
            midpoint = midpoint / np.linalg.norm(midpoint)

            # Add it to the vertex list.
            new_idx = len(vertices_list)
            vertices_list.append(midpoint)
            edge_midpoint_cache[edge_key] = new_idx

            return new_idx

        # Subdivide each triangle.
        for face in faces:
            v0, v1, v2 = face

            # Get the midpoints of the three edges.
            #       v0
            #      /  \
            #     a----c
            #    / \  / \
            #   v1--b----v2
            a = get_or_create_midpoint(v0, v1)
            b = get_or_create_midpoint(v1, v2)
            c = get_or_create_midpoint(v2, v0)

            # Split the original triangle into four smaller triangles.
            new_faces.append([v0, a, c])
            new_faces.append([a, v1, b])
            new_faces.append([c, b, v2])
            new_faces.append([a, b, c])

        faces = np.array(new_faces, dtype=np.int32)

    return np.array(vertices_list, dtype=np.float32)


def get_panorama_cameras_v2(subdivisions=0):
    vertices = subdivide_icosahedron(subdivisions=subdivisions)
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.numpy.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)


def rotate_around_z_axis(points, angle_deg):
    """
    Rotate 3D points clockwise around the Z axis.

    Args:
        points: 3D point array with shape (N, 3), where each row is (x, y, z).
        angle_deg: Rotation angle in degrees. Positive values rotate clockwise.

    Returns:
        rotated_points: Rotated 3D point array with shape (N, 3).
    """
    # Convert the angle to radians.
    angle_rad = np.radians(angle_deg)

    # Build the clockwise rotation matrix around the Z axis.
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # In a right-handed coordinate system, clockwise rotation equals negative counterclockwise rotation.
    rotation_matrix = np.array([
        [cos_theta, sin_theta, 0],  # X component
        [-sin_theta, cos_theta, 0],  # Y component
        [0, 0, 1]  # Z component, unchanged
    ])

    # Apply the rotation to each point by matrix multiplication.
    rotated_points = np.dot(points, rotation_matrix.T)

    return rotated_points


def directions_to_spherical_uv(directions: np.ndarray):
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(directions[..., 2]) / np.pi
    return np.stack([u, v], axis=-1)


def split_panorama_image(image: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, h: int, w: int, interp):
    height, width = image.shape[:2]
    safe_height = height // 2
    safe_width = int(round(safe_height / h * w))
    if interp == cv2.INTER_AREA: # remap does not support area downsampling; remap to a safe resolution first to avoid frequency artifacts.
        uv = utils3d.numpy.image_uv(width=safe_width, height=safe_height)
    else:
        uv = utils3d.numpy.image_uv(width=w, height=h)
    splitted_images = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(utils3d.numpy.unproject_cv(uv, extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
        pixels = utils3d.numpy.uv_to_pixel(spherical_uv, width=width, height=height).astype(np.float32)
        if interp == cv2.INTER_AREA: # remap does not support area downsampling; remap to a safe resolution first to avoid frequency artifacts.
            splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            splitted_image = cv2.resize(splitted_image, (w, h), interpolation=interp)
        else:
            splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        splitted_images.append(splitted_image)
    return splitted_images


def split_panorama_depth(depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, h: int, w: int, distance_to_depth=False):
    height, width = depth.shape[:2]
    depth = torch.tensor(depth, dtype=torch.float32)[None, None]
    uv = utils3d.numpy.image_uv(width=w, height=h)
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    splitted_depths = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(utils3d.numpy.unproject_cv(uv, extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
        pixels = utils3d.numpy.uv_to_pixel(spherical_uv, width=width, height=height).astype(np.float32)
        pixels = torch.tensor(pixels, dtype=torch.float32)[None, ...]  # [1,h,w,2]
        pixels[..., 0] /= width
        pixels[..., 1] /= height
        pixels = pixels * 2 - 1.0
        splitted_depth = F.grid_sample(depth, grid=pixels, mode="nearest", align_corners=True)

        if distance_to_depth:
            fx = intrinsics[i][0, 0] * w
            fy = intrinsics[i][1, 1] * h
            cx = intrinsics[i][0, 2] * w
            cy = intrinsics[i][1, 2] * h
            x_cam = (u_grid - cx) / fx
            y_cam = (v_grid - cy) / fy
            z_cam = np.ones_like(x_cam)
            rays_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float32)  # (H, W, 3)
            ray_length = np.linalg.norm(rays_cam, axis=-1).astype(np.float32)
            splitted_depth = splitted_depth * (z_cam[None, None] / ray_length[None, None])

        splitted_depths.append(splitted_depth)
    return torch.cat(splitted_depths, dim=0).float()


def smooth_south_pole_depth(depth_map, smooth_height_ratio=0.03):
    """
    Smooth depth near the panorama south pole (bottom region) to fix left-right inconsistencies.

    Args:
        depth_map: Depth map (H, W).
        smooth_height_ratio: Height ratio of the smoothing region. The default 0.03 means the bottom 0.03 * H region.

    Returns:
        Smoothed depth map.
    """
    height, width = depth_map.shape
    smooth_height = int(height * smooth_height_ratio)

    if smooth_height == 0:
        return depth_map

    # Copy the depth map to avoid modifying the input.
    smoothed_depth = depth_map.copy()

    # Compute the reference depth from the last 3 rows when possible, otherwise from the bottom row.
    if smooth_height > 3:
        # Use the last 3 rows to compute the average depth.
        reference_rows = depth_map[-3:, :]
        reference_data = reference_rows.flatten()
    else:
        # Use the bottom row.
        reference_data = depth_map[-1, :]

    # Filter outliers, including invalid, overly large, or overly small depth values.
    valid_mask = np.isfinite(reference_data) & (reference_data > 0)

    if np.any(valid_mask):
        valid_depths = reference_data[valid_mask]

        # Use quantiles to filter extreme outliers.
        lower_bound, upper_bound = np.quantile(valid_depths, [0.1, 0.9])

        # Further remove overly large or small depth values.
        depth_filter_mask = (valid_depths >= lower_bound) & (valid_depths <= upper_bound)

        if np.any(depth_filter_mask):
            avg_depth = np.mean(valid_depths[depth_filter_mask])
        else:
            # Fall back to the median if all values are filtered out.
            avg_depth = np.median(valid_depths)
    else:
        avg_depth = np.nanmean(reference_data)

    # Set the bottom row to the average value.
    smoothed_depth[-1, :] = avg_depth

    # Smooth upward to the specified height.
    for i in range(1, smooth_height):
        y_idx = height - 1 - i  # Index moving upward from the bottom.
        if y_idx < 0:
            break

        # The closer to the bottom, the stronger the smoothing.
        weight = (smooth_height - i) / smooth_height

        # Smooth the current row.
        current_row = depth_map[y_idx, :]
        valid_mask = np.isfinite(current_row) & (current_row > 0)

        if np.any(valid_mask):
            valid_row_depths = current_row[valid_mask]

            # Apply outlier filtering to the current row as well.
            if len(valid_row_depths) > 1:
                q25, q75 = np.quantile(valid_row_depths, [0.25, 0.75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                depth_filter_mask = (valid_row_depths >= lower_bound) & (valid_row_depths <= upper_bound)

                if np.any(depth_filter_mask):
                    row_avg = np.mean(valid_row_depths[depth_filter_mask])
                else:
                    row_avg = np.median(valid_row_depths)
            else:
                row_avg = valid_row_depths[0] if len(valid_row_depths) > 0 else avg_depth

            # Linearly interpolate between the original depth and the row average.
            smoothed_depth[y_idx, :] = (1 - weight) * current_row + weight * row_avg

    return smoothed_depth


def solve_lsmr_gpu(A, b, x0=None):
    """GPU-accelerated LSMR."""
    # Move to GPU.
    A_gpu = cp_csr_matrix(A)
    b_gpu = cp.asarray(b)
    x0_gpu = cp.asarray(x0) if x0 is not None else None

    # Solve on GPU.
    x_gpu, *_ = cp_lsmr(A_gpu, b_gpu, atol=1e-5, btol=1e-5, x0=x0_gpu)

    # Move back to CPU.
    return cp.asnumpy(x_gpu)


def merge_panorama_depth_gpu(width: int, height: int, distance_maps: List[np.ndarray], pred_masks: List[np.ndarray], extrinsics: List[np.ndarray], intrinsics: List[np.ndarray]):
    if max(width, height) > 256:
        panorama_depth_init, _ = merge_panorama_depth_gpu(width // 2, height // 2, distance_maps, pred_masks, extrinsics, intrinsics)
        panorama_depth_init = cv2.resize(panorama_depth_init, (width, height), cv2.INTER_LINEAR)
    else:
        panorama_depth_init = None

    uv = utils3d.numpy.image_uv(width=width, height=height)
    spherical_directions = spherical_uv_to_directions(uv)  # [h,w,3]

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    for i in range(len(distance_maps)):
        projected_uv, projected_depth = utils3d.numpy.project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)

        projected_pixels = utils3d.numpy.uv_to_pixel(np.clip(projected_uv, 0, 1), width=distance_maps[i].shape[1], height=distance_maps[i].shape[0]).astype(np.float32)

        log_splitted_distance = np.log(distance_maps[i])
        panorama_log_distance_map = np.where(projection_valid_mask,
                                             cv2.remap(log_splitted_distance, projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        panorama_pred_mask = projection_valid_mask & (
                cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)

        # calculate gradient map
        padded = np.pad(panorama_log_distance_map, ((0, 0), (0, 1)), mode='wrap')
        grad_x, grad_y = padded[:, :-1] - padded[:, 1:], padded[:-1, :] - padded[1:, :]

        padded = np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode='wrap')
        mask_x, mask_y = padded[:, :-1] & padded[:, 1:], padded[:-1, :] & padded[1:, :]

        panorama_log_distance_grad_maps.append((grad_x, grad_y))
        panorama_grad_masks.append((mask_x, mask_y))

        # calculate laplacian map
        padded = np.pad(panorama_log_distance_map, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        laplacian = convolve(padded, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))[1:-1, 1:-1]

        padded = np.pad(panorama_pred_mask, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        mask = convolve(padded.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))[1:-1, 1:-1] == 5

        panorama_log_distance_laplacian_maps.append(laplacian)
        panorama_laplacian_masks.append(mask)

        panorama_pred_masks.append(panorama_pred_mask)

    panorama_log_distance_grad_x = np.stack([grad_map[0] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_log_distance_grad_y = np.stack([grad_map[1] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_grad_mask_x = np.stack([mask_map[0] for mask_map in panorama_grad_masks], axis=0)
    panorama_grad_mask_y = np.stack([mask_map[1] for mask_map in panorama_grad_masks], axis=0)

    panorama_log_distance_grad_x = np.sum(panorama_log_distance_grad_x * panorama_grad_mask_x, axis=0) / np.sum(panorama_grad_mask_x, axis=0).clip(1e-3)
    panorama_log_distance_grad_y = np.sum(panorama_log_distance_grad_y * panorama_grad_mask_y, axis=0) / np.sum(panorama_grad_mask_y, axis=0).clip(1e-3)

    panorama_laplacian_maps = np.stack(panorama_log_distance_laplacian_maps, axis=0)
    panorama_laplacian_masks = np.stack(panorama_laplacian_masks, axis=0)
    panorama_laplacian_map = np.sum(panorama_laplacian_maps * panorama_laplacian_masks, axis=0) / np.sum(panorama_laplacian_masks, axis=0).clip(1e-3)

    grad_x_mask = np.any(panorama_grad_mask_x, axis=0).reshape(-1)
    grad_y_mask = np.any(panorama_grad_mask_y, axis=0).reshape(-1)
    grad_mask = np.concatenate([grad_x_mask, grad_y_mask])
    laplacian_mask = np.any(panorama_laplacian_masks, axis=0).reshape(-1)

    # Solve overdetermined system
    A = vstack([
        grad_equation(width, height, wrap_x=True, wrap_y=False)[grad_mask],
        poisson_equation(width, height, wrap_x=True, wrap_y=False)[laplacian_mask],
    ])
    b = np.concatenate([
        panorama_log_distance_grad_x.reshape(-1)[grad_x_mask],
        panorama_log_distance_grad_y.reshape(-1)[grad_y_mask],
        panorama_laplacian_map.reshape(-1)[laplacian_mask]
    ])
    x = solve_lsmr_gpu(A, b, x0=np.log(panorama_depth_init).reshape(-1) if panorama_depth_init is not None else None)

    panorama_depth = np.exp(x).reshape(height, width).astype(np.float32)
    panorama_mask = np.any(panorama_pred_masks, axis=0)

    return panorama_depth, panorama_mask


def pred_pano_depth(model, image: Image.Image, scale=1.0, resize_to=1920, remove_pano_depth_nan=True):
    """
    last_layer_mask: Previous-layer object mask.
    last_layer_depth: Previous-layer depth.
    """
    print("\t - Predicting pano depth with moge")
    image_origin = np.array(image)
    height_origin, width_origin = image_origin.shape[:2]

    image, height, width = image_origin, height_origin, width_origin
    if resize_to is not None:
        _height, _width = min(resize_to, int(resize_to * height_origin / width_origin)), min(resize_to, int(resize_to * width_origin / height_origin))
        if _height < height_origin:
            print(f"\t - Resizing image from {width_origin}x{height_origin} to {_width}x{_height} for pano depth prediction")
            image = cv2.resize(image_origin, (_width, _height), cv2.INTER_AREA)
            height, width = _height, _width

    splitted_extrinsics, splitted_intriniscs = get_panorama_cameras_v2(subdivisions=1)
    splitted_resolution = 512
    splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intriniscs, splitted_resolution, splitted_resolution, interp=cv2.INTER_AREA)

    # infer moge depth
    num_splitted_images = len(splitted_images)
    splitted_distance_maps = [None] * num_splitted_images
    splitted_masks = [None] * num_splitted_images

    indices_to_process_model = []
    skipped_count = 0

    for i in range(num_splitted_images):
        indices_to_process_model.append(i)

    pred_count = 0
    # Process images that require model inference in batches
    inference_batch_size = 1
    for i in range(0, len(indices_to_process_model), inference_batch_size):
        batch_indices = indices_to_process_model[i: i + inference_batch_size]
        if not batch_indices:
            continue

        current_batch_images = [splitted_images[k] for k in batch_indices]
        current_batch_intrinsics = [splitted_intriniscs[k] for k in batch_indices]

        image_tensor = torch.tensor(
            np.stack(current_batch_images) / 255,
            dtype=torch.float32,
            device=next(model.parameters()).device,
        ).permute(0, 3, 1, 2)

        fov_x, _ = np.rad2deg(  # fov_y is not used by model.infer
            utils3d.numpy.intrinsics_to_fov(np.array(current_batch_intrinsics))
        )
        fov_x_tensor = torch.tensor(
            fov_x, dtype=torch.float32, device=next(model.parameters()).device
        )

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            output = model.infer(image_tensor, fov_x=fov_x_tensor, apply_mask=False)

        batch_distance_maps = output["points"].norm(dim=-1).cpu().numpy()
        batch_masks = output["mask"].cpu().numpy()

        for batch_idx, original_idx in enumerate(batch_indices):
            splitted_distance_maps[original_idx] = batch_distance_maps[batch_idx]
            splitted_masks[original_idx] = batch_masks[batch_idx]
            pred_count += 1

    if (pred_count + skipped_count) == 0:  # Avoid division by zero if num_splitted_images is 0
        skip_ratio_info = "N/A (no images to process)"
    else:
        skip_ratio_info = f"{skipped_count / (pred_count + skipped_count):.2%}"

    print(f"\t 🔍 Predicted {pred_count} splitted images, skipped {skipped_count} splitted images. Skip ratio: {skip_ratio_info}")

    # merge moge depth
    merging_width, merging_height = width, height
    panorama_depth, panorama_mask = merge_panorama_depth_gpu(
        merging_width,
        merging_height,
        splitted_distance_maps,
        splitted_masks,
        splitted_extrinsics,
        splitted_intriniscs,
    )

    panorama_depth = panorama_depth.astype(np.float32)
    # Align the left and right depths in the bottom region of pano depth.
    if remove_pano_depth_nan:
        # for depth inpainting, remove nan
        panorama_depth[~panorama_mask] = 1.0 * np.nanquantile(panorama_depth, 0.999)  # Sky depth.
    panorama_depth = cv2.resize(panorama_depth, (width_origin, height_origin), cv2.INTER_LINEAR)
    panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (width_origin, height_origin), cv2.INTER_NEAREST) > 0

    # Smooth south-pole (bottom region) depth to fix left-right inconsistencies.
    print("\t - Smoothing south pole depth for consistency")
    panorama_depth = smooth_south_pole_depth(panorama_depth, smooth_height_ratio=0.05)

    rays = torch.from_numpy(spherical_uv_to_directions(utils3d.numpy.image_uv(width=width_origin, height=height_origin))).to(next(model.parameters()).device)

    panorama_depth = (
            torch.from_numpy(panorama_depth).to(next(model.parameters()).device) * scale
    )

    return {
        "rgb": torch.from_numpy(image_origin).to(next(model.parameters()).device),
        "distance": panorama_depth,
        "rays": rays,
        "mask": panorama_mask,
        "splitted_masks": splitted_masks,
        "splitted_distance_maps": splitted_distance_maps,
    }


# Panorama depth stitching based on normal constraints.
def compute_spherical_ray_derivatives(spherical_directions: np.ndarray):
    """
    Compute partial derivatives of panorama ray directions with respect to spherical coordinates.

    Args:
        spherical_directions: (H, W, 3) Ray direction for each panorama pixel in world coordinates as unit vectors.

    Returns:
        dray_dtheta: (H, W, 3) Partial derivative of rays with respect to azimuth theta.
        dray_dphi: (H, W, 3) Partial derivative of rays with respect to elevation phi.
    """
    H, W, _ = spherical_directions.shape

    # Recover spherical coordinates from ray directions.
    # Assumes x-right, y-up, z-forward coordinates; adjust if the actual convention differs.
    ray = spherical_directions

    # Azimuth theta and elevation phi.
    # ray = (cos(φ)sin(θ), sin(φ), cos(φ)cos(θ))
    # Or adapt to the coordinate-system definition in use.

    # Method 1: numerical computation, which is more robust.
    u = (np.arange(W) + 0.5) / W  # [0, 1]
    v = (np.arange(H) + 0.5) / H  # [0, 1]
    u_grid, v_grid = np.meshgrid(u, v)

    theta = (u_grid - 0.5) * 2 * np.pi  # [-π, π]
    phi = (0.5 - v_grid) * np.pi  # [π/2, -π/2]

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # ∂ray/∂θ, assuming standard equirectangular mapping.
    # Adjust according to how spherical_directions is actually computed.
    dray_dtheta = np.stack([
        cos_phi * cos_theta,
        np.zeros_like(theta),
        -cos_phi * sin_theta
    ], axis=-1)

    # ∂ray/∂φ
    dray_dphi = np.stack([
        -sin_phi * sin_theta,
        cos_phi,
        -sin_phi * cos_theta
    ], axis=-1)

    return dray_dtheta, dray_dphi


def normal_to_log_distance_gradient(
        panorama_normal: np.ndarray,
        spherical_directions: np.ndarray,
        width: int,
        height: int
) -> tuple:
    """
    Compute log-distance gradients from panorama normals.

    Derivation:
        Surface point P = d * ray
        Tangent vector ∂P/∂θ = (∂d/∂θ) * ray + d * (∂ray/∂θ)
        The normal is perpendicular to the tangent: n · ∂P/∂θ = 0
        => ∂(log d)/∂θ = -(n · ∂ray/∂θ) / (n · ray)

    Args:
        panorama_normal: (H, W, 3) Panorama normal map in world coordinates.
        spherical_directions: (H, W, 3) Ray directions in world coordinates.
        width, height: Panorama dimensions.

    Returns:
        grad_x: (H, W) x-direction gradient corresponding to log_d[j] - log_d[j+1].
        grad_y: (H-1, W) y-direction gradient corresponding to log_d[i] - log_d[i+1].
        valid_mask: (H, W) Valid region.
    """
    H, W = height, width

    # Get the ray-direction partial derivatives.
    dray_dtheta, dray_dphi = compute_spherical_ray_derivatives(spherical_directions)

    # Compute dot products.
    n_dot_ray = np.sum(panorama_normal * spherical_directions, axis=-1)  # (H, W)
    n_dot_dray_dtheta = np.sum(panorama_normal * dray_dtheta, axis=-1)  # (H, W)
    n_dot_dray_dphi = np.sum(panorama_normal * dray_dphi, axis=-1)  # (H, W)

    # Validity check: normals cannot be perpendicular to the view direction.
    eps = 1e-4
    valid_mask = np.abs(n_dot_ray) > eps

    # Safe division.
    n_dot_ray_safe = np.where(valid_mask, n_dot_ray, 1.0)

    # Continuous-space log(d) gradients with respect to angles.
    # ∂(log d)/∂θ = -(n · ∂ray/∂θ) / (n · ray)
    dlogd_dtheta = -n_dot_dray_dtheta / n_dot_ray_safe  # (H, W)
    dlogd_dphi = -n_dot_dray_dphi / n_dot_ray_safe  # (H, W)

    # Convert to discrete pixel gradients.
    # In this code, grad_x = log_d[j] - log_d[j+1] = -∂(log d)/∂θ * Δθ.
    # Δθ = 2π / W, the θ change per pixel.
    # grad_y = log_d[i] - log_d[i+1] = -∂(log d)/∂φ * Δφ = ∂(log d)/∂φ * (π/H)

    delta_theta = 2 * np.pi / W
    delta_phi = np.pi / H

    # Pixel-scale continuous gradients.
    pixel_dlogd_dx = -dlogd_dtheta * delta_theta  # Corresponds to log_d[j] - log_d[j+1].
    pixel_dlogd_dy = dlogd_dphi * delta_phi  # Corresponds to log_d[i] - log_d[i+1].

    # Discrete gradients, averaging both sides at boundaries.
    # X direction (wrap).
    padded_grad_x = np.pad(pixel_dlogd_dx, ((0, 0), (0, 1)), mode='wrap')
    grad_x = (padded_grad_x[:, :-1] + padded_grad_x[:, 1:]) / 2  # (H, W)

    # Y direction (no wrap).
    grad_y = (pixel_dlogd_dy[:-1, :] + pixel_dlogd_dy[1:, :]) / 2  # (H-1, W)

    # Valid masks for gradients
    padded_valid = np.pad(valid_mask, ((0, 0), (0, 1)), mode='wrap')
    mask_x = padded_valid[:, :-1] & padded_valid[:, 1:]  # (H, W)
    mask_y = valid_mask[:-1, :] & valid_mask[1:, :]  # (H-1, W)

    return grad_x, grad_y, mask_x, mask_y, valid_mask


def grad_equation_separate(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False):
    """
    Return gradient equation matrices for the x and y directions separately.

    Returns:
        grad_eq_x: x-direction gradient matrix, with H * W rows if wrap_x, otherwise H * (W - 1).
        grad_eq_y: y-direction gradient matrix, with (H - 1) * W rows if wrap_y is false.
    """
    grid_index = np.arange(width * height).reshape(height, width)

    # X direction.
    if wrap_x:
        grid_x = np.pad(grid_index, ((0, 0), (0, 1)), mode='wrap')
    else:
        grid_x = grid_index

    n_grad_x = grid_x.shape[0] * (grid_x.shape[1] - 1)
    data_x = np.concatenate([
        np.ones((grid_x.shape[0], grid_x.shape[1] - 1), dtype=np.float32).reshape(-1, 1),
        -np.ones((grid_x.shape[0], grid_x.shape[1] - 1), dtype=np.float32).reshape(-1, 1),
    ], axis=1).reshape(-1)
    indices_x = np.concatenate([
        grid_x[:, :-1].reshape(-1, 1),
        grid_x[:, 1:].reshape(-1, 1),
    ], axis=1).reshape(-1)
    indptr_x = np.arange(0, n_grad_x * 2 + 1, 2)
    grad_eq_x = csr_array((data_x, indices_x, indptr_x), shape=(n_grad_x, height * width))

    # Y direction, usually without wrapping.
    if wrap_y:
        grid_y = np.pad(grid_index, ((0, 1), (0, 0)), mode='wrap')
    else:
        grid_y = grid_index

    n_grad_y = (grid_y.shape[0] - 1) * grid_y.shape[1]
    data_y = np.concatenate([
        np.ones((grid_y.shape[0] - 1, grid_y.shape[1]), dtype=np.float32).reshape(-1, 1),
        -np.ones((grid_y.shape[0] - 1, grid_y.shape[1]), dtype=np.float32).reshape(-1, 1),
    ], axis=1).reshape(-1)
    indices_y = np.concatenate([
        grid_y[:-1, :].reshape(-1, 1),
        grid_y[1:, :].reshape(-1, 1),
    ], axis=1).reshape(-1)
    indptr_y = np.arange(0, n_grad_y * 2 + 1, 2)
    grad_eq_y = csr_array((data_y, indices_y, indptr_y), shape=(n_grad_y, height * width))

    return grad_eq_x, grad_eq_y










def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions


def convert_rgbd2pcd_panorama(
        rgb: torch.Tensor,  # (H, W, 3) RGB image, values [0, 1]
        distance: torch.Tensor,  # (H, W) Distance map
        rays: torch.Tensor,  # (H, W, 3) Ray directions (unit vectors ideally)
        excluded_region_mask: Optional[torch.Tensor] = None,  # (H, W) Optional boolean mask
        max_size: int = 4096,  # Max dimension for resizing
        device: Literal["cuda", "cpu"] = "cuda",  # Computation device
        dropout_pcd=False
):
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        excluded_region_mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        max_size: Maximum size (height or width) to resize inputs to.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    assert (
            rgb.shape[:2] == distance.shape[:2] == rays.shape[:2]
    ), "Input shapes must match"

    mask = excluded_region_mask

    if mask is not None:
        assert (
                mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2]
        ), "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    rgb = rgb.to(device)
    distance = distance.to(device)
    rays = rays.to(device)
    if mask is not None:
        mask = mask.to(device)

    H, W = distance.shape
    if max(H, W) > max_size:
        scale = max_size / max(H, W)
    else:
        scale = 1.0

    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    distance_nchw = distance.unsqueeze(0).unsqueeze(0)
    rays_nchw = rays.permute(2, 0, 1).unsqueeze(0)

    rgb_resized = (
        F.interpolate(
            rgb_nchw,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )

    distance_resized = (
        F.interpolate(
            distance_nchw,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        .squeeze(0)
        .squeeze(0)
    )

    rays_resized_nchw = F.interpolate(
        rays_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )

    # IMPORTANT: Renormalize ray directions after interpolation
    rays_resized = rays_resized_nchw.squeeze(0).permute(1, 2, 0)
    rays_norm = torch.linalg.norm(rays_resized, dim=-1, keepdim=True)
    rays_resized = rays_resized / (rays_norm + 1e-8)

    if mask is not None:
        mask_resized = (
            F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),  # Needs float for interpolation
                scale_factor=scale,
                mode="nearest",  # Or 'nearest' if sharp boundaries are critical
                # align_corners=False,
                recompute_scale_factor=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        mask_resized = mask_resized > 0.5  # Convert back to boolean
    else:
        mask_resized = None

    # --- Calculate 3D Vertices ---
    # Vertex position = origin + distance * ray_direction
    # Assuming origin is (0, 0, 0)
    distance_flat = distance_resized.reshape(-1, 1)  # (H*W, 1)
    rays_flat = rays_resized.reshape(-1, 3)  # (H*W, 3)
    vertices = distance_flat * rays_flat  # (H*W, 3)
    vertex_colors = rgb_resized.reshape(-1, 3)  # (H*W, 3)
    if mask_resized is not None:
        mask_resized = mask_resized.reshape(-1, )
        vertices = vertices[~mask_resized]
        vertex_colors = vertex_colors[~mask_resized]

    # downsample
    if dropout_pcd and vertices.shape[0] > 1_000_000:
        rdx = np.arange(vertices.shape[0])
        np.random.shuffle(rdx)
        rdx = rdx[:1_000_000]
        vertices = vertices[rdx]
        vertex_colors = vertex_colors[rdx]

    pcd = trimesh.PointCloud(vertices=vertices.cpu().numpy(), colors=vertex_colors.cpu().numpy())

    return pcd


def convert_rgbd2pcd_multi_scale_panorama(
        rgb: torch.Tensor,  # (H, W, 3) RGB image, values [0, 1]
        distance: torch.Tensor,  # (H, W) Distance map
        rays: torch.Tensor,  # (H, W, 3) Ray directions (unit vectors ideally)
        excluded_region_mask: Optional[torch.Tensor] = None,  # (H, W) Optional boolean mask
        device: Literal["cuda", "cpu"] = "cuda",  # Computation device
        depth_intervals=[0, 1, 2, 4, 8]
):
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        excluded_region_mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    assert (
            rgb.shape[:2] == distance.shape[:2] == rays.shape[:2]
    ), "Input shapes must match"

    mask = excluded_region_mask

    if mask is not None:
        assert (
                mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2]
        ), "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    rgb = rgb.to(device)
    distance = distance.to(device)
    rays = rays.to(device)
    if mask is not None:
        mask = mask.to(device)

    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    distance_nchw = distance.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    rays_nchw = rays.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    median_distance = torch.median(distance).item()

    total_points = []
    total_colors = []

    for i in tqdm(range(1, len(depth_intervals)), desc="Processing depth intervals"):
        if i == len(depth_intervals) - 1:
            interval_mask = distance_nchw > (median_distance * depth_intervals[i - 1])
        else:
            interval_mask = ((median_distance * depth_intervals[i - 1]) < distance_nchw) & (distance_nchw <= (median_distance * depth_intervals[i]))

        # pointclouds number ∝ depth^2
        resize_scale = depth_intervals[i]
        if interval_mask.sum() == 0:
            continue

        rgb_resized = (
            F.interpolate(
                rgb_nchw,
                scale_factor=resize_scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )

        distance_resized = (
            F.interpolate(
                distance_nchw,
                scale_factor=resize_scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        rays_resized_nchw = F.interpolate(
            rays_nchw,
            scale_factor=resize_scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )

        interval_mask_resized = F.interpolate(interval_mask.float(),
                                              scale_factor=resize_scale,
                                              mode="nearest",
                                              recompute_scale_factor=False).bool().squeeze(0).squeeze(0)

        # IMPORTANT: Renormalize ray directions after interpolation
        rays_resized = rays_resized_nchw.squeeze(0).permute(1, 2, 0)
        rays_norm = torch.linalg.norm(rays_resized, dim=-1, keepdim=True)
        rays_resized = rays_resized / (rays_norm + 1e-8)

        if mask is not None:
            mask_resized = (
                F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),  # Needs float for interpolation
                    scale_factor=resize_scale,
                    mode="nearest",  # Or 'nearest' if sharp boundaries are critical
                    # align_corners=False,
                    recompute_scale_factor=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
            mask_resized = mask_resized > 0.5  # Convert back to boolean
        else:
            mask_resized = None

        mask_resized = mask_resized & interval_mask_resized

        # --- Calculate 3D Vertices ---
        # Vertex position = origin + distance * ray_direction
        # Assuming origin is (0, 0, 0)
        distance_flat = distance_resized.reshape(-1, 1)  # (H*W, 1)
        rays_flat = rays_resized.reshape(-1, 3)  # (H*W, 3)
        vertices = distance_flat * rays_flat  # (H*W, 3)
        vertex_colors = rgb_resized.reshape(-1, 3)  # (H*W, 3)
        if mask_resized is not None:
            mask_resized = mask_resized.reshape(-1, )
            vertices = vertices[~mask_resized]
            vertex_colors = vertex_colors[~mask_resized]

        total_points.append(vertices)
        total_colors.append(vertex_colors)
        print(f"Depth interval: {depth_intervals[i]}, Number of points: {vertices.shape[0]}")

    vertices = torch.cat(total_points, dim=0)
    vertex_colors = torch.cat(total_colors, dim=0)
    pcd = trimesh.PointCloud(vertices=vertices.cpu().numpy(), colors=vertex_colors.cpu().numpy())

    return pcd


def convert_rgbd2pcd_panorama_da360(depth, rgb, mask=None, dropout_pcd=True):
    h, w = depth.shape
    Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    Theta = np.repeat(Theta, w, axis=1)
    Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
    Phi = -np.repeat(Phi, h, axis=0)

    X = depth * np.sin(Theta) * np.sin(Phi)
    Y = depth * np.cos(Theta)
    Z = depth * np.sin(Theta) * np.cos(Phi)

    if mask is None:
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()
        R = rgb[:, :, 0].flatten()
        G = rgb[:, :, 1].flatten()
        B = rgb[:, :, 2].flatten()
    else:
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        R = rgb[:, :, 0][mask]
        G = rgb[:, :, 1][mask]
        B = rgb[:, :, 2][mask]

    XYZ = np.stack([X, Y, Z], axis=1)
    RGB = np.stack([R, G, B], axis=1)

    # downsample
    if dropout_pcd and XYZ.shape[0] > 1_000_000:
        rdx = np.arange(XYZ.shape[0])
        np.random.shuffle(rdx)
        rdx = rdx[:1_000_000]
        XYZ = XYZ[rdx]
        RGB = RGB[rdx]

    pcd = trimesh.PointCloud(vertices=XYZ, colors=RGB)
    return pcd


def _generate_faces_numpy(H: int, W: int, mask: Optional[torch.Tensor]) -> np.ndarray:
    """
    Pure NumPy implementation, 2-3x faster than the PyTorch version.
    """
    # Precompute all vertex indices.
    idx = np.arange(H * W, dtype=np.int32).reshape(H, W)

    # Four corners of each quad, with horizontal wrapping.
    tl = idx[:-1, :]  # top-left
    tr = idx[:-1, :].copy()
    tr[:, :-1] = idx[:-1, 1:]
    tr[:, -1] = idx[:-1, 0]  # wrap
    bl = idx[1:, :]  # bottom-left
    br = idx[1:, :].copy()
    br[:, :-1] = idx[1:, 1:]
    br[:, -1] = idx[1:, 0]  # wrap

    # Apply mask.
    if mask is not None:
        mask_np = mask.cpu().numpy()
        # Check whether any of the four corners is masked.
        m_tl = mask_np[:-1, :]
        m_tr = np.roll(mask_np[:-1, :], -1, axis=1)
        m_bl = mask_np[1:, :]
        m_br = np.roll(mask_np[1:, :], -1, axis=1)

        keep = ~(m_tl | m_tr | m_bl | m_br)

        tl, tr, bl, br = tl[keep], tr[keep], bl[keep], br[keep]
    else:
        tl, tr, bl, br = tl.ravel(), tr.ravel(), bl.ravel(), br.ravel()

    # Build triangles, two per quad.
    n = len(tl)
    faces = np.empty((2 * n, 3), dtype=np.int32)
    faces[0::2] = np.column_stack([tl, tr, bl])
    faces[1::2] = np.column_stack([tr, br, bl])

    return faces


def convert_rgbd2mesh_panorama(
        rgb: torch.Tensor,  # (H, W, 3) RGB image, values [0, 1]
        distance: torch.Tensor,  # (H, W) Distance map
        rays: torch.Tensor,  # (H, W, 3) Ray directions (unit vectors ideally)
        excluded_region_mask: Optional[torch.Tensor] = None,  # (H, W) Optional boolean mask
        max_size: int = 4096,  # Max dimension for resizing
        device: Literal["cuda", "cpu"] = "cuda",  # Computation device
        connect_boundary_max_dist: Optional[float] = 0.5,  # Max distance to bridge boundary vertices
        connect_boundary_repeat_times: int = 2
) -> o3d.geometry.TriangleMesh:
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        excluded_region_mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        max_size: Maximum size (height or width) to resize inputs to.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    """Optimized version: about 3-5x faster."""
    H, W = distance.shape
    scale = min(1.0, max_size / max(H, W))
    need_resize = scale < 1.0

    # ========== 1. Data preparation with asynchronous transfers. ==========
    rgb = rgb.to(device, non_blocking=True)
    distance = distance.to(device, non_blocking=True)
    rays = rays.to(device, non_blocking=True)
    if excluded_region_mask is not None:
        mask = excluded_region_mask.to(device, non_blocking=True)
    else:
        mask = None

    # ========== 2. Scaling. ==========
    if need_resize:
        H_new, W_new = int(H * scale), int(W * scale)

        # Combine rgb and rays to reduce interpolation calls.
        combined = torch.cat([rgb, rays], dim=-1).permute(2, 0, 1).unsqueeze(0)
        combined_resized = F.interpolate(combined, size=(H_new, W_new), mode='bilinear', align_corners=False)
        combined_resized = combined_resized.squeeze(0).permute(1, 2, 0)

        rgb_resized = combined_resized[..., :3]
        rays_resized = F.normalize(combined_resized[..., 3:], dim=-1)

        distance_resized = F.interpolate(
            distance[None, None], size=(H_new, W_new), mode='bilinear', align_corners=False
        ).squeeze()

        if mask is not None:
            mask_resized = F.max_pool2d(
                mask[None, None].float(),
                kernel_size=int(1 / scale),
                stride=int(1 / scale)
            ).squeeze().bool()
            # Ensure the size matches.
            if mask_resized.shape != (H_new, W_new):
                mask_resized = F.interpolate(
                    mask[None, None].float(), size=(H_new, W_new), mode='nearest'
                ).squeeze().bool()
        else:
            mask_resized = None
    else:
        H_new, W_new = H, W
        rgb_resized = rgb
        rays_resized = F.normalize(rays, dim=-1)
        distance_resized = distance
        mask_resized = mask

    # ========== 3. Compute vertices on GPU. ==========
    vertices = (distance_resized.unsqueeze(-1) * rays_resized).reshape(-1, 3)
    vertex_colors = rgb_resized.reshape(-1, 3)

    # Synchronize and convert to NumPy.
    torch.cuda.synchronize() if device == 'cuda' else None
    vertices_np = vertices.cpu().numpy().astype(np.float64)
    colors_np = vertex_colors.cpu().numpy().astype(np.float64)

    # ========== 4. Generate faces on CPU/NumPy, which is faster. ==========
    faces_np = _generate_faces_numpy(H_new, W_new, mask_resized)

    # ========== 5. Create Open3D Mesh. ==========
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)

    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()

    # ========== 6. Boundary handling. ==========
    if connect_boundary_max_dist is not None and connect_boundary_max_dist > 0:
        mesh = _fill_small_boundary_spikes(mesh, connect_boundary_max_dist, connect_boundary_repeat_times)
        # Recompute normals after potential modification, if mesh still valid
        if mesh.has_triangles() and mesh.has_vertices():
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()  # Also computes triangle normals if vertex normals are computed

    return mesh


def _fill_small_boundary_spikes(
        mesh: o3d.geometry.TriangleMesh,
        max_bridge_dist: float,
        repeat_times: int = 3
) -> o3d.geometry.TriangleMesh:
    print(f"\t - DEBUG: Filling small boundary spikes with max_bridge_dist: {max_bridge_dist} and repeat_times: {repeat_times}")
    for iteration in range(repeat_times):
        if not mesh.has_triangles() or not mesh.has_vertices():
            return mesh

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # 1. Identify boundary edges
        edge_to_triangle_count = defaultdict(int)

        for tri_idx, tri in enumerate(triangles):
            for i in range(3):
                v1_idx, v2_idx = tri[i], tri[(i + 1) % 3]
                edge = tuple(sorted((v1_idx, v2_idx)))
                edge_to_triangle_count[edge] += 1

        boundary_edges = [edge for edge, count in edge_to_triangle_count.items() if count == 1]

        if not boundary_edges:
            return mesh

        # 2. Create an adjacency list for boundary vertices using only boundary edges
        boundary_adj = defaultdict(list)
        for v1_idx, v2_idx in boundary_edges:
            boundary_adj[v1_idx].append(v2_idx)
            boundary_adj[v2_idx].append(v1_idx)

        # 3. Process boundary vertices with new smooth filling algorithm
        new_triangles_list = []
        edge_added = defaultdict(bool)

        # print(f"DEBUG: Found {len(boundary_edges)} boundary edges.")
        # print(f"DEBUG: Max bridge distance set to: {max_bridge_dist}")

        new_triangles_added_count = 0

        for v_curr_idx, neighbors in boundary_adj.items():
            if len(neighbors) != 2:  # Only process vertices with exactly 2 boundary neighbors
                continue

            v_a_idx, v_b_idx = neighbors[0], neighbors[1]

            # Skip if these vertices already form a triangle
            potential_edge = tuple(sorted((v_a_idx, v_b_idx)))
            if edge_to_triangle_count[potential_edge] > 0 or edge_added[potential_edge]:
                continue

            # Calculate distances
            v_curr_coord = vertices[v_curr_idx]
            v_a_coord = vertices[v_a_idx]
            v_b_coord = vertices[v_b_idx]

            dist_a_b = np.linalg.norm(v_a_coord - v_b_coord)

            # Skip if distance exceeds threshold
            if dist_a_b > max_bridge_dist:
                continue

            # Create simple triangle (v_a, v_b, v_curr)
            new_triangles_list.append([v_a_idx, v_b_idx, v_curr_idx])
            new_triangles_added_count += 1
            edge_added[potential_edge] = True

            # Mark edges as processed
            edge_added[tuple(sorted((v_curr_idx, v_a_idx)))] = True
            edge_added[tuple(sorted((v_curr_idx, v_b_idx)))] = True

        # 4. Now process multi-step connections for better smoothing
        # First build boundary chains for multi-step connections
        boundary_loops = []
        visited_vertices = set()

        # Find boundary vertices with exactly 2 neighbors (part of continuous chains)
        chain_starts = [v for v in boundary_adj if len(boundary_adj[v]) == 2 and v not in visited_vertices]

        for start_vertex in chain_starts:
            if start_vertex in visited_vertices:
                continue

            chain = []
            curr_vertex = start_vertex

            # Follow the chain in one direction
            while curr_vertex not in visited_vertices:
                visited_vertices.add(curr_vertex)
                chain.append(curr_vertex)

                next_candidates = [n for n in boundary_adj[curr_vertex] if n not in visited_vertices]
                if not next_candidates:
                    break

                curr_vertex = next_candidates[0]

            if len(chain) >= 3:
                boundary_loops.append(chain)

        # print(f"DEBUG: Found {len(boundary_loops)} boundary chains for smoothing.")

        # Process each boundary chain for multi-step smoothing
        for chain in boundary_loops:
            chain_length = len(chain)

            # Skip very small chains
            if chain_length < 3:
                continue

            # Compute multi-step connections
            max_step = min(8, chain_length - 1)

            for i in range(chain_length):
                anchor_idx = chain[i]
                anchor_coord = vertices[anchor_idx]

                for step in range(3, max_step + 1):
                    if i + step >= chain_length:
                        break

                    far_idx = chain[i + step]
                    far_coord = vertices[far_idx]

                    # Check distance criteria
                    dist_anchor_far = np.linalg.norm(anchor_coord - far_coord)
                    if dist_anchor_far > max_bridge_dist * step:
                        continue

                    # Check if anchor and far are already connected
                    edge_anchor_far = tuple(sorted((anchor_idx, far_idx)))
                    if edge_to_triangle_count[edge_anchor_far] > 0 or edge_added[edge_anchor_far]:
                        continue

                    # Create fan triangles
                    fan_valid = True
                    fan_triangles = []

                    prev_mid_idx = anchor_idx

                    for j in range(1, step):
                        mid_idx = chain[i + j]

                        if prev_mid_idx != anchor_idx:
                            tri_edge1 = tuple(sorted((anchor_idx, mid_idx)))
                            tri_edge2 = tuple(sorted((prev_mid_idx, mid_idx)))

                            # Check if edges already exist (not created by our fan)
                            if (edge_to_triangle_count[tri_edge1] > 0 and not edge_added[tri_edge1]) or \
                                    (edge_to_triangle_count[tri_edge2] > 0 and not edge_added[tri_edge2]):
                                fan_valid = False
                                break

                            fan_triangles.append([anchor_idx, prev_mid_idx, mid_idx])

                        prev_mid_idx = mid_idx

                    # Add final triangle to connect to far_idx
                    if fan_valid:
                        fan_triangles.append([anchor_idx, prev_mid_idx, far_idx])

                    # Add all fan triangles if valid
                    if fan_valid and fan_triangles:
                        for triangle in fan_triangles:
                            v_a, v_b, v_c = triangle
                            edge_ab = tuple(sorted((v_a, v_b)))
                            edge_bc = tuple(sorted((v_b, v_c)))
                            edge_ac = tuple(sorted((v_a, v_c)))

                            new_triangles_list.append(triangle)
                            new_triangles_added_count += 1

                            edge_added[edge_ab] = True
                            edge_added[edge_bc] = True
                            edge_added[edge_ac] = True

                        # Once we've added a fan, move to the next anchor
                        break

        # print(f"DEBUG: Total new triangles added in iteration {iteration}: {new_triangles_added_count}")

        if new_triangles_added_count == 0:
            break

        # Update the mesh with new triangles
        if new_triangles_list:
            all_triangles_np = np.vstack((triangles, np.array(new_triangles_list, dtype=np.int32)))

            final_mesh = o3d.geometry.TriangleMesh()
            final_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            final_mesh.triangles = o3d.utility.Vector3iVector(all_triangles_np)

            if mesh.has_vertex_colors():
                final_mesh.vertex_colors = mesh.vertex_colors

            # Clean up the mesh
            final_mesh.remove_degenerate_triangles()
            final_mesh.remove_unreferenced_vertices()
            mesh = final_mesh

    return mesh


def get_view_point_from_panorama_point(global_pcd, w2c, K, image_h, image_w):
    # Get valid points corresponding to the current view.
    projected_uv, projected_depth = utils3d.numpy.project_cv(global_pcd.vertices, extrinsics=w2c, intrinsics=K)
    projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)
    projected_uv = projected_uv[projection_valid_mask]
    projected_uv[:, 0] = (projected_uv[:, 0] * image_w).round()
    projected_uv[:, 1] = (projected_uv[:, 1] * image_h).round()
    projected_uv = projected_uv.astype(np.int64)
    projected_uv[:, 0] = np.clip(projected_uv[:, 0], 0, image_w - 1)
    projected_uv[:, 1] = np.clip(projected_uv[:, 1], 0, image_h - 1)
    projected_depth = projected_depth[projection_valid_mask]

    projected_uv_1d = projected_uv[:, 1] * image_w + projected_uv[:, 0]  # Convert coordinates to 1D.
    original_indices = np.arange(projected_uv_1d.shape[0])
    # Double sort: for equal 1D coordinates, sort by depth ascending so only the nearest point is kept for each u, v.
    sorted_indices = np.lexsort((-projected_depth, projected_uv_1d))
    projected_uv_1d_sorted = projected_uv_1d[sorted_indices]
    sub_uv_1d = projected_uv_1d_sorted[:-1] - projected_uv_1d_sorted[1:]  # Offset subtraction; nonzero entries have the minimum depth.
    final_valid_indices = original_indices[sorted_indices][:-1][(sub_uv_1d != 0)]

    projected_points = global_pcd.vertices[projection_valid_mask][final_valid_indices]
    projected_uv = projected_uv[final_valid_indices]
    projected_colors = global_pcd.colors[projection_valid_mask][final_valid_indices, :3]

    return projected_points, projected_colors, projected_uv


def smooth_sky_depth_boundary(
        depth: torch.Tensor,
        sky_mask: torch.Tensor,
        transition_width: int = 50,
        depth_max: float = None,
        method: str = 'mean'  # 'mean', 'median', 'gaussian'
) -> torch.Tensor:
    """
    Smooth the depth transition between sky and foreground.

    Args:
        depth: Depth map, with sky regions already set to depth_max.
        sky_mask: Sky mask, where True indicates sky.
        transition_width: Transition-region width in pixels.
        depth_max: Sky depth value.
        method: Boundary diffusion method.
            - 'mean': Weighted mean diffusion, recommended.
            - 'median': Boundary median.
            - 'gaussian': Gaussian-blur diffusion.

    Returns:
        Depth map after boundary smoothing.
    """
    # 1. Dimension handling.
    original_dim = depth.dim()
    if original_dim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
        sky_mask = sky_mask.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:
        depth = depth.unsqueeze(0)
        sky_mask = sky_mask.unsqueeze(0)

    device = depth.device
    dtype = depth.dtype

    if depth_max is None:
        depth_max = depth.max().item()

    sky_mask_np = sky_mask.squeeze().cpu().numpy().astype(np.uint8)
    depth_np = depth.squeeze().cpu().numpy()

    if sky_mask_np.sum() == 0 or sky_mask_np.sum() == sky_mask_np.size:
        return _restore_dim(depth, original_dim)

    # ---------------------------------------------------------
    # 2. Compute the distance from each sky pixel to the boundary.
    # ---------------------------------------------------------
    dist_to_foreground = cv2.distanceTransform(sky_mask_np, cv2.DIST_L2, 5)

    # ---------------------------------------------------------
    # 3. Get boundary depth values according to the selected method.
    # ---------------------------------------------------------
    sky_mask_tensor = sky_mask.bool()

    if method == 'mean':
        boundary_depth = _mean_diffusion(depth, sky_mask_tensor, transition_width)
    elif method == 'median':
        boundary_depth = _median_boundary(depth_np, sky_mask_np, depth_max, device, dtype, depth.shape)
    elif method == 'gaussian':
        boundary_depth = _gaussian_diffusion(depth, sky_mask_tensor, depth_max, transition_width)
    else:
        raise ValueError(f"Unknown method: {method}")

    # ---------------------------------------------------------
    # 4. Linear interpolation: use diffused depth at the boundary and depth_max deeper in the sky.
    # ---------------------------------------------------------
    t = np.clip(dist_to_foreground / max(transition_width, 1), 0, 1)
    t = t * t * (3 - 2 * t)  # smoothstep
    t = torch.from_numpy(t).to(device=device, dtype=dtype).view_as(depth)

    interpolated = boundary_depth * (1 - t) + depth_max * t

    # ---------------------------------------------------------
    # 5. Clamp the range and modify only sky regions.
    # ---------------------------------------------------------
    interpolated = torch.clamp(interpolated, max=depth_max)
    result = torch.where(sky_mask_tensor, interpolated, depth)

    return _restore_dim(result, original_dim)


def _mean_diffusion(depth: torch.Tensor, sky_mask: torch.Tensor, transition_width: int) -> torch.Tensor:
    """
    Weighted mean diffusion, recommended.
    Preserves local depth characteristics while avoiding extreme values.
    """
    device = depth.device
    dtype = depth.dtype

    # Valid-region mask, where non-sky equals 1.
    valid_mask = (~sky_mask).float()

    # Initialize sky regions to 0.
    flood_depth = depth.clone()
    flood_depth[sky_mask] = 0

    kernel_size = 31
    pad = kernel_size // 2
    max_iterations = (transition_width // pad) + 10

    for _ in range(max_iterations):
        # Compute the sum of neighboring depths.
        depth_sum = F.avg_pool2d(flood_depth, kernel_size, stride=1, padding=pad) * (kernel_size ** 2)
        # Compute the number of valid neighboring pixels.
        valid_count = F.avg_pool2d(valid_mask, kernel_size, stride=1, padding=pad) * (kernel_size ** 2)

        # Mean = total sum / valid count.
        mean_depth = depth_sum / (valid_count + 1e-8)

        # Update only sky pixels whose neighborhoods contain valid pixels.
        can_update = sky_mask & (valid_count > 0)

        # Check whether any pixels still need updates.
        newly_filled = can_update & (valid_mask < 0.5)
        if not newly_filled.any():
            break

        # Update.
        flood_depth = torch.where(can_update, mean_depth, flood_depth)
        valid_mask = torch.where(can_update, torch.ones_like(valid_mask), valid_mask)

    # Fallback: fill any remaining zero regions with the global foreground mean.
    still_zero = (flood_depth == 0) & sky_mask
    if still_zero.any():
        fg_mean = depth[~sky_mask].mean()
        flood_depth = torch.where(still_zero, fg_mean, flood_depth)

    return flood_depth


def _median_boundary(depth_np: np.ndarray, sky_mask_np: np.ndarray,
                     depth_max: float, device, dtype, shape) -> torch.Tensor:
    """
    Boundary median method.
    Extract boundary depths and use their median as the uniform transition value.
    """
    # Extract the boundary: non-sky pixels adjacent to sky.
    kernel = np.ones((3, 3), np.uint8)
    dilated_sky = cv2.dilate(sky_mask_np, kernel)
    boundary_mask = (dilated_sky == 1) & (sky_mask_np == 0)

    boundary_depths = depth_np[boundary_mask]

    if len(boundary_depths) > 0:
        # Filter out values that are already depth_max.
        valid_depths = boundary_depths[boundary_depths < depth_max * 0.99]
        if len(valid_depths) > 0:
            median_depth = np.median(valid_depths)
        else:
            median_depth = np.median(boundary_depths)
    else:
        median_depth = depth_np[sky_mask_np == 0].mean()

    return torch.full(shape, median_depth, device=device, dtype=dtype)


def _gaussian_diffusion(depth: torch.Tensor, sky_mask: torch.Tensor,
                        depth_max: float, transition_width: int) -> torch.Tensor:
    """
    Gaussian-blur diffusion.
    Blur foreground depth first, then diffuse into sky regions.
    """
    device = depth.device
    dtype = depth.dtype

    # Fill sky with the foreground mean to avoid depth_max affecting the blur.
    fg_mean = depth[~sky_mask].mean()
    temp_depth = depth.clone()
    temp_depth[sky_mask] = fg_mean

    # Gaussian blur.
    blur_size = max(transition_width // 2, 3)
    if blur_size % 2 == 0:
        blur_size += 1

    from torchvision.transforms.functional import gaussian_blur
    blurred = gaussian_blur(temp_depth, kernel_size=[blur_size, blur_size], sigma=[blur_size / 4])

    # Iterative diffusion using min-pooling on blurred values, after extreme values have been smoothed.
    flood_depth = blurred.clone()
    flood_depth[sky_mask] = float('inf')

    kernel_size = 31
    pad = kernel_size // 2

    for _ in range(transition_width // pad + 5):
        if not torch.isinf(flood_depth[sky_mask]).any():
            break
        next_step = -F.max_pool2d(-flood_depth, kernel_size, stride=1, padding=pad)
        is_inf = torch.isinf(flood_depth)
        flood_depth = torch.where(is_inf, next_step, flood_depth)

    flood_depth = torch.where(torch.isinf(flood_depth), fg_mean, flood_depth)

    return flood_depth


def _restore_dim(tensor: torch.Tensor, original_dim: int) -> torch.Tensor:
    if original_dim == 2:
        return tensor.squeeze(0).squeeze(0)
    elif original_dim == 3:
        return tensor.squeeze(0)
    return tensor


def erp_distance_ray_to_normal(distance_map, ray_directions,
                               smooth_sigma=0.0,
                               facing_camera=True):
    """
    Convert an ERP distance map and ray directions to a world-coordinate normal map.

    Args:
        distance_map: (H, W) Distance map.
        ray_directions: (H, W, 3) Ray direction for each pixel, as unit vectors emitted from the origin.
        smooth_sigma: Smoothing sigma. 0 means no smoothing.
        facing_camera: True makes normals face the camera (origin); False makes them face outward.

    Returns:
        normal_map: (H, W, 3) World-coordinate normals in the range [-1, 1].
        normal_rgb: (H, W, 3) RGB visualization in the range [0, 255].

    Coordinate system (OpenCV):
        X: right
        Y: down
        Z: forward
    """
    # 0. Optional: smooth the distance map.
    if smooth_sigma > 0:
        distance_map = cv2.GaussianBlur(
            distance_map.astype(np.float32), (0, 0), smooth_sigma
        )

    # 1. Compute 3D point coordinates: P = ray * distance.
    d = distance_map.astype(np.float64)
    points = ray_directions * d[..., np.newaxis]  # (H, W, 3)

    # Split x, y, and z.
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    # 2. Compute tangent vectors.
    # Horizontal direction Tu uses cyclic boundaries because ERP wraps left to right.
    x_right = np.roll(x, -1, axis=1)
    y_right = np.roll(y, -1, axis=1)
    z_right = np.roll(z, -1, axis=1)

    x_left = np.roll(x, 1, axis=1)
    y_left = np.roll(y, 1, axis=1)
    z_left = np.roll(z, 1, axis=1)

    Tu_x = x_right - x_left
    Tu_y = y_right - y_left
    Tu_z = z_right - z_left

    # Vertical direction Tv does not wrap, so boundaries need special handling.
    x_down = np.roll(x, -1, axis=0)
    y_down = np.roll(y, -1, axis=0)
    z_down = np.roll(z, -1, axis=0)

    x_up = np.roll(x, 1, axis=0)
    y_up = np.roll(y, 1, axis=0)
    z_up = np.roll(z, 1, axis=0)

    Tv_x = x_down - x_up
    Tv_y = y_down - y_up
    Tv_z = z_down - z_up

    # Handle top and bottom boundaries near the poles.
    # Top (v=0): one-sided difference.
    Tv_x[0, :] = x[1, :] - x[0, :]
    Tv_y[0, :] = y[1, :] - y[0, :]
    Tv_z[0, :] = z[1, :] - z[0, :]

    # Bottom (v=H-1): one-sided difference.
    Tv_x[-1, :] = x[-1, :] - x[-2, :]
    Tv_y[-1, :] = y[-1, :] - y[-2, :]
    Tv_z[-1, :] = z[-1, :] - z[-2, :]

    # 3. Compute normals with a cross product: N = Tu × Tv.
    normal_x = Tu_y * Tv_z - Tu_z * Tv_y
    normal_y = Tu_z * Tv_x - Tu_x * Tv_z
    normal_z = Tu_x * Tv_y - Tu_y * Tv_x

    # 4. Normalize.
    norm = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
    norm = np.where(norm > 1e-10, norm, 1e-10)

    normal_x = normal_x / norm
    normal_y = normal_y / norm
    normal_z = normal_z / norm

    # 5. Ensure normals face the correct direction.
    if facing_camera:
        # Normals should face the camera (origin).
        # Check dot(normal, -ray) > 0, equivalent to dot(normal, ray) < 0.
        dot_product = (normal_x * ray_directions[..., 0] +
                       normal_y * ray_directions[..., 1] +
                       normal_z * ray_directions[..., 2])

        # If dot > 0, flip the normals.
        flip_mask = dot_product > 0
        normal_x = np.where(flip_mask, -normal_x, normal_x)
        normal_y = np.where(flip_mask, -normal_y, normal_y)
        normal_z = np.where(flip_mask, -normal_z, normal_z)

    # 6. Stack into (H, W, 3).
    normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)

    # 7. Convert to an RGB visualization.
    # [-1, 1] → [0, 255]
    normal_rgb = ((normal_map + 1.0) / 2.0 * 255).astype(np.uint8)

    return normal_map, normal_rgb
