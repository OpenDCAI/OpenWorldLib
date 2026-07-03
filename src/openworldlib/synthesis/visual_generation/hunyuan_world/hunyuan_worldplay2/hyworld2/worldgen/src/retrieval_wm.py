import collections
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import Tuple, List

import cv2
import einops
import numpy as np
import torch
import torch.distributed as dist
import trimesh
import utils3d
from PIL import Image
from moge.model.v2 import MoGeModel
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import RANSACRegressor
from torchvision import transforms
from tqdm import tqdm

from .depth_alignment import get_guided_depth_infos_v2, ConstrainedLinearRegression
from .general_utils import (
    point_padding,
    load_16bit_png_depth,
    save_16bit_png_depth,
    compute_normal_angles,
    rank0_log,
    color_print,
    sample_align_nframe,
    colorize_depth
)
from .pointcloud import depth2pcd


def statistical_outlier_removal(points, colors, nb_neighbors=20, std_ratio=2.0):
    """
    KNN-based Statistical Outlier Removal.
    For each point, compute the average distance to its K nearest neighbors. If that distance exceeds
    the global mean plus std_ratio times the global standard deviation, the point is treated as an outlier.

    Args:
        points: [N, 3] np.ndarray, point-cloud coordinates.
        colors: [N, 3] np.ndarray, point-cloud colors.
        nb_neighbors: int, number of KNN neighbors.
        std_ratio: float, threshold multiplier for the standard deviation.

    Returns:
        filtered_points: [M, 3] np.ndarray, filtered point-cloud coordinates.
        filtered_colors: [M, 3] np.ndarray, filtered point-cloud colors.
        inlier_mask: [N] bool np.ndarray, inlier mask.
    """
    if points.shape[0] <= nb_neighbors:
        return points, colors, np.ones(points.shape[0], dtype=bool)

    tree = cKDTree(points)
    # Query nb_neighbors + 1 neighbors including the point itself, then use the latter nb_neighbors distances.
    dists, _ = tree.query(points, k=nb_neighbors + 1)
    # dists[:, 0] is the self-distance (=0), so use dists[:, 1:] to compute the mean distance.
    mean_dists = np.mean(dists[:, 1:], axis=1)  # [N]

    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    threshold = global_mean + std_ratio * global_std

    inlier_mask = mean_dists <= threshold
    return points[inlier_mask], colors[inlier_mask], inlier_mask


def compute_depth_percentile_map(depth, depth_mask):
    """
    Compute each pixel's depth percentile within the valid region.

    Args:
        depth: [h, w] numpy array, depth map where smaller is nearer and larger is farther; valid values are > 0.
        depth_mask: [h, w] bool numpy array, where True indicates the valid region.

    Returns:
        percentile_map: [h, w] numpy array, depth percentile of each pixel (0-100).
                        Regions where mask is False are set to 0.
    """
    h, w = depth.shape
    percentile_map = np.zeros((h, w), dtype=np.float32)

    # Get depth values from the valid region.
    valid_depths = depth[depth_mask]  # [N]

    if len(valid_depths) == 0:
        return percentile_map

    # Sort valid depths.
    sorted_depths = np.sort(valid_depths)
    n_valid = len(sorted_depths)

    # Use searchsorted to find each depth's rank.
    # 'right' finds the first position greater than the value.
    ranks = np.searchsorted(sorted_depths, depth[depth_mask], side='right')

    # Compute percentile: rank / n * 100.
    percentiles = (ranks / n_valid) * 100.0

    # Fill the result.
    percentile_map[depth_mask] = percentiles

    return percentile_map


def calculate_camera_distance(cam1_extrinsic, cam2_extrinsic):
    """Calculate distance between two camera poses using translation and rotation

    Args:
        cam1_extrinsic: Camera 1 extrinsic matrix [B, 4, 4] or [4, 4]
        cam2_extrinsic: Camera 2 extrinsic matrix [B, 4, 4] or [4, 4]

    Returns:
        total_dist: [B] or scalar - Combined translation and rotation distance
    """
    # Handle both batched and single input
    is_batched = cam1_extrinsic.dim() == 3
    if not is_batched:
        cam1_extrinsic = cam1_extrinsic.unsqueeze(0)
        cam2_extrinsic = cam2_extrinsic.unsqueeze(0)

    # Extract translation vectors
    t1 = cam1_extrinsic[:, :3, 3]  # [B, 3]
    t2 = cam2_extrinsic[:, :3, 3]  # [B, 3]

    # Calculate translation distance
    translation_dist = torch.norm(t1 - t2, dim=1)  # [B]

    # Extract rotation matrices
    R1 = cam1_extrinsic[:, :3, :3]  # [B, 3, 3]
    R2 = cam2_extrinsic[:, :3, :3]  # [B, 3, 3]

    # Calculate rotation distance using Frobenius norm
    rotation_dist = torch.norm(R1 - R2, p='fro', dim=(1, 2))  # [B]

    # Combine translation and rotation distances
    total_dist = translation_dist + 0.1 * rotation_dist  # [B]

    if not is_batched:
        total_dist = total_dist.item()

    return total_dist


def get_camera_frustum_corners(K, extrinsic, image_width, image_height, depth_range=(0.1, 100.0)):
    """Get the 8 corners of camera frustum in world coordinates

    Args:
        K: Intrinsic matrix [B, 3, 3] or [3, 3]
        extrinsic: Extrinsic matrix (world to camera) [B, 4, 4] or [4, 4]
        image_width: Image width (scalar or tensor [B])
        image_height: Image height (scalar or tensor [B])
        depth_range: (near, far) depth range

    Returns:
        corners: [B, 8, 3] or [8, 3] tensor of frustum corners in world coordinates
    """
    near, far = depth_range

    # Handle both batched and single input
    is_batched = K.dim() == 3
    if not is_batched:
        K = K.unsqueeze(0)
        extrinsic = extrinsic.unsqueeze(0)

    batch_size = K.shape[0]
    device = K.device

    # Get camera center and rotation
    c2w = torch.inverse(extrinsic)  # [B, 4, 4] Camera to world
    camera_center = c2w[:, :3, 3]  # [B, 3]
    R = c2w[:, :3, :3]  # [B, 3, 3]

    # Get inverse intrinsic
    K_inv = torch.inverse(K)  # [B, 3, 3]

    # Handle scalar or tensor image dimensions
    if not isinstance(image_width, torch.Tensor):
        image_width = torch.tensor([image_width] * batch_size, device=device, dtype=torch.float32)
    if not isinstance(image_height, torch.Tensor):
        image_height = torch.tensor([image_height] * batch_size, device=device, dtype=torch.float32)

    # Define image corners in pixel coordinates for all batches
    # corners_2d: [B, 4, 3]
    corners_2d = torch.stack([
        torch.stack([torch.zeros(batch_size, device=device), torch.zeros(batch_size, device=device), torch.ones(batch_size, device=device)], dim=1),
        torch.stack([image_width, torch.zeros(batch_size, device=device), torch.ones(batch_size, device=device)], dim=1),
        torch.stack([image_width, image_height, torch.ones(batch_size, device=device)], dim=1),
        torch.stack([torch.zeros(batch_size, device=device), image_height, torch.ones(batch_size, device=device)], dim=1)
    ], dim=1)  # [B, 4, 3]

    # Unproject to normalized camera coordinates
    # K_inv: [B, 3, 3], corners_2d: [B, 4, 3]
    ray_dirs_cam = torch.bmm(corners_2d, K_inv.transpose(1, 2))  # [B, 4, 3]

    # Normalize ray directions
    ray_dirs_cam = ray_dirs_cam / torch.norm(ray_dirs_cam, dim=2, keepdim=True)  # [B, 4, 3]

    # Scale to near and far depths
    corners_cam_near = ray_dirs_cam * near  # [B, 4, 3]
    corners_cam_far = ray_dirs_cam * far  # [B, 4, 3]

    # Transform to world coordinates
    # R: [B, 3, 3], corners_cam_near: [B, 4, 3]
    corners_world_near = torch.bmm(corners_cam_near, R.transpose(1, 2)) + camera_center.unsqueeze(1)  # [B, 4, 3]
    corners_world_far = torch.bmm(corners_cam_far, R.transpose(1, 2)) + camera_center.unsqueeze(1)  # [B, 4, 3]

    # Combine near and far corners
    corners_world = torch.cat([corners_world_near, corners_world_far], dim=1)  # [B, 8, 3]

    if not is_batched:
        corners_world = corners_world.squeeze(0)  # [8, 3]

    return corners_world


def calculate_frustum_volume_overlap(corners1, corners2):
    """Calculate approximate overlap between two frustums using their corner points

    Args:
        corners1: [B, 8, 3] or [8, 3] tensor of frustum 1 corners
        corners2: [B, 8, 3] or [8, 3] tensor of frustum 2 corners

    Returns:
        overlap_score: [B] or scalar - Approximate overlap score (0 to 1)
    """
    # Handle both batched and single input
    is_batched = corners1.dim() == 3
    if not is_batched:
        corners1 = corners1.unsqueeze(0)
        corners2 = corners2.unsqueeze(0)

    # Calculate bounding boxes for both frustums
    min1 = torch.min(corners1, dim=1)[0]  # [B, 3]
    max1 = torch.max(corners1, dim=1)[0]  # [B, 3]
    min2 = torch.min(corners2, dim=1)[0]  # [B, 3]
    max2 = torch.max(corners2, dim=1)[0]  # [B, 3]

    # Calculate intersection of bounding boxes
    intersection_min = torch.max(min1, min2)  # [B, 3]
    intersection_max = torch.min(max1, max2)  # [B, 3]

    # Check if there is an intersection
    intersection_valid = torch.all(intersection_max > intersection_min, dim=1)  # [B]

    # Calculate volumes
    intersection_dims = torch.clamp(intersection_max - intersection_min, min=0.0)  # [B, 3]
    intersection_volume = torch.prod(intersection_dims, dim=1)  # [B]

    volume1 = torch.prod(max1 - min1, dim=1)  # [B]
    volume2 = torch.prod(max2 - min2, dim=1)  # [B]

    # Calculate overlap ratio (intersection over union)
    union_volume = volume1 + volume2 - intersection_volume  # [B]
    overlap_score = intersection_volume / (union_volume + 1e-8)  # [B]

    # Set overlap to 0 where there's no valid intersection
    overlap_score = torch.where(intersection_valid, overlap_score, torch.zeros_like(overlap_score))

    if not is_batched:
        overlap_score = overlap_score.item()

    return overlap_score


def calculate_fov_overlap(cam1_intrinsic, cam1_extrinsic, cam2_intrinsic, cam2_extrinsic, image_width, image_height, near, far):
    """Calculate FOV overlap between two cameras using frustum intersection

    This function constructs frustums from near and far planes and calculates their overlap.
    The overlap ratio represents the proportion of cam1's view that is also visible in cam2.
    Supports batched computation for efficiency.

    Args:
        cam1_intrinsic: Intrinsic matrix of camera 1 [B, 3, 3] or [3, 3]
        cam1_extrinsic: Extrinsic matrix of camera 1 [B, 4, 4] or [4, 4]
        cam2_intrinsic: Intrinsic matrix of camera 2 [B, 3, 3] or [3, 3]
        cam2_extrinsic: Extrinsic matrix of camera 2 [B, 4, 4] or [4, 4]
        image_width: Image width (scalar or tensor [B])
        image_height: Image height (scalar or tensor [B])

    Returns:
        overlap_ratio: [B] or scalar - Ratio of overlapping view (0 to 1)
        angle_between_cameras: [B] or scalar - Angle between camera viewing directions (degrees)
    """
    # Handle both batched and single input
    is_batched = cam1_intrinsic.dim() == 3
    if not is_batched:
        cam1_intrinsic = cam1_intrinsic.unsqueeze(0)
        cam1_extrinsic = cam1_extrinsic.unsqueeze(0)
        cam2_intrinsic = cam2_intrinsic.unsqueeze(0)
        cam2_extrinsic = cam2_extrinsic.unsqueeze(0)

    # Get camera centers and viewing directions
    c2w1 = torch.inverse(cam1_extrinsic)  # [B, 4, 4]
    c2w2 = torch.inverse(cam2_extrinsic)  # [B, 4, 4]

    # Camera viewing direction is the negative z-axis in camera space
    cam1_view_dir = c2w1[:, :3, 2]  # [B, 3] Third column of rotation matrix
    cam2_view_dir = c2w2[:, :3, 2]  # [B, 3]

    # Calculate angle between viewing directions
    cos_angle = torch.sum(cam1_view_dir * cam2_view_dir, dim=1)  # [B]
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle_between = torch.rad2deg(torch.acos(cos_angle))  # [B]

    # Construct frustums using near and far planes
    depth_range = (near, far)  # near and far planes

    # Get frustum corners for both cameras (batched)
    corners1 = get_camera_frustum_corners(
        cam1_intrinsic, cam1_extrinsic, image_width, image_height, depth_range
    )  # [B, 8, 3]
    corners2 = get_camera_frustum_corners(
        cam2_intrinsic, cam2_extrinsic, image_width, image_height, depth_range
    )  # [B, 8, 3]

    # Calculate frustum overlap (batched)
    overlap_ratio = calculate_frustum_volume_overlap(corners1, corners2)  # [B]

    if not is_batched:
        overlap_ratio = overlap_ratio.item() if isinstance(overlap_ratio, torch.Tensor) else overlap_ratio
        angle_between = angle_between.item()

    return overlap_ratio, angle_between


def find_closest_camera_in_view(target_extrinsic, ref_extrinsics, target_intrinsic, ref_intrinsics,
                                image_width, image_height, method="distance", near=0.1, far=5.0, angle_penalty=False,
                                shortcut_index=None, topk_return=0):
    """Find the camera in reference views that is closest to the target camera

    Args:
        target_extrinsic: Target camera extrinsic matrix [4, 4]
        ref_extrinsics: Reference camera extrinsic matrices [N, 4, 4]
        target_intrinsic: Target camera intrinsic matrix [3, 3]
        ref_intrinsics: Reference camera intrinsic matrices [N, 3, 3]
        image_width: Image width
        image_height: Image height
        method: "distance" for pose-based matching, "fov_overlap" for FOV overlap-based matching

    Returns:
        best_idx: Index of the closest camera in ref_extrinsics
        best_score: Score of the best match
    """
    num_refs = ref_extrinsics.shape[0]

    if num_refs == 0:
        return None, float('inf') if method == "distance" else -1.0

    # Expand target to match batch size
    target_extrinsic_batch = target_extrinsic.unsqueeze(0).expand(num_refs, -1, -1)  # [N, 4, 4]

    if method == "distance":
        # Batch calculate distances
        distances = calculate_camera_distance(target_extrinsic_batch, ref_extrinsics)  # [N]

        # Find the camera with minimum distance
        min_distance_idx = torch.argmin(distances)
        min_distance = distances[min_distance_idx].item()

        return min_distance_idx.item(), min_distance

    elif method == "fov_overlap":
        # Expand target intrinsic to match batch size
        target_intrinsic_batch = target_intrinsic.unsqueeze(0).expand(num_refs, -1, -1)  # [N, 3, 3]

        # Batch calculate FOV overlap
        overlap_ratios, angle_betweens = calculate_fov_overlap(
            target_intrinsic_batch, target_extrinsic_batch,
            ref_intrinsics, ref_extrinsics,
            image_width, image_height,
            near=near, far=far
        )  # overlap_ratios: [N], angle_betweens: [N]

        angle_betweens[angle_betweens < 0] = -angle_betweens[angle_betweens < 0]
        angle_betweens[angle_betweens > 180] = 360 - angle_betweens[angle_betweens > 180]

        if angle_penalty:
            # Penalize cameras with large angle difference
            overlap_ratios = overlap_ratios * torch.clip(torch.exp(((-angle_betweens + 90) / 180.0) * 5.0), 0.0, 1.0)

        if shortcut_index is not None:
            overlap_ratios[shortcut_index] += 1.0

        if topk_return == 0:
            # Find the camera with maximum overlap
            max_overlap_idx = torch.argmax(overlap_ratios)
            max_overlap = overlap_ratios[max_overlap_idx].item()
            angle_between = angle_betweens[max_overlap_idx].item()

            return max_overlap_idx.item(), max_overlap, angle_between
        else:
            max_overlap_indices = torch.topk(overlap_ratios, topk_return, dim=0, sorted=True, largest=True).indices
            return max_overlap_indices.tolist()

    else:
        raise ValueError(f"Unknown method: {method}. Use 'distance' or 'fov_overlap'")


class CameraSelector:
    """Representative camera selector that fuses camera extrinsics and image features."""

    def __init__(
            self,
            feature_extractor='dinov2',
            device: str = 'cuda'
    ):
        self.device = device
        self.feature_extractor = feature_extractor
        self.model = None
        self.transform = None

        self._load_model(feature_extractor)

    def _load_model(self, extractor: str):
        """Load the pretrained model."""
        if extractor == 'dinov2':
            from transformers import AutoImageProcessor, AutoModel
            model_path = 'facebook/dinov2-base'
            self.processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
        else:
            raise ValueError(f"Unknown feature extractor: {extractor}")

    @torch.no_grad()
    def extract_image_features(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract image features.

        Args:
            images: List of (H, W, 3) RGB images

        Returns:
            features: (N, D) feature vectors.
        """

        features = []
        for img in images:
            # RGB -> PIL
            img_pil = Image.fromarray(img)

            # Extract
            if self.feature_extractor == 'dinov2':
                with torch.no_grad():
                    inputs = self.processor(images=img_pil, return_tensors="pt")
                    inputs.pixel_values = inputs.pixel_values.to(self.device)
                    feat = self.model(pixel_values=inputs.pixel_values).pooler_output  # (1, 768)
            else:
                raise ValueError(f"Unknown feature extractor: {self.feature_extractor}")

            features.append(feat.cpu().numpy().flatten())

        return np.array(features)

    def compute_image_quality_scores(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Compute image quality/information scores.
        Used to bias FPS sampling toward high-quality images.
        """
        scores = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Sharpness, measured by Laplacian variance.
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Information entropy.
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist = hist / hist.sum() + 1e-10
            entropy = -np.sum(hist * np.log2(hist))

            # Contrast.
            contrast = gray.std()

            # Combined score.
            score = 0.4 * np.log1p(sharpness) + 0.3 * entropy + 0.3 * (contrast / 128.0)
            scores.append(score)

        return np.array(scores)

    def select(
            self,
            extrinsics: np.ndarray,
            images: List[np.ndarray],
            topk: int,
            camera_weight: float = 0.3,
            image_weight: float = 0.7,
            quality_bias: float = 0.1,  # Quality preference weight.
    ) -> Tuple[np.ndarray, dict]:
        """
        Select representative cameras by combining camera extrinsics and image features.

        Args:
            extrinsics: (N, 4, 4) world2camera matrices.
            images: List of (H, W, 3) RGB images
            topk: Number to select.
            camera_weight: Camera feature weight.
            image_weight: Image feature weight.
            quality_bias: Image quality preference; larger values favor higher-quality images.

        Returns:
            indices: Selected indices.
            info: Additional information.
        """
        N = len(images)
        images = [im[0] for im in images]
        assert extrinsics.shape[0] == N
        assert topk <= N

        # 1. Extract camera features.
        camera_features, positions = self._extract_camera_features(extrinsics)

        # 2. Extract image features.
        image_features = self.extract_image_features(images)

        # 3. Compute image quality scores.
        quality_scores = self.compute_image_quality_scores(images)

        # 4. Normalize.
        camera_features = self._normalize(camera_features)
        image_features = self._normalize(image_features)

        # 5. Fuse features.
        combined_features = np.concatenate([
            camera_features * camera_weight,
            image_features * image_weight
        ], axis=1)

        # 6. FPS with quality bias.
        indices = self._quality_aware_fps(combined_features, quality_scores, topk, quality_bias)

        info = {
            'positions': positions,
            'quality_scores': quality_scores,
            'selected_quality_scores': quality_scores[indices],
        }

        return indices, info

    def _extract_camera_features(self, extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract camera features, including position and orientation."""
        N = extrinsics.shape[0]
        positions = []
        orientations = []

        for i in range(N):
            w2c = extrinsics[i]
            R_mat = w2c[:3, :3]
            t = w2c[:3, 3]

            cam_pos = -R_mat.T @ t
            positions.append(cam_pos)

            quat = R.from_matrix(R_mat).as_quat()
            if quat[3] < 0:
                quat = -quat
            orientations.append(quat)

        positions = np.array(positions)
        orientations = np.array(orientations)
        features = np.concatenate([positions, orientations], axis=1)

        return features, positions

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize."""
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        return (features - mean) / std

    def _quality_aware_fps(
            self,
            features: np.ndarray,
            quality_scores: np.ndarray,
            k: int,
            quality_bias: float
    ) -> np.ndarray:
        """
        Quality-aware farthest point sampling.
        Add a quality penalty to distances so low-quality images are less likely to be selected.
        """
        N = features.shape[0]
        selected = []
        distances = np.full(N, np.inf)

        # Normalize quality scores.
        q_min, q_max = quality_scores.min(), quality_scores.max()
        q_normalized = (quality_scores - q_min) / (q_max - q_min + 1e-8)

        # Quality adjustment factor: low-quality images have smaller effective distances.
        quality_factor = 1.0 + quality_bias * (q_normalized - 0.5)

        # Start from the point closest to the centroid while favoring higher quality.
        centroid = features.mean(axis=0)
        centroid_dist = np.linalg.norm(features - centroid, axis=1)
        start_score = -centroid_dist + quality_bias * q_normalized
        current = np.argmax(start_score)
        selected.append(current)

        for _ in range(k - 1):
            dist_to_current = np.linalg.norm(features - features[current], axis=1)
            distances = np.minimum(distances, dist_to_current)

            # Adjusted distances: high-quality images get larger effective distances.
            adjusted_distances = distances * quality_factor

            # Set already selected points to -inf.
            adjusted_distances[selected] = -np.inf

            current = np.argmax(adjusted_distances)
            selected.append(current)

        return np.array(selected)


def voxel_downsample_fixed(points, colors, voxel_size):
    """
    Downsample a point cloud with a fixed voxel_size using a pure NumPy hash implementation without Open3D.
    Use the first point that falls into each voxel as the representative point to avoid Open3D object
    construction/destruction and data-format conversion overhead.
    Args:
        points: numpy array [N, 3]
        voxel_size: float, voxel size.
        colors: numpy array [N, 3] or None, point-cloud colors, uint8 or float.
    Returns:
        If colors is None: ds_points numpy array [M, 3].
        If colors is not None: (ds_points [M, 3], ds_colors [M, 3]).
    """
    # Discretize point coordinates into integer voxel coordinates for fast hash-based deduplication.
    voxel_coords = np.floor(points / voxel_size).astype(np.int64)
    # Encode 3D voxel coordinates as a single int64 hash value.
    hash_keys = voxel_coords[:, 0] * 116101979 + voxel_coords[:, 1] * 104729 + voxel_coords[:, 2]
    # Use the first point index in each unique voxel as the representative point.
    _, unique_indices = np.unique(hash_keys, return_index=True)
    ds_points = points[unique_indices]
    if colors is not None:
        return ds_points, colors[unique_indices]
    return ds_points


def adaptive_voxel_downsample(points, colors=None, N_points=1_000_000, tol=0.2, max_iter=10):
    """
    Adaptive voxel downsampling: use hash-accelerated binary search for voxel_size so the downsampled
    point count is close to N_points.
    During search, use spatial hashing to quickly count unique voxels; after selecting voxel_size, call
    voxel_downsample_fixed to perform downsampling.
    Args:
        points: numpy array [N, 3], point-cloud coordinates.
        colors: numpy array [N, 3] or None, point-cloud colors, uint8 or float.
        N_points: Target number of points.
        tol: Allowed relative error, default 20%.
        max_iter: Maximum number of binary-search iterations.
    Returns:
        ds_points: Downsampled point-cloud coordinates [M, 3].
        ds_colors: Downsampled colors [M, 3], if input colors is not None.
    """

    n_total = points.shape[0]
    if n_total <= N_points:
        return (points, colors, 0.003) if colors is not None else (points, None, 0.003)

    # Compute bbox extent in pure NumPy to set the binary-search upper bound.
    bbox_extent = points.max(axis=0) - points.min(axis=0)
    voxel_lo = 1e-4
    voxel_hi = float(bbox_extent.max()) / 100

    best_voxel_size = (voxel_lo + voxel_hi) / 2.0
    for _ in range(max_iter):
        voxel_mid = (voxel_lo + voxel_hi) / 2.0
        # Hash counting: discretize coordinates into integer voxels and deduplicate with NumPy unique.
        voxel_coords = np.floor(points / voxel_mid).astype(np.int64)
        # Encode 3D voxel coordinates as one int64 hash value to speed up unique.
        hash_keys = voxel_coords[:, 0] * 116101979 + voxel_coords[:, 1] * 104729 + voxel_coords[:, 2]
        n_voxels = np.unique(hash_keys).shape[0]

        if abs(n_voxels - N_points) / N_points <= tol:
            best_voxel_size = voxel_mid
            break

        if n_voxels > N_points:
            voxel_lo = voxel_mid
        else:
            voxel_hi = voxel_mid
        best_voxel_size = voxel_mid

    # Finally call voxel_downsample_fixed to get the actual downsampled result.
    result = voxel_downsample_fixed(points, colors, best_voxel_size)
    if colors is not None:
        ds_points, ds_colors = result
    else:
        ds_points = result
        ds_colors = None
    rank0_log(f"Voxel downsample: {n_total} -> {ds_points.shape[0]} points (target: {N_points}, voxel_size: {best_voxel_size:.6f})")

    return ds_points, ds_colors, best_voxel_size


class PanoramaMemoryBank:
    def __init__(self, root_path, image_width, image_height, device,
                 nframe=21, max_reference=8, align_nframe=8,
                 rank=0, world_size=1,
                 moge_model=None, sam3_model=None, sam3_processor=None,
                 camera_selector="dinov2",
                 results_name=None,
                 valid_threshold=0.3,
                 apply_normal=True,
                 pts_num=2_000_000,
                 percentile_threshold=20,
                 kb_anomaly_percentile=90,
                 pcd_nb_neighbors=10,
                 pcd_std_ratio=2.0):
        # loading panorama info
        self.root_path = root_path
        self.image_width = image_width
        self.image_height = image_height
        self.pcd_nb_neighbors = pcd_nb_neighbors
        self.pcd_std_ratio = pcd_std_ratio
        self.meta_info = json.load(open(f"{root_path}/meta_info.json"))
        self.global_pcd = trimesh.load(f"{root_path}/render_results/global_pcd.ply")

        # Voxel-downsample global_pcd to determine the initial voxel_size.
        global_pcd_points_sampled, global_pcd_colors_sampled, voxel_size = adaptive_voxel_downsample(
            self.global_pcd.vertices.astype(np.float64),
            colors=self.global_pcd.colors[:, :3],
            N_points=pts_num
        )
        self.global_pcd_sampled = trimesh.PointCloud(vertices=global_pcd_points_sampled, colors=global_pcd_colors_sampled)
        self.voxel_size = voxel_size
        self.sky_pcd_sampled = None
        if os.path.exists(f"{root_path}/render_results/sky_pcd.ply"):
            sky_pcd = trimesh.load(f"{root_path}/render_results/sky_pcd.ply")
            if hasattr(sky_pcd, "vertices") and sky_pcd.vertices.shape[0] > 0:
                sky_pcd_points_sampled, sky_pcd_colors_sampled = voxel_downsample_fixed(points=sky_pcd.vertices, colors=sky_pcd.colors[:, :3], voxel_size=voxel_size)
                self.sky_pcd_sampled = trimesh.PointCloud(vertices=sky_pcd_points_sampled, colors=sky_pcd_colors_sampled)

        self.global_normal = None
        if apply_normal:
            if os.path.exists(f"{root_path}/render_results/global_normal.npy"):
                self.global_normal = np.load(f"{root_path}/render_results/global_normal.npy")
                self.global_normal = torch.from_numpy(self.global_normal).to(dtype=torch.float32)
            else:
                rank0_log(f"No global normal found in {root_path}/render_results/global_normal.npy!", "WARNING")
        ground_mask = (np.array(Image.open(f"{root_path}/render_results/sky_mask.png")) / 255).astype(np.bool_)
        full_depth = torch.load(f"{root_path}/render_results/full_depth_prediction.pt", weights_only=False, map_location="cpu")
        self.min_d_ = full_depth["distance"][ground_mask].min().item()
        self.max_d_ = full_depth["distance"][ground_mask].max().item()
        self.max_d = self.max_d_ * 1.1
        self.depth_median = full_depth["distance"][ground_mask].median().item()
        self.nframe = nframe
        self.align_nframe = align_nframe  # Frames used for alignment, uniformly sampled.
        self.valid_threshold = valid_threshold  # Reasonable coverage ratio for guidance depth.
        self.percentile_threshold = percentile_threshold  # percentile threshold for depth
        self.kb_anomaly_percentile = kb_anomaly_percentile  # Percentile threshold for abnormal k,b detection; P90 means P90 of max_relative_deviation is the inlier upper bound.

        self.max_reference = max_reference
        self.camera_selector = CameraSelector(camera_selector, device=device)

        if results_name is None:
            self.results_path = "generation_bank"
        else:
            self.results_path = f"generation_bank_{results_name}"

        # predefine 2d points
        x = torch.arange(image_width).float()
        y = torch.arange(image_height).float()
        points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        points = einops.rearrange(points, 'w h c -> (h w) c')
        self.points = point_padding(points).to(device)

        # build panoramic memory bank
        memory_bank_path = f"{root_path}/render_results/pano_bank"
        memory_cameras = json.load(open(f"{memory_bank_path}/cameras.json"))

        # We share the same memory bank. Do not store intermediate results for now; preload them into memory.
        ref_image_list = glob(f"{memory_bank_path}/images/*.png")
        ref_image_list = sorted(ref_image_list)
        ref_depth_list = glob(f"{memory_bank_path}/depths/*.png")
        ref_depth_list = sorted(ref_depth_list)

        tasks = [(i, d, memory_cameras) for i, d in zip(ref_image_list, ref_depth_list)]

        def _load_item(args):
            img_p, dep_p, cam_dict = args
            key = img_p.split('/')[-1].split('.')[0]
            view_id, traj_id = img_p.split('/')[-4], img_p.split('/')[-3]
            fname = f"{view_id}/{traj_id}/{key}"
            return (
                np.array(cam_dict[key]['extrinsic']),
                np.array(cam_dict[key]['intrinsic']),
                Image.open(img_p).convert('RGB'),
                load_16bit_png_depth(dep_p),
                fname
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            # map keeps result order consistent with task order.
            results = list(tqdm(executor.map(_load_item, tasks), total=len(tasks), desc="Loading memory bank...", disable=rank != 0))

        if results:
            self.ref_w2cs, self.ref_Ks, self.ref_frames, self.ref_depths, self.fnames = map(list, zip(*results))
        else:
            self.ref_w2cs, self.ref_Ks, self.ref_frames, self.ref_depths, self.fnames = [], [], [], [], []

        self.ref_w2cs = torch.from_numpy(np.stack(self.ref_w2cs, axis=0)).to(dtype=torch.float32, device=device)
        self.ref_Ks = torch.from_numpy(np.stack(self.ref_Ks, axis=0)).to(dtype=torch.float32, device=device)
        self.mem_size = len(self.ref_frames)
        self.align_start_index = self.mem_size
        rank0_log(f"Initialized panorama memory size: {self.mem_size}")

        rank0_log(f"Initializing Moge Model...")
        if moge_model is None:
            self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
        else:
            self.moge_model = moge_model
        self.moge_model.eval()

        rank0_log(f"Initializing SAM3 Model...")
        if sam3_model is None or sam3_processor is None:
            from transformers import Sam3VideoModel, Sam3VideoProcessor
            self.sam3_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
            self.sam3_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
        else:
            self.sam3_model = sam3_model
            self.sam3_processor = sam3_processor

        self.min_depth = 0.05
        self.device = device
        self.rank = rank
        self.world_size = world_size

    def export_pcd(self, save_path, N_points=2_000_000):
        """
        Export point clouds after adaptive voxel downsampling with adaptive_voxel_downsample, then save with trimesh.
        For indoor scenes, use only aligned_pcd, but supplement pole-adjacent points where xy is near 0 from global_pcd.
        """
        if self.rank == 0:
            full_pcds = []
            full_colors = []
            for vname in self.global_points.keys():
                full_pcds.append(self.global_points[vname]["points"])
                full_colors.append(self.global_points[vname]["colors"])
            if len(full_pcds) == 0:
                full_pcds = np.zeros((1, 3), dtype=np.float32)
                full_colors = np.zeros((1, 3), dtype=np.uint8)
            else:
                full_pcds = np.concatenate(full_pcds, axis=0)
                full_colors = np.concatenate(full_colors, axis=0)

            aligned_points, aligned_colors = voxel_downsample_fixed(
                full_pcds, colors=full_colors, voxel_size=self.voxel_size
            )
            origin_aligned_num = aligned_points.shape[0]

            if aligned_points.shape[0] > N_points:
                rdv = np.random.choice(aligned_points.shape[0], N_points, replace=False)
                aligned_points = aligned_points[rdv]
                aligned_colors = aligned_colors[rdv]

            aligned_pcd = trimesh.PointCloud(vertices=aligned_points, colors=aligned_colors)
            aligned_pcd.export(f"{save_path}/aligned_pcd.ply")

            # Process global_pcd: for indoor scenes, keep only point clouds near the poles.
            global_pcd_vertices = self.global_pcd_sampled.vertices  # [N, 3]
            global_pcd_colors = self.global_pcd_sampled.colors[:, :3] if self.global_pcd_sampled.colors.shape[1] >= 3 else self.global_pcd_sampled.colors

            if self.meta_info["scene_type"] == "indoor":
                # Compute each point's distance to the origin in the xy plane.
                xy_dist = np.sqrt(global_pcd_vertices[:, 0] ** 2 + global_pcd_vertices[:, 1] ** 2)
                # Use Gaussian decay as the sampling probability, with sigma = depth_median / 3.
                # When xy_dist = depth_median, probability = exp(-9/2) ≈ 0.011, close to 0.
                sigma = self.depth_median / 3.0
                sample_prob = np.exp(-0.5 * (xy_dist / sigma) ** 2)
                # Normalize to a probability distribution.
                sample_prob = sample_prob / sample_prob.sum()
                # Sampling count: min(original count, reasonable upper bound), avoiding more samples than total points.
                n_global = global_pcd_vertices.shape[0]
                n_sample = min(n_global, max(int(n_global * 0.1), 10000))  # Keep at most 20% of global points.
                sampled_indices = np.random.choice(n_global, size=n_sample, replace=False, p=sample_prob)
                global_pcd_export = trimesh.PointCloud(
                    vertices=global_pcd_vertices[sampled_indices],
                    colors=global_pcd_colors[sampled_indices]
                )
                color_print(f"[Indoor] Global PCD: {n_global} -> {n_sample} points (polar补充, sigma={sigma:.4f}, depth_median={self.depth_median:.4f})", "info")
            else:
                global_pcd_export = self.global_pcd_sampled

            global_pcd_export.export(f"{save_path}/global_pcd.ply")

            if self.sky_pcd_sampled is not None:
                self.sky_pcd_sampled.export(f"{save_path}/sky_pcd.ply")

            # Save point-cloud counts and voxel_size information.
            pcd_info = {
                "scene_type": self.meta_info["scene_type"],
                "global_pcd": {"num": global_pcd_export.vertices.shape[0], "voxel_size": self.voxel_size},
                "aligned_pcd": {"num_voxel": origin_aligned_num, "num_final": aligned_pcd.vertices.shape[0], "voxel_size": self.voxel_size},
                "sky_pcd": {"num": 0 if self.sky_pcd_sampled is None else self.sky_pcd_sampled.vertices.shape[0], "voxel_size": self.voxel_size}
            }

            with open(f"{save_path}/pcd_info.json", "w") as f:
                json.dump(pcd_info, f)

    def retrieval(self, tar_w2cs_full, tar_Ks_full, view_id=None, traj_id=None):
        """
        Return: retrieved_frames, ref_index, ref_index_dict
        """

        # For certain aerial tracking trajectories, always use part of the previous generation as retrieval results.
        if ("wonder" in view_id or "target" in view_id) and traj_id > "traj0":
            shortcut_type = "aerial"
            shortcut_indices = [i for i, fname in enumerate(self.fnames) if fname.startswith(f"{view_id}/traj0")]
            rank0_log(f"Using {len(shortcut_indices)} history from {view_id}/traj0 ...")
            shortcut_indices = shortcut_indices[3::2]
        elif "view" in view_id and traj_id in ("traj0", "traj1"):
            shortcut_type = "regular"
            shortcut_indices = [i for i, fname in enumerate(self.fnames) if fname.startswith(f"{view_id}/traj2")]
            rank0_log(f"Using {len(shortcut_indices)} history from {view_id}/traj2 ...")
            if traj_id == "traj0":
                shortcut_indices = shortcut_indices[3::2]
            else:
                shortcut_indices = shortcut_indices[:len(shortcut_indices) // 2][3::2]
        else:
            shortcut_type = "none"
            shortcut_indices = []

        if tar_w2cs_full.shape[0] > self.nframe:
            tar_w2cs = tar_w2cs_full[0::4]
            tar_Ks = tar_Ks_full[0::4]
        else:
            tar_w2cs = tar_w2cs_full
            tar_Ks = tar_Ks_full

        # Track ref_index based on best_idx changes
        retrieval_map = dict()
        ref_index_dict = collections.defaultdict(dict)
        retrieved_frames = []
        retrieved_w2cs = []
        retrieved_Ks = []
        ref_w2cs = []
        retrieved_global_indices = []  # Indices in the global list corresponding to retrieval results.

        for i in range(1, tar_w2cs.shape[0]):
            if len(shortcut_indices) > 0:
                if shortcut_type == "aerial" and i not in (4, 8, 12, 16):
                    shortcut_index = shortcut_indices[int(i / tar_w2cs.shape[0] * len(shortcut_indices))]
                elif shortcut_type == "regular" and i in (1, 2, 3, 5, 7, 9, 13, 17):
                    shortcut_index = shortcut_indices[int(i / tar_w2cs.shape[0] * len(shortcut_indices))]
                else:
                    shortcut_index = None
            else:
                shortcut_index = None

            best_idx, best_score, angle_diff = find_closest_camera_in_view(
                tar_w2cs[i],
                self.ref_w2cs,
                tar_Ks[i],
                self.ref_Ks,
                self.image_width,
                self.image_height,
                method="fov_overlap",
                near=0.1,
                far=max(self.depth_median * 8, 0.15),
                angle_penalty=True,
                shortcut_index=shortcut_index,
            )

            # Track ref_index when best_idx changes from previous frame
            if best_idx not in retrieval_map:
                retrieval_map[best_idx] = i - 1  # key: best_idx, value: the frame index in retrieved_frames corresponding to best_idx, recording only the earliest occurrence.
            ref_index_dict[retrieval_map[best_idx]][i] = {"score": best_score, "angle_diff": angle_diff}  # key: frame index in retrieved_frames, value: target position corresponding to that retrieval frame.

            retrieved_frames.append(np.array(self.ref_frames[best_idx])[None])
            retrieved_w2cs.append(self.ref_w2cs[best_idx])
            retrieved_Ks.append(self.ref_Ks[best_idx])
            retrieved_global_indices.append(best_idx)

        if len(ref_index_dict) > self.max_reference:
            rank0_log(f"Too many references. {len(ref_index_dict)} > {self.max_reference}")
            retrieved_w2cs = torch.stack(retrieved_w2cs)
            retrieved_Ks = torch.stack(retrieved_Ks)
            indices, _ = self.camera_selector.select(retrieved_w2cs.cpu().numpy(), retrieved_frames, topk=self.max_reference,
                                                     camera_weight=0.3, image_weight=0.7, quality_bias=0.1)
            retrieved_w2cs = retrieved_w2cs[indices]
            retrieved_Ks = retrieved_Ks[indices]
            retrieved_frames_selected = [retrieved_frames[i] for i in indices]
            retrieved_frames = []
            retrieved_global_indices_selected = [retrieved_global_indices[i] for i in indices]
            retrieved_global_indices = []

            # Reassign retrieval results once.
            retrieval_map = dict()
            ref_index_dict = collections.defaultdict(dict)

            for i in range(1, tar_w2cs.shape[0]):
                best_idx, best_score, angle_diff = find_closest_camera_in_view(
                    tar_w2cs[i],
                    retrieved_w2cs,
                    tar_Ks[i],
                    retrieved_Ks,
                    self.image_width,
                    self.image_height,
                    method="fov_overlap",
                    near=0.1,
                    far=max(self.depth_median * 8, 0.15),
                    angle_penalty=True
                )

                # Track ref_index when best_idx changes from previous frame
                if best_idx not in retrieval_map:
                    retrieval_map[best_idx] = i - 1  # key: best_idx, value: the frame index in retrieved_frames corresponding to best_idx, recording only the earliest occurrence.
                ref_index_dict[retrieval_map[best_idx]][i] = {"score": best_score, "angle_diff": angle_diff}  # key: frame index in retrieved_frames, value: target position corresponding to that retrieval frame.

                retrieved_frames.append(np.array(retrieved_frames_selected[best_idx]))
                ref_w2cs.append(retrieved_w2cs[best_idx])  # Reorder retrieved_w2cs.
                retrieved_global_indices.append(retrieved_global_indices_selected[best_idx])
        else:
            ref_w2cs = retrieved_w2cs  # copy retrieved_w2cs

        # TODO: currently only returns ref_index_list.
        ref_index_list = list(ref_index_dict.keys())

        retrieved_frames = np.concatenate(retrieved_frames, axis=0)
        ref_index = torch.tensor(ref_index_list, dtype=torch.long)
        ref_w2cs = torch.stack(ref_w2cs)[ref_index]

        return retrieved_frames, ref_index, ref_index_dict, ref_w2cs, retrieved_global_indices

    def update_memory(self, gen_frames, tar_w2cs_full, tar_Ks_full, view_id=None, traj_id=None):
        """
        Only update memory images in memory, processed by all processes at the same time; no alignment or IO operations.
        view_path: traj*, points.ply, projected_xy.npy, start_frame.png
        gen_frames: [PIL.Image] * N
        """
        assert tar_w2cs_full.shape[0] == tar_Ks_full.shape[0] == len(gen_frames)

        # Downsample frames that need updates in the memory bank, skipping the first frame.
        nframe = len(gen_frames)
        indices = sample_align_nframe(nframe, self.align_nframe)
        updated_tar_w2cs = tar_w2cs_full[indices]
        updated_tar_Ks = tar_Ks_full[indices]
        gen_frames = [gen_frames[idx] for idx in indices]

        # Update memory bank cache.
        self.ref_w2cs = torch.cat([self.ref_w2cs, updated_tar_w2cs.to(self.device)], dim=0)
        self.ref_Ks = torch.cat([self.ref_Ks, updated_tar_Ks.to(self.device)], dim=0)
        self.ref_frames.extend(gen_frames)

        self.ref_depths.extend([None] * len(indices))  # Use None as a placeholder before alignment.
        for index in indices:
            self.fnames.append(f"{view_id}/{traj_id}/{str(index).zfill(4)}")

    def apply_worldmirror(self, skip_exist=True):
        self.world_mirror_dir = f"{self.root_path}/render_results/{self.results_path}/world_mirror_data"

        if not (skip_exist and os.path.exists(f"{self.world_mirror_dir}/cameras.json")):
            os.makedirs(f"{self.world_mirror_dir}/images", exist_ok=True)

            # Multiprocess assignment: each rank processes its own frames.
            process_list = np.arange(len(self.fnames))[self.rank::self.world_size]

            # Camera dictionary for this rank.
            local_camera_dict = {"extrinsics": [], "intrinsics": []}
            # Collect image-save tasks as (image, save_path).
            save_tasks = []
            for gi in process_list:
                fname = self.fnames[gi]
                view_id, traj_id, frame_id = fname.split("/")
                if view_id.startswith("render_results"):
                    camera_id = f"pano-{frame_id}"
                else:
                    camera_id = f"{view_id}-{traj_id}-{frame_id}"

                local_camera_dict["extrinsics"].append({
                    "camera_id": camera_id,
                    "matrix": self.ref_w2cs[gi].inverse().cpu().numpy().tolist()
                })
                local_camera_dict["intrinsics"].append({
                    "camera_id": camera_id,
                    "matrix": self.ref_Ks[gi].cpu().numpy().tolist()
                })
                save_tasks.append((self.ref_frames[gi], f"{self.world_mirror_dir}/images/{camera_id}.png"))

            # Use multithreading to speed up image saving.
            def _save_image(args):
                img, path = args
                img.save(path)

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(_save_image, save_tasks))

            color_print(f"[Rank{self.rank}] Saved {len(save_tasks)} world mirror images.", "info")

            # Synchronize camera_dict across all ranks.
            all_camera_dicts = [None] * self.world_size
            dist.all_gather_object(all_camera_dicts, local_camera_dict)

            # Rank 0 merges and saves cameras.json.
            if self.rank == 0:
                merged_camera_dict = {"num_cameras": 0, "extrinsics": [], "intrinsics": []}
                for rank_dict in all_camera_dicts:
                    if rank_dict is not None:
                        merged_camera_dict["extrinsics"].extend(rank_dict["extrinsics"])
                        merged_camera_dict["intrinsics"].extend(rank_dict["intrinsics"])
                merged_camera_dict["num_cameras"] = len(merged_camera_dict["extrinsics"])
                with open(f"{self.world_mirror_dir}/cameras.json", "w") as f:
                    json.dump(merged_camera_dict, f, indent=2)
                color_print(f"[Rank0] Saved cameras.json with {merged_camera_dict['num_cameras']} cameras.", "info")

            dist.barrier()

        self.name_map = {}
        merged_camera_dict = json.load(open(f"{self.world_mirror_dir}/cameras.json", "r"))
        world_mirror_cam_ids = []
        for cam in merged_camera_dict["extrinsics"]:
            world_mirror_cam_ids.append(cam["camera_id"])
        world_mirror_cam_ids.sort()
        for i in range(len(world_mirror_cam_ids)):
            camera_id = world_mirror_cam_ids[i]
            if camera_id.startswith("pano-"):
                view_id, traj_id, fname_id = "render_results", "pano_bank", camera_id.split("-")[1]
            else:
                view_id, traj_id, fname_id = camera_id.split("-")
            self.name_map[f"{view_id}/{traj_id}/{fname_id}"] = str(i).zfill(4)

        # Rank 0 runs World Mirror inference.
        torch.cuda.empty_cache()
        if self.rank == 0:
            if not (skip_exist and os.path.exists(f"{self.world_mirror_dir}/name_map.json")):
                wm_cmd = [
                    "torchrun", f"--nproc_per_node={self.world_size}", "-m", "worldrecon.pipeline",
                    "--input_path", f"{self.world_mirror_dir}/images",
                    "--prior_cam_path", f"{self.world_mirror_dir}/cameras.json",
                    "--strict_output_path", f"{self.world_mirror_dir}/results",
                    "--target_size", "832",
                    "--log_time",
                    "--no_interactive",
                    "--no_save_gs",
                    "--no_save_normal",
                    "--no_save_points",
                    "--no_sky_mask",
                    "--no_edge_mask",
                    "--use_fsdp",
                    "--enable_bf16",
                    "--disable_heads", "normal", "points", "gs"
                ]
                color_print(f"[Rank0] Running World Mirror inference: {' '.join(wm_cmd)}", "info")
                result = subprocess.run(wm_cmd, cwd="..")

                if result.returncode != 0:
                    raise RuntimeError(f"World Mirror inference failed with return code {result.returncode}")

                # save name_map
                with open(f"{self.world_mirror_dir}/name_map.json", "w") as w:
                    json.dump(self.name_map, w, indent=2)

                color_print(f"[Rank0] World Mirror inference completed successfully.", "info")

        torch.cuda.empty_cache()
        dist.barrier()

    def _detect_kb_anomalies(self, global_kb_summary):
        """
        Detect abnormal k,b values based on anchor depths.
        Core idea: instead of directly filtering k and b, evaluate whether each (k, b) pair's mapping effect
        at several fixed anchor depths, aligned_inv_depth = k * anchor_inv_depth + b, is consistent with
        the global median. This turns the coupled k,b problem into one-dimensional outlier detection in
        the effect space.

        Args:
            global_kb_summary: Globally synchronized {video_name: {local_i: {"k": ..., "b": ..., ...}}} dictionary.

        Returns:
            inlier_threshold: float, P{kb_anomaly_percentile} threshold of max_relative_deviation.
            frame_deviations: dict, {(video_name, local_i): max_relative_deviation} maximum relative deviation per frame.
        """
        # Build anchors: dense between min_d_ and depth_median * 4, sparse at farther depths.
        # Do not add anchors beyond max_d_.
        anchor_far = min(8 * self.depth_median, self.max_d_)
        anchor_depths_list = [
            self.min_d_,  # Nearest depth.
            self.min_d_ + (self.depth_median - self.min_d_) * 0.25,  # Near 1/4.
            self.min_d_ + (self.depth_median - self.min_d_) * 0.5,  # Near 1/2.
            self.min_d_ + (self.depth_median - self.min_d_) * 0.75,  # Near 3/4.
            self.depth_median,  # Median depth.
        ]
        # Multiples of median depth, truncated at max_d_.
        for multiplier in [1.5, 2, 3, 4]:
            d = self.depth_median * multiplier
            if d >= self.max_d_:
                break
            anchor_depths_list.append(d)
        # Far anchors: mid-to-far transition and farthest point, added only when not duplicated.
        mid_far = (anchor_depths_list[-1] + anchor_far) / 2
        if mid_far < anchor_far and mid_far < self.max_d_:
            anchor_depths_list.append(mid_far)
        anchor_depths_list.append(anchor_far)
        anchor_depths = np.array(anchor_depths_list)
        anchor_inv_depths = 1.0 / anchor_depths  # Convert to inv_depth space, consistent with RANSAC fitting.

        if self.rank == 0:
            color_print(f"[Anchor Depths] min_d_={self.min_d_:.4f}, depth_median={self.depth_median:.4f}, "
                        f"max_d_={self.max_d_:.4f}, anchor_far={anchor_far:.4f}", "info")
            color_print(f"[Anchor Depths] depths = {anchor_depths.tolist()}", "info")
            color_print(f"[Anchor InvDepths] inv_depths = {anchor_inv_depths.tolist()}", "info")

        # Collect all globally valid k,b values.
        all_valid_ks = []
        all_valid_bs = []
        all_valid_video_names = []
        all_valid_frame_ids = []
        for vname in sorted(global_kb_summary.keys()):
            for local_i, fdata in global_kb_summary[vname].items():
                if fdata["k"] is not None:
                    all_valid_ks.append(fdata["k"])
                    all_valid_bs.append(fdata["b"])
                    all_valid_video_names.append(vname)
                    all_valid_frame_ids.append(local_i)

        all_valid_ks = np.array(all_valid_ks)
        all_valid_bs = np.array(all_valid_bs)
        N_valid = len(all_valid_ks)

        if N_valid > 0:
            # Compute aligned_inv_depth = k * anchor_inv_depth + b for each (k,b) pair at every anchor.
            # effect_matrix: [N_valid, num_anchors]
            effect_matrix = all_valid_ks[:, None] * anchor_inv_depths[None, :] + all_valid_bs[:, None]

            # Global median effect.
            median_effect = np.median(effect_matrix, axis=0)  # [num_anchors]

            # Deviation of each frame from the median at each anchor.
            deviation = np.abs(effect_matrix - median_effect[None, :])  # [N_valid, num_anchors]

            # Maximum per-frame deviation; a large deviation at any anchor is considered abnormal.
            max_deviation = deviation.max(axis=1)  # [N_valid]

            # Also compute relative deviation as a percentage of the median effect.
            relative_deviation = deviation / (np.abs(median_effect[None, :]) + 1e-8)  # [N_valid, num_anchors]
            max_relative_deviation = relative_deviation.max(axis=1)  # [N_valid]

            # ===== Output statistics for analysis. =====
            if self.rank == 0:
                color_print(f"\n{'=' * 80}", "info")
                color_print(f"[KB Anomaly Detection] 全局统计 (N_valid={N_valid})", "info")
                color_print(f"{'=' * 80}", "info")

                # Global k,b statistics.
                color_print(f"[Global k] min={all_valid_ks.min():.6f}, max={all_valid_ks.max():.6f}, "
                            f"median={np.median(all_valid_ks):.6f}, mean={all_valid_ks.mean():.6f}, "
                            f"std={all_valid_ks.std():.6f}", "info")
                color_print(f"[Global b] min={all_valid_bs.min():.6f}, max={all_valid_bs.max():.6f}, "
                            f"median={np.median(all_valid_bs):.6f}, mean={all_valid_bs.mean():.6f}, "
                            f"std={all_valid_bs.std():.6f}", "info")

                # Median effect at each anchor, i.e. the "standard" mapping result.
                color_print(f"\n[Median Effect at Anchors] (aligned_inv_depth = k * anchor_inv_depth + b)", "info")
                for ai, (ad, aid, me) in enumerate(zip(anchor_depths, anchor_inv_depths, median_effect)):
                    # The aligned_depth corresponding to the median effect is 1 / median_effect.
                    aligned_d = 1.0 / me if abs(me) > 1e-8 else float('inf')
                    color_print(f"  Anchor[{ai}]: depth={ad:.4f} -> inv_depth={aid:.4f} -> median_aligned_inv_depth={me:.6f} (aligned_depth={aligned_d:.4f})", "info")

                # Deviation distribution statistics.
                color_print(f"\n[Max Absolute Deviation Distribution]", "info")
                percentiles = [50, 75, 90, 95, 99, 100]
                pct_values = np.percentile(max_deviation, percentiles)
                for p, v in zip(percentiles, pct_values):
                    color_print(f"  P{p}: {v:.6f}", "info")

                color_print(f"\n[Max Relative Deviation Distribution] (相对于中位数效果的百分比)", "info")
                pct_rel_values = np.percentile(max_relative_deviation, percentiles)
                for p, v in zip(percentiles, pct_rel_values):
                    color_print(f"  P{p}: {v:.4%}", "info")

                # Statistics grouped by video.
                color_print(f"\n[Per-Video Deviation Statistics]", "info")
                unique_videos = sorted(set(all_valid_video_names))
                for vname in unique_videos:
                    vid_mask = np.array([v == vname for v in all_valid_video_names])
                    vid_max_dev = max_deviation[vid_mask]
                    vid_max_rel_dev = max_relative_deviation[vid_mask]
                    vid_ks = all_valid_ks[vid_mask]
                    vid_bs = all_valid_bs[vid_mask]
                    color_print(f"  {vname} ({vid_mask.sum()} frames): "
                                f"k=[{vid_ks.min():.4f},{vid_ks.max():.4f}], "
                                f"b=[{vid_bs.min():.4f},{vid_bs.max():.4f}], "
                                f"max_abs_dev=[{vid_max_dev.min():.6f},{vid_max_dev.max():.6f}], "
                                f"max_rel_dev=[{vid_max_rel_dev.min():.4%},{vid_max_rel_dev.max():.4%}]", "info")

                # Output the top 10 frames with the largest deviations.
                color_print(f"\n[Top-10 Outlier Candidates] (按 max_absolute_deviation 排序)", "info")
                top_indices = np.argsort(max_deviation)[::-1][:10]
                for rank_i, idx in enumerate(top_indices):
                    color_print(f"  #{rank_i + 1}: video={all_valid_video_names[idx]}, frame_id={all_valid_frame_ids[idx]}, "
                                f"k={all_valid_ks[idx]:.6f}, b={all_valid_bs[idx]:.6f}, "
                                f"max_abs_dev={max_deviation[idx]:.6f}, max_rel_dev={max_relative_deviation[idx]:.4%}", "info")

            # Compute the inlier threshold based on kb_anomaly_percentile using relative deviation.
            inlier_threshold = float(np.percentile(max_relative_deviation, self.kb_anomaly_percentile))
            if self.rank == 0:
                color_print(f"[KB Anomaly] Using P{self.kb_anomaly_percentile} as inlier threshold (relative): {inlier_threshold:.4%}", "info")
                n_inliers = int((max_relative_deviation <= inlier_threshold).sum())
                n_outliers = N_valid - n_inliers
                color_print(f"[KB Anomaly] Inliers: {n_inliers}, Outliers: {n_outliers}", "info")

            # Build the per-frame deviation dictionary using relative deviation.
            frame_deviations = {}
            for i in range(N_valid):
                frame_deviations[(all_valid_video_names[i], all_valid_frame_ids[i])] = float(max_relative_deviation[i])

            return inlier_threshold, frame_deviations

        # If there are no valid frames, return an infinite threshold and an empty dictionary.
        return float('inf'), {}

    def alignment(self, debug_mode=False):
        # =====================================================================
        # Phase 1: Build the global video mapping and assign videos to different ranks.
        # =====================================================================
        rank0_log(f"Starting alignment of Scene {self.root_path}")
        video_names = []
        global_video_indices_map = dict()

        # 1a. Group pano_bank frames into virtual videos by align_nframe.
        pano_indices = list(range(self.align_start_index))  # All frame indices in pano_bank.
        n_pano = len(pano_indices)
        if n_pano > 0:
            n_splits = max(1, math.ceil(n_pano / self.align_nframe))
            split_size = math.ceil(n_pano / n_splits)
            for split_idx in range(n_splits):
                start = split_idx * split_size
                end = min(start + split_size, n_pano)
                group_indices = pano_indices[start:end]
                video_key = f"pano/split_{split_idx}"
                video_names.append(video_key)
                global_video_indices_map[video_key] = group_indices

        # 1b. Process normal video frames after align_start_index.
        for i, fname in enumerate(self.fnames):
            if i < self.align_start_index:
                continue
            view_id, traj_id, frame_id = fname.split("/")
            if f"{view_id}/{traj_id}" not in global_video_indices_map:
                video_names.append(f"{view_id}/{traj_id}")  # Preserve the original ordering of video_names.
                global_video_indices_map[f"{view_id}/{traj_id}"] = [i]
            else:
                global_video_indices_map[f"{view_id}/{traj_id}"].append(i)

        video_names_rank = video_names[self.rank::self.world_size]

        # =====================================================================
        # Phase 2: Preprocessing -- precompute MoGe depth and SAM3 sky masks by video.
        #   Cache results in video_align_cache[video_name] to avoid uneven compute latency in the sync loop.
        # =====================================================================
        video_align_cache = {}
        for video_name in video_names_rank:
            global_indices = global_video_indices_map[video_name]
            gen_tensor = []
            gen_frames = []
            for idx in global_indices:
                gen_frames.append(self.ref_frames[idx])
                gen_tensor.append(transforms.ToTensor()(self.ref_frames[idx]))
            gen_tensor = torch.stack(gen_tensor, dim=0)  # [f,c,h,w] 0~1
            updated_tar_w2cs = self.ref_w2cs[global_indices]
            updated_tar_Ks = self.ref_Ks[global_indices]
            N_align = len(gen_tensor)

            # Set save path.
            view_id, traj_id = video_name.split("/")
            save_path = f"{self.root_path}/render_results/{self.results_path}/{view_id}/{traj_id}"
            os.makedirs(f"{save_path}/depths", exist_ok=True)

            # Estimate monocular depth.
            mono_depths = []
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                for i in range(N_align):
                    moge_pred = self.moge_model.infer(gen_tensor[i:i + 1])
                    fname = self.fnames[global_indices[i]]
                    if os.path.exists(f"{self.world_mirror_dir}/results/depth/depth_{self.name_map[fname]}.npy"):
                        depth_wm = np.load(f"{self.world_mirror_dir}/results/depth/depth_{self.name_map[fname]}.npy")
                        depth_wm = cv2.resize(depth_wm, (moge_pred['depth'].shape[2], moge_pred['depth'].shape[1]), interpolation=cv2.INTER_NEAREST)
                        moge_pred['depth'] = torch.from_numpy(depth_wm).unsqueeze(0).to(self.device)
                    else:
                        color_print(f"{self.world_mirror_dir}/results/depth/depth_{self.name_map[fname]} depth not exist.", "warning")
                    mono_depths.append(moge_pred)

            # Use SAM3 to remove the sky mask.
            if self.meta_info["scene_type"] == "outdoor":
                video_frames = []
                for frame in gen_frames:
                    video_frames.append(np.array(frame))
                video_frames = np.stack(video_frames)
                inference_session = self.sam3_processor.init_video_session(
                    video=video_frames,
                    inference_device=self.device,
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=torch.bfloat16,
                )
                inference_session = self.sam3_processor.add_text_prompt(
                    inference_session=inference_session,
                    text="sky",
                )
                outputs_per_frame = {}
                for model_outputs in self.sam3_model.propagate_in_video_iterator(inference_session=inference_session,
                                                                                 max_frame_num_to_track=video_frames.shape[0],
                                                                                 show_progress_bar=False):
                    processed_outputs = self.sam3_processor.postprocess_outputs(inference_session, model_outputs)
                    outputs_per_frame[model_outputs.frame_idx] = processed_outputs
                for frame_idx, processed_outputs in outputs_per_frame.items():
                    if processed_outputs['masks'].shape[0] != 0:
                        mono_depths[frame_idx]["mask"] = (mono_depths[frame_idx]["mask"][0] & ~processed_outputs["masks"][0])[None]

            # Initialize the cache for the current video.
            video_align_cache[video_name] = {
                "global_indices": global_indices,
                "gen_frames": gen_frames,
                "mono_depths": mono_depths,
                "updated_tar_w2cs": updated_tar_w2cs,
                "updated_tar_Ks": updated_tar_Ks,
                "save_path": save_path,
                "frames": {},  # local_i -> per-frame alignment results
            }

            # START alignment!
            for local_i in range(N_align):
                gi = global_indices[local_i]
                fname = self.fnames[gi].split("/")[-1]

                mono_depth_mask = mono_depths[local_i]["mask"][0]
                mono_depth = mono_depths[local_i]["depth"][0].clone().detach()
                mono_depth[~mono_depth_mask] = 0

                # Precompute masks that depend only on mono_depth so every branch can store them in the cache.
                mono_edge_mask = ~torch.from_numpy(utils3d.numpy.depth_edge(mono_depth.cpu().numpy(), rtol=0.05)).to(self.device).bool()
                depth_filter = torch.median(mono_depth[mono_depth > 0]) * 8
                far_mask = mono_depth > depth_filter

                if debug_mode:
                    os.makedirs(f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}", exist_ok=True)
                    gen_frames[local_i].save(f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}/{fname}-image.png")
                    mask_ = mono_depths[local_i]["mask"][0].cpu().numpy()
                    mask_ = Image.fromarray((mask_ * 255).astype(np.uint8))
                    mask_.save(f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}/{fname}-mono-mask.png")

                # == Global PCD Rendering (obtaining guided_depth) ==
                first_align_result = "success"
                guided_depth, guided_depth_mask, guided_normal = None, None, None
                try:
                    guided_depth, guided_depth_mask, guided_normal = get_guided_depth_infos_v2(w2c=updated_tar_w2cs[local_i], K=updated_tar_Ks[local_i],
                                                                                               prev_points3d=self.global_pcd.vertices, prev_normal=self.global_normal,
                                                                                               height=self.image_height, width=self.image_width, device=self.device)
                    guided_depth_np = guided_depth.cpu().numpy()
                    # Compute percentile maps for guided depth and mono depth.
                    guided_mono_mask = (guided_depth_mask & mono_depth_mask).cpu().numpy()
                    mono_depth_np = mono_depth.cpu().numpy()
                    guided_depth_percentile = compute_depth_percentile_map(guided_depth_np, guided_mono_mask)
                    mono_depth_percentile = compute_depth_percentile_map(mono_depth_np, guided_mono_mask)
                    percentile_mask = np.abs(guided_depth_percentile - mono_depth_percentile) > self.percentile_threshold
                    percentile_mask = torch.from_numpy(percentile_mask).bool().to(self.device)
                    guided_depth_mask = guided_depth_mask & ~percentile_mask

                    if debug_mode:
                        # Visualize debug outputs.
                        mono_depth_color = compute_depth_percentile_map(mono_depth_np, mono_depth_mask.cpu().numpy())
                        mono_normal_vis = mono_depths[local_i]["normal"][0]  # [H, W, 3], value range [-1, 1].
                        if isinstance(mono_normal_vis, torch.Tensor):
                            mono_normal_vis = mono_normal_vis.cpu().numpy()
                        mono_normal_rgb = ((mono_normal_vis + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)  # [H, W, 3]

                        # Save visualization results.
                        os.makedirs(f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}", exist_ok=True)
                        colorize_depth(guided_depth_percentile, colormap="turbo", save_path=f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}/{fname}-guided-percentile.png")
                        colorize_depth(mono_depth_percentile, colormap="turbo", save_path=f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}/{fname}-mono-percentile.png")
                        colorize_depth(mono_depth_color, colormap="turbo", save_path=f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}/{fname}-mono.png")
                        Image.fromarray(mono_normal_rgb).save(f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}/{fname}-mono-normal.png")
                        colorize_depth(np.abs(guided_depth_percentile - mono_depth_percentile), colormap="turbo", show_colorbar=True, colorbar_label="Percentile Abs Sub",
                                       save_path=f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}/{fname}-percentile-abssub.png")
                        percentile_mask_pil = Image.fromarray(percentile_mask.cpu().numpy().astype(np.uint8) * 255)
                        percentile_mask_pil.save(f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}/{fname}-percentile-mask.png")
                        colorize_depth(guided_depth_np, save_path=f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}/{fname}-guided-depth.png")

                    if guided_depth_mask.float().mean() < self.valid_threshold:
                        color_print(f"[Alignment Rank{self.rank}] [Frame {view_id}/{traj_id}/{fname}] [Warning] Guidance mask ratio: {guided_depth_mask.float().mean():.4f} <= {self.valid_threshold}", "warning")
                        first_align_result = "warning"
                    else:
                        color_print(f"[Alignment Rank{self.rank}] [Frame {view_id}/{traj_id}/{fname}] [Success] Guidance mask ratio: {guided_depth_mask.float().mean():.4f},"
                                    f" depth percentile error ratio: {percentile_mask.float().sum() / (guided_mono_mask.sum() + 1e-7):.5f}", "info")
                except Exception as e:
                    color_print(f"[Alignment Rank{self.rank}] [Frame {view_id}/{traj_id}/{fname}] [Failed] Error in rendering guidance depth with Exception: {e}...", "error")
                    first_align_result = "failed"

                if first_align_result in ("warning", "failed"):  # If guidance depth rendering fails, record the failed frame.
                    video_align_cache[video_name]["frames"][local_i] = {
                        "gi": gi, "fname": fname, "k": None, "b": None,
                        "fail_reason": first_align_result,
                        "mono_depth": mono_depth.cpu(),
                        "mono_depth_mask": mono_depth_mask.cpu(),
                        "mono_edge_mask": mono_edge_mask.cpu(),
                        "far_mask": far_mask.cpu(),
                        "depth_filter": float(depth_filter.item()),
                    }
                    continue

                # normal update mask; depth alignment avoids samples with normal angles greater than 90 degrees.
                # align normal map to the world coordinate
                pred_normal_camera = mono_depths[local_i]["normal"][0]  # [h, w, 3]
                tar_c2w = updated_tar_w2cs[local_i].inverse()
                pred_normal_world = tar_c2w[:3, :3].to(self.device) @ pred_normal_camera.reshape(-1, 3).T
                pred_normal_world = pred_normal_world.T[:, :3]
                if guided_normal is not None:
                    pred_normal = pred_normal_world.reshape(self.image_height, self.image_width, 3)
                    normal_angle_diff = compute_normal_angles(guided_normal, pred_normal)
                    normal_mask = (normal_angle_diff <= 90)
                else:
                    normal_mask = torch.ones_like(mono_depth_mask).bool()

                valid_mask = guided_depth_mask & mono_depth_mask & mono_edge_mask & normal_mask

                if debug_mode:
                    os.makedirs(f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}", exist_ok=True)
                    valid_mask_np = valid_mask.cpu().numpy()
                    valid_mask_pil = Image.fromarray(valid_mask_np.astype(np.uint8) * 255)
                    valid_mask_pil.save(f"{self.root_path}/render_results/{self.results_path}/tmp_debug/{view_id}-{traj_id}/{fname}-valid-mask.png")

                if valid_mask.float().mean() < 0.05:
                    color_print(f"Rank{self.rank}: Too little overlapping masks are detected for {view_id}/{traj_id}/{fname}. skip...", "error")
                    video_align_cache[video_name]["frames"][local_i] = {
                        "gi": gi, "fname": fname, "k": None, "b": None,
                        "fail_reason": "too_little_overlap",
                        "mono_depth": mono_depth.cpu(),
                        "mono_depth_mask": mono_depth_mask.cpu(),
                        "mono_edge_mask": mono_edge_mask.cpu(),
                        "far_mask": far_mask.cpu(),
                        "depth_filter": float(depth_filter.item()),
                    }
                    continue

                # Initialize with least squares.
                ransac = RANSACRegressor(
                    min_samples=100,
                    estimator=ConstrainedLinearRegression(min_coef=1e-4, max_bias=[-10.0, 10.0]),
                    stop_probability=0.995,
                    random_state=42
                )

                mono_inv_depth = 1.0 / mono_depth
                guided_inv_depth = 1.0 / guided_depth
                near_mask = valid_mask & (mono_depth <= depth_filter)
                try:
                    _ = ransac.fit(mono_inv_depth[near_mask].cpu().numpy().reshape(-1, 1), guided_inv_depth[near_mask].cpu().numpy().reshape(-1, 1))
                except:
                    color_print(f"[Rank{self.rank}] RANSAC failed for {view_id}/{traj_id}/{fname}, continue...", "error")
                    video_align_cache[video_name]["frames"][local_i] = {
                        "gi": gi, "fname": fname, "k": None, "b": None,
                        "fail_reason": "ransac_failed",
                        "mono_depth": mono_depth.cpu(),
                        "mono_depth_mask": mono_depth_mask.cpu(),
                        "mono_edge_mask": mono_edge_mask.cpu(),
                        "far_mask": far_mask.cpu(),
                        "depth_filter": float(depth_filter.item()),
                    }
                    continue
                k = ransac.estimator_.model.coef_[0][0]
                b = ransac.estimator_.model.intercept_[0]

                # Store the current frame's k,b and masks in video_align_cache.
                video_align_cache[video_name]["frames"][local_i] = {
                    "gi": gi,
                    "fname": fname,
                    "k": float(k),
                    "b": float(b),
                    # The following masks are needed later to generate aligned depth and point clouds.
                    "mono_depth": mono_depth.cpu(),  # [H, W] Monocular depth.
                    "mono_depth_mask": mono_depth_mask.cpu(),  # [H, W] Valid monocular-depth mask.
                    "mono_edge_mask": mono_edge_mask.cpu(),  # [H, W] Edge mask, recalculated later with final k,b.
                    "far_mask": far_mask.cpu(),  # [H, W] Far-region mask.
                    "depth_filter": float(depth_filter.item()),  # Depth filtering threshold.
                }

            n_success = sum(1 for f in video_align_cache[video_name]['frames'].values() if f['k'] is not None)
            n_failed = sum(1 for f in video_align_cache[video_name]['frames'].values() if f['k'] is None)
            color_print(f"[Rank{self.rank}] Video {video_name}: {n_success} success, {n_failed} failed, {N_align} total frames.", "info")

        # =====================================================================
        # Phase 3: Synchronize k,b results in video_align_cache across processes.
        #   Only synchronize lightweight data (k, b, gi, fname); keep heavy data (mask, depth) local.
        # =====================================================================
        # Build lightweight synchronization data containing only k, b, and frame identifiers.
        local_kb_summary = {}  # video_name -> {local_i: {"gi", "fname", "k", "b"}}
        for video_name, cache in video_align_cache.items():
            local_kb_summary[video_name] = {}
            for local_i, frame_data in cache["frames"].items():
                local_kb_summary[video_name][local_i] = {
                    "gi": frame_data["gi"],
                    "fname": frame_data["fname"],
                    "k": frame_data["k"],
                    "b": frame_data["b"],
                }

        # Synchronize lightweight k,b data across ranks.
        all_kb_summaries = [None] * self.world_size
        dist.all_gather_object(all_kb_summaries, local_kb_summary)

        # Merge k,b results from all ranks into a global dictionary.
        global_kb_summary = {}  # video_name -> {local_i: {"gi", "fname", "k", "b"}}
        for rank_summary in all_kb_summaries:
            if rank_summary is not None:
                for video_name, frames_kb in rank_summary.items():
                    assert video_name not in global_kb_summary, f"Video {video_name} appears in multiple ranks!"
                    global_kb_summary[video_name] = frames_kb

        # Print synchronization statistics.
        total_aligned_frames = sum(len(v) for v in global_kb_summary.values())
        total_videos = len(global_kb_summary)
        rank0_log(f"Alignment Phase 3: Synced k,b from {total_videos} videos, {total_aligned_frames} frames across {self.world_size} ranks.")

        if self.rank == 0:
            for vname in sorted(global_kb_summary.keys()):
                frames_kb = global_kb_summary[vname]
                valid_ks = [v["k"] for v in frames_kb.values() if v["k"] is not None]
                valid_bs = [v["b"] for v in frames_kb.values() if v["b"] is not None]
                n_failed = sum(1 for v in frames_kb.values() if v["k"] is None)
                if valid_ks:
                    color_print(f"  Video {vname}: {len(valid_ks)} success, {n_failed} failed, "
                                f"k=[{min(valid_ks):.4f}, {max(valid_ks):.4f}], "
                                f"b=[{min(valid_bs):.4f}, {max(valid_bs):.4f}]", "info")
                else:
                    color_print(f"  Video {vname}: 0 success, {n_failed} failed (all frames failed)", "error")

        dist.barrier()

        # =====================================================================
        # Phase 4: Detect abnormal k,b values based on anchor depths.
        # =====================================================================
        inlier_threshold, frame_deviations = self._detect_kb_anomalies(global_kb_summary)

        dist.barrier()

        # =====================================================================
        # Phase 5: Classify each frame on this rank as inlier/outlier and determine the final k,b.
        #   - inlier: directly use its own k,b
        #   - outlier / originally failed frame: reuse k,b from the nearest inlier frame in the same video
        #   - if no frames in the same video are inliers, abandon that video
        # =====================================================================
        abandoned_videos = []  # Record abandoned videos where all frames are outliers or failed.

        for video_name, cache in video_align_cache.items():
            frames = cache["frames"]
            if not frames:
                abandoned_videos.append((video_name, "no_frames"))
                continue

            # Sort by local_i to keep frame order consistent.
            sorted_local_is = sorted(frames.keys())

            # Determine whether each frame is an inlier.
            # Inlier condition: k is not None and max_relative_deviation <= inlier_threshold.
            inlier_map = {}  # local_i -> bool
            for li in sorted_local_is:
                fdata = frames[li]
                if fdata["k"] is None:
                    inlier_map[li] = False
                else:
                    dev = frame_deviations.get((video_name, li), float('inf'))
                    inlier_map[li] = (dev <= inlier_threshold)

            # Collect local_i and k,b for all inlier frames in this video.
            inlier_frames = [(li, frames[li]["k"], frames[li]["b"]) for li in sorted_local_is if inlier_map[li]]

            if not inlier_frames:
                # All frames in this video are non-inliers, so abandon it.
                abandoned_videos.append((video_name, "all_outlier"))
                color_print(f"[Rank{self.rank}] Video {video_name}: ALL frames are outlier/failed, abandoning.", "error")
                continue

            inlier_local_is = np.array([x[0] for x in inlier_frames])

            # Determine the final k,b to use for each frame.
            n_self = 0  # Number of frames using their own k,b.
            n_borrowed = 0  # Number of frames borrowing nearby inlier k,b.
            for li in sorted_local_is:
                fdata = frames[li]
                # Inliers use their own k,b.
                if inlier_map[li]:
                    # Inliers use their own k,b.
                    fdata["final_k"] = fdata["k"]
                    fdata["final_b"] = fdata["b"]
                    fdata["kb_source"] = "self"
                    n_self += 1
                else:
                    # For an outlier or failed frame, find the nearest inlier frame in the same video.
                    distances = np.abs(inlier_local_is - li)
                    nearest_idx = np.argmin(distances)
                    nearest_li, nearest_k, nearest_b = inlier_frames[nearest_idx]
                    fdata["final_k"] = nearest_k
                    fdata["final_b"] = nearest_b
                    fdata["kb_source"] = f"borrowed_from_{nearest_li}"
                    n_borrowed += 1

            # color_print(f"[Rank{self.rank}] Video {video_name}: {n_self} inlier (self k,b), "
            #             f"{n_borrowed} outlier/failed (borrowed k,b), "
            #             f"{len(inlier_frames)} inlier frames available.", "info")

        # Synchronize the list of abandoned videos across ranks and report the aggregate.
        all_abandoned = [None] * self.world_size
        dist.all_gather_object(all_abandoned, abandoned_videos)
        global_abandoned = []
        for rank_abandoned in all_abandoned:
            if rank_abandoned:
                global_abandoned.extend(rank_abandoned)

        if self.rank == 0 and global_abandoned:
            color_print(f"\n{'=' * 80}", "error")
            color_print(f"[CRITICAL] Total {len(global_abandoned)} videos abandoned due to severe alignment issues:", "error")
            for vname, reason in global_abandoned:
                color_print(f"  {vname}: {reason}", "error")
            color_print(f"{'=' * 80}\n", "error")

        dist.barrier()

        # =====================================================================
        # Phase 6: Generate aligned depth, update_mask, and point clouds with final_k and final_b.
        # =====================================================================
        abandoned_video_names = set(vname for vname, _ in abandoned_videos)
        video_aligned_data = {}  # video_name -> {"points": np.ndarray, "colors": np.ndarray}
        video_camera_dicts = {}  # video_name -> {fname: {"intrinsic": ..., "extrinsic": ...}}

        eps = 1e-6
        for video_name, cache in video_align_cache.items():
            if video_name in abandoned_video_names:
                color_print(f"[Rank{self.rank}] Skipping abandoned video {video_name}.", "warning")
                continue

            frames = cache["frames"]
            gen_frames = cache["gen_frames"]
            updated_tar_w2cs = cache["updated_tar_w2cs"]
            updated_tar_Ks = cache["updated_tar_Ks"]
            save_path = cache["save_path"]

            video_aligned_data[video_name] = {"points": None, "colors": None}
            video_camera_dicts[video_name] = {}

            sorted_local_is = sorted(frames.keys())
            for local_i in sorted_local_is:
                fdata = frames[local_i]
                gi = fdata["gi"]
                fname = fdata["fname"]
                final_k = fdata.get("final_k")
                final_b = fdata.get("final_b")

                if final_k is None or final_b is None:
                    # This should not happen because abandoned videos were skipped; keep this as a safeguard.
                    color_print(f"[Rank{self.rank}] Frame {video_name}/{fname}: no final_k/b, skipping.", "warning")
                    continue

                # Restore mono_depth and masks from the cache, moving CPU tensors to GPU.
                mono_depth = fdata["mono_depth"].to(self.device)  # [H, W]
                mono_depth_mask = fdata["mono_depth_mask"].to(self.device)  # [H, W]
                mono_edge_mask = fdata["mono_edge_mask"].to(self.device)  # [H, W]
                far_mask = fdata["far_mask"].to(self.device)  # [H, W]

                # Compute aligned_depth with final_k and final_b.
                aligned_depth = 1.0 / torch.clamp_min((1.0 / torch.clamp_min(mono_depth, eps)) * final_k + final_b, eps)

                # Compute aligned_edge_mask and final_mask.
                aligned_edge_mask = ~torch.from_numpy(utils3d.numpy.depth_edge(aligned_depth.cpu().numpy(), rtol=0.1)).to(self.device).bool()
                combined_edge_mask = mono_edge_mask & aligned_edge_mask & mono_depth_mask
                final_mask = (aligned_depth >= self.min_depth) & combined_edge_mask & (aligned_depth <= self.max_d)
                aligned_depth[~final_mask] = 0

                if final_mask.float().sum() < 10:
                    color_print(f"[Rank{self.rank}] Too few valid points for {video_name}/{fname} after alignment, skipping.", "warning")
                    continue

                # Save aligned depth.
                save_16bit_png_depth(aligned_depth.cpu().numpy(), f"{save_path}/depths/{fname}.png")

                # Compute update_mask and point cloud.
                update_mask = mono_depth_mask & ~far_mask
                update_mask = final_mask & update_mask

                # Update ref_depths.
                aligned_depth_for_ref = aligned_depth.clone()
                aligned_depth_for_ref[~update_mask] = 0
                self.ref_depths[gi] = aligned_depth_for_ref.cpu().numpy()

                # Generate point cloud.
                rgb_colors = torch.from_numpy(np.array(gen_frames[local_i]).reshape(-1, 3)).to(self.device, dtype=torch.float32)
                update_points3d, update_rgb = depth2pcd(
                    updated_tar_w2cs[local_i], updated_tar_Ks[local_i],
                    self.points.clone(), aligned_depth, rgb_colors, update_mask
                )
                update_points3d = update_points3d.cpu().numpy()
                update_rgb = update_rgb.cpu().numpy().astype(np.uint8)

                # Aggregate at video level.
                if video_aligned_data[video_name]["points"] is None:
                    video_aligned_data[video_name]["points"] = update_points3d
                    video_aligned_data[video_name]["colors"] = update_rgb
                else:
                    video_aligned_data[video_name]["points"] = np.concatenate([video_aligned_data[video_name]["points"], update_points3d], axis=0)
                    video_aligned_data[video_name]["colors"] = np.concatenate([video_aligned_data[video_name]["colors"], update_rgb], axis=0)

                # Record camera information by directly referencing global ref_Ks/ref_w2cs to avoid redundant storage.
                video_camera_dicts[video_name][fname] = {
                    "intrinsic": self.ref_Ks[gi].cpu().numpy().tolist(),
                    "extrinsic": self.ref_w2cs[gi].cpu().numpy().tolist()
                }

            n_aligned = len(video_camera_dicts.get(video_name, {}))
            color_print(f"[Rank{self.rank}] Video {video_name}: {n_aligned} frames aligned with depth & pointcloud.", "info")

        # =====================================================================
        # Phase 6.5: Filter outlier points after video-level aggregation with Statistical Outlier Removal.
        # =====================================================================
        for video_name, vdata in video_aligned_data.items():
            if vdata["points"] is None or vdata["points"].shape[0] == 0:
                continue
            n_before = vdata["points"].shape[0]
            filtered_points, filtered_colors, _ = statistical_outlier_removal(
                vdata["points"], vdata["colors"],
                nb_neighbors=self.pcd_nb_neighbors, std_ratio=self.pcd_std_ratio
            )
            n_after = filtered_points.shape[0]
            n_removed = n_before - n_after
            video_aligned_data[video_name]["points"] = filtered_points
            video_aligned_data[video_name]["colors"] = filtered_colors
            if n_removed > 0:
                color_print(
                    f"[Rank{self.rank}] SOR filter {video_name}: {n_before} -> {n_after} points "
                    f"(removed {n_removed}, {n_removed / n_before * 100:.1f}%)",
                    "info"
                )

        # =====================================================================
        # Phase 7: Save cameras.json and synchronize point-cloud data across ranks.
        # =====================================================================
        for video_name, cam_dict in video_camera_dicts.items():
            if len(cam_dict) == 0:
                continue
            vid_save_path = video_align_cache[video_name]["save_path"]
            with open(f"{vid_save_path}/cameras.json", "w") as f:
                json.dump(cam_dict, f, indent=2)

        # Synchronize video_aligned_data across ranks.
        all_video_aligned_data_list = [None] * self.world_size
        dist.all_gather_object(all_video_aligned_data_list, video_aligned_data)

        # Merge: different ranks process different videos, so update directly.
        self.global_points = {}
        for rank_data in all_video_aligned_data_list:
            if rank_data is not None:
                for vname, vdata in rank_data.items():
                    if vdata["points"] is None:
                        continue
                    assert vname not in self.global_points, f"Video {vname} appears in multiple ranks!"
                    self.global_points[vname] = {"points": vdata["points"], "colors": vdata["colors"]}

        if self.rank == 0:
            total_points = sum(v["points"].shape[0] for v in self.global_points.values())
            color_print(f"[Phase 7] Global points merged: {len(self.global_points)} videos, {total_points} total points.", "info")

        # Save each video's point cloud in debug mode.
        if debug_mode:
            for video_name, data in video_aligned_data.items():
                if data["points"] is None or data["points"].shape[0] == 0:
                    continue
                view_id, traj_id = video_name.split("/")
                temp_points = data["points"]
                temp_colors = data["colors"]
                if temp_points.shape[0] > 500_000:
                    downsampled_indices = np.random.choice(temp_points.shape[0], 500_000, replace=False)
                    temp_points = temp_points[downsampled_indices]
                    temp_colors = temp_colors[downsampled_indices]
                os.makedirs(f"{self.root_path}/render_results/{self.results_path}/tmp_debug/pointclouds", exist_ok=True)
                temp_pcd = trimesh.PointCloud(vertices=temp_points, colors=temp_colors)
                temp_pcd.export(f"{self.root_path}/render_results/{self.results_path}/tmp_debug/pointclouds/{view_id}-{traj_id}-pcd.ply")

        dist.barrier()