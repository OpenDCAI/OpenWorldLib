"""
Utilities for geometry operations.

References: DUSt3R, MoGe
"""

from numbers import Number
from typing import Tuple, Union

import torch
import einops
import numpy as np
from mirror_src.utils.warnings import no_warnings


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = depthmap > 0.0
    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(
    depthmap, camera_intrs, camera_poses, **kw
):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrs: a 3x3 matrix
        - camera_poses: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrs)

    R_cam2world = camera_poses[:3, :3]
    t_cam2world = camera_poses[:3, 3]

    # Express in absolute coordinates (invalid depth values)
    X_world = (
        np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]
    )
    return X_world, valid_mask


@torch.no_grad()
def calculate_unprojected_mask(views, target_views):
    '''Calcuate the loss mask for the target views in the batch'''
    target_depth = torch.stack([torch.from_numpy(view["depthmap"]) for view in target_views], dim=0)
    target_intrinsics = torch.stack([torch.from_numpy(view["camera_intrs"]) for view in target_views], dim=0)
    target_c2w = torch.stack([torch.from_numpy(view["camera_poses"]) for view in target_views], dim=0)
    context_depth = torch.stack([torch.from_numpy(view["depthmap"]) for view in views], dim=0)
    context_intrinsics = torch.stack([torch.from_numpy(view["camera_intrs"]) for view in views], dim=0)
    context_c2w = torch.stack([torch.from_numpy(view["camera_poses"]) for view in views], dim=0)

    target_intrinsics = target_intrinsics[..., :3, :3]
    context_intrinsics = context_intrinsics[..., :3, :3]

    mask = calculate_in_frustum_mask(
        target_depth[None], target_intrinsics[None], target_c2w[None],
        context_depth[None], context_intrinsics[None], context_c2w[None]
    )[0]
    return mask


@torch.no_grad()
def calculate_in_frustum_mask(depth_1, intrinsics_1, c2w_1, depth_2, intrinsics_2, c2w_2):
    """
    A function that takes in the depth, intrinsics and c2w matrices of two sets
    of views, and then works out which of the pixels in the first set of views
    has a direct corresponding pixel in any of views in the second set

    Args:
        depth_1: (b, v1, h, w)
        intrinsics_1: (b, v1, 3, 3)
        c2w_1: (b, v1, 4, 4)
        depth_2: (b, v2, h, w)
        intrinsics_2: (b, v2, 3, 3)
        c2w_2: (b, v2, 4, 4)

    Returns:
        torch.Tensor: valid mask with shape (b, v1, v2, h, w).
    """

    _, v1, h, w = depth_1.shape
    _, v2, _, _ = depth_2.shape

    # Unproject the depth to get the 3D points in world space
    points_3d = unproject_depth(depth_1[..., None], intrinsics_1, c2w_1)  # (b, v1, h, w, 3)

    # Project the 3D points into the pixel space of all the second views simultaneously
    camera_points = world_space_to_camera_space(points_3d, c2w_2)  # (b, v1, v2, h, w, 3)
    points_2d = camera_space_to_pixel_space(camera_points, intrinsics_2)  # (b, v1, v2, h, w, 2)

    # Calculate the depth of each point
    rendered_depth = camera_points[..., 2]  # (b, v1, v2, h, w)

    # We use three conditions to determine if a point should be masked

    # Condition 1: Check if the points are in the frustum of any of the v2 views
    in_frustum_mask = (
        (points_2d[..., 0] > 0) &
        (points_2d[..., 0] < w) &
        (points_2d[..., 1] > 0) &
        (points_2d[..., 1] < h)
    )  # (b, v1, v2, h, w)
    in_frustum_mask = in_frustum_mask.any(dim=-3)  # (b, v1, h, w)

    # Condition 2: Check if the points have non-zero (i.e. valid) depth in the input view
    non_zero_depth = depth_1 > 1e-6

    # Condition 3: Check if the points have matching depth to any of the v2
    # views torch.nn.functional.grid_sample expects the input coordinates to
    # be normalized to the range [-1, 1], so we normalize first
    points_2d[..., 0] /= w
    points_2d[..., 1] /= h
    points_2d = points_2d * 2 - 1
    matching_depth = torch.ones_like(rendered_depth, dtype=torch.bool)
    for b in range(depth_1.shape[0]):
        for i in range(v1):
            for j in range(v2):
                depth = einops.rearrange(depth_2[b, j], 'h w -> 1 1 h w')
                coords = einops.rearrange(points_2d[b, i, j], 'h w c -> 1 h w c')
                sampled_depths = torch.nn.functional.grid_sample(depth, coords, align_corners=False)[0, 0]
                matching_depth[b, i, j] = torch.isclose(rendered_depth[b, i, j], sampled_depths, atol=1e-1)

    matching_depth = matching_depth.any(dim=-3)  # (..., v1, h, w)

    mask = in_frustum_mask & non_zero_depth & matching_depth
    return mask


# --- Projections ---

def homogenize_points(points):
    """Append a '1' along the final dimension of the tensor (i.e. convert xyz->xyz1)"""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def normalize_homogenous_points(points):
    """Normalize the point vectors"""
    return points / points[..., -1:]


def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):
    """
    Convert pixel space points to camera space points.

    Args:
        pixel_space_points (torch.Tensor): Pixel space points with shape (h, w, 2)
        depth (torch.Tensor): Depth map with shape (b, v, h, w, 1)
        intrinsics (torch.Tensor): Camera intrinsics with shape (b, v, 3, 3)

    Returns:
        torch.Tensor: Camera space points with shape (b, v, h, w, 3).
    """
    pixel_space_points = homogenize_points(pixel_space_points)
    camera_space_points = torch.einsum('b v i j , h w j -> b v h w i', intrinsics.inverse(), pixel_space_points)
    camera_space_points = camera_space_points * depth
    return camera_space_points


def camera_space_to_world_space(camera_space_points, c2w):
    """
    Convert camera space points to world space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum('b v i j , b v h w j -> b v h w i', c2w, camera_space_points)
    return world_space_points[..., :3]


def camera_space_to_pixel_space(camera_space_points, intrinsics):
    """
    Convert camera space points to pixel space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v1, v2, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 3, 3)

    Returns:
        torch.Tensor: World space points with shape (b, v1, v2, h, w, 2).
    """
    camera_space_points = normalize_homogenous_points(camera_space_points)
    pixel_space_points = torch.einsum('b u i j , b v u h w j -> b v u h w i', intrinsics, camera_space_points)
    return pixel_space_points[..., :2]



def world_space_to_camera_space(world_space_points, c2w):
    """
    Convert world space points to pixel space points.

    Args:
        world_space_points (torch.Tensor): World space points with shape (b, v1, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 4, 4)

    Returns:
        torch.Tensor: Camera space points with shape (b, v1, v2, h, w, 3).
    """
    world_space_points = homogenize_points(world_space_points)
    camera_space_points = torch.einsum('b u i j , b v h w j -> b v u h w i', c2w.inverse(), world_space_points)
    return camera_space_points[..., :3]


def unproject_depth(depth, intrinsics, c2w):
    """
    Turn the depth map into a 3D point cloud in world space

    Args:
        depth: (b, v, h, w, 1)
        intrinsics: (b, v, 3, 3)
        c2w: (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """

    # Compute indices of pixels
    h, w = depth.shape[-3], depth.shape[-2]
    x_grid, y_grid = torch.meshgrid(
        torch.arange(w, device=depth.device, dtype=torch.float32),
        torch.arange(h, device=depth.device, dtype=torch.float32),
        indexing='xy'
    )  # (h, w), (h, w)

    # Compute coordinates of pixels in camera space
    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)  # (..., h, w, 2)
    camera_points = pixel_space_to_camera_space(pixel_space_points, depth, intrinsics)  # (..., h, w, 3)

    # Convert points to world space
    world_points = camera_space_to_world_space(camera_points, c2w)  # (..., h, w, 3)

    return world_points


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5

    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5

    return K


def angle_diff_vec3_numpy(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12):
    """
    Compute angle difference between 3D vectors using NumPy.

    Args:
        v1 (np.ndarray): First vector of shape (..., 3)
        v2 (np.ndarray): Second vector of shape (..., 3)
        eps (float, optional): Small epsilon value for numerical stability. Defaults to 1e-12.

    Returns:
        np.ndarray: Angle differences in radians
    """
    return np.arctan2(
        np.linalg.norm(np.cross(v1, v2, axis=-1), axis=-1) + eps, (v1 * v2).sum(axis=-1)
    )


@no_warnings(category=RuntimeWarning)
def points_to_normals(
    point: np.ndarray, mask: np.ndarray = None, edge_threshold: float = None
) -> np.ndarray:
    """
    Calculate normal map from point map. Value range is [-1, 1].

    Args:
        point (np.ndarray): shape (height, width, 3), point map
        mask (optional, np.ndarray): shape (height, width), dtype=bool. Mask of valid depth pixels. Defaults to None.
        edge_threshold (optional, float): threshold for the angle (in degrees) between the normal and the view direction. Defaults to None.

    Returns:
        normal (np.ndarray): shape (height, width, 3), normal map.
    """
    height, width = point.shape[-3:-1]
    has_mask = mask is not None

    if mask is None:
        mask = np.ones_like(point[..., 0], dtype=bool)
    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask
    mask = mask_pad

    pts = np.zeros((height + 2, width + 2, 3), dtype=point.dtype)
    pts[1:-1, 1:-1, :] = point
    up = pts[:-2, 1:-1, :] - pts[1:-1, 1:-1, :]
    left = pts[1:-1, :-2, :] - pts[1:-1, 1:-1, :]
    down = pts[2:, 1:-1, :] - pts[1:-1, 1:-1, :]
    right = pts[1:-1, 2:, :] - pts[1:-1, 1:-1, :]
    normal = np.stack(
        [
            np.cross(up, left, axis=-1),
            np.cross(left, down, axis=-1),
            np.cross(down, right, axis=-1),
            np.cross(right, up, axis=-1),
        ]
    )
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    valid = (
        np.stack(
            [
                mask[:-2, 1:-1] & mask[1:-1, :-2],
                mask[1:-1, :-2] & mask[2:, 1:-1],
                mask[2:, 1:-1] & mask[1:-1, 2:],
                mask[1:-1, 2:] & mask[:-2, 1:-1],
            ]
        )
        & mask[None, 1:-1, 1:-1]
    )
    if edge_threshold is not None:
        view_angle = angle_diff_vec3_numpy(pts[None, 1:-1, 1:-1, :], normal)
        view_angle = np.minimum(view_angle, np.pi - view_angle)
        valid = valid & (view_angle < np.deg2rad(edge_threshold))

    normal = (normal * valid[..., None]).sum(axis=0)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    if has_mask:
        normal_mask = valid.any(axis=0)
        normal = np.where(normal_mask[..., None], normal, 0)
        return normal, normal_mask
    else:
        return normal


def sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    """
    Create a sliding window view of the input array along a specified axis.

    This function creates a memory-efficient view of the input array with sliding windows
    of the specified size and stride. The window dimension is appended to the end of the
    output array's shape. This is useful for operations like convolution, pooling, or
    any analysis that requires examining local neighborhoods in the data.

    Args:
        x (np.ndarray): Input array with shape (..., axis_size, ...)
        window_size (int): Size of the sliding window
        stride (int): Stride of the sliding window (step size between consecutive windows)
        axis (int, optional): Axis to perform sliding window over. Defaults to -1 (last axis)

    Returns:
        np.ndarray: View of the input array with shape (..., n_windows, ..., window_size),
                   where n_windows = (axis_size - window_size + 1) // stride

    Raises:
        AssertionError: If window_size is larger than the size of the specified axis

    Example:
        >>> x = np.array([1, 2, 3, 4, 5, 6])
        >>> sliding_window_1d(x, window_size=3, stride=2)
        array([[1, 2, 3],
               [3, 4, 5]])
    """
    assert x.shape[axis] >= window_size, (
        f"kernel_size ({window_size}) is larger than axis_size ({x.shape[axis]})"
    )
    axis = axis % x.ndim
    shape = (
        *x.shape[:axis],
        (x.shape[axis] - window_size + 1) // stride,
        *x.shape[axis + 1 :],
        window_size,
    )
    strides = (
        *x.strides[:axis],
        stride * x.strides[axis],
        *x.strides[axis + 1 :],
        x.strides[axis],
    )
    x_sliding = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return x_sliding


def sliding_window_nd(
    x: np.ndarray,
    window_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    axis: Tuple[int, ...],
) -> np.ndarray:
    """
    Create sliding windows along multiple dimensions of the input array.

    This function applies sliding_window_1d sequentially along multiple axes to create
    N-dimensional sliding windows. This is useful for operations that need to examine
    local neighborhoods in multiple dimensions simultaneously.

    Args:
        x (np.ndarray): Input array
        window_size (Tuple[int, ...]): Size of the sliding window for each axis
        stride (Tuple[int, ...]): Stride of the sliding window for each axis
        axis (Tuple[int, ...]): Axes to perform sliding window over

    Returns:
        np.ndarray: Array with sliding windows along the specified dimensions.
                   The window dimensions are appended to the end of the shape.

    Note:
        The length of window_size, stride, and axis tuples must be equal.

    Example:
        >>> x = np.random.rand(10, 10)
        >>> windows = sliding_window_nd(x, window_size=(3, 3), stride=(2, 2), axis=(-2, -1))
        >>> # Creates 3x3 sliding windows with stride 2 in both dimensions
    """
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x


def sliding_window_2d(
    x: np.ndarray,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    axis: Tuple[int, int] = (-2, -1),
) -> np.ndarray:
    """
    Create 2D sliding windows over the input array.

    Convenience function for creating 2D sliding windows, commonly used for image
    processing operations like convolution, pooling, or patch extraction.

    Args:
        x (np.ndarray): Input array
        window_size (Union[int, Tuple[int, int]]): Size of the 2D sliding window.
                                                  If int, same size is used for both dimensions.
        stride (Union[int, Tuple[int, int]]): Stride of the 2D sliding window.
                                             If int, same stride is used for both dimensions.
        axis (Tuple[int, int], optional): Two axes to perform sliding window over.
                                         Defaults to (-2, -1) (last two dimensions).

    Returns:
        np.ndarray: Array with 2D sliding windows. The window dimensions (height, width)
                   are appended to the end of the shape.

    Example:
        >>> image = np.random.rand(100, 100)
        >>> patches = sliding_window_2d(image, window_size=8, stride=4)
        >>> # Creates 8x8 patches with stride 4 from the image
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, axis)


def max_pool_1d(
    x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1
):
    """
    Perform 1D max pooling on the input array.

    Max pooling reduces the dimensionality of the input by taking the maximum value
    within each sliding window. This is commonly used in neural networks and signal
    processing for downsampling and feature extraction.

    Args:
        x (np.ndarray): Input array
        kernel_size (int): Size of the pooling kernel
        stride (int): Stride of the pooling operation
        padding (int, optional): Amount of padding to add on both sides. Defaults to 0.
        axis (int, optional): Axis to perform max pooling over. Defaults to -1.

    Returns:
        np.ndarray: Max pooled array with reduced size along the specified axis

    Note:
        - For floating point arrays, padding is done with np.nan values
        - For integer arrays, padding is done with the minimum value of the dtype
        - np.nanmax is used to handle NaN values in the computation

    Example:
        >>> x = np.array([1, 3, 2, 4, 5, 1, 2])
        >>> max_pool_1d(x, kernel_size=3, stride=2)
        array([3, 5, 2])
    """
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == "f" else np.iinfo(x.dtype).min
        padding_arr = np.full(
            (*x.shape[:axis], padding, *x.shape[axis + 1 :]),
            fill_value=fill_value,
            dtype=x.dtype,
        )
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = sliding_window_1d(x, kernel_size, stride, axis)
    max_pool = np.nanmax(a_sliding, axis=-1)
    return max_pool


def max_pool_nd(
    x: np.ndarray,
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    axis: Tuple[int, ...],
) -> np.ndarray:
    """
    Perform N-dimensional max pooling on the input array.

    This function applies max_pool_1d sequentially along multiple axes to perform
    multi-dimensional max pooling. This is useful for downsampling multi-dimensional
    data while preserving the most important features.

    Args:
        x (np.ndarray): Input array
        kernel_size (Tuple[int, ...]): Size of the pooling kernel for each axis
        stride (Tuple[int, ...]): Stride of the pooling operation for each axis
        padding (Tuple[int, ...]): Amount of padding for each axis
        axis (Tuple[int, ...]): Axes to perform max pooling over

    Returns:
        np.ndarray: Max pooled array with reduced size along the specified axes

    Note:
        The length of kernel_size, stride, padding, and axis tuples must be equal.
        Max pooling is applied sequentially along each axis in the order specified.

    Example:
        >>> x = np.random.rand(10, 10, 10)
        >>> pooled = max_pool_nd(x, kernel_size=(2, 2, 2), stride=(2, 2, 2),
        ...                      padding=(0, 0, 0), axis=(-3, -2, -1))
        >>> # Reduces each dimension by half with 2x2x2 max pooling
    """
    for i in range(len(axis)):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], axis[i])
    return x


def max_pool_2d(
    x: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    axis: Tuple[int, int] = (-2, -1),
):
    """
    Perform 2D max pooling on the input array.

    Convenience function for 2D max pooling, commonly used in computer vision
    and image processing for downsampling images while preserving important features.

    Args:
        x (np.ndarray): Input array
        kernel_size (Union[int, Tuple[int, int]]): Size of the 2D pooling kernel.
                                                  If int, same size is used for both dimensions.
        stride (Union[int, Tuple[int, int]]): Stride of the 2D pooling operation.
                                             If int, same stride is used for both dimensions.
        padding (Union[int, Tuple[int, int]]): Amount of padding for both dimensions.
                                              If int, same padding is used for both dimensions.
        axis (Tuple[int, int], optional): Two axes to perform max pooling over.
                                         Defaults to (-2, -1) (last two dimensions).

    Returns:
        np.ndarray: 2D max pooled array with reduced size along the specified axes

    Example:
        >>> image = np.random.rand(64, 64)
        >>> pooled = max_pool_2d(image, kernel_size=2, stride=2, padding=0)
        >>> # Reduces image size from 64x64 to 32x32 with 2x2 max pooling
    """
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return max_pool_nd(x, kernel_size, stride, padding, axis)


@no_warnings(category=RuntimeWarning)
def depth_edge(
    depth: np.ndarray,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth.

    Args:
        depth (np.ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    if mask is None:
        diff = max_pool_2d(
            depth, kernel_size, stride=1, padding=kernel_size // 2
        ) + max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2)
    else:
        diff = max_pool_2d(
            np.where(mask, depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ) + max_pool_2d(
            np.where(mask, -depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol

    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


def depth_aliasing(
    depth: np.ndarray,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the map that indicates the aliasing of x depth map. The aliasing is defined as the pixels which neither close to the maximum nor the minimum of its neighbors.
    Args:
        depth (np.ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    if mask is None:
        diff_max = (
            max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) - depth
        )
        diff_min = (
            max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2) + depth
        )
    else:
        diff_max = (
            max_pool_2d(
                np.where(mask, depth, -np.inf),
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
            - depth
        )
        diff_min = (
            max_pool_2d(
                np.where(mask, -depth, -np.inf),
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
            + depth
        )
    diff = np.minimum(diff_max, diff_min)

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


@no_warnings(category=RuntimeWarning)
def normals_edge(
    normals: np.ndarray, tol: float, kernel_size: int = 3, mask: np.ndarray = None
) -> np.ndarray:
    """
    Compute the edge mask from normal map.

    Args:
        normal (np.ndarray): shape (..., height, width, 3), normal map
        tol (float): tolerance in degrees

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    assert normals.ndim >= 3 and normals.shape[-1] == 3, (
        "normal should be of shape (..., height, width, 3)"
    )
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    padding = kernel_size // 2
    normals_window = sliding_window_2d(
        np.pad(
            normals,
            (
                *([(0, 0)] * (normals.ndim - 3)),
                (padding, padding),
                (padding, padding),
                (0, 0),
            ),
            mode="edge",
        ),
        window_size=kernel_size,
        stride=1,
        axis=(-3, -2),
    )
    if mask is None:
        angle_diff = np.arccos(
            (normals[..., None, None] * normals_window).sum(axis=-3)
        ).max(axis=(-2, -1))
    else:
        mask_window = sliding_window_2d(
            np.pad(
                mask,
                (*([(0, 0)] * (mask.ndim - 3)), (padding, padding), (padding, padding)),
                mode="edge",
            ),
            window_size=kernel_size,
            stride=1,
            axis=(-3, -2),
        )
        angle_diff = np.where(
            mask_window,
            np.arccos((normals[..., None, None] * normals_window).sum(axis=-3)),
            0,
        ).max(axis=(-2, -1))

    angle_diff = max_pool_2d(
        angle_diff, kernel_size, stride=1, padding=kernel_size // 2
    )
    edge = angle_diff > np.deg2rad(tol)
    return edge
