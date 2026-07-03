from pathlib import Path
import struct
from openworldlib.base_models.three_dimensions.point_clouds.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from openworldlib.base_models.three_dimensions.point_clouds.vggt.vggt.utils.geometry import closed_form_inverse_se3
import math
import os

import cv2
import numpy as np
from PIL import Image
import PIL

import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


def crop_image_depth_and_intrinsic_by_pp(
    image,
    depth_map,
    intrinsic,
    target_shape,
    track=None,
    filepath=None,
    strict=False,
    conf_map=None,
):
    """
    Crops the given image and depth map around the camera's principal point, as defined by `intrinsic`.
    Specifically:
      - Ensures that the crop is centered on (cx, cy).
      - Optionally pads the image (and depth map) if `strict=True` and the result is smaller than `target_shape`.
      - Shifts the camera intrinsic matrix (and `track` if provided) accordingly.

    Args:
        image (np.ndarray):
            Input image array of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array of shape (H, W), or None if not available.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3). The principal point is assumed to be at (intrinsic[1,2], intrinsic[0,2]).
        target_shape (tuple[int, int]):
            Desired output shape.
        track (np.ndarray or None):
            Optional array of shape (N, 2). Interpreted as (x, y) pixel coordinates. Will be shifted after cropping.
        filepath (str or None):
            An optional file path for debug logging (only used if strict mode triggers warnings).
        strict (bool):
            If True, will zero-pad to ensure the exact target_shape even if the cropped region is smaller.

    Raises:
        AssertionError:
            If the input image is smaller than `target_shape`.
        ValueError:
            If the cropped image is larger than `target_shape` (in strict mode), which should not normally happen.

    Returns:
        tuple:
            (cropped_image, cropped_depth_map, updated_intrinsic, updated_track)

            - cropped_image (np.ndarray): Cropped (and optionally padded) image.
            - cropped_depth_map (np.ndarray or None): Cropped (and optionally padded) depth map.
            - updated_intrinsic (np.ndarray): Intrinsic matrix adjusted for the crop.
            - updated_track (np.ndarray or None): Track array adjusted for the crop, or None if track was not provided.
    """
    original_size = np.array(image.shape)
    intrinsic = np.copy(intrinsic)
    if original_size[0] < target_shape[0]:
        error_message = (
            f"Width check failed: original width {original_size[0]} "
            f"is less than target width {target_shape[0]}."
        )
        print(error_message)
        raise AssertionError(error_message)
    if original_size[1] < target_shape[1]:
        error_message = (
            f"Height check failed: original height {original_size[1]} "
            f"is less than target height {target_shape[1]}."
        )
        print(error_message)
        raise AssertionError(error_message)

    # Identify principal point (cx, cy) from intrinsic
    cx = intrinsic[1, 2]
    cy = intrinsic[0, 2]
    # Compute how far we can crop in each direction
    if strict:
        half_x = min((target_shape[0] / 2), cx)
        half_y = min((target_shape[1] / 2), cy)
    else:
        half_x = min((target_shape[0] / 2), cx, original_size[0] - cx)
        half_y = min((target_shape[1] / 2), cy, original_size[1] - cy)

    # Compute starting indices
    start_x = math.floor(cx) - math.floor(half_x)
    start_y = math.floor(cy) - math.floor(half_y)

    assert start_x >= 0
    assert start_y >= 0

    # Compute ending indices
    if strict:
        end_x = start_x + target_shape[0]
        end_y = start_y + target_shape[1]
    else:
        end_x = start_x + 2 * math.floor(half_x)
        end_y = start_y + 2 * math.floor(half_y)

    # Perform the crop
    image = image[start_x:end_x, start_y:end_y, :]
    if depth_map is not None:
        depth_map = depth_map[start_x:end_x, start_y:end_y]
    if conf_map is not None:
        conf_map = conf_map[start_x:end_x, start_y:end_y]
    # Shift the principal point in the intrinsic
    intrinsic[1, 2] = intrinsic[1, 2] - start_x
    intrinsic[0, 2] = intrinsic[0, 2] - start_y

    # Adjust track if provided
    if track is not None:
        track[:, 1] = track[:, 1] - start_x
        track[:, 0] = track[:, 0] - start_y

    # If strict, zero-pad if the new shape is smaller than target_shape
    if strict:
        if (image.shape[:2] != target_shape).any():
            current_h, current_w = image.shape[:2]
            target_h, target_w = target_shape[0], target_shape[1]
            pad_h = target_h - current_h
            pad_w = target_w - current_w
            if pad_h < 0 or pad_w < 0:
                raise ValueError(
                    f"The cropped image is bigger than the target shape: "
                    f"cropped=({current_h},{current_w}), "
                    f"target=({target_h},{target_w})."
                )
            image = np.pad(
                image,
                pad_width=((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if depth_map is not None:
                depth_map = np.pad(
                    depth_map,
                    pad_width=((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )
            if conf_map is not None:
                conf_map = np.pad(
                    conf_map,
                    pad_width=((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

    return image, depth_map, intrinsic, track, conf_map


def normalize_scene(
    extrinsics: torch.Tensor,
    first_moge_world=None,
    first_moge_mask=None,
) -> torch.Tensor:
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    assert device == torch.device("cpu")

    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    new_extrinsics = torch.matmul(
        extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1)
    )  # (B,N,4,4)

    R = extrinsics[:, 0, :3, :3]
    t = extrinsics[:, 0, :3, 3]
    first_moge_world = first_moge_world.to(torch.float32)
    first_moge_world = (first_moge_world @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    point_masks = first_moge_mask.to(torch.bool).to(first_moge_world.device)
    final_mask = torch.zeros_like(point_masks)
    dist = first_moge_world.norm(dim=-1)
    valid_dists = dist[point_masks]
    if valid_dists.numel() > 0:
        outlier_threshold = torch.quantile(valid_dists, 0.95)
        final_mask = point_masks & (dist <= outlier_threshold).to(point_masks.dtype).to(point_masks.device)

    dist_sum = (dist * final_mask).sum(dim=[1, 2, 3])
    valid_count = final_mask.sum(dim=[1, 2, 3])
    avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)

    new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)

    return new_extrinsics


def get_intrinsic_matrix(camera) -> np.ndarray:
    """
    Build and return the 3x3 intrinsic matrix K from Camera object.

    Args:
        camera: Camera object with fx, fy, cx, cy attributes

    Returns:
        np.ndarray: 3x3 intrinsic matrix
    """
    K = np.eye(3)
    K[0, 0] = camera.fx
    K[1, 1] = camera.fy
    K[0, 2] = camera.cx
    K[1, 2] = camera.cy
    return K


def batch_depth_to_world(prediction, extrinsics, intrinsics):
    """
    Convert batch depth maps to world coordinates.

    Args:
        prediction: Model prediction containing depth
        extrinsics: Camera extrinsic matrices
        intrinsics: Camera intrinsic matrices

    Returns:
        tuple: (world_points, masks)
    """
    prediction["depth"][torch.isinf(prediction["depth"]) | torch.isnan(prediction["depth"])] = 0
    depths = prediction["depth"].unsqueeze(0).cpu().numpy()
    extrinsics = extrinsics.cpu().numpy()
    intrinsics = intrinsics.cpu().numpy()
    world_points_all = []
    masks_all = []
    for f in range(depths.shape[0]):
        wp, _, mask = depth_to_world_coords_points(depths[f], extrinsics[f], intrinsics[f])
        world_points_all.append(wp)
        masks_all.append(mask)

    world_points_all = torch.from_numpy(np.stack(world_points_all))
    masks_all = torch.from_numpy(np.stack(masks_all))
    return world_points_all, masks_all


def save_video_imageio(frames_np, output_path, fps=16):
    """
    Save video frames using imageio.

    Args:
        frames_np: Video frames tensor
        output_path: Output video file path
        fps: Frames per second
    """
    try:
        import imageio

        imageio.mimwrite(
            str(output_path),
            frames_np,
            fps=fps,
            quality=8,
            macro_block_size=1,
        )
        print(f"[OK] Method 5 (imageio): Saved to {output_path}")
    except Exception as e:
        print(f"[FAIL] Method 5 (imageio) failed: {e}")


def resize_by_short_side_and_update_intrinsics(
    image,
    depth_map,
    intrinsic,
    short_side_target,
    track=None,
    pixel_center=True,
    conf_map=None,
):
    long_side_target = short_side_target * 592.0 / 336.0

    original_h, original_w = image.shape[:2]
    if original_h > original_w:
        scale_h = long_side_target / original_h
        scale_w = short_side_target / original_w
    else:
        scale_h = short_side_target / original_h
        scale_w = long_side_target / original_w

    scale = max(scale_h, scale_w)

    intrinsic = np.copy(intrinsic)

    image_pil = Image.fromarray(image)
    new_w = int(round(original_w * scale))
    new_h = int(round(original_h * scale))
    output_resolution = (new_w, new_h)
    resample_filter = lanczos if scale < 1 else bicubic
    image_pil = image_pil.resize(output_resolution, resample=resample_filter)
    image = np.array(image_pil)

    if depth_map is not None:
        depth_map = cv2.resize(
            depth_map,
            output_resolution,
            interpolation=cv2.INTER_NEAREST,
        )
    if conf_map is not None:
        conf_map = cv2.resize(
            conf_map,
            output_resolution,
            interpolation=cv2.INTER_NEAREST,
        )

    if pixel_center:
        intrinsic[0, 2] += 0.5
        intrinsic[1, 2] += 0.5
    intrinsic[:2, :] *= scale
    if track is not None:
        track *= scale
    if pixel_center:
        intrinsic[0, 2] -= 0.5
        intrinsic[1, 2] -= 0.5

    if depth_map is not None:
        assert image.shape[:2] == depth_map.shape[:2], (
            f"Resized image shape {image.shape[:2]} "
            f"does not match depth shape {depth_map.shape[:2]}"
        )

    return image, depth_map, intrinsic, track, conf_map


def resize_image_depth_and_intrinsic(
    image,
    depth_map,
    intrinsic,
    target_shape,
    original_size,
    track=None,
    pixel_center=True,
    safe_bound=4,
    rescale_aug=True,
):
    """
    Resizes the given image and depth map (if provided) to slightly larger than `target_shape`,
    updating the intrinsic matrix (and track array if present). Optionally uses random rescaling
    to create some additional margin (based on `rescale_aug`).

    Steps:
      1. Compute a scaling factor so that the resized result is at least `target_shape + safe_bound`.
      2. Apply an optional triangular random factor if `rescale_aug=True`.
      3. Resize the image with LANCZOS if downscaling, BICUBIC if upscaling.
      4. Resize the depth map with nearest-neighbor.
      5. Update the camera intrinsic and track coordinates (if any).

    Args:
        image (np.ndarray):
            Input image array (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array (H, W), or None if unavailable.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3).
        target_shape (np.ndarray or tuple[int, int]):
            Desired final shape (height, width).
        original_size (np.ndarray or tuple[int, int]):
            Original size of the image in (height, width).
        track (np.ndarray or None):
            Optional (N, 2) array of pixel coordinates. Will be scaled.
        pixel_center (bool):
            If True, accounts for 0.5 pixel center shift during resizing.
        safe_bound (int or float):
            Additional margin (in pixels) to add to target_shape before resizing.
        rescale_aug (bool):
            If True, randomly increase the `safe_bound` within a certain range to simulate augmentation.

    Returns:
        tuple:
            (resized_image, resized_depth_map, updated_intrinsic, updated_track)

            - resized_image (np.ndarray): The resized image.
            - resized_depth_map (np.ndarray or None): The resized depth map.
            - updated_intrinsic (np.ndarray): Camera intrinsic updated for new resolution.
            - updated_track (np.ndarray or None): Track array updated or None if not provided.

    Raises:
        AssertionError:
            If the shapes of the resized image and depth map do not match.
    """
    if rescale_aug:
        random_boundary = np.random.triangular(0, 0, 0.3)
        safe_bound = safe_bound + random_boundary * target_shape.max()

    resize_scales = (target_shape + safe_bound) / original_size
    max_resize_scale = np.max(resize_scales)
    intrinsic = np.copy(intrinsic)

    # Convert image to PIL for resizing
    image = Image.fromarray(image)
    input_resolution = np.array(image.size)
    output_resolution = np.floor(input_resolution * max_resize_scale).astype(int)
    image = image.resize(
        tuple(output_resolution), resample=lanczos if max_resize_scale < 1 else bicubic
    )
    image = np.array(image)

    if depth_map is not None:
        depth_map = cv2.resize(
            depth_map,
            output_resolution,
            fx=max_resize_scale,
            fy=max_resize_scale,
            interpolation=cv2.INTER_NEAREST,
        )

    actual_size = np.array(image.shape[:2])
    actual_resize_scale = np.max(actual_size / original_size)

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5

    intrinsic[:2, :] = intrinsic[:2, :] * actual_resize_scale

    if track is not None:
        track = track * actual_resize_scale

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5

    assert image.shape[:2] == depth_map.shape[:2]
    return image, depth_map, intrinsic, track


def threshold_depth_map(
    depth_map: np.ndarray,
    max_percentile: float = 99,
    min_percentile: float = 1,
    max_depth: float = -1,
) -> np.ndarray:
    """
    Thresholds a depth map using percentile-based limits and optional maximum depth clamping.

    Steps:
      1. If `max_depth > 0`, clamp all values above `max_depth` to zero.
      2. Compute `max_percentile` and `min_percentile` thresholds using nanpercentile.
      3. Zero out values above/below these thresholds, if thresholds are > 0.

    Args:
        depth_map (np.ndarray):
            Input depth map (H, W).
        max_percentile (float):
            Upper percentile (0-100). Values above this will be set to zero.
        min_percentile (float):
            Lower percentile (0-100). Values below this will be set to zero.
        max_depth (float):
            Absolute maximum depth. If > 0, any depth above this is set to zero.
            If <= 0, no maximum-depth clamp is applied.

    Returns:
        np.ndarray:
            Depth map (H, W) after thresholding. Some or all values may be zero.
            Returns None if depth_map is None.
    """
    if depth_map is None:
        return None

    depth_map = depth_map.astype(float, copy=True)

    # Optional clamp by max_depth
    if max_depth > 0:
        depth_map[depth_map > max_depth] = 0.0

    # Percentile-based thresholds
    depth_max_thres = (
        np.nanpercentile(depth_map, max_percentile) if max_percentile > 0 else None
    )
    depth_min_thres = (
        np.nanpercentile(depth_map, min_percentile) if min_percentile > 0 else None
    )

    # Apply the thresholds if they are > 0
    if depth_max_thres is not None and depth_max_thres > 0:
        depth_map[depth_map > depth_max_thres] = 0.0
    if depth_min_thres is not None and depth_min_thres > 0:
        depth_map[depth_map < depth_min_thres] = 0.0

    return depth_map


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a depth map to world coordinates (HxWx3) given the camera extrinsic and intrinsic.
    Returns both the world coordinates and the intermediate camera coordinates,
    as well as a mask for valid depth.

    Args:
        depth_map (np.ndarray):
            Depth map of shape (H, W).
        extrinsic (np.ndarray):
            Extrinsic matrix of shape (3, 4), representing the camera pose in OpenCV convention (camera-from-world).
        intrinsic (np.ndarray):
            Intrinsic matrix of shape (3, 3).
        eps (float):
            Small epsilon for thresholding valid depth.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (world_coords_points, cam_coords_points, point_mask)

            - world_coords_points: (H, W, 3) array of 3D points in world frame.
            - cam_coords_points: (H, W, 3) array of 3D points in camera frame.
            - point_mask: (H, W) boolean array where True indicates valid (non-zero) depth.
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # The extrinsic is camera-from-world, so invert it to transform
    # camera->world
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """
    Unprojects a depth map into camera coordinates, returning (H, W, 3).

    Args:
        depth_map (np.ndarray):
            Depth map of shape (H, W).
        intrinsic (np.ndarray):
            3x3 camera intrinsic matrix.
            Assumes zero skew and standard OpenCV layout:
            [ fx   0   cx ]
            [  0  fy   cy ]
            [  0   0    1 ]

    Returns:
        np.ndarray:
            An (H, W, 3) array, where each pixel is mapped to (x, y, z) in the camera frame.
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))

    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


def rotate_90_degrees(
    image, depth_map, extri_opencv, intri_opencv, clockwise=True, track=None
):
    """
    Rotates the input image, depth map, and camera parameters by 90 degrees.

    Applies one of two 90-degree rotations:
    - Clockwise
    - Counterclockwise (if clockwise=False)

    The extrinsic and intrinsic matrices are adjusted accordingly to maintain
    correct camera geometry. Track coordinates are also updated if provided.

    Args:
        image (np.ndarray):
            Input image of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map of shape (H, W), or None if not available.
        extri_opencv (np.ndarray):
            Extrinsic matrix (3x4) in OpenCV convention.
        intri_opencv (np.ndarray):
            Intrinsic matrix (3x3).
        clockwise (bool):
            If True, rotates the image 90 degrees clockwise; else 90 degrees counterclockwise.
        track (np.ndarray or None):
            Optional (N, 2) track array. Will be rotated accordingly.

    Returns:
        tuple:
            (
                rotated_image,
                rotated_depth_map,
                new_extri_opencv,
                new_intri_opencv,
                new_track
            )

            Where each is the updated version after the rotation.
    """
    image_height, image_width = image.shape[:2]

    # Rotate the image and depth map
    rotated_image, rotated_depth_map = rotate_image_and_depth_rot90(
        image, depth_map, clockwise
    )
    # Adjust the intrinsic matrix
    new_intri_opencv = adjust_intrinsic_matrix_rot90(
        intri_opencv, image_width, image_height, clockwise
    )

    if track is not None:
        new_track = adjust_track_rot90(track, image_width, image_height, clockwise)
    else:
        new_track = None

    # Adjust the extrinsic matrix
    new_extri_opencv = adjust_extrinsic_matrix_rot90(extri_opencv, clockwise)

    return (
        rotated_image,
        rotated_depth_map,
        new_extri_opencv,
        new_intri_opencv,
        new_track,
    )


def rotate_image_and_depth_rot90(image, depth_map, clockwise):
    """
    Rotates the given image and depth map by 90 degrees (clockwise or counterclockwise),
    using a transpose+flip pattern.

    Args:
        image (np.ndarray):
            Input image of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map of shape (H, W), or None if not available.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        tuple:
            (rotated_image, rotated_depth_map)
    """
    rotated_depth_map = None
    if clockwise:
        rotated_image = np.transpose(image, (1, 0, 2))  # Transpose height and width
        rotated_image = np.flip(rotated_image, axis=1)  # Flip horizontally
        if depth_map is not None:
            rotated_depth_map = np.transpose(depth_map, (1, 0))
            rotated_depth_map = np.flip(rotated_depth_map, axis=1)
    else:
        rotated_image = np.transpose(image, (1, 0, 2))  # Transpose height and width
        rotated_image = np.flip(rotated_image, axis=0)  # Flip vertically
        if depth_map is not None:
            rotated_depth_map = np.transpose(depth_map, (1, 0))
            rotated_depth_map = np.flip(rotated_depth_map, axis=0)
    return np.copy(rotated_image), np.copy(rotated_depth_map)


def adjust_extrinsic_matrix_rot90(extri_opencv, clockwise):
    """
    Adjusts the extrinsic matrix (3x4) for a 90-degree rotation of the image.

    The rotation is in the image plane. This modifies the camera orientation
    accordingly. The function applies either a clockwise or counterclockwise
    90-degree rotation.

    Args:
        extri_opencv (np.ndarray):
            Extrinsic matrix (3x4) in OpenCV convention.
        clockwise (bool):
            If True, rotate extrinsic for a 90-degree clockwise image rotation;
            otherwise, counterclockwise.

    Returns:
        np.ndarray:
            A new 3x4 extrinsic matrix after the rotation.
    """
    R = extri_opencv[:, :3]
    t = extri_opencv[:, 3]

    if clockwise:
        R_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    else:
        R_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    new_R = np.dot(R_rotation, R)
    new_t = np.dot(R_rotation, t)
    new_extri_opencv = np.hstack((new_R, new_t.reshape(-1, 1)))
    return new_extri_opencv


def adjust_intrinsic_matrix_rot90(
    intri_opencv,
    image_width,
    image_height,
    clockwise,
):
    """
    Adjusts the intrinsic matrix (3x3) for a 90-degree rotation of the image in the image plane.

    Args:
        intri_opencv (np.ndarray):
            Intrinsic matrix (3x3).
        image_width (int):
            Original width of the image.
        image_height (int):
            Original height of the image.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        np.ndarray:
            A new 3x3 intrinsic matrix after the rotation.
    """
    fx, fy, cx, cy = (
        intri_opencv[0, 0],
        intri_opencv[1, 1],
        intri_opencv[0, 2],
        intri_opencv[1, 2],
    )

    new_intri_opencv = np.eye(3)
    if clockwise:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = image_height - cy
        new_intri_opencv[1, 2] = cx
    else:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = cy
        new_intri_opencv[1, 2] = image_width - cx

    return new_intri_opencv


def adjust_track_rot90(track, image_width, image_height, clockwise):
    """
    Adjusts a track (N, 2) for a 90-degree rotation of the image in the image plane.

    Args:
        track (np.ndarray):
            (N, 2) array of pixel coordinates, each row is (x, y).
        image_width (int):
            Original image width.
        image_height (int):
            Original image height.
        clockwise (bool):
            Whether the rotation is 90 degrees clockwise or counterclockwise.

    Returns:
        np.ndarray:
            A new track of shape (N, 2) after rotation.
    """
    if clockwise:
        # (x, y) -> (y, image_width - 1 - x)
        new_track = np.stack((track[:, 1], image_width - 1 - track[:, 0]), axis=-1)
    else:
        # (x, y) -> (image_height - 1 - y, x)
        new_track = np.stack((image_height - 1 - track[:, 1], track[:, 0]), axis=-1)

    return new_track


def read_image_cv2(path: str, rgb: bool = True) -> np.ndarray:
    """
    Reads an image from disk using OpenCV, returning it as an RGB image array (H, W, 3).

    Args:
        path (str):
            File path to the image.
        rgb (bool):
            If True, convert the image to RGB.
            If False, leave the image in BGR/grayscale.

    Returns:
        np.ndarray or None:
            A numpy array of shape (H, W, 3) if successful,
            or None if the file does not exist or could not be read.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"File does not exist or is empty: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image={path}. Retrying...")
        img = cv2.imread(path)
        if img is None:
            print("Retry failed.")
            return None

    if rgb:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def read_depth(path: str, scale_adjustment=1.0) -> np.ndarray:
    """
    Reads a depth map from disk in either .exr or .png format. The .exr is loaded using OpenCV
    with the environment variable OPENCV_IO_ENABLE_OPENEXR=1. The .png is assumed to be a 16-bit
    PNG (converted from half float).

    Args:
        path (str):
            File path to the depth image. Must end with .exr or .png.
        scale_adjustment (float):
            A multiplier for adjusting the loaded depth values (default=1.0).

    Returns:
        np.ndarray:
            A float32 array (H, W) containing the loaded depth. Zeros or non-finite values
            may indicate invalid regions.

    Raises:
        ValueError:
            If the file extension is not supported.
    """
    if path.lower().endswith(".exr"):
        # Ensure OPENCV_IO_ENABLE_OPENEXR is set to "1"
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
        d[d > 1e9] = 0.0
    elif path.lower().endswith(".png"):
        d = load_16big_png_depth(path)
    else:
        raise ValueError(f'unsupported depth file name "{path}"')

    d = d * scale_adjustment
    d[~np.isfinite(d)] = 0.0

    return d


def load_16big_png_depth(depth_png: str) -> np.ndarray:
    """
    Loads a 16-bit PNG as a half-float depth map (H, W), returning a float32 NumPy array.

    Implementation detail:
      - PIL loads 16-bit data as 32-bit "I" mode.
      - We reinterpret the bits as float16, then cast to float32.

    Args:
        depth_png (str):
            File path to the 16-bit PNG.

    Returns:
        np.ndarray:
            A float32 depth array of shape (H, W).
    """
    with Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(
                np.array(depth_pil, dtype=np.uint16),
                dtype=np.float16,
            )
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


class Camera:
    """Camera class for storing camera parameters and transformations."""

    def __init__(self, entry):
        """
        Initialize Camera from entry data.

        Args:
            entry: List containing camera parameters [id, fx, fy, cx, cy, 0.0, 0.0, w2c_matrix...]
        """
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def _infer_intrinsics(data, image_size, K=None):
    """
    Infer camera intrinsics from data or provided matrix.

    Args:
        data: Dictionary containing focal_length
        image_size: Tuple of (height, width) or None
        K: Pre-defined intrinsic matrix or None

    Returns:
        tuple: (fx, fy, cx, cy)
    """
    if K is not None:
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        return fx, fy, cx, cy
    fx = fy = float(data.get("focal_length", 500))
    H, W = image_size
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    return fx, fy, cx, cy


def cameras_json_to_camera_list(data, image_size=None, K=None):
    """
    Convert camera JSON data to list of Camera objects.

    Args:
        data: Dictionary containing 'cameras_interp' key
        image_size: Tuple of (height, width) or None
        K: Pre-defined intrinsic matrix or None

    Returns:
        list: List of Camera objects
    """
    fx, fy, cx, cy = _infer_intrinsics(data, image_size=image_size, K=K)
    mats = data.get("cameras_interp")
    if mats is None:
        trajectory = data.get("trajectory", {})
        num_frames = int(trajectory.get("num_frames", data.get("num_frames", 17)))
        step = float(trajectory.get("step", 0.02))
        axis = trajectory.get("axis", "z")
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        mats = []
        for idx in range(num_frames):
            c2w = np.eye(4, dtype=np.float64)
            c2w[axis_index, 3] = step * idx
            mats.append(c2w.tolist())

    cam_list = []
    for idx, c2w in enumerate(mats):
        c2w = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
        w2c = np.linalg.inv(c2w)
        w2c_3x4 = w2c[:3, :]
        entry = [idx, fx, fy, cx, cy, 0.0, 0.0] + w2c_3x4.flatten().tolist()
        cam_list.append(Camera(entry))
    return cam_list


def _to_uint8_colors(colors):
    if colors.dtype == np.uint8:
        return colors
    c = colors.astype(np.float32)
    if c.max() <= 1.0:
        c = c * 255.0
    c = np.clip(c, 0, 255).astype(np.uint8)
    return c


def save_colored_pointcloud_ply(
    points,
    colors,
    out_path,
    stride=1,
    max_points=None,
    valid_mask=None,
    save_first_frame=True,
):
    assert points.ndim == 4 and points.shape[-1] == 3, "points should be in the shape of [F,H,W,3]"
    assert colors.shape == points.shape, "colors should has the same size as points"
    F, H, W, _ = points.shape
    if not save_first_frame:
        points = points[1:]
        colors = colors[1:]
        valid_mask = valid_mask[1:]

    valid_mask = valid_mask[:, ::stride, ::stride]
    if valid_mask is not None:
        pts = points[:, ::stride, ::stride, :][valid_mask].reshape(-1, 3)
        cols = colors[:, ::stride, ::stride, :][valid_mask].reshape(-1, 3)
    else:
        pts = points[:, ::stride, ::stride, :].reshape(-1, 3)
        cols = colors[:, ::stride, ::stride, :].reshape(-1, 3)

    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    cols = cols[valid]

    if (max_points is not None) and (pts.shape[0] > max_points):
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]

    cols = _to_uint8_colors(cols)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_vertices = pts.shape[0]

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {num_vertices}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header\n",
        ]
    ).encode("ascii")

    with open(out_path, "wb") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(
            pts.astype(np.float32),
            cols.astype(np.uint8),
        ):
            f.write(
                struct.pack(
                    "<fffBBB",
                    float(x),
                    float(y),
                    float(z),
                    int(r),
                    int(g),
                    int(b),
                )
            )


def get_pointclouds(prediction, fix_first_frame=False):
    B, F, H, W, _ = prediction["world_points"].shape

    # === compute extrinsic and intrinsic ===
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        prediction["pose_enc"], (H, W), pose_encoding_type="absT_quaR_FoV"
    )  # extrinsic: (B,F,3,4), intrinsic: (B,F,3,3)

    extrinsic = extrinsic[0].to(torch.float32).cpu().numpy()
    intrinsic = intrinsic[0].to(torch.float32).cpu().numpy()
    depths = prediction["depth"][0].squeeze(-1).cpu().numpy()  # [F,H,W]
    if fix_first_frame:
        extrinsic[0] = np.eye(3, 4)

    recon_worldpoints = []
    for f in range(F):
        wp, _, mask = depth_to_world_coords_points(depths[f], extrinsic[f], intrinsic[f])
        recon_worldpoints.append(wp)
    recon_worldpoints = np.stack(recon_worldpoints)  # [F,H,W,3]

    return recon_worldpoints
