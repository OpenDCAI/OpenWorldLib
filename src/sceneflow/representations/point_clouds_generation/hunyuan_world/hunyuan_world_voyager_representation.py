import torch
import numpy as np
import os
import cv2
import json
import imageio
import pyexr
from typing import Dict, Union

from ...base_representation import BaseRepresentation
# import the corresponding depth model
from ....base_models.three_dimensions.depth.moge.model.v1 import MoGeModel
from ....base_models.three_dimensions.depth.depth_anything.depth_anything_v1.dpt import DepthAnything
from ....base_models.three_dimensions.depth.depth_anything.depth_anything_v1.adapter import DepthAnythingAdapter

"""
# Use DepthAnything as the depth model
representation = Depth2PointCloudRepresentation.from_pretrained(
    pretrained_model_path="/path/to/depth_anything_vitl14.pth",  # or HuggingFace repo ID
    device="cuda",
    depth_model_name='depthanything',
    encoder='vitl',  # optional: 'vits', 'vitb', 'vitl'
)
"""


DEPTH_MODEL_DICT = {
    'moge_v1': MoGeModel,
    'depthanything': DepthAnythingAdapter,
}


# from VGGT: https://github.com/facebookresearch/vggt/blob/main/vggt/utils/geometry.py
def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4).

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points


def render_from_cameras_videos(points, colors, extrinsics, intrinsics, height, width):
    
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    
    render_list = []
    mask_list = []
    depth_list = []
    # Render from each camera
    for frame_idx in range(len(extrinsics)):
        # Get corresponding camera parameters
        extrinsic = extrinsics[frame_idx]
        intrinsic = intrinsics[frame_idx]
        
        camera_coords = (extrinsic @ homogeneous_points.T).T[:, :3]
        projected = (intrinsic @ camera_coords.T).T
        uv = projected[:, :2] / projected[:, 2].reshape(-1, 1)
        depths = projected[:, 2]    
        
        pixel_coords = np.round(uv).astype(int)  # pixel_coords (h*w, 2)      
        valid_pixels = (  # valid_pixels (h*w, )      valid_pixels is the valid pixels in width and height
            (pixel_coords[:, 0] >= 0) & 
            (pixel_coords[:, 0] < width) & 
            (pixel_coords[:, 1] >= 0) & 
            (pixel_coords[:, 1] < height)
        )
        
        pixel_coords_valid = pixel_coords[valid_pixels]  # (h*w, 2) to (valid_count, 2)
        colors_valid = colors[valid_pixels]
        depths_valid = depths[valid_pixels]
        uv_valid = uv[valid_pixels]
        
        
        valid_mask = (depths_valid > 0) & (depths_valid < 60000) # & normal_angle_mask
        colors_valid = colors_valid[valid_mask]
        depths_valid = depths_valid[valid_mask]
        pixel_coords_valid = pixel_coords_valid[valid_mask]

        # Initialize depth buffer
        depth_buffer = np.full((height, width), np.inf)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Vectorized depth buffer update
        if len(pixel_coords_valid) > 0:
            rows = pixel_coords_valid[:, 1]
            cols = pixel_coords_valid[:, 0]
                            
            # Sort by depth (near to far)
            sorted_idx = np.argsort(depths_valid)
            rows = rows[sorted_idx]
            cols = cols[sorted_idx]
            depths_sorted = depths_valid[sorted_idx]
            colors_sorted = colors_valid[sorted_idx]

            # Vectorized depth buffer update
            depth_buffer[rows, cols] = np.minimum(
                depth_buffer[rows, cols], 
                depths_sorted
            )
            
            # Get the minimum depth index for each pixel
            flat_indices = rows * width + cols  # Flatten 2D coordinates to 1D index
            unique_indices, idx = np.unique(flat_indices, return_index=True)
            
            # Recover 2D coordinates from flattened indices
            final_rows = unique_indices // width
            final_cols = unique_indices % width
            
            image[final_rows, final_cols] = colors_sorted[idx, :3].astype(np.uint8)

        mask = np.zeros_like(depth_buffer, dtype=np.uint8)
        mask[depth_buffer != np.inf] = 255
        
        render_list.append(image)
        mask_list.append(mask)
        depth_list.append(depth_buffer)
    
    return render_list, mask_list, depth_list


class HunyuanWorldVoyagerRepresentation(BaseRepresentation):
    def __init__(self, depth_model=None):
        super().__init__()
        self.depth_model = depth_model

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_path,
                        device=None,
                        depth_model_name='moge_v1',
                        **kwargs):
        depth_model_name = depth_model_name
        if depth_model_name not in DEPTH_MODEL_DICT:
            raise ValueError(f"Unsupported depth model: {depth_model_name}. "
                           f"Available models: {list(DEPTH_MODEL_DICT.keys())}")
        
        if depth_model_name == 'moge_v1':
            pretrained_model_path = os.path.join(pretrained_model_path, 'model.pt')
            depth_model_class = DEPTH_MODEL_DICT[depth_model_name]
            depth_model = depth_model_class.from_pretrained(
                pretrained_model_path, 
                local_files_only=kwargs.get('local_files_only', False),
            ).to(device)
        elif depth_model_name == 'depthanything':
            # Initialize DepthAnything model directly (HuggingFace or local repo id)
            encoder = kwargs.get('encoder', 'vitl')
            model_id_or_path = pretrained_model_path or f"LiheYoung/depth_anything_{encoder}14"
            depth_core = DepthAnything.from_pretrained(model_id_or_path)
            depth_model = DepthAnythingAdapter(depth_core, device=device)
        else:
            depth_model_class = DEPTH_MODEL_DICT[depth_model_name]
            depth_model = depth_model_class.from_pretrained(
                pretrained_model_path, 
                local_files_only=kwargs.get('local_files_only', False),
            ).to(device)

        instance = cls(depth_model=depth_model)
        return instance
    
    def get_representation(self, data):
        input_image = data['image']
        input_image_tensor = data['image_tensor']
        # camera parameters
        extrinsics = data['extrinsics']
        intrinsics = data['intrinsics']

        with torch.no_grad():
            output = self.depth_model.infer(input_image_tensor)

        depth = np.array(output['depth'].detach().cpu())
        depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4

        # 反向投影点云
        point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
        points = point_map.reshape(-1, 3)
        colors = input_image.reshape(-1, 3) if hasattr(input_image, 'reshape') else input_image.view(-1, 3)
        return points, colors, depth
    
    def render_video(self, points, colors, extrinsics, intrinsics, height, width):
        render_list, mask_list, depth_list = render_from_cameras_videos(
            points, colors, extrinsics, intrinsics, height=height, width=width
        )
        return render_list, mask_list, depth_list
    
    def save_representation_video(
        self, render_list, mask_list, depth_list, render_output_dir,
        separate=True, ref_image=None, ref_depth=None,
        Width=512, Height=512,
        min_percentile=2, max_percentile=98
    ):
        video_output_dir = os.path.join(render_output_dir)
        os.makedirs(video_output_dir, exist_ok=True)
        video_input_dir = os.path.join(render_output_dir, "video_input")
        os.makedirs(video_input_dir, exist_ok=True)

        value_list = []
        for i, (render, mask, depth) in enumerate(zip(render_list, mask_list, depth_list)):

            # Sky part is the region where depth_max is, also included in mask
            mask = mask > 0
            # depth_max = np.max(depth)
            # non_sky_mask = (depth != depth_max)
            # mask = mask & non_sky_mask
            depth[mask] = 1 / (depth[mask] + 1e-6)
            depth_values = depth[mask]
            
            min_percentile = np.percentile(depth_values, 2)
            max_percentile = np.percentile(depth_values, 98)
            value_list.append((min_percentile, max_percentile))

            depth[mask] = (depth[mask] - min_percentile) / (max_percentile - min_percentile)
            depth[~mask] = depth[mask].min()
            

            # resize to 512x512
            render = cv2.resize(render, (Width, Height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize((mask.astype(np.float32) * 255).astype(np.uint8), \
                (Width, Height), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (Width, Height), interpolation=cv2.INTER_LINEAR)

            # Save mask as png
            mask_path = os.path.join(video_input_dir, f"mask_{i:04d}.png")
            imageio.imwrite(mask_path, mask)
            
            if separate:
                render_path = os.path.join(video_input_dir, f"render_{i:04d}.png")
                imageio.imwrite(render_path, render)
                depth_path = os.path.join(video_input_dir, f"depth_{i:04d}.exr")
                pyexr.write(depth_path, depth)  
            else:
                render = np.concatenate([render, depth], axis=-3)
                render_path = os.path.join(video_input_dir, f"render_{i:04d}.png")
                imageio.imwrite(render_path, render)

            if i == 0:
                if separate:
                    ref_image_path = os.path.join(video_output_dir, f"ref_image.png")
                    imageio.imwrite(ref_image_path, ref_image)
                    ref_depth_path = os.path.join(video_output_dir, f"ref_depth.exr")
                    pyexr.write(ref_depth_path, depth) 
                else:
                    ref_image = np.concatenate([ref_image, depth], axis=-3)
                    ref_image_path = os.path.join(video_output_dir, f"ref_image.png")
                    imageio.imwrite(ref_image_path, ref_image)

        with open(os.path.join(video_output_dir, f"depth_range.json"), "w") as f:
            json.dump(value_list, f)