import random
import time
from collections import defaultdict
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import colormaps
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_dims, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img


# ============ 从深度图计算法线 (可视化专用) ============
def depth_to_normal(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    从深度图计算法线图

    Args:
        depth: (H, W) 深度图
        K: (3, 3) 相机内参

    Returns:
        normal: (H, W, 3) 法线图，值域 [0, 1]
    """
    H, W = depth.shape

    # 获取相机内参
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 创建像素坐标网格
    u = torch.arange(W, device=depth.device, dtype=depth.dtype)
    v = torch.arange(H, device=depth.device, dtype=depth.dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')  # (H, W)

    # 反投影到相机坐标系
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 计算梯度 (使用 Sobel 或简单差分)
    # dz/du, dz/dv
    dz_du = torch.zeros_like(z)
    dz_dv = torch.zeros_like(z)

    # 简单差分
    dz_du[:, 1:] = z[:, 1:] - z[:, :-1]
    dz_dv[1:, :] = z[1:, :] - z[:-1, :]

    # dx/du, dy/dv (从反投影公式推导)
    dx_du = z / fx + (u - cx) * dz_du / fx
    dy_dv = z / fy + (v - cy) * dz_dv / fy

    # 计算法线: n = normalize(cross(du, dv))
    # du = (dx_du, 0, dz_du)
    # dv = (0, dy_dv, dz_dv)
    # cross = (0*dz_dv - dz_du*dy_dv, dz_du*0 - dx_du*dz_dv, dx_du*dy_dv - 0*0)
    #       = (-dz_du*dy_dv, -dx_du*dz_dv, dx_du*dy_dv)

    normal = torch.stack([
        -dz_du * dy_dv,
        -dx_du * dz_dv,
        dx_du * dy_dv
    ], dim=-1)  # (H, W, 3)

    # 归一化
    normal = F.normalize(normal, dim=-1, eps=1e-6)

    # 处理无效区域（深度为0或边界）
    invalid_mask = (depth < 1e-6) | (depth > 1e6)
    normal[invalid_mask] = torch.tensor([0.0, 0.0, 1.0], device=depth.device)

    # 转换到 [0, 1] 范围用于可视化
    normal_vis = (normal + 1.0) / 2.0  # [-1, 1] -> [0, 1]

    return normal_vis


def load_16bit_png_depth(depth_png: str) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.

    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.

    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization

    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    # Pad inputs to avoid boundary issues
    padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
    pts = F.pad(point_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='constant', value=0).permute(0, 2, 3, 1)

    # Get neighboring points for each pixel
    center = pts[:, 1:-1, 1:-1, :]  # B,H,W,3
    up = pts[:, :-2, 1:-1, :]
    left = pts[:, 1:-1, :-2, :]
    down = pts[:, 2:, 1:-1, :]
    right = pts[:, 1:-1, 2:, :]

    # Compute direction vectors from center to neighbors
    up_dir = up - center
    left_dir = left - center
    down_dir = down - center
    right_dir = right - center

    # Compute four cross products for different normal directions
    n1 = torch.cross(up_dir, left_dir, dim=-1)  # up x left
    n2 = torch.cross(left_dir, down_dir, dim=-1)  # left x down
    n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
    n4 = torch.cross(right_dir, up_dir, dim=-1)  # right x up

    # Validity masks - require both direction pixels to be valid
    v1 = padded_mask[:, :-2, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
    v2 = padded_mask[:, 1:-1, :-2] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:, 1:-1]
    v3 = padded_mask[:, 2:, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
    v4 = padded_mask[:, 1:-1, 2:] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2, 1:-1]

    # Stack normals and validity masks
    normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
    valids = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

    # Normalize normal vectors
    normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def robust_normal_from_xyz(xyz, masks):
    """
    Compute surface normals from xyz map using 4-quadrant cross products
    and robustly aggregating them to avoid smoothing over edges.

    Args:
        xyz: [B, H, W, 3]
        masks: [B, H, W]
    Returns:
        normals: [B, 3, H, W]
        valid: [B, 1, H, W]
    """
    B, H, W, C = xyz.shape
    # Get 4 normals for each pixel (up-left, left-down, etc)
    normals_4, valids_4 = point_map_to_normal(xyz, masks)
    # normals_4: [4, B, H, W, 3]
    # valids_4: [4, B, H, W]

    # Simple average of valid normals for now
    # (A more advanced version could check agreement between normals)

    sum_normals = torch.sum(normals_4 * valids_4.unsqueeze(-1), dim=0)  # [B, H, W, 3]
    count_valids = torch.sum(valids_4, dim=0).unsqueeze(-1)  # [B, H, W, 1]

    avg_normal = sum_normals / (count_valids + 1e-8)
    avg_normal = F.normalize(avg_normal, p=2, dim=-1)  # Normalize

    valid_final = (count_valids > 0).float()

    # Re-orient normals to point towards camera if needed (usually z < 0)
    # But point_map_to_normal implementation doesn't strictly enforce orientation
    # relative to camera center, it just does cross product.
    # Standard convention is normal points to viewer.
    # Let's check dot product with view direction.
    # View direction is -xyz (if camera at 0).

    view_dir = -xyz
    dot = torch.sum(avg_normal * view_dir, dim=-1, keepdim=True)
    avg_normal = torch.where(dot < 0, -avg_normal, avg_normal)

    # Apply mask
    avg_normal = avg_normal * valid_final

    # Permute to [B, 3, H, W]
    return avg_normal.permute(0, 3, 1, 2).contiguous(), valid_final.permute(0, 3, 1, 2) > 0.5


class Depth2Normal(nn.Module):
    """Layer to compute surface normal from depth map
    """

    def __init__(self, ):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(Depth2Normal, self).__init__()

    def init_img_coor(self, height, width, device="cuda"):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                               torch.arange(0, width, dtype=torch.float32, device=device)], indexing='ij')
        meshgrid = torch.stack((x, y))

        # generate homogeneous pixel coordinates
        ones = torch.ones((1, 1, height * width), device=device)
        xy = meshgrid.reshape(2, -1).unsqueeze(0)
        xy = torch.cat([xy, ones], 1)

        return xy

    def back_projection(self, depth, inv_K, xy, img_like_out=False, scale=1.0):
        """
        Args:
            depth (Nx1xHxW): depth map
            inv_K (Nx4x4): inverse camera intrinsics
            img_like_out (bool): if True, the output shape is Nx4xHxW; else Nx4x(HxW)
        Returns:
            points (Nx4x(HxW)): 3D points in homogeneous coordinates
        """
        B, C, H, W = depth.shape
        depth = depth.contiguous()

        points = torch.matmul(inv_K[:, :3, :3], xy)
        points = depth.view(depth.shape[0], 1, -1) * points
        depth_descale = points[:, 2:3, :] / scale
        points = torch.cat((points[:, 0:2, :], depth_descale), dim=1)
        # points = torch.cat([points, ones], 1)

        if img_like_out:
            points = points.reshape(depth.shape[0], 3, H, W)
        return points

    def forward(self, depth, intrinsics, masks, scale):
        """
        Args:
            depth (Nx1xHxW): depth map
            #inv_K (Nx4x4): inverse camera intrinsics
            intrinsics (Nx4x4): camera intrinsics
        Returns:
            normal (Nx3xHxW): normalized surface normal
            mask (Nx1xHxW): valid mask for surface normal
        """
        B, C, H, W = depth.shape
        xy = self.init_img_coor(height=H, width=W, device=depth.device)
        inv_K = intrinsics.inverse()

        xyz = self.back_projection(depth, inv_K, xy, scale=scale)  # [N, 4, HxW]

        xyz = xyz.view(depth.shape[0], 3, H, W)
        xyz = xyz[:, :3].permute(0, 2, 3, 1).contiguous()  # [b, h, w, c]

        normals, normal_masks = robust_normal_from_xyz(xyz, masks.squeeze(1))
        # normals, normal_masks = get_surface_normalv2(xyz, mask_valid=masks.squeeze())
        # normal_masks = normal_masks & masks
        return normals, normal_masks


def perturb_cameras(
    c2ws: np.ndarray,
    c2w_eval: np.ndarray,
    yaw_angle_deg: float = 5.0,
    pitch_angle_deg: float = 5.0,
    translation_scale: float = 0.05,
    seed: int = None,
) -> np.ndarray:
    """
    对单个验证相机（OpenCV 坐标系）进行扰动：
      - 绕相机局部 y 轴旋转 yaw（水平偏航）
      - 绕相机局部 x 轴旋转 pitch（俯仰）
      - 不绕相机局部 z 轴旋转（无 roll）
      - xyz 位移扰动（以全局 c2ws 的场景范围为参考）

    OpenCV 相机坐标系约定：
        x → 右,  y → 下,  z → 前（光轴方向）

    Args:
        c2ws:              (N, 4, 4) 全局训练相机外参（c2w），用于估算场景尺度
        c2w_eval:          (4, 4)   单个验证相机外参（c2w）
        yaw_angle_deg:     绕相机 y 轴（水平）最大扰动角度（度）
        pitch_angle_deg:   绕相机 x 轴（俯仰）最大扰动角度（度）
        translation_scale: 位移扰动幅度系数，实际位移 = translation_scale * scene_scale
        seed:              随机种子，None 表示不固定

    Returns:
        c2w_perturbed: (4, 4) 扰动后的验证相机外参
    """
    assert c2ws.ndim == 3 and c2ws.shape[1:] == (4, 4), "c2ws 应为 (N, 4, 4)"
    assert c2w_eval.shape == (4, 4), "c2w_eval 应为 (4, 4)"

    rng = np.random.default_rng(seed)

    # ================================================================== #
    # 1. 根据全局 c2ws 估算场景尺度
    # ================================================================== #
    camera_positions = c2ws[:, :3, 3]                       # (N, 3)
    scene_center = camera_positions.mean(axis=0)            # (3,)
    dists = np.linalg.norm(camera_positions - scene_center, axis=1)
    scene_scale = float(np.max(dists))                      # 场景尺度

    c2w = c2w_eval.copy().astype(np.float64)
    rot = c2w[:3, :3]   # (3, 3)  相机朝向
    pos = c2w[:3, 3]    # (3,)    相机位置

    # ================================================================== #
    # 2. 在相机局部坐标系中构造 Yaw / Pitch 旋转矩阵（无 Roll）
    # ================================================================== #
    yaw   = np.deg2rad(rng.uniform(-yaw_angle_deg,   yaw_angle_deg))
    pitch = np.deg2rad(rng.uniform(-pitch_angle_deg, pitch_angle_deg))

    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # 绕局部 y 轴旋转 yaw（OpenCV: y 朝下）
    R_y = np.array([
        [ cy, 0.0,  sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0,  cy],
    ], dtype=np.float64)

    # 绕局部 x 轴旋转 pitch（OpenCV: x 朝右）
    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  cp, -sp],
        [0.0,  sp,  cp],
    ], dtype=np.float64)

    # 先 yaw 再 pitch（可按需调换顺序，小角度下差异极小）
    R_local = R_y @ R_x                              # (3, 3) 局部旋转

    # ================================================================== #
    # 3a. 更新相机朝向：右乘 = 在相机局部坐标系中旋转
    # ================================================================== #
    rot_perturbed = rot @ R_local                     # (3, 3)

    # ================================================================== #
    # 3b. 更新相机位置：绕场景中心做等价的世界坐标系旋转
    # ================================================================== #
    #   R_local 是局部旋转，其等价的世界旋转为:
    #       R_world = rot @ R_local @ rot^T
    R_world = rot @ R_local @ rot.T                   # (3, 3)

    pos_centered = pos - scene_center
    pos_perturbed = R_world @ pos_centered + scene_center

    # ================================================================== #
    # 3c. xyz 位移扰动
    # ================================================================== #
    max_trans = translation_scale * scene_scale
    delta_t = rng.uniform(-max_trans, max_trans, size=(3,))

    # ================================================================== #
    # 4. 组装扰动后的 c2w
    # ================================================================== #
    c2w_perturbed = c2w.copy()
    c2w_perturbed[:3, :3] = rot_perturbed
    c2w_perturbed[:3, 3]  = pos_perturbed + delta_t

    return c2w_perturbed.astype(np.float32)