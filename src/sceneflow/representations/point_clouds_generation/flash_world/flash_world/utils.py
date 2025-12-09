from io import BytesIO
import math
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import importlib
from plyfile import PlyData, PlyElement

import copy

class EmbedContainer(nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = nn.Parameter(tensor)
    
    def forward(self):
        return self.tensor

@torch.no_grad
def zero_init(module):
    if type(module) is torch.nn.Conv2d or type(module) is torch.nn.Linear:
        module.weight.zero_()
        module.bias.zero_()
    return module

def import_str(string):
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

"""
from https://github.com/Kai-46/minFM/blob/main/utils/ema.py
Exponential Moving Average (EMA) utilities for PyTorch models.

This module provides utilities for maintaining and updating EMA models,
which are commonly used to improve model stability and generalization
in training deep neural networks. It supports both regular tensors and
DTensors (from FSDP-wrapped models).
"""
class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        # 判断是否是FSDP模型
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(fsdp_module, FSDP):
            with FSDP.summon_full_params(fsdp_module, writeback=False):
                for n, p in fsdp_module.module.named_parameters():
                    self.shadow[n] = p.detach().clone().float().cpu()
        else:
            for n, p in fsdp_module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(fsdp_module, FSDP):
            with FSDP.summon_full_params(fsdp_module, writeback=False):
                for n, p in fsdp_module.module.named_parameters():
                    self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)
        else:
            for n, p in fsdp_module.named_parameters():
                print(n, self.shadow[n])
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        # load EMA weights into an (unwrapped) copy of the generator
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))

def create_raymaps(cameras, h, w):
    rays_o, rays_d = create_rays(cameras, h, w)
    raymaps = torch.cat([rays_d, rays_o - (rays_o * rays_d).sum(dim=-1, keepdim=True) * rays_d], dim=-1)
    return raymaps

# def create_raymaps(cameras, h, w):
#     rays_o, rays_d = create_rays(cameras, h, w)
#     raymaps = torch.cat([rays_d, torch.cross(rays_d, rays_o, dim=-1)], dim=-1)
#     return raymaps

class EMANorm(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.register_buffer('magnitude_ema', torch.ones([]))
        self.beta = beta

    def forward(self, x):
        if self.training:
            magnitude_cur = x.detach().to(torch.float32).square().mean()
            self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema.to(torch.float32), self.beta))
        input_gain = self.magnitude_ema.rsqrt()
        x = x.mul(input_gain)
        return x
    
class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000, time_factor: float = 1000.0, zero_weight: bool = True):
        super().__init__()
        self.max_period = max_period
        self.time_factor = time_factor
        self.dim = dim
        if zero_weight:
            self.weight = nn.Parameter(torch.zeros(dim))
        else:
            self.weight = None

    def forward(self, t):
        if self.weight is None:
            return timestep_embedding(t, self.dim, self.max_period, self.time_factor)
        else:
            return timestep_embedding(t, self.dim, self.max_period, self.time_factor) * self.weight.unsqueeze(0)

# @torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def timestep_embedding(t, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

# from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)

@torch.amp.autocast(device_type="cuda", enabled=False)
def normalize_cameras(cameras, return_meta=False, ref_w2c=None, T_norm=None, n_frame=None):
    B, N = cameras.shape[:2]
     
    c2ws = torch.zeros(B, N, 3, 4, device=cameras.device)

    c2ws[..., :3, :3] = quaternion_to_matrix(cameras[..., 0:4])
    c2ws[..., :, 3] = cameras[..., 4:7]

    _c2ws = c2ws

    ref_w2c = torch.inverse(matrix_to_square(_c2ws[:, :1])) if ref_w2c is None else ref_w2c
    _c2ws = (ref_w2c.repeat(1, N, 1, 1) @ matrix_to_square(_c2ws))[..., :3, :]

    if n_frame is not None:
        T_norm = _c2ws[..., :n_frame, :3, 3].norm(dim=-1).max(dim=1)[0][..., None, None] if T_norm is None else T_norm
    else:
        T_norm = _c2ws[..., :3, 3].norm(dim=-1).max(dim=1)[0][..., None, None] if T_norm is None else T_norm

    _c2ws[..., :3, 3] = _c2ws[..., :3, 3] / (T_norm + 1e-2)

    R = matrix_to_quaternion(_c2ws[..., :3, :3])
    T = _c2ws[..., :3, 3]
    cameras = torch.cat([R.float(), T.float(), cameras[..., 7:]], dim=-1)

    if return_meta:
        return cameras, ref_w2c, T_norm
    else:
        return cameras

def create_rays(cameras, h, w, uv_offset=None):
    prefix_shape = cameras.shape[:-1]
    cameras = cameras.flatten(0, -2)
    device = cameras.device
    N = cameras.shape[0]

    c2w = torch.eye(4, device=device)[None].repeat(N, 1, 1)
    c2w[:, :3, :3] = quaternion_to_matrix(cameras[:, :4])
    c2w[:, :3, 3] = cameras[:, 4:7]

    # fx, fy, cx, cy should be divided by original H, W
    fx, fy, cx, cy = cameras[:, 7:].chunk(4, -1)

    fx, cx = fx * w, cx * w
    fy, cy = fy * h, cy * h

    inds = torch.arange(0, h*w, device=device).expand(N, h*w)
        
    i = inds % w + 0.5
    j = torch.div(inds, w, rounding_mode='floor') + 0.5

    u = i / cx + (uv_offset[..., 0].reshape(N, h*w) if uv_offset is not None else 0) 
    v = j / cy + (uv_offset[..., 1].reshape(N, h*w) if uv_offset is not None else 0) 

    zs = - torch.ones_like(i)
    xs = - (u - 1) * cx / fx * zs
    ys = (v - 1) * cy / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)

    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)

    rays_o = c2w[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)

    rays_o = rays_o.reshape(*prefix_shape, h, w, 3)
    rays_d = rays_d.reshape(*prefix_shape, h, w, 3)

    return rays_o, rays_d

def matrix_to_square(mat):
    l = len(mat.shape)
    if l==3:
        return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],1,1).to(mat.device)],dim=1)
    elif l==4:
        return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],mat.shape[1],1,1).to(mat.device)],dim=2)

def export_gaussians(gaussians, opacity_threshold=0.00, T_norm=None, ply_path=None, spz_path=None):

    sh_degree = int(math.sqrt((gaussians.shape[-1] - sum([3, 1, 3, 4])) / 3 - 1))

    xyz, opacity, scale, rotation, feature = gaussians.float().split([3, 1, 3, 4, (sh_degree + 1)**2 * 3], dim=-1)
     
    means3D = xyz.contiguous().float()
    opacity = opacity.contiguous().float()
    scales = scale.contiguous().float()
    rotations = rotation.contiguous().float()
    shs = feature.contiguous().float() # [N, 1, 3]

    # print(means3D.shape, opacity.shape, scales.shape, rotations.shape, shs.shape)

    # prune by opacity
    if opacity_threshold > 0:
        mask = opacity[..., 0] >= opacity_threshold
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        print("Gaussian percentage: ", mask.float().mean())

    if T_norm is not None:
        means3D = means3D * T_norm.item()
        scales = scales * T_norm.item()

    # invert activation to make it compatible with the original ply format
    opacity = torch.log(opacity/(1-opacity))
    scales = torch.log(scales + 1e-8)

    xyzs = means3D.detach() # .cpu().numpy()
    f_dc = shs.detach().flatten(start_dim=1).contiguous() #.cpu().numpy()
    opacities = opacity.detach() #.cpu().numpy()
    scales = scales.detach() #.cpu().numpy()
    rotations = rotations.detach() #.cpu().numpy()

    """spz
    Data Layout
    The Python bindings maintain the same data layout as the C++ library:

    Positions: [x1, y1, z1, x2, y2, z2, ...]
    Scales: [sx1, sy1, sz1, sx2, sy2, sz2, ...] (log-scale)
    Rotations: [x1, y1, z1, w1, x2, y2, z2, w2, ...] (quaternions)
    Alphas: [a1, a2, a3, ...] (before sigmoid activation)
    Colors: [r1, g1, b1, r2, g2, b2, ...] (base RGB)
    Spherical Harmonics: Coefficient-major order, e.g., for degree 1: [sh1n1_r, sh1n1_g, sh1n1_b, sh10_r, sh10_g, sh10_b, sh1p1_r, sh1p1_g, sh1p1_b, ...]
    """
    if spz_path is not None:
        import spz

        cloud = spz.GaussianCloud()
        cloud.sh_degree = sh_degree

        cloud.positions = xyzs.flatten().cpu().numpy()
        cloud.scales = scales.flatten().cpu().numpy()
        cloud.rotations = rotations[:, [3, 0, 1, 2]].flatten().cpu().numpy()
        cloud.alphas = opacities.flatten().cpu().numpy()
        cloud.colors = f_dc[..., :3].flatten().cpu().numpy()
        cloud.sh = f_dc[..., 3:].flatten().cpu().numpy()

        spz.save_spz(cloud, spz.PackOptions(), spz_path)
    
    if ply_path is not None:
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        attributes = torch.cat((xyzs, f_dc, opacities, scales, rotations), dim=1).cpu().numpy()

        elements = np.rec.fromarrays([attributes[:, i] for i in range(attributes.shape[1])], names=l, formats=['f4'] * len(l))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(ply_path)

@torch.amp.autocast(device_type="cuda", enabled=False)
def quaternion_slerp(
    q0, q1, fraction, spin: int = 0, shortestpath: bool = True
):
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    d = (q0 * q1).sum(-1)
    if shortestpath:
        # invert rotation
        d[d < 0.0] = -d[d < 0.0]
        q1[d < 0.0] = q1[d < 0.0]

    _d = d.clamp(0, 1.0)

    # theta = torch.arccos(d) * fraction
    # q2 = q1 - q0 * d
    # q2 = q2 / (q2.norm(dim=-1) + 1e-10)
    
    # return torch.cos(theta) * q0 + torch.sin(theta) * q2

    angle = torch.acos(_d) + spin * math.pi
    isin = 1.0 / (torch.sin(angle)+ 1e-10)
    q0_ = q0 * (torch.sin((1.0 - fraction) * angle) * isin)[..., None]
    q1_ = q1 * (torch.sin(fraction * angle) * isin)[..., None]

    q = q0_ + q1_

    q[angle < 1e-5] = q0[angle < 1e-5]
    # q[fraction < 1e-5] = q0[fraction < 1e-5]
    # q[fraction > 1 - 1e-5] = q1[fraction > 1 - 1e-5]
    # q[(d.abs() - 1).abs() < 1e-5] = q0[(d.abs() - 1).abs() < 1e-5]

    return q

def sample_from_two_pose(pose_a, pose_b, fraction, noise_strengths=[0, 0]):
    """
    Args:
        pose_a: first pose
        pose_b: second pose
        fraction
    """

    quat_a = pose_a[..., :4]
    quat_b = pose_b[..., :4]

    dot = torch.sum(quat_a * quat_b, dim=-1, keepdim=True)
    quat_b = torch.where(dot < 0, -quat_b, quat_b)

    quaternion = quaternion_slerp(quat_a, quat_b, fraction)
    quaternion = torch.nn.functional.normalize(quaternion + torch.randn_like(quaternion) * noise_strengths[0], dim=-1)

    T = (1 - fraction)[:, None] * pose_a[..., 4:] + fraction[:, None] * pose_b[..., 4:]
    T = T + torch.randn_like(T) * noise_strengths[1]

    new_pose = pose_a.clone()
    new_pose[..., :4] = quaternion
    new_pose[..., 4:] = T
    return new_pose

def sample_from_dense_cameras(dense_cameras, t, noise_strengths=[0, 0, 0, 0]):
    N, C = dense_cameras.shape
    M = t.shape
    
    left = torch.floor(t * (N-1)).long().clamp(0, N-2)
    right = left + 1
    fraction = t * (N-1) - left

    a = torch.gather(dense_cameras, 0, left[..., None].repeat(1, C))
    b = torch.gather(dense_cameras, 0, right[..., None].repeat(1, C))

    new_pose = sample_from_two_pose(a[:, :7], 
                                    b[:, :7], fraction, noise_strengths=noise_strengths[:2])

    new_ins = (1 - fraction)[:, None] * a[:, 7:] + fraction[:, None] * b[:, 7:]

    return torch.cat([new_pose, new_ins], dim=1)

