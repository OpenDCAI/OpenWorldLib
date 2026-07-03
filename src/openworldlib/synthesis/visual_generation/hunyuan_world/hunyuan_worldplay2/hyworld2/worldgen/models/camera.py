import einops
import torch
import torch.nn.functional as F


@torch.amp.autocast("cuda", enabled=False)
def batch_sample_rays(intrinsic, extrinsic, image_h=None, image_w=None):
    ''' get rays
    Args:
        intrinsic: [BF, 3, 3],
        extrinsic: [BF, 4, 4],
        h, w: int
        # normalize: let the first camera R=I
    Returns:
        rays_o, rays_d: [BF, N, 3]
    '''

    # FIXME: PPU does not support inverse in GPU
    device = intrinsic.device
    B = intrinsic.shape[0]

    c2w = torch.inverse(extrinsic)[:, :3, :4].to(device)  # [BF,3,4]
    x = torch.arange(image_w, device=device).float() - 0.5
    y = torch.arange(image_h, device=device).float() + 0.5
    points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
    points = einops.repeat(points, 'w h c -> b (h w) c', b=B)
    points = torch.cat([points, torch.ones_like(points)[:, :, 0:1]], dim=-1)
    directions = points @ intrinsic.inverse().to(device).transpose(-1, -2) * 1  # depth is 1

    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)  # [BF,N,3]
    rays_o = c2w[..., :3, 3]  # [BF, 3]

    rays_o = rays_o[:, None, :].expand_as(rays_d)  # [BF, N, 3]

    return rays_o, rays_d


@torch.amp.autocast("cuda", enabled=False)
def embed_rays(rays_o, rays_d, nframe):
    if len(rays_o.shape) == 4:  # [b,f,n,3]
        rays_o = einops.rearrange(rays_o, "b f n c -> (b f) n c")
        rays_d = einops.rearrange(rays_d, "b f n c -> (b f) n c")
    cross_od = torch.cross(rays_o, rays_d, dim=-1)
    cam_emb = torch.cat([rays_d, cross_od], dim=-1)
    cam_emb = einops.rearrange(cam_emb, "(b f) n c -> b f n c", f=nframe)
    return cam_emb


@torch.amp.autocast("cuda", enabled=False)
def camera_center_normalization(w2c, nframe, camera_scale=2.0, is_w2c=False):
    # copy from SEVA, w2c: [BF, 4, 4]
    # ensure the first view is eye matrix
    w2c = w2c.float()
    c2w_view0 = w2c[::nframe].inverse()  # [B,4,4]
    c2w_view0 = c2w_view0.repeat_interleave(nframe, dim=0)  # [BF,4,4]
    if is_w2c: # BUGFIX, w2c should be right multiplied by c2w_view0
        w2c = w2c @ c2w_view0
    else: # this is a bug for 'w2c' to keep consistency with previous version
        w2c = c2w_view0 @ w2c

    # camera centering
    c2w = torch.linalg.inv(w2c)
    camera_dist_2med = torch.norm(c2w[:, :3, 3] - c2w[:, :3, 3].median(0, keepdim=True).values, dim=-1)
    valid_mask = camera_dist_2med <= torch.clamp(torch.quantile(camera_dist_2med, 0.97) * 10, max=1e6)
    c2w[:, :3, 3] -= c2w[valid_mask, :3, 3].mean(0, keepdim=True)
    w2c = torch.linalg.inv(c2w)

    # camera normalization
    camera_dists = c2w[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1, dtype=camera_dists.dtype, device=camera_dists.device),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    w2c[:, :3, 3] *= translation_scaling_factor
    c2w[:, :3, 3] *= translation_scaling_factor

    return w2c


def get_camera_embedding(intrinsic, extrinsic, f, h, w, normalize=True, is_w2c=False):
    if normalize:
        extrinsic = camera_center_normalization(extrinsic, nframe=f, is_w2c=is_w2c)

    rays_o, rays_d = batch_sample_rays(intrinsic, extrinsic, image_h=h, image_w=w)
    camera_embedding = embed_rays(rays_o, rays_d, nframe=f)
    camera_embedding = einops.rearrange(camera_embedding, "b f (h w) c -> b c f h w", h=h, w=w)

    if (~torch.isfinite(camera_embedding)).sum() > 0:
        print("Error camera!!!")
        camera_embedding = torch.zeros_like(camera_embedding)

    return camera_embedding

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


@torch.amp.autocast("cuda", enabled=False)
def unified_camera_normalization(w2c, w2c_ref, camera_scale=2.0):
    w2c = w2c.float()
    w2c_ref = w2c_ref.float()

    num_target = w2c.shape[0]
    # num_ref = w2c_ref.shape[0]

    # Concatenate target and reference w2c for unified coordinate system reset
    combined_w2c = torch.cat([w2c, w2c_ref], dim=0)  # [f + f_ref, 4, 4]
    nframe = combined_w2c.shape[0]

    # Step 1: Ensure the first view is eye matrix (use first target frame as reference)
    c2w_view0 = combined_w2c[0:1].inverse()  # [1, 4, 4] - use first target frame
    c2w_view0 = c2w_view0.repeat(nframe, 1, 1)  # [f + f_ref, 4, 4]
    combined_w2c = combined_w2c @ c2w_view0

    # Step 2: Camera centering (based on ALL cameras)
    combined_c2w = torch.linalg.inv(combined_w2c)
    target_c2w = combined_c2w[:num_target]
    camera_dist_2med = torch.norm(
        target_c2w[:, :3, 3] - target_c2w[:, :3, 3].median(0, keepdim=True).values,
        dim=-1
    )
    valid_mask = camera_dist_2med <= torch.clamp(
        torch.quantile(camera_dist_2med, 0.97) * 10,
        max=1e6
    )
    center_offset = target_c2w[valid_mask, :3, 3].mean(0, keepdim=True)
    combined_c2w[:, :3, 3] -= center_offset
    combined_w2c = torch.linalg.inv(combined_c2w)

    # Step 3: Compute translation_scaling_factor ONLY from target frames
    target_c2w = combined_c2w[:num_target]  # [f, 4, 4]
    camera_dists = target_c2w[:, :3, 3].clone()

    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1, dtype=camera_dists.dtype, device=camera_dists.device),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )

    # Step 4: Apply the same scaling factor to ALL cameras
    combined_w2c[:, :3, 3] *= translation_scaling_factor
    combined_c2w[:, :3, 3] *= translation_scaling_factor

    # Split back to target and reference
    w2c_normalized = combined_w2c[:num_target]
    w2c_ref_normalized = combined_w2c[num_target:]

    return w2c_normalized, w2c_ref_normalized
