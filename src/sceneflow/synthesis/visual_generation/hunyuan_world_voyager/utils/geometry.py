
import numpy as np
import torch


DEFAULT_FOV_RAD = 0.9424777960769379  # 54 degrees by default


def get_camera_dist(
    source_c2ws: torch.Tensor,  # N x 3 x 4
    target_c2ws: torch.Tensor,  # M x 3 x 4
    mode: str = "translation",
):
    if mode == "rotation":
        dists = torch.acos(
            (
                (
                    torch.matmul(
                        source_c2ws[:, None, :3, :3],
                        target_c2ws[None, :, :3, :3].transpose(-1, -2),
                    )
                    .diagonal(offset=0, dim1=-2, dim2=-1)
                    .sum(-1)
                    - 1
                )
                / 2
            ).clamp(-1, 1)
        ) * (180 / torch.pi)
    elif mode == "translation":
        dists = torch.norm(
            source_c2ws[:, None, :3, 3] - target_c2ws[None, :, :3, 3], dim=-1
        )
    else:
        raise NotImplementedError(
            f"Mode {mode} is not implemented for finding nearest source indices."
        )
    return dists


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


def to_hom_pose(pose):
    # get homogeneous coordinates of the input pose
    if pose.shape[-2:] == (3, 4):
        pose_hom = torch.eye(4, device=pose.device)[
            None].repeat(pose.shape[0], 1, 1)
        pose_hom[:, :3, :] = pose
        return pose_hom
    return pose


def get_default_intrinsics(
    fov_rad=DEFAULT_FOV_RAD,
    aspect_ratio=1.0,
):
    if not isinstance(fov_rad, torch.Tensor):
        fov_rad = torch.tensor(
            [fov_rad] if isinstance(fov_rad, (int, float)) else fov_rad
        )
    if aspect_ratio >= 1.0:  # W >= H
        focal_x = 0.5 / torch.tan(0.5 * fov_rad)
        focal_y = focal_x * aspect_ratio
    else:  # W < H
        focal_y = 0.5 / torch.tan(0.5 * fov_rad)
        focal_x = focal_y / aspect_ratio
    intrinsics = focal_x.new_zeros((focal_x.shape[0], 3, 3))
    intrinsics[:, torch.eye(3, device=focal_x.device, dtype=bool)] = torch.stack(
        [focal_x, focal_y, torch.ones_like(focal_x)], dim=-1
    )
    intrinsics[:, :, -1] = torch.tensor(
        [0.5, 0.5, 1.0], device=focal_x.device, dtype=focal_x.dtype
    )
    return intrinsics


def get_image_grid(img_h, img_w):
    # add 0.5 is VERY important especially when your img_h and img_w
    # is not very large (e.g., 72)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    y_range = torch.arange(img_h, dtype=torch.float32).add_(0.5)
    x_range = torch.arange(img_w, dtype=torch.float32).add_(0.5)
    Y, X = torch.meshgrid(y_range, x_range, indexing="ij")  # [H,W]
    xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
    return to_hom(xy_grid)  # [HW,3]


def img2cam(X, cam_intr):
    cam_intr = torch.from_numpy(np.linalg.inv(np.array(cam_intr))).float()
    return X @ cam_intr.transpose(-1, -2)


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = torch.from_numpy(np.linalg.inv(
        np.array(to_hom_pose(pose)))).float()[..., :3, :4]
    return X_hom @ pose_inv.transpose(-1, -2)


def get_center_and_ray(img_h, img_w, pose, intr):  # [HW,2]
    # given the intrinsic/extrinsic matrices, get the camera center and ray directions]
    # assert(opt.camera.model=="perspective")

    # compute center and ray
    grid_img = get_image_grid(img_h, img_w)  # [HW,3]
    grid_3D_cam = img2cam(grid_img.to(intr.device), intr.float())  # [B,HW,3]
    center_3D_cam = torch.zeros_like(grid_3D_cam)  # [B,HW,3]

    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D_cam, pose)  # [B,HW,3]
    center_3D = cam2world(center_3D_cam, pose)  # [B,HW,3]
    ray = grid_3D - center_3D  # [B,HW,3]

    return center_3D, ray, grid_3D_cam


def get_plucker_coordinates(
    extrinsics_src,
    extrinsics,
    intrinsics=None,
    fov_rad=DEFAULT_FOV_RAD,
    target_size=[72, 72],
):
    if intrinsics is None:
        intrinsics = get_default_intrinsics(fov_rad).to(extrinsics.device)
    else:
        if not (
            torch.all(intrinsics[:, :2, -1] >= 0)
            and torch.all(intrinsics[:, :2, -1] <= 1)
        ):
            intrinsics[:,
                       :2] /= intrinsics.new_tensor(target_size).view(1, -1, 1) * 8
        # you should ensure the intrisics are expressed in
        # resolution-independent normalized image coordinates just performing a
        # very simple verification here checking if principal points are
        # between 0 and 1
        assert (
            torch.all(intrinsics[:, :2, -1] >= 0)
            and torch.all(intrinsics[:, :2, -1] <= 1)
        ), "Intrinsics should be expressed in resolution-independent normalized image coordinates."

    c2w_src = torch.from_numpy(np.linalg.inv(np.array(extrinsics_src))).float()
    # transform coordinates from the source camera's coordinate system to the coordinate system of the respective camera
    extrinsics_rel = torch.einsum(
        "vnm,vmp->vnp", extrinsics, c2w_src[None].repeat(
            extrinsics.shape[0], 1, 1)
    )

    intrinsics[:, :2] *= extrinsics.new_tensor(
        [
            target_size[1],  # w
            target_size[0],  # h
        ]
    ).view(1, -1, 1)
    centers, rays, grid_cam = get_center_and_ray(
        img_h=target_size[0],
        img_w=target_size[1],
        pose=extrinsics_rel[:, :3, :],
        intr=intrinsics,
    )

    rays = torch.nn.functional.normalize(rays, dim=-1)
    plucker = torch.cat((rays, torch.cross(centers, rays, dim=-1)), dim=-1)
    plucker = plucker.permute(0, 2, 1).reshape(
        plucker.shape[0], -1, *target_size)
    return plucker
