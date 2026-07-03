import contextlib
import os
import sys

import einops
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from pytorch3d.renderer import (
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
    PointsRasterizationSettings
)
from pytorch3d.structures import Pointclouds

from .general_utils import split_n_into_d_parts


def points_padding(points):
    padding = torch.ones_like(points)[..., 0:1]
    points = torch.cat([points, padding], dim=-1)
    return points


class PointsZbufRenderer(PointsRenderer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, point_clouds, **kwargs):
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def point_rendering(K, w2cs, points, colors, device, h, w, background_color=[0, 0, 0],
                    render_radius=0.008, points_per_pixel=8, return_depth=False):
    """
    only support batchsize=1
    :param K: [F,3,3]
    :param w2cs: [F,4,4] opencv
    :param points: [N,3]
    :param colors: [N,3]
    :param background_color: [-1,-1,-1]~[1,1,1]
    :param mask: [1,1,H,W] 0 or 1
    :return: render_rgbs, render_masks
    """
    nframe = w2cs.shape[0]

    # depth contract
    K = K.to(device)
    w2cs = w2cs.to(device)
    c2ws = w2cs.inverse()

    if type(points) != torch.Tensor:
        points = torch.tensor(points, dtype=torch.float32)
    if type(colors) != torch.Tensor:
        colors = torch.tensor(colors, dtype=torch.float32)
    point_cloud = Pointclouds(points=[points.to(device)], features=[colors.to(device)]).extend(nframe)

    # convert opencv to opengl coordinate
    c2ws[:, :, 0] = - c2ws[:, :, 0]
    c2ws[:, :, 1] = - c2ws[:, :, 1]
    w2cs = c2ws.inverse()

    focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1)
    principal_point = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1)
    image_shapes = torch.tensor([[h, w]]).repeat(nframe, 1)
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point,
                                 R=c2ws[:, :3, :3], T=w2cs[:, :3, -1], in_ndc=False,
                                 image_size=image_shapes, device=device)

    raster_settings = PointsRasterizationSettings(
        image_size=(h, w),
        radius=render_radius,
        points_per_pixel=points_per_pixel
    )

    renderer = PointsZbufRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color)
    )

    with suppress_stdout_stderr():
        render_rgbs, zbuf = renderer(point_cloud)  # rgb:[f,h,w,3]

    if not return_depth:
        render_masks = (zbuf[..., 0:1] == -1).float()  # [f,h,w,1]
        render_rgbs = einops.rearrange(render_rgbs, "f h w c -> f c h w")  # [f,3,h,w]
        render_masks = einops.rearrange(render_masks, "f h w c -> f c h w")  # [f,1,h,w]

        return render_rgbs, render_masks
    else:
        render_depth = einops.rearrange(zbuf, "f h w c -> f c h w")  # [f,1,h,w]
        return render_rgbs, render_depth


def multi_gpu_point_rendering(image, Ks, w2cs, render_points, render_colors, image_h, image_w, device, device_num,
                              render_radius=0.008, points_per_pixel=20, slice_size=4, local_rank=0, replace_first_frame=True):
    image_tensor = (transforms.ToTensor()(image) * 2 - 1)[None]

    if type(Ks) != torch.Tensor:
        Ks_tensor = torch.tensor(Ks).float()
    else:
        Ks_tensor = Ks

    if type(w2cs) != torch.Tensor:
        w2cs_tensor = torch.tensor(w2cs).float()
    else:
        w2cs_tensor = w2cs

    ### multi-gpu rendering start ###
    pcd_renders, pcd_mask = [], []
    n_per_gpu_list = split_n_into_d_parts(Ks_tensor.shape[0], device_num)
    cumsum_gpu_list = np.cumsum(n_per_gpu_list)

    if local_rank == 0:
        Ks_tensor = Ks_tensor[:cumsum_gpu_list[0]]
        w2cs_tensor = w2cs_tensor[:cumsum_gpu_list[0]]
    else:
        Ks_tensor = Ks_tensor[cumsum_gpu_list[local_rank - 1]:cumsum_gpu_list[local_rank]]
        w2cs_tensor = w2cs_tensor[cumsum_gpu_list[local_rank - 1]:cumsum_gpu_list[local_rank]]

    gather_pcd_renders_r = [torch.zeros((n_per_gpu_list[j], 1, image_h, image_w), dtype=torch.float32, device=device) for j in range(device_num)]
    gather_pcd_renders_g = [torch.zeros((n_per_gpu_list[j], 1, image_h, image_w), dtype=torch.float32, device=device) for j in range(device_num)]
    gather_pcd_renders_b = [torch.zeros((n_per_gpu_list[j], 1, image_h, image_w), dtype=torch.float32, device=device) for j in range(device_num)]
    gather_pcd_mask = [torch.zeros((n_per_gpu_list[j], 1, image_h, image_w), dtype=torch.float32, device=device) for j in range(device_num)]

    slice_times = w2cs_tensor.shape[0] // slice_size
    if w2cs_tensor.shape[0] % slice_size != 0:
        slice_times += 1

    # for si in tqdm(range(slice_times), desc="final rendering..."):
    for si in range(slice_times):
        pcd_renders_, pcd_mask_ = point_rendering(K=Ks_tensor[si * slice_size:(si + 1) * slice_size],
                                                  w2cs=w2cs_tensor[si * slice_size:(si + 1) * slice_size],
                                                  points=render_points, colors=render_colors,
                                                  h=image_h, w=image_w, render_radius=render_radius, points_per_pixel=points_per_pixel,
                                                  device=device, background_color=[0, 0, 0])

        pcd_renders.append(pcd_renders_)
        pcd_mask.append(pcd_mask_)

    pcd_renders = torch.cat(pcd_renders, dim=0).to(torch.float32)  # [f,3,h,w]
    pcd_mask = torch.cat(pcd_mask, dim=0).to(torch.float32)  # [f,1,h,w]

    dist.barrier()
    dist.all_gather(gather_pcd_renders_r, pcd_renders[:, 0:1].contiguous())
    dist.all_gather(gather_pcd_renders_g, pcd_renders[:, 1:2].contiguous())
    dist.all_gather(gather_pcd_renders_b, pcd_renders[:, 2:3].contiguous())
    dist.all_gather(gather_pcd_mask, pcd_mask)
    dist.barrier()

    gather_pcd_renders_r = torch.cat(gather_pcd_renders_r, dim=0)
    gather_pcd_renders_g = torch.cat(gather_pcd_renders_g, dim=0)
    gather_pcd_renders_b = torch.cat(gather_pcd_renders_b, dim=0)
    gather_pcd_renders = torch.cat([gather_pcd_renders_r, gather_pcd_renders_g, gather_pcd_renders_b], dim=1)

    # gather_pcd_renders = torch.cat(gather_pcd_renders, dim=0)
    gather_pcd_mask = torch.cat(gather_pcd_mask, dim=0)

    if replace_first_frame:
        gather_pcd_renders[0:1] = image_tensor
        gather_pcd_mask[0:1] = 0
    ### multi-gpu rendering end ###

    return gather_pcd_renders, gather_pcd_mask


def depth2pcd(w2c, K, points2d, depth, colors, mask):
    points3d = w2c.inverse() @ points_padding((K.inverse() @ points2d.T).T * depth.reshape(-1, 1)).T
    points3d = points3d.T[:, :3]
    points3d = points3d[mask.reshape(-1)]
    colors = colors[mask.reshape(-1)]

    return points3d, colors