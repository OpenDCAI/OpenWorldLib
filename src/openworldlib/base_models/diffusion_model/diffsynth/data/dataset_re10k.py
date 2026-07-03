import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

from torch.utils.data.dataset import Dataset
from packaging import version as pver
from einops import rearrange
from ....three_dimensions.point_clouds.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri

from PIL import Image

class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        if torch.rand(1).item() < self.p:
            return torch.zeros(n_image, dtype=torch.bool)
        else:
            return torch.ones(n_image, dtype=torch.bool)

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)


class Camera(object):
    def __init__(self, entry):
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
def create_camera_params_from_batch(extrinsics_np, intrinsics_np):
    cam_params = []
    for i, (ext_mat, int_mat) in enumerate(zip(extrinsics_np, intrinsics_np)):
        fx, fy = int_mat[0, 0], int_mat[1, 1]
        cx, cy = int_mat[0, 2], int_mat[1, 2]
        w2c_mat = ext_mat
        entry = [i, fx, fy, cx, cy, 0, 0] + w2c_mat.flatten().tolist()
        camera_obj = Camera(entry)
        cam_params.append(camera_obj)
    return cam_params


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


class RealEstate10KPoseProcessor:
    def __init__(
            self,
            sample_stride=4,
            minimum_sample_stride=1,
            sample_n_frames=16,
            relative_pose=False,
            zero_t_first_frame=False,
            sample_size=[256, 384],
            rescale_fxy=False,
            shuffle_frames=False,
            use_flip=False,
            return_clip_name=False,
            is_i2v=False
    ):
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames
        self.return_clip_name = return_clip_name
        self.is_i2v = is_i2v

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        if use_flip:
            pixel_transforms = [transforms.Resize(sample_size),
                                RandomHorizontalFlipWithPose(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        else:
            pixel_transforms = [transforms.Resize(sample_size),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.pixel_transforms = pixel_transforms
        self.shuffle_frames = shuffle_frames
        self.use_flip = use_flip

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def load_cameras(self, pose_file):
        with open(pose_file, 'r') as f:
            poses = f.readlines()
        if 'youtube' in poses[0]:
            poses = [pose.strip().split(' ') for pose in poses[1:]]
        else:
            poses = [pose.strip().split(' ') for pose in poses]
        cam_params = [[float(x) for x in pose] for pose in poses]
        cam_params = [Camera(cam_param) for cam_param in cam_params]
        return cam_params

    def get_plucker_embedding(self, pose_file, image_path=None):
        cam_params = self.load_cameras(pose_file)
        assert len(cam_params) >= self.sample_n_frames
        total_frames = len(cam_params)

        current_sample_stride = self.sample_stride

        if total_frames < self.sample_n_frames * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.sample_n_frames)
            current_sample_stride = random.randint(self.minimum_sample_stride, maximum_sample_stride)

        cropped_length = self.sample_n_frames * current_sample_stride
        start_frame_ind = 0
        # start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.sample_n_frames
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)

        if self.shuffle_frames:
            perm = np.random.permutation(self.sample_n_frames)
            frame_indices = frame_indices[perm]

        cam_params = [cam_params[indice] for indice in frame_indices]
        if self.rescale_fxy:
            ori_w, ori_h = Image.open(image_path).size
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:       # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            else:                                          # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]
        intrinsics = np.asarray([[cam_param.fx * self.sample_size[1],
                                  cam_param.fy * self.sample_size[0],
                                  cam_param.cx * self.sample_size[1],
                                  cam_param.cy * self.sample_size[0]]
                                 for cam_param in cam_params], dtype=np.float32)
        intrinsics = torch.as_tensor(intrinsics)[None]                  # [1, n_frame, 4]
        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        c2w = torch.as_tensor(c2w_poses)[None]                          # [1, n_frame, 4, 4]
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool, device=c2w.device)
        # plucker_embedding = ray_condition(intrinsics, c2w, self.sample_size[0], self.sample_size[1], device='cpu',
        #                                   flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()
        plucker_embedding = ray_condition(intrinsics, c2w, self.sample_size[0], self.sample_size[1], device='cpu',
                                          flip_flag=flip_flag)
        return plucker_embedding

    def get_plucker_embedding_direct_from_cam_params(self, pose_enc, image_size,  image_path=None):
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_size ,pose_encoding_type="absT_quaR_FoV")
        extrinsics_np = extrinsic.cpu().numpy().squeeze(0)  # Shape: (5, 3, 4)
        intrinsics_np = intrinsic.cpu().numpy().squeeze(0)  # Shape: (5, 3, 3)
        cam_params= create_camera_params_from_batch(extrinsics_np, intrinsics_np)
        assert len(cam_params) >= self.sample_n_frames
        total_frames = len(cam_params)

        current_sample_stride = self.sample_stride

        if total_frames < self.sample_n_frames * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.sample_n_frames)
            current_sample_stride = random.randint(self.minimum_sample_stride, maximum_sample_stride)

        cropped_length = self.sample_n_frames * current_sample_stride
        start_frame_ind = 0
        # start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.sample_n_frames
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)

        if self.shuffle_frames:
            perm = np.random.permutation(self.sample_n_frames)
            frame_indices = frame_indices[perm]

        cam_params = [cam_params[indice] for indice in frame_indices]
        if self.rescale_fxy:
            ori_w, ori_h = Image.open(image_path).size
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:       # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            else:                                          # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]
        intrinsics = np.asarray([[cam_param.fx * self.sample_size[1],
                                  cam_param.fy * self.sample_size[0],
                                  cam_param.cx * self.sample_size[1],
                                  cam_param.cy * self.sample_size[0]]
                                 for cam_param in cam_params], dtype=np.float32)
        intrinsics = torch.as_tensor(intrinsics)[None]                  # [1, n_frame, 4]
        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        c2w = torch.as_tensor(c2w_poses)[None]                          # [1, n_frame, 4, 4]
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool, device=c2w.device)
        # plucker_embedding = ray_condition(intrinsics, c2w, self.sample_size[0], self.sample_size[1], device='cpu',
        #                                   flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()
        plucker_embedding = ray_condition(intrinsics, c2w, self.sample_size[0], self.sample_size[1], device='cpu',
                                          flip_flag=flip_flag)
        return plucker_embedding
