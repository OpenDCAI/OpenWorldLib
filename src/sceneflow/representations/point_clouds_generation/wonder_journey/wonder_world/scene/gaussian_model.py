#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from torch import nn
from plyfile import PlyData, PlyElement
from io import BytesIO
from tqdm import tqdm
import torch.nn.functional as F

# 导入原版3DGS的模块（假设原版代码在 original_gaussian 模块中）
from ......base_models.three_dimensions.point_clouds.gaussian_splatting.scene.gaussian_model import GaussianModel as OriginalGaussianModel
from ......base_models.three_dimensions.point_clouds.gaussian_splatting.utils.general_utils import (
    strip_symmetric, build_scaling_rotation, get_expon_lr_func, build_rotation
)
from ......base_models.three_dimensions.point_clouds.gaussian_splatting.utils.graphics_utils import BasicPointCloud
from ..utils.general_utils import normal2rotation, rotation2normal
from simple_knn._C import distCUDA2


class WonderWorldGaussianModel(OriginalGaussianModel):
    """
    wonderworld 3DGS模型，继承自原版GaussianModel并添加以下功能：
    1. 3D滤波器支持
    2. 前序高斯点冻结与累积
    3. 可见性过滤
    4. 天空点标记
    5. 删除掩码管理
    6. 修改的激活函数
    """
    
    def __init__(self, sh_degree: int, previous_gaussian=None, 
                 floater_dist2_threshold=0.0002, optimizer_type="default"):
        """
        Args:
            sh_degree: 球谐函数的最大阶数
            previous_gaussian: ModifiedGaussianModel对象；冻结其所有3DGS粒子用于渲染
            floater_dist2_threshold: 浮动点距离阈值
            optimizer_type: 优化器类型（保持与原版兼容）
        """
        # 先调用父类初始化
        super().__init__(sh_degree, optimizer_type)
        
        # 添加魔改特有的属性
        self.floater_dist2_threshold = floater_dist2_threshold
        self.filter_3D = torch.empty(0).cuda()
        
        # 初始化前序高斯点的参数（冻结的历史数据）
        if previous_gaussian is not None:
            self._xyz_prev = torch.cat([
                previous_gaussian._xyz.detach(), 
                previous_gaussian._xyz_prev
            ], dim=0)
            self._features_dc_prev = torch.cat([
                previous_gaussian._features_dc.detach(), 
                previous_gaussian._features_dc_prev
            ], dim=0)
            self._scaling_prev = torch.cat([
                previous_gaussian._scaling.detach(), 
                previous_gaussian._scaling_prev
            ], dim=0)
            self._rotation_prev = torch.cat([
                previous_gaussian._rotation.detach(), 
                previous_gaussian._rotation_prev
            ], dim=0)
            self._opacity_prev = torch.cat([
                previous_gaussian._opacity.detach(), 
                previous_gaussian._opacity_prev
            ], dim=0)
            self.filter_3D_prev = torch.cat([
                previous_gaussian.filter_3D.detach(), 
                previous_gaussian.filter_3D_prev
            ], dim=0)
            self.visibility_filter_all = previous_gaussian.visibility_filter_all
            self.is_sky_filter = previous_gaussian.is_sky_filter
            self.delete_mask_all = previous_gaussian.delete_mask_all
        else:
            self._xyz_prev = torch.empty(0).cuda()
            self._features_dc_prev = torch.empty(0).cuda()
            self._scaling_prev = torch.empty(0).cuda()
            self._rotation_prev = torch.empty(0).cuda()
            self._opacity_prev = torch.empty(0).cuda()
            self.filter_3D_prev = torch.empty(0).cuda()
            self.visibility_filter_all = torch.empty(0, dtype=torch.bool).cuda()
            self.is_sky_filter = torch.empty(0, dtype=torch.bool).cuda()
            self.delete_mask_all = torch.empty(0, dtype=torch.bool).cuda()
    
    def setup_functions(self):
        """重写激活函数设置"""
        # 保留父类的缩放和协方差激活函数
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.rotation_activation = torch.nn.functional.normalize
        
        # 修改不透明度激活函数（使用tanh替代sigmoid）
        self.opacity_activation = lambda x: (torch.tanh(x) * 0.51).clamp(-0.5, 0.5) + 0.5
        self.inverse_opacity_activation = lambda y: torch.atanh((y - 0.5) / 0.51)
        
        # 添加颜色激活函数
        self.color_activation = lambda x: (torch.tanh(x) * 0.51).clamp(-0.5, 0.5) + 0.5
        self.inverse_color_activation = lambda y: torch.atanh((y - 0.5) / 0.51)
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=(0., 0.99))
        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    @property
    def get_scaling_with_3D_filter(self):
        """获取应用3D滤波器后的缩放"""
        scales = self.get_scaling
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_opacity_with_3D_filter(self):
        """获取应用3D滤波器后的不透明度"""
        opacity = self.get_opacity
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]
    
    # ========== 包含所有点（当前+历史）的属性 ==========
    
    @property
    def get_xyz_all(self):
        """获取所有点的位置（当前+历史）"""
        return torch.cat([self._xyz, self._xyz_prev], dim=0)
    
    @property
    def get_features_all(self):
        """获取所有点的特征（当前+历史）"""
        features_dc = torch.cat([self._features_dc, self._features_dc_prev], dim=0)
        return features_dc
    
    @property
    def get_scaling_all(self):
        """获取所有点的缩放（当前+历史）"""
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_prev], dim=0))
    
    @property
    def get_scaling_with_3D_filter_all(self):
        """获取所有点应用3D滤波器后的缩放"""
        scales = self.get_scaling_all
        scales = torch.square(scales) + torch.square(
            torch.cat([self.filter_3D, self.filter_3D_prev], dim=0)
        )
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation_all(self):
        """获取所有点的旋转（当前+历史）"""
        return self.rotation_activation(torch.cat([self._rotation, self._rotation_prev], dim=0))
    
    @property
    def get_opacity_all(self):
        """获取所有点的不透明度（当前+历史）"""
        return self.opacity_activation(torch.cat([self._opacity, self._opacity_prev], dim=0))
    
    @property
    def get_opacity_with_3D_filter_all(self):
        """获取所有点应用3D滤波器后的不透明度"""
        opacity = self.get_opacity_all
        scales = self.get_scaling_all
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(
            torch.cat([self.filter_3D, self.filter_3D_prev], dim=0)
        )
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]
    
    def get_covariance_all(self, scaling_modifier=1):
        """获取所有点的协方差（当前+历史）"""
        return self.covariance_activation(
            self.get_scaling_all, 
            scaling_modifier, 
            torch.cat([self._rotation, self._rotation_prev], dim=0)
        )
    
    # ========== 3D滤波器计算 ==========
    
    @torch.no_grad()
    def compute_3D_filter(self, cameras, initialize_scaling=False):
        """
        计算3D滤波器以防止走样
        
        Args:
            cameras: 相机列表
            initialize_scaling: 是否初始化缩放参数
        """
        print("Computing 3D filter")
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        focal_length = 0.
        for camera in cameras:
            # 转换到相机空间
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # 投影到屏幕空间
            valid_depth = xyz_cam[:, 2] > 0.2
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # 扩展筛选范围（类似论文中的切线空间滤波）
            in_screen = torch.logical_and(
                torch.logical_and(
                    x >= -0.15 * camera.image_width, 
                    x <= camera.image_width * 1.15
                ),
                torch.logical_and(
                    y >= -0.15 * camera.image_height, 
                    y <= 1.15 * camera.image_height
                )
            )
            
            valid = torch.logical_and(valid_depth, in_screen)
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
            
            # 计算法向量方向的滤波器调整
            screen_normal = torch.tensor([[0, 0, -1]], device=xyz.device, dtype=torch.float32)
            point_normals_in_screen = rotation2normal(self.get_rotation) @ R
            
            point_normals_in_screen_xoz = F.normalize(point_normals_in_screen[:, [0, 2]], dim=1)
            screen_normal_xoz = F.normalize(screen_normal[:, [0, 2]], dim=1)
            cos_xz = torch.sum(point_normals_in_screen_xoz * screen_normal_xoz, dim=1)
            
            point_normals_in_screen_yoz = F.normalize(point_normals_in_screen[:, [1, 2]], dim=1)
            screen_normal_yoz = F.normalize(screen_normal[:, [1, 2]], dim=1)
            cos_yz = torch.sum(point_normals_in_screen_yoz * screen_normal_yoz, dim=1)
        
        distance[~valid_points] = distance[valid_points].max()
        
        # 计算3D滤波器
        filter_3D = distance / focal_length
        self.filter_3D = filter_3D[..., None]
        
        x_scale = distance / focal_length / cos_xz.clamp(min=1e-1)
        y_scale = distance / focal_length / cos_yz.clamp(min=1e-1)
        
        if initialize_scaling:
            print('Initializing scaling...')
            nyquist_scales = self.filter_3D.clone().repeat(1, 3)
            nyquist_scales[:, 0:1] = x_scale[..., None]
            nyquist_scales[:, 1:2] = y_scale[..., None]
            nyquist_scales *= 0.7
            scaling = torch.log(nyquist_scales)
            
            optimizable_tensors = self.replace_tensor_to_optimizer(scaling, 'scaling')
            self._scaling = optimizable_tensors['scaling']
    
    # ========== 点云创建（处理浮动点） ==========
    
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, is_sky: bool = False):
        """
        从点云创建高斯点，过滤浮动点
        
        Args:
            pcd: 基础点云
            spatial_lr_scale: 空间学习率缩放
            is_sky: 是否为天空点
        """
        # 计算距离并过滤浮动点
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 
            0.0000001
        )
        floater_mask = dist2 > self.floater_dist2_threshold
        print(f"Floater ratio: {floater_mask.float().mean().item()*100:.2f}%")
        dist2 = dist2[~floater_mask]
        
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()[~floater_mask]
        
        # 使用修改的颜色激活函数
        fused_color = self.inverse_color_activation(
            (torch.tensor(np.asarray(pcd.colors)).float().cuda() * 1.01).clamp(0, 1)
        )[~floater_mask]
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        
        print(f"Number of points at initialisation: {fused_point_cloud.shape[0]}")
        
        # 初始化缩放（z方向设为0）
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales[:, 2] = torch.log(torch.tensor(0.0))
        
        # 从法向量计算旋转
        normals = pcd.normals
        rots = normal2rotation(torch.from_numpy(normals).to(torch.float32)).to("cuda")[~floater_mask]
        
        # 使用修改的不透明度激活函数
        opacities = self.inverse_opacity_activation(
            0.15 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )
        
        # 如果已有点，则追加；否则初始化
        if self._xyz.numel() == 0:
            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._features_dc = nn.Parameter(
                features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
            )
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
        else:
            print(f"Adding these points to the existing model that has {self.get_xyz.shape[0]} points")
            self._xyz = nn.Parameter(
                torch.cat((self._xyz, fused_point_cloud), dim=0).requires_grad_(True)
            )
            self._features_dc = nn.Parameter(
                torch.cat((self._features_dc, features[:, :, 0:1].transpose(1, 2).contiguous()), dim=0).requires_grad_(True)
            )
            self._scaling = nn.Parameter(
                torch.cat((self._scaling, scales), dim=0).requires_grad_(True)
            )
            self._rotation = nn.Parameter(
                torch.cat((self._rotation, rots), dim=0).requires_grad_(True)
            )
            self._opacity = nn.Parameter(
                torch.cat((self._opacity, opacities), dim=0).requires_grad_(True)
            )
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # 更新可见性和天空滤波器
        visibility_filter_current = torch.ones((fused_point_cloud.shape[0]), device="cuda").bool()
        visibility_filter_prev = self.visibility_filter_all
        self.visibility_filter_all = torch.cat((visibility_filter_current, visibility_filter_prev), dim=0)
        
        delete_mask_current = torch.zeros((fused_point_cloud.shape[0]), device="cuda").bool()
        delete_mask_prev = self.delete_mask_all
        self.delete_mask_all = torch.cat((delete_mask_current, delete_mask_prev), dim=0)
        
        is_sky_filter_prev = self.is_sky_filter
        if is_sky:
            is_sky_filter_current = torch.ones((self.get_xyz.shape[0]), dtype=torch.bool, device="cuda")
        else:
            is_sky_filter_current = torch.zeros((self.get_xyz.shape[0]), dtype=torch.bool, device="cuda")
        self.is_sky_filter = torch.cat((is_sky_filter_current, is_sky_filter_prev), dim=0)
    
    # ========== 点管理功能 ==========
    
    @torch.no_grad()
    def delete_points(self, tdgs_cam):
        """标记屏幕内的非天空点为删除"""
        xyz = self.get_xyz_all
        R = torch.tensor(tdgs_cam.R, device=xyz.device, dtype=torch.float32)
        T = torch.tensor(tdgs_cam.T, device=xyz.device, dtype=torch.float32)
        
        xyz_cam = xyz @ R + T[None, :]
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        z = torch.clamp(z, min=0.001)
        
        x = x / z * tdgs_cam.focal_x + tdgs_cam.image_width / 2.0
        y = y / z * tdgs_cam.focal_y + tdgs_cam.image_height / 2.0
        
        in_screen_x = torch.logical_and(x >= 0, x < tdgs_cam.image_width)
        in_screen_y = torch.logical_and(y >= 0, y < tdgs_cam.image_height)
        in_screen = torch.logical_and(in_screen_x, in_screen_y)
        
        delete_mask = torch.logical_and(in_screen, ~self.is_sky_filter)
        self.delete_mask_all = self.delete_mask_all | delete_mask
    
    @torch.no_grad()
    def set_inscreen_points_to_visible(self, tdgs_cam):
        """将屏幕内的点标记为可见"""
        xyz = self.get_xyz_all
        R = torch.tensor(tdgs_cam.R, device=xyz.device, dtype=torch.float32)
        T = torch.tensor(tdgs_cam.T, device=xyz.device, dtype=torch.float32)
        
        xyz_cam = xyz @ R + T[None, :]
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        z = torch.clamp(z, min=0.001)
        
        x = x / z * tdgs_cam.focal_x + tdgs_cam.image_width / 2.0
        y = y / z * tdgs_cam.focal_y + tdgs_cam.image_height / 2.0
        
        in_screen = torch.logical_and(x >= 0, x < tdgs_cam.image_width)
        self.visibility_filter_all = self.visibility_filter_all | in_screen
    
    # ========== 保存和加载（支持3D滤波器） ==========
    
    def save_ply_with_filter(self, path):
        """保存PLY文件（包含3D滤波器）"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        filters_3D = self.filter_3D.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=False)]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation, filters_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        PlyData([el]).write(path)
    
    def load_ply_with_filter(self, path):
        """加载PLY文件（包含3D滤波器）"""
        plydata = PlyData.read(path)
        
        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(
            np.asarray(plydata.elements[0]["filter_3D"]), 
            dtype=torch.float, 
            device="cuda"
        )[:, None]
        
        self.active_sh_degree = self.max_sh_degree
    
    def construct_list_of_attributes(self, exclude_filter=False, use_higher_freq=True):
        """构建属性列表"""
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if not exclude_filter:
            l.append('filter_3D')
        return l
    
    # ========== Splat数据导出 ==========
    
    def yield_splat_data(self, path):
        """导出splat格式数据（过滤删除点和天空点）"""
        print('Yielding splat data...')
        
        def apply_activation(x):
            return np.clip(np.tanh(x) * 0.51, -0.5, 0.5) + 0.5
        
        # 过滤掉删除的点
        filter_all = ~self.delete_mask_all
        filter_all = filter_all.cpu()
        
        xyz = torch.cat([self._xyz.detach(), self._xyz_prev], dim=0).cpu().numpy()
        xyz = xyz[filter_all]
        normals = np.zeros_like(xyz)
        f_dc = torch.cat([
            self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous(),
            self._features_dc_prev.detach().transpose(1, 2).flatten(start_dim=1).contiguous()
        ], dim=0).cpu().numpy()
        f_dc = f_dc[filter_all]
        opacities = torch.cat([self._opacity.detach(), self._opacity_prev.detach()], dim=0).cpu().numpy()
        opacities = opacities[filter_all]
        scale = torch.cat([self._scaling.detach(), self._scaling_prev.detach()], dim=0).cpu().numpy()
        scale = scale[filter_all]
        rotation = torch.cat([self._rotation.detach(), self._rotation_prev.detach()], dim=0).cpu().numpy()
        rotation = rotation[filter_all]
        filters_3D = torch.cat([self.filter_3D.detach(), self.filter_3D_prev.detach()], dim=0).cpu().numpy()
        filters_3D = filters_3D[filter_all]
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=False, use_higher_freq=False)]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation, filters_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        vert = el
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"]) * apply_activation(vert["opacity"])
        )
        buffer = BytesIO()
        
        for idx in tqdm(sorted_indices):
            v = el[idx]
            position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
            scales = np.exp(
                np.array([v["scale_0"], v["scale_1"], v["scale_2"]], dtype=np.float32)
            )
            rot = np.array(
                [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
            color = np.array([
                apply_activation(v["f_dc_0"]),
                apply_activation(v["f_dc_1"]),
                apply_activation(v["f_dc_2"]),
                apply_activation(v["opacity"]),
            ])
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(
                ((rot / np.linalg.norm(rot)) * 128 + 128)
                .clip(0, 255)
                .astype(np.uint8)
                .tobytes()
            )
        
        splat_data = buffer.getvalue()
        with open(path, 'wb') as f:
            f.write(splat_data)
        print('Splat data yielded')
        return splat_data
    
    # ========== 重写prune_points以支持可见性过滤器 ==========
    
    def prune_points(self, mask):
        """修剪点并更新可见性过滤器"""
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        # 更新可见性和天空过滤器
        if len(valid_points_mask) < len(self.visibility_filter_all):
            current = self.visibility_filter_all[:len(valid_points_mask)]
            prev = self.visibility_filter_all[len(valid_points_mask):]
            self.visibility_filter_all = torch.cat((current[valid_points_mask], prev), dim=0)
            
            current_sky = self.is_sky_filter[:len(valid_points_mask)]
            prev_sky = self.is_sky_filter[len(valid_points_mask):]
            self.is_sky_filter = torch.cat((current_sky[valid_points_mask], prev_sky), dim=0)
            
            current_delete_mask = self.delete_mask_all[:len(valid_points_mask)]
            prev_delete_mask = self.delete_mask_all[len(valid_points_mask):]
            self.delete_mask_all = torch.cat((current_delete_mask[valid_points_mask], prev_delete_mask), dim=0)
        else:
            self.visibility_filter_all = self.visibility_filter_all[valid_points_mask]
            self.is_sky_filter = self.is_sky_filter[valid_points_mask]
            self.delete_mask_all = self.delete_mask_all[valid_points_mask]
    
    # ========== 重写reset_opacity以支持3D滤波器 ==========
    
    def reset_opacity(self):
        """重置不透明度（考虑3D滤波器）"""
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter) * 0.01)
        
        # 反向应用3D滤波器
        scales = self.get_scaling
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = self.inverse_opacity_activation(opacities_new)
        
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    self.optimizer.state[group['params'][0]] = stored_state
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def densification_postfix(self, new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        n_added_points = new_xyz.shape[0] - self.get_xyz.shape[0]

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if n_added_points > 0:
            assert len(self.visibility_filter_all) == 0, 'We have not yet implemented visibility filter densification.'

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
