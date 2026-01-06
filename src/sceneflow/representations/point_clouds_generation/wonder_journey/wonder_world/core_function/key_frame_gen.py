import copy
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import skimage.measure
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from scipy.ndimage import label
import matplotlib.pyplot as plt

from kornia.geometry import PinholeCamera
from kornia.morphology import dilation, erosion
from torchvision.transforms import ToTensor, ToPILImage

from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
)
from pytorch3d.renderer.points.compositor import _add_background_color_to_images
from pytorch3d.structures import Pointclouds

# 假设这些工具函数在你的目录下存在
from ..utils.utils import functbl, save_depth_map, SimpleLogger, soft_stitching
from ..utils.segment_utils import refine_disp_with_segments_2

BG_COLOR = (1, 0, 0)

# ---------------------------------------------------------------------------- #
#                               Renderer Classes                               #
# ---------------------------------------------------------------------------- #

class PointsRenderer(torch.nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(self, point_clouds, return_z=False, return_bg_mask=False, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        zbuf = fragments.zbuf.permute(0, 3, 1, 2)
        fragment_idx = fragments.idx.long().permute(0, 3, 1, 2)
        background_mask = fragment_idx[:, 0] < 0 
        
        images = self.compositor(
            fragment_idx,
            zbuf,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )
        images = images.permute(0, 2, 3, 1) # [B, H, W, C]

        ret = [images]
        if return_z: ret.append(fragments.zbuf)
        if return_bg_mask: ret.append(background_mask)
        return ret if len(ret) > 1 else images


class SoftmaxImportanceCompositor(torch.nn.Module):
    def __init__(self, background_color=None, softmax_scale=1.0) -> None:
        super().__init__()
        self.background_color = background_color
        self.scale = softmax_scale

    def forward(self, fragments, zbuf, ptclds, **kwargs) -> torch.Tensor:
        background_color = kwargs.get("background_color", self.background_color)
        zbuf_processed = zbuf.clone()
        zbuf_processed[zbuf_processed < 0] = - 1e-4
        importance = 1.0 / (zbuf_processed + 1e-6)
        weights = torch.softmax(importance * self.scale, dim=1)

        fragments_flat = fragments.flatten()
        gathered = ptclds[:, fragments_flat]
        gathered_features = gathered.reshape(ptclds.shape[0], fragments.shape[0], fragments.shape[1], fragments.shape[2], fragments.shape[3])
        images = (weights[None, ...] * gathered_features).sum(dim=2).permute(1, 0, 2, 3)

        if background_color is not None:
            return _add_background_color_to_images(fragments, images, background_color)
        return images


# ---------------------------------------------------------------------------- #
#                               KeyframeGen Class                              #
# ---------------------------------------------------------------------------- #

class KeyframeGen(torch.nn.Module):
    def __init__(self, config, wonder_world_synthesis, depth_model, mask_generator,
                 segment_model=None, segment_processor=None, normal_estimator=None,
                 rotation_path=None, inpainting_resolution=None):
        super().__init__()

        self.rendered_image_latest = torch.zeros(1, 3, 512, 512)
        self.rendered_depth_latest = torch.zeros(1, 1, 512, 512)
        
        # --- Configuration & Models ---
        self.config = config
        self.device = config["device"]
        self.wonder_world_synthesis = wonder_world_synthesis
        self.depth_model = depth_model
        self.normal_estimator = normal_estimator
        self.mask_generator = mask_generator
        self.segment_model = segment_model
        self.segment_processor = segment_processor
        
        # --- Parameters ---
        self.inpainting_resolution = config["inpainting_resolution_gen"]
        self.init_focal_length = config["init_focal_length"]
        self.depth_shift = config['depth_shift']
        self.very_far_depth = config['sky_hard_depth'] * 2
        self.sky_hard_depth = config['sky_hard_depth']
        self.sky_erode_kernel_size = config['sky_erode_kernel_size']
        self.rotation_range_theta = config['rotation_range']
        self.interp_frames = config['frames']
        self.camera_speed = config["camera_speed"]
        self.camera_speed_multiplier_rotation = config["camera_speed_multiplier_rotation"]
        self.negative_inpainting_prompt = config['negative_inpainting_prompt']
        
        # --- State / Archives ---
        self.kf_idx = 0
        self.current_pc = None
        self.current_pc_sky = None
        self.current_pc_layer = None
        self.current_pc_latest = None
        self.current_pc_layer_latest = None
        self.current_camera = None
        
        # Data buffers
        self.images = []
        self.images_layer = []
        self.rendered_images = []
        self.rendered_depths = []
        self.inpaint_input_images = []
        self.disparities = []
        self.depths = []
        self.masks = []
        self.post_masks = []
        self.cameras = []
        self.cameras_archive = []
        self.no_loss_masks = []
        self.no_loss_masks_layer = []
        self.sky_mask_list = []
        
        # Latest states
        self.image_latest = torch.zeros(1, 3, 512, 512)
        self.depth_latest = torch.zeros(1, 1, 512, 512)
        self.disparity_latest = torch.zeros(1, 1, 512, 512)
        self.sky_mask_latest = torch.zeros(1, 1, 512, 512)
        self.mask_latest = torch.zeros(1, 1, 512, 512)
        self.post_mask_latest = torch.zeros(1, 1, 512, 512)
        self.mask_disocclusion = torch.zeros(1, 1, 512, 512)
        self.border_mask = None
        self.border_image = None
        self.background_image = None
        
        # --- Initialization ---
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        self.run_dir = Path(config["runs_dir"]) / f"Gen-{dt_string}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'images' / "frames").mkdir(parents=True, exist_ok=True)
        self.logger = SimpleLogger(self.run_dir / "log.txt")

        # Pre-calculate 2D points for unprojection
        x = torch.arange(512).float() + 0.5
        y = torch.arange(512).float() + 0.5
        self.points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        self.points = rearrange(self.points, "h w c -> (h w) c").to(self.device)

        # Camera Setup
        self.scene_cameras_idx = []
        self.center_camera_idx = None
        self.generate_cameras(rotation_path)

    # ------------------------------------------------------------------------ #
    #                             Camera Logic                                 #
    # ------------------------------------------------------------------------ #

    @torch.no_grad()
    def get_camera_at_origin(self, big_view=False):
        K = torch.zeros((1, 4, 4), device=self.device)
        focal = 500 if big_view else self.init_focal_length
        cx = 768 if big_view else 256
        cy = 256
        
        K[0, 0, 0] = focal
        K[0, 1, 1] = focal
        K[0, 0, 2] = cx
        K[0, 1, 2] = cy
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        return PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device)

    @torch.no_grad()
    def set_current_camera(self, camera, archive_camera=False):
        self.current_camera = camera
        if archive_camera:
            self.cameras_archive.append(copy.deepcopy(camera))

    @torch.no_grad()
    def transform_all_cam_to_current_cam(self, center=False):
        if not self.cameras: return
        
        if not center:
            inv_current_camera_RT = self.cameras[-1].get_world_to_view_transform().inverse().get_matrix()
        else:
            inv_current_camera_RT = self.cameras[self.center_camera_idx].get_world_to_view_transform().inverse().get_matrix()
            
        for cam in self.cameras:
            cam_RT = cam.get_world_to_view_transform().get_matrix()
            new_cam_RT = inv_current_camera_RT @ cam_RT
            cam.R = new_cam_RT[:, :3, :3]
            cam.T = new_cam_RT[:, 3, :3]

    @torch.no_grad()
    def generate_cameras(self, rotation_path):
        print("-- generating 360-degree cameras...")
        camera = self.get_camera_at_origin()
        self.cameras.append(copy.deepcopy(camera))
        self.scene_cameras_idx.append(len(self.cameras) - 1)
        self.transform_all_cam_to_current_cam()
        
        # Generate sequence
        move_left_count = 0
        move_right_count = 0
        for rotation in rotation_path:
            new_camera = copy.deepcopy(self.cameras[-1])
            
            # Logic for rotation/translation based on path codes (0, 1, 2, etc.)
            # Simplified for brevity, keeping core logic
            if rotation == 0:
                forward_speed_multiplier = -1.0
                right_multiplier = 0
                camera_speed = self.camera_speed
                if move_left_count != 0 or move_right_count != 0:
                    new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                    move_left_count = 0; move_right_count = 0
            elif abs(rotation) == 2:
                if rotation > 0: move_left_count += 1
                else: move_right_count += 1
                if (rotation > 0 and move_right_count != 0) or (rotation < 0 and move_left_count != 0):
                     new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                     move_right_count = 0; move_left_count = 0
                
                forward_speed_multiplier = 0; right_multiplier = 0; camera_speed = 0
                theta = torch.tensor(self.rotation_range_theta * rotation / 2)
                rotation_matrix = torch.tensor([[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]], device=self.device)
                new_camera.R[0] = rotation_matrix @ new_camera.R[0]
            elif abs(rotation) == 1:
                if move_left_count != 0 or move_right_count != 0:
                    new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                    move_left_count = 0; move_right_count = 0
                theta_frame = torch.tensor(self.rotation_range_theta / (self.interp_frames + 1)) * rotation
                sin = torch.sum(torch.stack([torch.sin(i*theta_frame) for i in range(1, self.interp_frames+2)]))
                cos = torch.sum(torch.stack([torch.cos(i*theta_frame) for i in range(1, self.interp_frames+2)]))
                forward_speed_multiplier = -1.0 / (self.interp_frames + 1) * cos.item()
                right_multiplier = -1.0 / (self.interp_frames + 1) * sin.item()
                camera_speed = self.camera_speed * self.camera_speed_multiplier_rotation
                theta = torch.tensor(self.rotation_range_theta * rotation)
                rotation_matrix = torch.tensor([[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]], device=self.device)
                new_camera.R[0] = rotation_matrix @ new_camera.R[0]
            elif rotation == 3: continue

            move_dir = torch.tensor([[-right_multiplier, 0.0, -forward_speed_multiplier]], device=self.device)
            new_camera.T += camera_speed * move_dir
            self.cameras.append(copy.deepcopy(new_camera))

        self.center_camera_idx = 0
        self.transform_all_cam_to_current_cam(True)

    @torch.no_grad()
    def generate_sky_cameras(self):
        print("-- generating sky cameras...")
        cameras_cache = copy.deepcopy(self.cameras)
        init_len = len(self.cameras)
        for i in range(1):
            delta = -torch.tensor(torch.pi) / (8) * (i + 1)
            for camera_id in range(init_len):    
                self.center_camera_idx = camera_id
                self.transform_all_cam_to_current_cam(True)
                new_camera = copy.deepcopy(self.cameras[camera_id])                
                rotation_matrix = torch.tensor([[1, 0, 0], [0, torch.cos(delta), -torch.sin(delta)], [0, torch.sin(delta), torch.cos(delta)]], device=self.device)
                new_camera.R[0] = rotation_matrix @ new_camera.R[0]
                self.cameras.append(copy.deepcopy(new_camera))
        self.center_camera_idx = 0
        self.transform_all_cam_to_current_cam(True)
        self.sky_cameras = copy.deepcopy(self.cameras)
        self.cameras = cameras_cache

    # ------------------------------------------------------------------------ #
    #                         Depth & Normal & Inpaint                         #
    # ------------------------------------------------------------------------ #

    @torch.no_grad()
    def set_kf_param(self, inpainting_resolution, inpainting_prompt, adaptive_negative_prompt):
        self.inpainting_resolution = inpainting_resolution
        self.inpainting_prompt = inpainting_prompt
        self.adaptive_negative_prompt = adaptive_negative_prompt
        
        self.border_mask = torch.ones((1, 1, inpainting_resolution, inpainting_resolution)).to(self.device)
        self.border_size = (inpainting_resolution - 512) // 2
        self.border_mask[:, :, self.border_size : self.inpainting_resolution-self.border_size, self.border_size : self.inpainting_resolution-self.border_size] = 0
        self.border_image = torch.zeros(1, 3, inpainting_resolution, inpainting_resolution).to(self.device)

    @torch.no_grad()
    def get_normal(self, image):
        # Marigold normal estimation
        normal = self.normal_estimator(
            image * 2 - 1, num_inference_steps=10, processing_res=768, output_prediction_format='pt'
        ).to(dtype=torch.float32)
        return normal

    def get_depth(self, image, archive_output=False, target_depth=None, mask_align=None, mask_farther=None, diffusion_steps=30, guidance_steps=8):
        # Marigold depth estimation
        image_input = (image*255).byte().squeeze().permute(1, 2, 0)
        image_input = Image.fromarray(image_input.cpu().numpy())
        depth = self.depth_model(
            image_input, denoising_steps=diffusion_steps, ensemble_size=1, processing_res=0, match_input_res=True, batch_size=0,
            depth_conditioning=self.config['depth_conditioning'], target_depth=target_depth, mask_align=mask_align, mask_farther=mask_farther,
            guidance_steps=guidance_steps, logger=self.logger,
        )
        depth = depth[None, None, :].to(dtype=torch.float32) / 200
        depth = depth + self.depth_shift
        disparity = 1 / depth

        if archive_output:
            self.depth_latest = depth
            self.disparity_latest = disparity
        return depth, disparity

    @torch.no_grad()
    def inpaint(self, rendered_image, inpaint_mask, fill_mask=None, fill_mode='cv2_telea', self_guidance=False, inpainting_prompt=None, negative_prompt=None, mask_strategy=np.min, diffusion_steps=50):
        # Handle resolution padding
        self.wonder_world_synthesis.negative_inpainting_prompt = self.negative_inpainting_prompt
        inpainted_image = self.wonder_world_synthesis.inpaint(
            rendered_image=rendered_image, inpaint_mask=inpaint_mask, fill_mask=fill_mask, fill_mode=fill_mode,
            self_guidance=self_guidance, inpainting_prompt=inpainting_prompt, negative_prompt=negative_prompt,
            mask_strategy=mask_strategy, diffusion_steps=diffusion_steps, inpainting_resolution=self.inpainting_resolution,
            border_mask=self.border_mask, border_image=self.border_image, 
        )
        self.inpaint_input_image_latest = self.wonder_world_synthesis.inpaint_input_image_latest
        return inpainted_image

    # ------------------------------------------------------------------------ #
    #                       Point Cloud & Scene Management                     #
    # ------------------------------------------------------------------------ #

    @torch.no_grad()
    def get_current_pc(self, is_detach=False, get_sky=False, combine=False, get_layer=False):
        if combine:
            pc = self.get_combined_pc()
        elif get_sky:
            pc = self.current_pc_sky
        elif get_layer:
            pc = self.current_pc_layer
        else:
            pc = self.current_pc
            
        if is_detach and pc is not None:
            return {k: v.detach() for k, v in pc.items()}
        return pc

    @torch.no_grad()
    def get_current_pc_latest(self, get_layer=False):
        pc = self.current_pc_layer_latest if get_layer else self.current_pc_latest
        return {k: v.detach() for k, v in pc.items()}

    @torch.no_grad()
    def update_current_pc(self, points, colors, gen_sky=False, gen_layer=False, normals=None):
        if gen_sky:
            if self.current_pc_sky is None: self.current_pc_sky = {"xyz": points, "rgb": colors}
            else:
                self.current_pc_sky["xyz"] = torch.cat([self.current_pc_sky["xyz"], points], dim=0)
                self.current_pc_sky["rgb"] = torch.cat([self.current_pc_sky["rgb"], colors], dim=0)
        elif gen_layer:
            if self.current_pc_layer is None: self.current_pc_layer = {"xyz": points, "rgb": colors}
            else:
                self.current_pc_layer["xyz"] = torch.cat([self.current_pc_layer["xyz"], points], dim=0)
                self.current_pc_layer["rgb"] = torch.cat([self.current_pc_layer["rgb"], colors], dim=0)
            self.current_pc_layer_latest = {"xyz": points, "rgb": colors, 'normals': normals}
        else:
            if self.current_pc is None: self.current_pc = {"xyz": points, "rgb": colors}
            else:
                self.current_pc["xyz"] = torch.cat([self.current_pc["xyz"], points], dim=0)
                self.current_pc["rgb"] = torch.cat([self.current_pc["rgb"], colors], dim=0)
            self.current_pc_latest = {"xyz": points, "rgb": colors, 'normals': normals}

    @torch.no_grad()
    def get_combined_pc(self):
        parts = [self.current_pc, self.current_pc_sky]
        if self.current_pc_layer is not None: parts.append(self.current_pc_layer)
        return {
            "xyz": torch.cat([p["xyz"] for p in parts], dim=0),
            "rgb": torch.cat([p["rgb"] for p in parts], dim=0)
        }

    @torch.no_grad()
    def update_current_pc_by_kf(self, valid_mask=None, gen_layer=False, image=None, depth=None, camera=None):
        image = self.image_latest if image is None else image
        depth = self.depth_latest if depth is None else depth
        camera = self.current_camera if camera is None else camera
        
        kf_camera = convert_pytorch3d_kornia(camera, self.init_focal_length)
        point_depth = rearrange(depth, "b c h w -> (w h b) c")
        
        normals = self.get_normal(image[0])
        normals[:, 1:] *= -1  # OpenGL to OpenCV
        normals_world = kf_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')
        new_normals = rearrange(normals_world, 'b c z -> (z b) c')
        
        new_points_3d = kf_camera.unproject(self.points, point_depth)
        new_colors = rearrange(image, "b c h w -> (w h b) c")
        
        if valid_mask is not None:
            extract_mask = rearrange(valid_mask, "b c h w -> (w h b) c")[:, 0].bool()
            new_points_3d = new_points_3d[extract_mask]
            new_colors = new_colors[extract_mask]
            new_normals = new_normals[extract_mask]
        
        self.update_current_pc(new_points_3d, new_colors, normals=new_normals, gen_layer=gen_layer)
        return new_points_3d, new_colors

    @torch.no_grad()
    def archive_latest(self, idx=None, vmax=0.006):
        if idx is None: idx = self.kf_idx
        
        if self.config['gen_layer']:
            self.images_layer.append(self.image_latest)
            self.images.append(self.image_latest_init)
        else:
            self.images.append(self.image_latest)
            
        self.masks.append(self.mask_latest)
        self.post_masks.append(self.post_mask_latest)
        self.inpaint_input_images.append(self.inpaint_input_image_latest)
        self.depths.append(self.depth_latest)
        self.disparities.append(self.disparity_latest)
        self.rendered_images.append(self.rendered_image_latest)
        self.rendered_depths.append(self.rendered_depth_latest)
        self.sky_mask_list.append(~self.sky_mask_latest.bool())

        save_root = Path(self.run_dir) / "images" / "frames"
        ToPILImage()(self.image_latest[0]).save(save_root / f"{idx:03d}.png")
        if idx == 0:
            with open(Path(self.run_dir) / "config.yaml", "w") as f: OmegaConf.save(self.config, f)

    @torch.no_grad()
    def increment_kf_idx(self):
        self.kf_idx += 1

    # ------------------------------------------------------------------------ #
    #                           Rendering & Masks                              #
    # ------------------------------------------------------------------------ #

    @torch.no_grad()
    def render(self, archive_output=False, camera=None, render_visible=False, render_sky=False, big_view=False, render_fg=False):
        camera = self.current_camera if camera is None else camera
        raster_settings = PointsRasterizationSettings(image_size=1536 if big_view else 512, radius=0.003, points_per_pixel=8)
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
        )
        
        if render_sky: points, colors = self.current_pc_sky["xyz"], self.current_pc_sky["rgb"]
        elif render_fg: points, colors = self.current_pc["xyz"], self.current_pc["rgb"]
        else: 
            combined = self.get_combined_pc()
            points, colors = combined["xyz"], combined["rgb"]
            
        point_cloud = Pointclouds(points=[points], features=[colors])
        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)

        rendered_image = rearrange(images, "b h w c -> b c h w")
        inpaint_mask = bg_mask.float()[:, None, ...]
        rendered_depth = rearrange(zbuf[..., 0:1], "b h w c -> b c h w")
        rendered_depth[rendered_depth < 0] = 0

        if archive_output:
            self.rendered_image_latest = rendered_image
            self.rendered_depth_latest = rendered_depth
            self.mask_latest = inpaint_mask

        return {"rendered_image": rendered_image, "rendered_depth": rendered_depth, "inpaint_mask": inpaint_mask}

    @torch.no_grad()
    def recompose_image_latest_and_set_current_pc(self, scene_name=None):
        self.set_current_camera(self.get_camera_at_origin(), archive_camera=True)
        sem_map = self.update_sky_mask()
        render_output = self.render(render_sky=True)
        self.image_latest = soft_stitching(render_output["rendered_image"], self.image_latest, self.sky_mask_latest)

        ground_mask = self.generate_ground_mask(sem_map=sem_map)[None, None]
        depth_should_be_ground = self.compute_ground_depth(camera_height=0.0003)
        ground_outputable_mask = (depth_should_be_ground > 0.001) & (depth_should_be_ground < 0.006 * 0.8)

        self.get_depth(self.image_latest, archive_output=True, target_depth=depth_should_be_ground, mask_align=(ground_mask & ground_outputable_mask))
        self.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())

        if self.config['gen_layer']:
            self.generate_layer(pred_semantic_map=sem_map, scene_name=scene_name)
            depth_should_be = self.depth_latest_init
            mask_to_align_depth = ~(self.mask_disocclusion.bool()) & (depth_should_be < 0.006 * 0.8)
            mask_to_farther_depth = self.mask_disocclusion.bool() & (depth_should_be < 0.006)
            
            self.depth, self.disparity = self.get_depth(self.image_latest, archive_output=True, target_depth=depth_should_be, mask_align=mask_to_align_depth, mask_farther=mask_to_farther_depth)
            
            self.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy(), existing_mask=~(self.mask_disocclusion).bool().squeeze().cpu().numpy(), existing_disp=self.disparity_latest_init.squeeze().cpu().numpy())
            
            wrong_depth_mask = self.depth_latest < self.depth_latest_init
            self.depth_latest[wrong_depth_mask] = self.depth_latest_init[wrong_depth_mask] + 0.0001
            self.depth_latest = self.mask_disocclusion * self.depth_latest + (1-self.mask_disocclusion) * self.depth_latest_init
            self.update_sky_mask()
            self.update_current_pc_by_kf(image=self.image_latest, depth=self.depth_latest, valid_mask=~self.sky_mask_latest)
            self.update_current_pc_by_kf(image=self.image_latest_init, depth=self.depth_latest_init, valid_mask=self.mask_disocclusion, gen_layer=True)
        else:
            self.update_current_pc_by_kf(image=self.image_latest, depth=self.depth_latest, valid_mask=~self.sky_mask_latest)
        self.archive_latest()

    @torch.no_grad()
    def compute_ground_depth(self, camera_height=0.0003):
        focal_length = self.init_focal_length
        y_res = 512
        y_grid = torch.arange(y_res).view(1, 1, y_res, 1)
        denominator = torch.where(y_grid - 256 != 0, y_grid - 256, torch.tensor(1e-10))
        depth_map = (camera_height * focal_length) / denominator
        return depth_map.expand(-1, -1, -1, 512).to(self.device)

    @torch.no_grad()
    def update_sky_mask(self):
        sky_mask_latest, sem_seg = self.generate_sky_mask(self.image_latest, return_sem_seg=True)
        self.sky_mask_latest = sky_mask_latest[None, None, :]
        return sem_seg

    @torch.no_grad()
    def generate_sky_mask(self, input_image=None, return_sem_seg=False):
        image = ToPILImage()(input_image.squeeze() if input_image is not None else self.image_latest.squeeze())
        segmenter_input = {k: v.to("cuda") for k, v in self.segment_processor(image, ["semantic"], return_tensors="pt").items()}
        segment_output = self.segment_model(**segmenter_input)
        pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(segment_output, target_sizes=[image.size[::-1]])[0]
        
        sky_mask = pred_semantic_map == 2
        if self.sky_erode_kernel_size > 0:
            sky_mask = erosion(sky_mask.float()[None, None], kernel=torch.ones(self.sky_erode_kernel_size, self.sky_erode_kernel_size).to(self.device)).squeeze() > 0.5
        return (sky_mask, pred_semantic_map) if return_sem_seg else sky_mask

    @torch.no_grad()
    def generate_ground_mask(self, sem_map=None, input_image=None):
        if sem_map is None:
            _, sem_map = self.generate_sky_mask(input_image, return_sem_seg=True)
        ground_mask = (sem_map == 3) | (sem_map == 6) | (sem_map == 9) | (sem_map == 11) | (sem_map == 13) | (sem_map == 26) | (sem_map == 29) | (sem_map == 46) | (sem_map == 128)
        if self.config['ground_erode_kernel_size'] > 0:
            ground_mask = erosion(ground_mask.float()[None, None], kernel=torch.ones(self.config['ground_erode_kernel_size'], self.config['ground_erode_kernel_size']).to(self.device)).squeeze() > 0.5
        return ground_mask

    @torch.no_grad()
    def generate_grad_magnitude(self, disparity):
        # Simplified gradient calculation
        grad_x = cv2.Sobel(disparity, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(disparity, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = cv2.magnitude(grad_x, grad_y)
        grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return torch.from_numpy(grad_magnitude > 10)

    @torch.no_grad()
    def generate_layer(self, pred_semantic_map=None, scene_name=None):
        self.image_latest_init = copy.deepcopy(self.image_latest)
        self.depth_latest_init = copy.deepcopy(self.depth_latest)
        self.disparity_latest_init = copy.deepcopy(self.disparity_latest)
        
        if pred_semantic_map is None:
            _, pred_semantic_map = self.generate_sky_mask(return_sem_seg=True)

        unique_elements = torch.unique(pred_semantic_map)
        masks = {str(element.item()): (pred_semantic_map == element) for element in unique_elements}
        
        disparity_np = self.disparity_latest.squeeze().cpu().numpy()
        grad_magnitude_mask = self.generate_grad_magnitude(disparity_np)
        mask_disocclusion = np.full((512, 512), False, dtype=bool)
        dilation_kernel = torch.ones(9, 9).to(self.device)

        for id, mask in masks.items():
            if id in ['3', '6', '9', '11', '13', '26', '29', '46', '52', '128']: continue # Ground types
            mask = dilation((mask).float()[None, None], kernel=dilation_kernel).squeeze().cpu() > 0.5
            if id in ['4', '76', '83', '87']: # Objects
                mask_disocclusion |= mask.numpy(); continue
            
            labeled_array, num_features = label(mask)
            for i in range(1, num_features+1):
                mask_i = labeled_array==i
                if disparity_np[mask_i].mean() < np.percentile(disparity_np, 60): continue
                if grad_magnitude_mask[mask_i].float().mean() < 0.02: continue
                if mask_i.mean() < 0.001: continue
                mask_disocclusion |= mask_i

        inpainting_prompt = scene_name if scene_name is not None else 'road, building'
        mask_disocclusion = torch.from_numpy(mask_disocclusion)[None, None]
        self.mask_disocclusion = erosion(mask_disocclusion.float().to(self.device), kernel=dilation_kernel)
        inpaint_mask = self.mask_disocclusion > 0.5
        
        self.inpaint(self.image_latest, inpaint_mask=inpaint_mask, inpainting_prompt=inpainting_prompt, negative_prompt='tree, plant', mask_strategy=np.max, diffusion_steps=50)
        
        stitch_mask = erosion(mask_disocclusion.float().to(self.device), kernel=torch.ones(5, 5).to(self.device))
        self.image_latest = soft_stitching(self.image_latest, self.image_latest_init, stitch_mask, sigma=1, blur_size=3)

    @torch.no_grad()
    def refine_disp_with_segments(self, no_refine_mask=None, existing_mask=None, existing_disp=None):
        image_np = np.array(ToPILImage()(self.image_latest.squeeze()))
        masks = self.mask_generator.generate(image_np)
        sorted_mask = sorted([m for m in masks if m['area'] > 100], key=(lambda x: x['area']), reverse=False)
        
        disparity_np = self.disparity_latest.squeeze().cpu().numpy()
        refined_disparity = refine_disp_with_segments_2(disparity_np, sorted_mask, keep_threshold=10, no_refine_mask=no_refine_mask, existing_mask=existing_mask, existing_disp=existing_disp)

        self.depth_latest[0, 0] = torch.from_numpy(1 / refined_disparity).to(self.device)
        self.disparity_latest[0, 0] = torch.from_numpy(refined_disparity).to(self.device)

    # ------------------------------------------------------------------------ #
    #                       3DGS Data Conversion                               #
    # ------------------------------------------------------------------------ #

    @torch.no_grad()
    def _create_frame_data(self, image_tensor, camera, xyz_scale, no_loss_mask=None):
        image = ToPILImage()(image_tensor[0])
        transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
        transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
        transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale
        transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()
        opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
        transform_matrix = (transform_matrix_c2w_pt3d @ opengl_to_pt3d).cpu().numpy().tolist()
        return {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': no_loss_mask}

    @torch.no_grad()
    def convert_to_3dgs_traindata(self, xyz_scale=1.0, remove_threshold=None, use_no_loss_mask=True):
        train_datas = []
        W, H = 512, 512
        camera_angle_x = 2*np.arctan(W / (2*self.init_focal_length))

        # 1. Main Scene
        current_pc = self.get_current_pc(is_detach=True)
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        
        if remove_threshold:
            mask = np.linalg.norm(pcd_points, axis=0) < (remove_threshold * xyz_scale)
            pcd_points = pcd_points[:, mask]; pcd_colors = pcd_colors[mask]

        frames = [self._create_frame_data(img, self.cameras[i], xyz_scale, self.no_loss_masks[i][0] if use_no_loss_mask else None) for i, img in enumerate(self.images)]
        train_datas.append({'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H})
        
        # 2. Sky Scene
        current_pc = self.sky_pc_downsampled
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = (pcd_points / np.linalg.norm(pcd_points, axis=1, keepdims=True)).T
        
        frames = []
        for i, camera in enumerate(self.sky_cameras):
            self.current_camera = camera
            render_output = self.render(render_sky=True)
            if render_output['inpaint_mask'].mean() > 0:
                render_output['rendered_image'] = inpaint_cv2(render_output['rendered_image'], render_output['inpaint_mask'])
            frames.append(self._create_frame_data(render_output['rendered_image'], camera, xyz_scale, render_output['inpaint_mask'][0]))
            
        train_datas.append({'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'pcd_normals': pcd_normals, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H})
        
        # 3. Layer Scene (Optional)
        if self.config['gen_layer']:
            current_pc = self.get_current_pc(is_detach=True, get_layer=True)
            pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
            pcd_colors = current_pc["rgb"].cpu().numpy()
            frames = [self._create_frame_data(img, self.cameras[i], xyz_scale, self.no_loss_masks_layer[i][0] if use_no_loss_mask else None) for i, img in enumerate(self.images_layer)]
            train_datas.append({'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H})
            
        return train_datas

    @torch.no_grad()
    def convert_to_3dgs_traindata_latest(self, xyz_scale=1.0, use_no_loss_mask=False):
        W, H = 512, 512
        camera_angle_x = 2*np.arctan(W / (2*self.init_focal_length))
        current_pc = self.get_current_pc_latest()
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = current_pc['normals'].cpu().numpy()

        # Only latest frame
        i = len(self.images) - 1
        frames = [self._create_frame_data(self.images[i], self.cameras_archive[i], xyz_scale, self.no_loss_masks[i][0] if use_no_loss_mask else None)]
        return {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'pcd_normals': pcd_normals, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H}

    @torch.no_grad()
    def convert_to_3dgs_traindata_latest_layer(self, xyz_scale=1.0):
        W, H = 512, 512
        camera_angle_x = 2*np.arctan(W / (2*self.init_focal_length))
        i = len(self.images) - 1
        
        # Layer 1: Occluding Objects
        current_pc = self.get_current_pc_latest(get_layer=True)
        frames = [self._create_frame_data(self.images[i], self.cameras_archive[i], xyz_scale)]
        train_data = {'frames': frames, 'pcd_points': current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale, 
                      'pcd_colors': current_pc["rgb"].cpu().numpy(), 'pcd_normals': current_pc['normals'].cpu().numpy(), 'camera_angle_x': camera_angle_x, 'W': W, 'H': H}
        
        # Layer 2: Base Layer
        current_pc = self.get_current_pc_latest()
        frames = [self._create_frame_data(self.images_layer[i], self.cameras_archive[i], xyz_scale)]
        train_data_layer = {'frames': frames, 'pcd_points': current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale, 
                            'pcd_colors': current_pc["rgb"].cpu().numpy(), 'pcd_normals': current_pc['normals'].cpu().numpy(), 'camera_angle_x': camera_angle_x, 'W': W, 'H': H}
        
        return train_data, train_data_layer

    def generate_sky_pointcloud(self, syncdiffusion_model=None, image=None, mask=None, gen_sky=False, style=None,
                                image_height=512, image_width=6144, sky_text_prompt="blue sky",):
        # Simplified sky generation logic
        w_start = 256
        example_name = self.config["example_name"]
        imgs = []
        
        # This function has contained linear blending
        if self.background_image is None:
            img = self.wonder_world_synthesis.generation_360_data(input_image=image, sky_text_prompt=sky_text_prompt,
                                width=image_width, height=image_height, num_inference_steps=50, guidance_scale=7.5)
            self.background_image = img
        else:
            img = self.background_image
        
        # Point Cloud Generation from Panorama
        equatorial_radius = 0.02
        camera_angle_x = 2*np.arctan(512 / (2*self.init_focal_length))
        min_latitude = -camera_angle_x / 2 - (image_height / 512 - 1) * camera_angle_x
        max_latitude = camera_angle_x / 2
        
        lat, lon = torch.meshgrid(torch.linspace(min_latitude, max_latitude, image_height), 
                                  torch.linspace(-camera_angle_x/2, -camera_angle_x/2 + 2*np.pi, image_width), indexing='ij')
        
        x = -equatorial_radius * torch.cos(lat) * torch.sin(lon)
        z = equatorial_radius * torch.cos(lat) * torch.cos(lon)
        y = -equatorial_radius * torch.sin(lat)
        points = torch.stack((x, y, z), -1).reshape(-1, 3).to(self.device)
        colors = rearrange(ToTensor()(img).unsqueeze(0).to(self.device), "b c h w -> (h w b) c")
        
        # Filter ground
        sky_rows_idx = torch.where(mask.any(dim=1))[0]
        max_idx = sky_rows_idx.max().item()
        ground_threshold = -0.0003 if max_idx <= 255 else -0.003
        mask_above_ground = points[:, 1] >= ground_threshold
        
        self.update_current_pc(points[mask_above_ground], colors[mask_above_ground], gen_sky=True)
        
        # Downsampled version for training
        img_down = img.resize((int(image_width/2), int(image_height/2)), Image.Resampling.LANCZOS)
        lat_d, lon_d = torch.meshgrid(torch.linspace(min_latitude, max_latitude, int(image_height/2)), 
                                      torch.linspace(-camera_angle_x/2, -camera_angle_x/2 + 2*np.pi, int(image_width/2)), indexing='ij')
        x_d = -equatorial_radius * torch.cos(lat_d) * torch.sin(lon_d)
        z_d = equatorial_radius * torch.cos(lat_d) * torch.cos(lon_d)
        y_d = -equatorial_radius * torch.sin(lat_d)
        points_d = torch.stack((x_d, y_d, z_d), -1).reshape(-1, 3).to(self.device)
        colors_d = rearrange(ToTensor()(img_down).unsqueeze(0).to(self.device), "b c h w -> (h w b) c")
        mask_above_ground_d = points_d[:, 1] >= ground_threshold
        self.sky_pc_downsampled = {"xyz": points_d[mask_above_ground_d], "rgb": colors_d[mask_above_ground_d]}
        
        self.generate_sky_cameras()
        self.depth_latest[:] = self.sky_hard_depth
        self.disparity_latest[:] = 1. / self.sky_hard_depth

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #

def convert_pytorch3d_kornia(camera, focal_length, size=512):
    transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
    transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
    pt3d_to_kornia = torch.diag(torch.tensor([-1., -1, 1, 1], device=camera.device))
    transform_matrix_w2c_kornia = pt3d_to_kornia @ transform_matrix_w2c_pt3d
    
    extrinsics = transform_matrix_w2c_kornia.unsqueeze(0)
    h = torch.tensor([size], device="cuda")
    w = torch.tensor([size], device="cuda")
    K = torch.eye(4)[None].to("cuda")
    K[0, 0, 2] = size // 2; K[0, 1, 2] = size // 2
    K[0, 0, 0] = focal_length; K[0, 1, 1] = focal_length
    return PinholeCamera(K, extrinsics, h, w)

def inpaint_cv2(rendered_image, mask_diff):
    image_cv2 = (rendered_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    mask_cv2 = (mask_diff[0, 0].cpu().numpy() * 255).astype(np.uint8)
    inpainting = cv2.inpaint(image_cv2, mask_cv2, 3, cv2.INPAINT_TELEA)
    return torch.from_numpy(inpainting).permute(2, 0, 1).float().unsqueeze(0) / 255
