import time
import copy
import sys
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from kornia.morphology import dilation

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import skimage
from PIL import Image
from einops import rearrange
from kornia.geometry import PinholeCamera
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
)
from pytorch3d.renderer.points.compositor import _add_background_color_to_images
from pytorch3d.structures import Pointclouds
from torchvision.transforms import ToTensor, ToPILImage, Resize
from ..utils.midas_utils import dpt_transform, dpt_512_transform
from ..utils.utils import functbl, save_depth_map, rotate_pytorch3d_camera, translate_pytorch3d_camera, SimpleLogger, soft_stitching

from ..utils.segment_utils import refine_disp_with_segments_2, save_sam_anns
from typing import List, Optional, Tuple, Union
from kornia.morphology import erosion
from syncdiffusion.syncdiffusion_model import SyncDiffusion    # this one need change
import os
from utils.loss import l1_loss
import matplotlib.pyplot as plt
from scipy.ndimage import label

BG_COLOR=(1, 0, 0)

    
class PointsRenderer(torch.nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(self, point_clouds, return_z=False, return_bg_mask=False, return_fragment_idx=False, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        zbuf = fragments.zbuf.permute(0, 3, 1, 2)
        fragment_idx = fragments.idx.long().permute(0, 3, 1, 2)
        background_mask = fragment_idx[:, 0] < 0  # [B, H, W]
        images = self.compositor(
            fragment_idx,
            zbuf,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        ret = [images]
        if return_z:
            ret.append(fragments.zbuf)
        if return_bg_mask:
            ret.append(background_mask)
        if return_fragment_idx:
            ret.append(fragments.idx.long())
        if len(ret) == 1:
            ret = images
        return ret


class SoftmaxImportanceCompositor(torch.nn.Module):
    """
    Accumulate points using a softmax importance weighted sum.
    """

    def __init__(
        self, background_color: Optional[Union[Tuple, List, torch.Tensor]] = None, softmax_scale=1.0,
    ) -> None:
        super().__init__()
        self.background_color = background_color
        self.scale = softmax_scale

    def forward(self, fragments, zbuf, ptclds, **kwargs) -> torch.Tensor:
        """
        Composite features within a z-buffer using importance sum. Given a z-buffer
        with corresponding features and weights, these values are accumulated
        according to softmax(1/z * scale) to produce a final image.

        Args:
            fragments: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
                giving the indices of the nearest points at each pixel, sorted in z-order.
                Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
                feature of the kth closest point (along the z-direction) to pixel (y, x) in
                batch element n. 
            zbuf: float32 Tensor of shape (N, points_per_pixel, image_size,
                image_size) giving the depth value of each point in the z-buffer.
                Value -1 means no points assigned to the pixel.
            pt_clds: Packed feature tensor of shape (C, P) giving the features of each point
                (can use RGB for example).

        Returns:
            images: Tensor of shape (N, C, image_size, image_size)
                giving the accumulated features at each point.
        """
        background_color = kwargs.get("background_color", self.background_color)

        zbuf_processed = zbuf.clone()
        zbuf_processed[zbuf_processed < 0] = - 1e-4
        importance = 1.0 / (zbuf_processed + 1e-6)
        weights = torch.softmax(importance * self.scale, dim=1)

        fragments_flat = fragments.flatten()
        gathered = ptclds[:, fragments_flat]
        gathered_features = gathered.reshape(ptclds.shape[0], fragments.shape[0], fragments.shape[1], fragments.shape[2], fragments.shape[3])
        images = (weights[None, ...] * gathered_features).sum(dim=2).permute(1, 0, 2, 3)

        # images are of shape (N, C, H, W)
        # check for background color & feature size C (C=4 indicates rgba)
        if background_color is not None:
            return _add_background_color_to_images(fragments, images, background_color)
        return images


class FrameSyn(torch.nn.Module):
    def __init__(self, config, inpainter_pipeline, depth_model, normal_estimator=None):
        """ This module implement following tasks that are exactly the same in both keyframe generation and new view generation:
        1. Inpainting
        2. Depth estimation
        3. Add new points to a current point cloud

        But it does not implement:
        1. Camera control
        2. Rendering
        3. Initialize point cloud
        4. Anything else
        """
        super().__init__()

        ####### Set up placeholder attributes #######
        self.inpainting_prompt = None
        self.adaptive_negative_prompt = None
        self.current_pc = None
        self.current_pc_sky = None
        self.current_pc_layer = None
        self.current_pc_latest = None  # Only store the valid newly added points for the latest generated scene
        self.current_pc_layer_latest = None  # Only store the valid newly added points for the latest generated scene
        self.current_visible_pc = None
        self.current_visible_pc_init = None
        self.inpainting_resolution = None
        self.border_mask = None
        self.border_size = None
        self.border_image = None
        self.run_dir = None

        ####### Set up archives #######
        self.image_latest = torch.zeros(1, 3, 512, 512)
        self.sky_mask_latest = torch.zeros(1, 1, 512, 512)
        self.mask_latest = torch.zeros(1, 1, 512, 512)
        self.inpaint_input_image_latest = ToPILImage()(torch.zeros(3, 512, 512))
        self.depth_latest = torch.zeros(1, 1, 512, 512)
        self.disparity_latest = torch.zeros(1, 1, 512, 512)
        self.post_mask_latest = torch.zeros(1, 1, 512, 512)
        self.mask_disocclusion = torch.zeros(1, 1, 512, 512)
        
        self.kf_idx = 0
        self.images = []
        self.images_layer = []
        self.inpaint_input_images = []
        self.disparities = []
        self.depths = []
        self.masks = []
        self.post_masks = []
        self.cameras = []
        self.cameras_archive = []

        ####### Set up attributes #######
        self.config = config
        self.device = config["device"]

        self.inpainting_pipeline = inpainter_pipeline
        self.use_noprompt = False
        self.negative_inpainting_prompt = config['negative_inpainting_prompt']
        self.is_upper_mask_aggressive = False
        self.preservation_weight = config['preservation_weight']
        self.init_focal_length = config["init_focal_length"]

        self.decoder_learning_rate = config['decoder_learning_rate']
        self.dilate_mask_decoder_ft = config['dilate_mask_decoder_ft']

        self.depth_model = depth_model
        self.normal_estimator = normal_estimator
        self.depth_model_name = config['depth_model'].lower()
        self.depth_shift = config['depth_shift']
        self.very_far_depth = config['sky_hard_depth'] * 2

        # 2D pixel points to be used for unprojection and adding new points to PC
        x = torch.arange(512).float() + 0.5
        y = torch.arange(512).float() + 0.5
        self.points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        self.points = rearrange(self.points, "h w c -> (h w) c").to(self.device)

        self.points_3d_list = []
        self.colors_list = []
        self.floating_point_mask = None
        self.floating_point_mask_list = []
        self.sky_mask_list = []
        self.depth_cache = []
        self.floater_cluster_mask = torch.zeros(1, 1, 512, 512)
        
    @torch.no_grad()
    def set_frame_param(self, inpainting_resolution, inpainting_prompt, adaptive_negative_prompt):
        self.inpainting_resolution = inpainting_resolution
        self.inpainting_prompt = inpainting_prompt
        self.adaptive_negative_prompt = adaptive_negative_prompt

        # Create mask for inpainting of the right size, white area around the image in the middle
        self.border_mask = torch.ones(
            (1, 1, inpainting_resolution, inpainting_resolution)
        ).to(self.device)
        self.border_size = (inpainting_resolution - 512) // 2
        self.border_mask[:, :, self.border_size : self.inpainting_resolution-self.border_size, self.border_size : self.inpainting_resolution-self.border_size] = 0
        self.border_image = torch.zeros(
            1, 3, inpainting_resolution, inpainting_resolution
        ).to(self.device)
        
    @torch.no_grad()
    def get_normal(self, image):
        """
        args:
            image: [1, 3, 512, 512]
        """
        # Marigold-official-normal
        normal = self.normal_estimator(
            image * 2 - 1,
            num_inference_steps=10,
            processing_res=768,
            output_prediction_format='pt',
        ).to(dtype=torch.float32)  # [1, 3, H, W], [-1, 1]
        return normal
        
    def get_depth(self, image, archive_output=False, target_depth=None, mask_align=None, save_depth_to_cache=False, mask_farther=None, diffusion_steps=30, guidance_steps=8):
        """
        args:
            image: [1, 3, 512, 512]
            archive_output: if True, then save the depth and disparity to self.depth_latest and self.disparity_latest.
            target_depth: if not None, then use this target depth to condition the depth model.
            mask_align: if not None, then use this mask to align the depth map.
            save_depth_to_cache: if True, then save the depth map to cache.
        """
        assert self.depth_model is not None
        if self.depth_model_name == "midas":
            # MiDaS
            disparity = self.depth_model(dpt_transform(image))
            disparity = torch.nn.functional.interpolate(
                disparity.unsqueeze(1),
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            disparity = disparity.clip(1e-6, max=None)
            depth = 1 / disparity
        if self.depth_model_name == "midas_v3.1":
            img_transformed = dpt_512_transform(image)
            disparity = self.depth_model(img_transformed)
            disparity = torch.nn.functional.interpolate(
                disparity.unsqueeze(1),
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            disparity = disparity.clip(1e-6, max=None)
            depth = 1 / disparity
        elif self.depth_model_name == "zoedepth":
            # ZeoDepth
            depth = self.depth_model(image)['metric_depth']
        elif self.depth_model_name == "marigold":
            # Marigold
            image_input = (image*255).byte().squeeze().permute(1, 2, 0)
            image_input = Image.fromarray(image_input.cpu().numpy())
            depth = self.depth_model(
                image_input,
                denoising_steps=diffusion_steps,     # optional
                ensemble_size=1,       # optional
                processing_res=0,     # optional
                match_input_res=True,   # optional
                batch_size=0,           # optional
                color_map=None,   # optional
                show_progress_bar=True, # optional
                depth_conditioning=self.config['depth_conditioning'],
                target_depth=target_depth,
                mask_align=mask_align,
                mask_farther=mask_farther,
                guidance_steps=guidance_steps,
                # guidance_steps=20,
                logger=self.logger,
            )
                
            depth = depth[None, None, :].to(dtype=torch.float32)
            depth /= 200

        depth = depth + self.depth_shift
        disparity = 1 / depth

        if archive_output:
            self.depth_latest = depth
            self.disparity_latest = disparity
        
        if save_depth_to_cache:
            self.depth_cache.append(depth)
            
        return depth, disparity
    
    @torch.no_grad()
    def inpaint(self, rendered_image, inpaint_mask, fill_mask=None, fill_mode = 'cv2_telea', self_guidance=False, style=None, inpainting_prompt=None, negative_prompt=None, mask_strategy=np.min, diffusion_steps=50):
        # set resolution
        if self.inpainting_resolution > 512 and rendered_image.shape[-1] == 512:
            padded_inpainting_mask = self.border_mask.clone()
            padded_inpainting_mask[
                :, :, self.border_size : self.inpainting_resolution-self.border_size, self.border_size : self.inpainting_resolution-self.border_size
            ] = inpaint_mask
            padded_rendered_image = self.border_image.clone()
            padded_rendered_image[
                :, :, self.border_size : self.inpainting_resolution-self.border_size, self.border_size : self.inpainting_resolution-self.border_size
            ] = rendered_image
        else:
            padded_inpainting_mask = inpaint_mask
            padded_rendered_image = rendered_image

        # fill in image
        img = (padded_rendered_image[0].cpu().permute([1, 2, 0]).numpy() * 255).astype(np.uint8)
        fill_mask = padded_inpainting_mask if fill_mask is None else fill_mask
        fill_mask_ = (fill_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        mask = (padded_inpainting_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        img, _ = functbl[fill_mode](img, fill_mask_)

        # process mask original
        mask_block_size = 8
        mask_boundary = mask.shape[0] // 2
        mask_upper = skimage.measure.block_reduce(mask[:mask_boundary, :], (mask_block_size, mask_block_size), mask_strategy)
        mask_upper = mask_upper.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
        mask_lower = skimage.measure.block_reduce(mask[mask_boundary:, :], (mask_block_size, mask_block_size), mask_strategy)
        mask_lower = mask_lower.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
        mask = np.concatenate([mask_upper, mask_lower], axis=0)

        init_image = Image.fromarray(img)
        mask_image = Image.fromarray(mask)

        if inpainting_prompt is not None:
            self.inpainting_prompt = inpainting_prompt
        if negative_prompt is None:
            negative_prompt = self.adaptive_negative_prompt + self.negative_inpainting_prompt if self.adaptive_negative_prompt != None else self.negative_inpainting_prompt
      
        inpainted_image = self.inpainting_pipeline(
            prompt='' if self.use_noprompt else self.inpainting_prompt,
            negative_prompt=negative_prompt,
            image=init_image.resize((1024, 1024)),
            mask_image=mask_image.resize((1024, 1024)),
            num_inference_steps=diffusion_steps,
            guidance_scale=0 if self.use_noprompt else 8.0,
            height=1024,
            width=1024,
            self_guidance=self_guidance,
            # inpaint_mask=~padded_inpainting_mask.bool(),
            # rendered_image=padded_rendered_image,
        ).images[0]
        
        # [1, 3, 512, 512]
        inpainted_image = inpainted_image.resize((self.inpainting_resolution, self.inpainting_resolution))
        # inpainted_image = ToTensor()(inpainted_image).unsqueeze(0).to(self.device)
        inpainted_image = ToTensor()(inpainted_image).to(self.device)
        inpainted_image = (inpainted_image / 2 + 0.5).clamp(0, 1).to(torch.float32)[None]
            
        post_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() * 255
        
        self.post_mask_latest = post_mask
        self.inpaint_input_image_latest = init_image
        self.image_latest = inpainted_image

        return {"inpainted_image": inpainted_image,
                "padded_inpainting_mask": padded_inpainting_mask, "padded_rendered_image": padded_rendered_image}

    @torch.no_grad()
    def get_current_pc(self, is_detach=False, get_sky=False, combine=False, get_layer=False):
        # sky + foreground + layer
        if combine:
            if is_detach:
                return {k: v.detach() for k, v in self.get_combined_pc().items()}
            else:
                return self.get_combined_pc()
        # sky
        elif get_sky:
            if is_detach:
                return {k: v.detach() for k, v in self.current_pc_sky.items()}
            else:
                return self.current_pc_sky
        # layer
        elif get_layer:
            if is_detach:
                return {k: v.detach() for k, v in self.current_pc_layer.items()}
            else:
                return self.current_pc_layer
        # foreground
        else:
            if is_detach:
                return {k: v.detach() for k, v in self.current_pc.items()}
            else:
                return self.current_pc

    @torch.no_grad()
    def get_current_pc_latest(self, get_layer=False):
        if get_layer:
            return {k: v.detach() for k, v in self.current_pc_layer_latest.items()}
        else:
            return {k: v.detach() for k, v in self.current_pc_latest.items()}
                
    @torch.no_grad()
    def update_current_pc(self, points, colors, gen_sky=False, gen_layer=False, normals=None):
        if gen_sky:
            if self.current_pc_sky is None:   
                self.current_pc_sky = {"xyz": points, "rgb": colors}
            else:
                self.current_pc_sky["xyz"] = torch.cat([self.current_pc_sky["xyz"], points], dim=0)
                self.current_pc_sky["rgb"] = torch.cat([self.current_pc_sky["rgb"], colors], dim=0)
        elif gen_layer:
            if self.current_pc_layer is None:   
                self.current_pc_layer = {"xyz": points, "rgb": colors}
            else:
                self.current_pc_layer["xyz"] = torch.cat([self.current_pc_layer["xyz"], points], dim=0)
                self.current_pc_layer["rgb"] = torch.cat([self.current_pc_layer["rgb"], colors], dim=0)
            self.current_pc_layer_latest = {"xyz": points, "rgb": colors, 'normals': normals}
        else:
            if self.current_pc is None:
                self.current_pc = {"xyz": points, "rgb": colors}
            else:
                self.current_pc["xyz"] = torch.cat([self.current_pc["xyz"], points], dim=0)
                self.current_pc["rgb"] = torch.cat([self.current_pc["rgb"], colors], dim=0)
            self.current_pc_latest = {"xyz": points, "rgb": colors, 'normals': normals}
    
    @torch.no_grad()
    def get_combined_pc(self):
        if self.current_pc_layer is None:
            pc = {"xyz": torch.cat([self.current_pc["xyz"], self.current_pc_sky["xyz"]], dim=0), "rgb": torch.cat([self.current_pc["rgb"], self.current_pc_sky["rgb"]], dim=0)}
        else:
            pc = {"xyz": torch.cat([self.current_pc["xyz"], self.current_pc_sky["xyz"], self.current_pc_layer["xyz"]], dim=0), "rgb": torch.cat([self.current_pc["rgb"], self.current_pc_sky["rgb"], self.current_pc_layer["rgb"]], dim=0)}
        return pc
    
    
    @torch.no_grad()
    def push_away_inconsistent_points(self, inconsistent_point_index, depth, mask):
        h, w = depth.shape[2:]
        depth = rearrange(depth.clone(), "b c h w -> (w h b) c")
        extract_mask = rearrange(mask, "b c h w -> (w h b) c")[:, 0].bool()
        depth_extracted = depth[extract_mask]
        if inconsistent_point_index.shape[0] > 0:
            assert depth_extracted.shape[0] >= inconsistent_point_index.max() + 1
        depth_extracted[inconsistent_point_index] = self.very_far_depth
        depth[extract_mask] = depth_extracted
        depth = rearrange(depth, "(w h b) c -> b c h w", w=w, h=h)
        return depth
    
    @torch.no_grad()
    def archive_latest(self, idx=0, vmax=0.006):
        if self.config['gen_layer']:
            self.images_layer.append(self.image_latest)
            self.images.append(self.image_latest_init)
        else:
            self.images.append(self.image_latest)
        # render_output = self.render(render_sky=True)
        # render_output = self.render(render_ground=True)
        # self.images_ground.append(render_output['rendered_image'])
        self.masks.append(self.mask_latest)
        self.post_masks.append(self.post_mask_latest)
        self.inpaint_input_images.append(self.inpaint_input_image_latest)
        self.depths.append(self.depth_latest)
        self.disparities.append(self.disparity_latest)

        save_root = Path(self.run_dir) / "images"
        save_root.mkdir(exist_ok=True, parents=True)

        # (save_root / "inpaint_input_images").mkdir(exist_ok=True, parents=True)
        (save_root / "frames").mkdir(exist_ok=True, parents=True)
        (save_root / "frames_init").mkdir(exist_ok=True, parents=True)

        # self.inpaint_input_image_latest.save(save_root / "inpaint_input_images" / f"{idx:03d}.png")
        ToPILImage()(self.image_latest[0]).save(save_root / "frames" / f"{idx:03d}.png")
        if self.config['gen_layer']:
            ToPILImage()(self.image_latest_init[0]).save(save_root / "frames_init" / f"{idx:03d}.png")
            
        if idx == 0:
            with open(Path(self.run_dir) / "config.yaml", "w") as f:
                OmegaConf.save(self.config, f)

    @torch.no_grad()
    def increment_kf_idx(self):
        self.kf_idx += 1

    @torch.no_grad()
    def convert_to_3dgs_traindata(self, xyz_scale=1.0, remove_threshold=None, use_no_loss_mask=True):
        """
        args:
            xyz_scale: scale the xyz coordinates by this factor (so that the value range is better for 3DGS optimization and web-viewing).
            remove_threshold: Since 3DGS does not optimize very distant points well, we remove points whose distance to scene origin is greater than this threshold.
        """
        train_datas = []
        W, H = 512, 512
        camera_angle_x = 2*np.arctan(W / (2*self.init_focal_length))
        current_pc = self.get_current_pc(is_detach=True)
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()

        if remove_threshold is not None:
            remove_threshold_scaled = remove_threshold * xyz_scale
            mask = np.linalg.norm(pcd_points, axis=0) >= remove_threshold_scaled
            pcd_points = pcd_points[:, ~mask]
            pcd_colors = pcd_colors[~mask]

        frames = []

        for i, img in enumerate(self.images):
            image = ToPILImage()(img[0])
            no_loss_mask = self.no_loss_masks[i][0] if use_no_loss_mask else None
            transform_matrix_pt3d = self.cameras[i].get_world_to_view_transform().get_matrix()[0]
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale
            
            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d
            
            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': no_loss_mask}
            frames.append(frame)
        train_data = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H}
        train_datas.append(train_data)
        
        # current_pc = self.get_current_pc(is_detach=True, get_sky=True)
        current_pc = self.sky_pc_downsampled
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = pcd_points / np.linalg.norm(pcd_points, axis=1, keepdims=True)
        pcd_normals = pcd_normals.T
        
        frames = []

        for i, camera in enumerate(self.sky_cameras):
            self.current_camera = camera
            render_output = self.render(render_sky=True)
            
            if render_output['inpaint_mask'].mean() > 0:
                render_output['rendered_image'] = inpaint_cv2(render_output['rendered_image'], render_output['inpaint_mask'])
            no_loss_mask = render_output['inpaint_mask'][0]
            
            image = ToPILImage()(render_output['rendered_image'][0])
            save_root = Path(self.run_dir) / "images"
            # image.save(save_root / "sky_frames" / f"{i:03d}.png")
            
            transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale
            
            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d
            
            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': no_loss_mask}
            frames.append(frame)
        train_data_sky = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'pcd_normals': pcd_normals, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H}
        train_datas.append(train_data_sky)
        
        if self.config['gen_layer']:
            current_pc = self.get_current_pc(is_detach=True, get_layer=True)
            pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
            pcd_colors = current_pc["rgb"].cpu().numpy()
            
            frames = []

            for i, img in enumerate(self.images_layer):
                image = ToPILImage()(img[0])
                no_loss_mask = self.no_loss_masks_layer[i][0]  if use_no_loss_mask else None
                transform_matrix_pt3d = self.cameras[i].get_world_to_view_transform().get_matrix()[0]
                transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
                transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale
                
                transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

                opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
                transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d
                
                transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
                frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': no_loss_mask}
                frames.append(frame)
            train_data_layer = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H}
            train_datas.append(train_data_layer)
            
        return train_datas

    @torch.no_grad()
    def convert_to_3dgs_traindata_latest(self, xyz_scale=1.0, points_3d=None, colors=None, use_no_loss_mask=False, use_only_latest_frame=True):
        """
        args:
            xyz_scale: scale the xyz coordinates by this factor (so that the value range is better for 3DGS optimization and web-viewing).
        """
        W, H = 512, 512
        camera_angle_x = 2*np.arctan(W / (2*self.init_focal_length))
        current_pc = self.get_current_pc_latest()
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = current_pc['normals'].cpu().numpy()

        frames = []

        images = self.images
        for i, img in enumerate(images):
            if use_only_latest_frame and i != len(images) - 1:
                continue
            image = ToPILImage()(img[0])
            no_loss_mask = self.no_loss_masks[i][0] if use_no_loss_mask else None
            transform_matrix_pt3d = self.cameras_archive[i].get_world_to_view_transform().get_matrix()[0]
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale
            
            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d
            
            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': no_loss_mask}
            frames.append(frame)
        train_data = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'pcd_normals': pcd_normals, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H}
        
        return train_data

    @torch.no_grad()
    def convert_to_3dgs_traindata_latest_layer(self, xyz_scale=1.0, points_3d=None, colors=None, use_only_latest_frame=True):
        """
        args:
            xyz_scale: scale the xyz coordinates by this factor (so that the value range is better for 3DGS optimization and web-viewing).
        return:
            train_data: Original image and the point cloud of only occluding objects
            train_data_layer: Base image (original with inpainted regions) and the point cloud of the base layer
        """
        W, H = 512, 512
        camera_angle_x = 2*np.arctan(W / (2*self.init_focal_length))

        current_pc = self.get_current_pc_latest(get_layer=True)
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = current_pc['normals'].cpu().numpy()
        frames = []
        images = self.images
        for i, img in enumerate(images):
            if use_only_latest_frame and i != len(images) - 1:
                continue
            image = ToPILImage()(img[0])
            transform_matrix_pt3d = self.cameras_archive[i].get_world_to_view_transform().get_matrix()[0]
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale
            
            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d
            
            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': None}
            frames.append(frame)
        train_data = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'pcd_normals': pcd_normals, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H}
        
        current_pc = self.get_current_pc_latest()
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = current_pc['normals'].cpu().numpy()
        frames = []
        images = self.images_layer
        for i, img in enumerate(images):
            if use_only_latest_frame and i != len(images) - 1:
                continue
            image = ToPILImage()(img[0])
            transform_matrix_pt3d = self.cameras_archive[i].get_world_to_view_transform().get_matrix()[0]
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale
            
            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d
            
            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': None}
            frames.append(frame)
        train_data_layer = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'pcd_normals': pcd_normals, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H}
        
        return train_data, train_data_layer

    @torch.no_grad()
    def get_knn_mask(self, pad_width=1):
        """
        Clean depth map by removing floating points with KNN heuristic over multiple iterations.

        Args:
        - pad_width: Padding width for the depth map processing.

        Returns:
        - mask which indicates floating points after specified iterations.
        """      
        print("-- knn heuristic, removing floating points...")
        depth_map = self.depth_latest.squeeze().detach().cpu().numpy()
        height, width = depth_map.shape
        padded_depth_map = np.pad(depth_map, pad_width=pad_width, mode='constant', constant_values=0)
        cleaned_depth_map = np.zeros_like(depth_map)
        
        for dy in range(-pad_width, pad_width+1):
            for dx in range(-pad_width, pad_width+1):
                if dy == 0 and dx == 0:
                    continue
                neighbor_diff = np.abs(padded_depth_map[pad_width+dy:height+pad_width+dy, pad_width+dx:width+pad_width+dx] - depth_map)
                cleaned_depth_map += (neighbor_diff > 0.00001)

        knn_mask = torch.from_numpy(cleaned_depth_map == 8)
        print("-- floating points ratio: {}".format(knn_mask.float().mean()))
        ToPILImage()(knn_mask.float()).save(self.run_dir / 'images' / 'knn_masks' / f"{self.kf_idx:02d}_knn_mask.png")
        
        return knn_mask 

    @torch.no_grad()
    def update_current_pc_by_kf(self, valid_mask=None, gen_layer=False, image=None, depth=None, camera=None):
        """
        Use self.image_latest and self.depth_latest to update current_pc.
        args:
            valid_mask: if None, then use inpaint_mask (given by rendered_depth == 0) to extract new points.
                        if not None, should be [B, C, H, W], then just valid_mask to extract new points.
        """
        if image is None:
            image = self.image_latest
        if depth is None:
            depth = self.depth_latest
        if camera is None:
            camera = self.current_camera
        kf_camera = convert_pytorch3d_kornia(camera, self.init_focal_length)
        point_depth = rearrange(depth, "b c h w -> (w h b) c")
        normals = self.get_normal(image[0])
        normals[:, 1:] *= -1  # Marigold normal is opengl format; make it opencv format here


        normals_world = kf_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')

        normals = rearrange(normals_world, 'b c (h w) -> b c h w', h=512)
        new_normals = rearrange(normals, "b c h w -> (w h b) c")
        new_points_3d = kf_camera.unproject(self.points, point_depth)
        
        new_colors = rearrange(image, "b c h w -> (w h b) c")
        
        if valid_mask is not None:
            extract_mask = rearrange(valid_mask, "b c h w -> (w h b) c")[:, 0].bool()
            new_points_3d = new_points_3d[extract_mask]
            new_colors = new_colors[extract_mask]
            new_normals = new_normals[extract_mask]
        
        self.update_current_pc(new_points_3d, new_colors, normals=new_normals, gen_layer=gen_layer)
        return new_points_3d, new_colors

    
class KeyframeGen(FrameSyn):
    def __init__(self, config, inpainter_pipeline, depth_model, mask_generator,
                 segment_model=None, segment_processor=None, normal_estimator=None,
                 rotation_path=None, inpainting_resolution=None):
        """ This class is for generating keyframes. It inherits from FrameSyn. It implements the following tasks:
        1. Render
        2. Set cameras
        3. Initialize point cloud
        4. Post-process depth
        """
        super().__init__(config, inpainter_pipeline=inpainter_pipeline, depth_model=depth_model, normal_estimator=normal_estimator)
        
        ####### Set up placeholder attributes #######

        ####### Set up archives #######
        self.rendered_image_latest = torch.zeros(1, 3, 512, 512)
        self.rendered_depth_latest = torch.zeros(1, 1, 512, 512)
        self.no_loss_mask_latest = torch.zeros(1, 1, 512, 512).bool()
        self.no_loss_mask_latest_layer = torch.zeros(1, 1, 512, 512).bool()
        self.current_camera = None

        self.rendered_images = []
        self.rendered_depths = []
        self.no_loss_masks = []  # Indicating which pixels to remove for foreground in optimizing 3DGS
        self.no_loss_masks_layer = []  # Indicating which pixels to remove for layer in optimizing 3DGS
        
        ####### Set up attributes #######
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        run_dir_root = Path(config["runs_dir"])
        self.run_dir = run_dir_root / f"Gen-{dt_string}"
        self.logger = SimpleLogger(self.run_dir / "log.txt")
        self.mask_generator = mask_generator
        self.segment_model = segment_model
        self.segment_processor = segment_processor
        self.sky_hard_depth = config['sky_hard_depth']
        self.sky_erode_kernel_size = config['sky_erode_kernel_size']
        self.is_upper_mask_aggressive = False

        self.rotation_range_theta = config['rotation_range']
        self.interp_frames = config['frames']
        self.camera_speed = config["camera_speed"]
        self.camera_speed_multiplier_rotation = config["camera_speed_multiplier_rotation"]

        ####### Initialization functions #######
        (self.run_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'images' / "knn_masks").mkdir(exist_ok=True, parents=True)
        # (self.run_dir / 'images' / "floating_point_mask").mkdir(exist_ok=True, parents=True)
        (self.run_dir / 'images' / "depth_should_be").mkdir(exist_ok=True, parents=True)
        (self.run_dir / 'images' / "depth_conditioned").mkdir(exist_ok=True, parents=True)
        # (self.run_dir / 'images' / "floating_masked_images").mkdir(exist_ok=True, parents=True)
        (self.run_dir / 'images' / "layer").mkdir(exist_ok=True, parents=True)
        (self.run_dir / 'images' / "disparity_gradient").mkdir(exist_ok=True, parents=True)
        
        # rotation matrix of each scene
        self.scene_cameras_idx = []
        self.center_camera_idx = None
        self.generate_cameras(rotation_path)
        self.cameras_users = []
        self.inpainting_resolution = inpainting_resolution
        
    @torch.no_grad()
    def get_camera_at_origin(self, big_view=False):
        if big_view:
            K = torch.zeros((1, 4, 4), device=self.device)
            K[0, 0, 0] = 500
            K[0, 1, 1] = 500
            K[0, 0, 2] = 768
            K[0, 1, 2] = 256
            K[0, 2, 3] = 1
            K[0, 3, 2] = 1
            R = torch.eye(3, device=self.device).unsqueeze(0)
            T = torch.zeros((1, 3), device=self.device)
            camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device)
        else:
            K = torch.zeros((1, 4, 4), device=self.device)
            K[0, 0, 0] = self.init_focal_length
            K[0, 1, 1] = self.init_focal_length
            K[0, 0, 2] = 256
            K[0, 1, 2] = 256
            K[0, 2, 3] = 1
            K[0, 3, 2] = 1
            R = torch.eye(3, device=self.device).unsqueeze(0)
            T = torch.zeros((1, 3), device=self.device)
            camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device)
        return camera

    @torch.no_grad()
    def recompose_image_latest_and_set_current_pc(self, scene_name=None):
        self.set_current_camera(self.get_camera_at_origin(), archive_camera=True)
        sem_map = self.update_sky_mask()
        render_output = self.render(render_sky=True)
        self.image_latest = soft_stitching(render_output["rendered_image"], self.image_latest, self.sky_mask_latest)  # Replace generated sky with rendered sky

        ground_mask = self.generate_ground_mask(sem_map=sem_map)[None, None]
        depth_should_be_ground = self.compute_ground_depth(camera_height=0.0003)
        ground_outputable_mask = (depth_should_be_ground > 0.001) & (depth_should_be_ground < 0.006 * 0.8)

        with torch.no_grad():
            depth_guided, _ = self.get_depth(self.image_latest, archive_output=True, target_depth=depth_should_be_ground, mask_align=(ground_mask & ground_outputable_mask),
                                             diffusion_steps=30, guidance_steps=8)
        self.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())

        if self.config['gen_layer']:
            self.generate_layer(pred_semantic_map=sem_map, scene_name=scene_name)
            depth_should_be = self.depth_latest_init
            mask_to_align_depth = ~(self.mask_disocclusion.bool()) & (depth_should_be < 0.006 * 0.8)
            mask_to_farther_depth = self.mask_disocclusion.bool() & (depth_should_be < 0.006)
            with torch.no_grad():
                self.depth, self.disparity = self.get_depth(self.image_latest, archive_output=True, target_depth=depth_should_be, mask_align=mask_to_align_depth, mask_farther=mask_to_farther_depth,
                                                            diffusion_steps=30, guidance_steps=8)
            self.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy(),
                                             existing_mask=~(self.mask_disocclusion).bool().squeeze().cpu().numpy(),
                                             existing_disp=self.disparity_latest_init.squeeze().cpu().numpy())
            wrong_depth_mask = self.depth_latest<self.depth_latest_init
            self.depth_latest[wrong_depth_mask] = self.depth_latest_init[wrong_depth_mask] + 0.0001
            self.depth_latest = self.mask_disocclusion * self.depth_latest + (1-self.mask_disocclusion) * self.depth_latest_init
            self.update_sky_mask()
            self.update_current_pc_by_kf(image=self.image_latest, depth=self.depth_latest, valid_mask=~self.sky_mask_latest)  # Base only
            self.update_current_pc_by_kf(image=self.image_latest_init, depth=self.depth_latest_init, valid_mask=self.mask_disocclusion, gen_layer=True)  # Object layer
        else:
            self.update_current_pc_by_kf(image=self.image_latest, depth=self.depth_latest, valid_mask=~self.sky_mask_latest)
        self.archive_latest()

    @torch.no_grad()
    def compute_ground_depth(self, camera_height = 0.0003):
        """
        Compute the depth map in PyTorch, assuming that after camera unproject, all pixels will lie in the XoZ plane.
        return:
            analytic_depth: [1, 1, 512, 512] torch tensor containing depth values
        """
        focal_length = self.init_focal_length
        x_res, y_res = 512, 512
        y_principal = 256

        # Generate a grid of y-coordinate values aligned directly with its use in the final tensor
        y_grid = torch.arange(y_res).view(1, 1, y_res, 1)

        # Compute the depth using the formula d = h * f / (y - p_y)
        denominator = torch.where(y_grid - y_principal != 0, y_grid - y_principal, torch.tensor(1e-10))
        depth_map = (camera_height * focal_length) / denominator
        
        # Explicitly expand the last dimension to match x_res
        depth_map = depth_map.expand(-1, -1, -1, x_res)
        
        return depth_map.to(self.device)
    
    def generate_sky_pointcloud(self, syncdiffusion_model:SyncDiffusion=None, image=None, mask=None, gen_sky=False, style=None):
        image_height = 512
        image_width = 6144
        w_start = 256
        stride = 8
        anchor_view_idx = w_start // 8 // stride
        layers_panorama = 2
        num_inference_steps = 50
        guidance_scale = 7.5
        sync_weight = 80.0
        sync_decay_rate = 0.98
        sync_freq = 3
        sync_thres = 50
        
        example_name = self.config["example_name"]
        
        def linear_blend(images, overlap=100):
    
            # create blending field
            alpha = np.linspace(0, 1, overlap).reshape(overlap, 1, 1)
            
            for i, img in enumerate(images):
                img_new = np.array(img)
                if i != 0:
                    overlap_img2 = img_new[512-overlap:, :, :]
                    top_img = img_new[:512-overlap, :, :]
                    blend_overlap = overlap_img1 * (1 - alpha) + overlap_img2 * alpha

                    # combine the image
                    blended_image = np.concatenate((top_img, blend_overlap, bottom_img), axis=0)
                    img_old = blended_image
                else:
                    img_old = img_new
                
                overlap_img1 = img_old[:overlap, :, :]
                bottom_img = img_old[overlap:, :, :]
            
            blended_image = (blended_image).astype(np.uint8)
            return Image.fromarray(blended_image)
        
        imgs = []
        gen_layer_0 = (not os.path.exists(f'./examples/sky_images/{example_name}/sky_0.png')) or gen_sky
        gen_layer_1 = (not os.path.exists(f'./examples/sky_images/{example_name}/sky_1.png')) or gen_layer_0 or gen_sky
        gen_layer_2 = (not os.path.exists(f'./examples/sky_images/{example_name}/sky_2.png')) or gen_layer_1 or gen_sky
        
        for layer in range(layers_panorama):
            if layer == 0:
                if gen_layer_0:    
                    init_image = torch.zeros((1, 3, image_height, image_width))
                    init_image[:, :, :, w_start:w_start+image_height] = image
                    init_image = init_image.to(self.device)
                    ToPILImage()(init_image[0]).save(self.run_dir / f"{layer:02d}_init_image.png")

                    mask_image = torch.ones((1, 1, image_height, image_width))
                    mask_image[:, :, :, w_start:w_start+image_height] = 1-mask
                    mask_image = mask_image.to(self.device)
                    ToPILImage()(mask_image.float()[0]).save(self.run_dir / f"{layer:02d}_mask.png")
                    
                    # Inpaint init_image using inpaint_cv2()
                    mask_image_eroded = dilation(mask_image,
                                    kernel=torch.ones(10, 10).cuda()
                                    )
                    init_image = inpaint_cv2(init_image, mask_image_eroded)
                    init_image = init_image.to(self.device)
                    ToPILImage()(init_image[0]).save(self.run_dir / f"{layer:02d}_inpainted_init_image.png")

                    # Block-expand mask using an aggresive way
                    mask_ = (mask_image[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    mask_block_size = 8
                    mask_ = skimage.measure.block_reduce(mask_, (mask_block_size, mask_block_size), np.min)
                    mask_ = mask_.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
                    mask_image = ToTensor()(mask_).unsqueeze(0).to(self.device)
                    ToPILImage()(mask_image.float()[0]).save(self.run_dir / f"{layer:02d}_mask_blocky.png")    
                else:
                    img = Image.open(f'./examples/sky_images/{example_name}/sky_0.png')
                    # img.save(self.run_dir / f"{layer:02d}_synced_output.png")
                    imgs.append(img)
                    continue
            else:
                if gen_layer_1:
                    init_image = imgs[-1]
                    init_image = ToTensor()(init_image).unsqueeze(0).to(self.device)
                    toprows = init_image[:, :, :100, :]
                    remaining = init_image[:, :, 100:, :]
                    init_image = torch.cat((remaining, toprows), dim=-2)
                    ToPILImage()(init_image[0]).save(self.run_dir / f"{layer:02d}_init_image.png")
                    
                    mask_image = torch.ones((1, 1, image_height, image_width))
                    mask_image[:, :, -100:, :] = 0
                    mask_image = mask_image.to(self.device)
                    ToPILImage()(mask_image.float()[0]).save(self.run_dir / f"{layer:02d}_mask.png")
                else:
                    img = Image.open(f'./examples/sky_images/{example_name}/sky_1.png')
                    # img.save(self.run_dir / f"{layer:02d}_synced_output.png")
                    imgs.append(img)
                    continue
            
            print(f"[INFO] generating sky layer {layer} ...")
            prompts = f"sky, blue sky, horizon, distant hills. style: {style}" if layer == 0 else f"sky, blue sky, cloud. style: {style}"
            # prompts = f"sky, blue sky, cloud, horizon. style: {style}" if layer == 0 else f"sky, blue sky, cloud. style: {style}"
            img = syncdiffusion_model.sample(
                prompts = prompts,
                negative_prompts = 'tree, text',
                height = image_height,
                width = image_width,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                sync_weight = sync_weight,
                sync_decay_rate = sync_decay_rate,
                sync_freq = sync_freq,
                sync_thres = sync_thres,
                stride = stride,
                loop_closure = True,
                condition = True,
                inpaint_mask=mask_image,
                rendered_image=init_image,
                anchor_view_idx=anchor_view_idx,
            )

            # img.save(self.run_dir / f"{layer:02d}_synced_output.png")

            new_img = ToTensor()(img).unsqueeze(0).to(self.device)
            mask_image_ = mask_image.expand(-1, 3, -1, -1).bool()
            loss = F.mse_loss(new_img[~mask_image_], init_image[~mask_image_]).cpu().item()
            print(f"[INFO] Sky Loss: {loss}")

            # move conditioning image to the leftmost, and save it
            if layer == 0:
                new_img_ = torch.cat((new_img[:, :, :, w_start:], new_img[:, :, :, :w_start]), dim=-1)
                img = ToPILImage()(new_img_[0])
                img.save(self.run_dir / f"{layer:02d}_sky_leftmost.png")
            os.makedirs(f'./examples/sky_images/{example_name}', exist_ok=True)
            img.save(f'./examples/sky_images/{example_name}/sky_{layer}.png')
            imgs.append(img)
        img = linear_blend(imgs)
        
        image_height =  img.size[-1]  
        equatorial_radius = 0.02
        # range: FOV
        camera_angle_x = 2*np.arctan(512 / (2*self.init_focal_length))
        min_latitude = -camera_angle_x / 2 - (image_height / 512 - 1) * camera_angle_x # Starting latitude of the band
        max_latitude = camera_angle_x / 2 # Ending latitude of the band

        latitude = torch.linspace(min_latitude, max_latitude, image_height)
        longitude_offset = -camera_angle_x / 2  # The conditioning img is the leftmost, we need to offset the longitude to let the 3D points align with the panorama
        longitude = torch.linspace(longitude_offset, longitude_offset + 2 * np.pi, image_width)

        lat, lon = torch.meshgrid(latitude, longitude, indexing='ij')
        
        # Pytorch3d coord system: +x: left, +y: up, +z: forward
        x = -equatorial_radius * torch.cos(lat) * torch.sin(lon)
        z = equatorial_radius * torch.cos(lat) * torch.cos(lon)
        y = -equatorial_radius * torch.sin(lat)

        points = torch.stack((x, y, z), -1)

        # Flatten the points for batch processing
        points_flat = points.reshape(-1, 3)

        # Assuming 'self.device' is the PyTorch device you want to use
        new_points_3d = points_flat.to(self.device)
        
        # img.save(self.run_dir / "sky.png")
        
        image_latest = ToTensor()(img).unsqueeze(0).to(self.device)
        colors = rearrange(image_latest, "b c h w -> (h w b) c")

        # Remove points below the ground height
        sky_rows_idx = torch.where(mask.any(dim=1))[0]
        max_idx = sky_rows_idx.max().item()
        ground_threshold = -0.0003 if max_idx <= 255 else -0.003
        mask_above_ground = new_points_3d[:, 1] >= ground_threshold
        new_points_3d = new_points_3d[mask_above_ground]
        colors = colors[mask_above_ground]
        
        # ########## DEBUG: layer-wise ablation ############
        # new_points_3d = new_points_3d[mask_above_ground * layer_mask.bool()]
        # colors = colors[mask_above_ground * layer_mask.bool()]
        # ToPILImage()((mask_above_ground * layer_mask.bool()).reshape(924, 6144).float()).save('test.png')
        # ########## DEBUG: layer-wise ablation ############
        
        self.update_current_pc(new_points_3d, colors, gen_sky=True)
        # return

        # generate the upper part of the sky
        self.depth_latest[:] = self.sky_hard_depth
        self.disparity_latest[:] = 1. / self.sky_hard_depth
        self.depth_latest = self.depth_latest.to(self.device)
        self.disparity_latest = self.disparity_latest.to(self.device)

        ########## Generate downsampled points ############
        image_height_down, image_width_down = int(image_height / 2), int(image_width / 2)
        img_down = img.resize((image_width_down, image_height_down), Image.Resampling.LANCZOS)
        latitude_down = torch.linspace(min_latitude, max_latitude, image_height_down)
        longitude_offset = -camera_angle_x / 2  # The conditioning img is the leftmost, we need to offset the longitude to let the 3D points align with the panorama
        longitude_down = torch.linspace(longitude_offset, longitude_offset + 2 * np.pi, image_width_down)

        lat_down, lon_down = torch.meshgrid(latitude_down, longitude_down, indexing='ij')
        
        x_down = -equatorial_radius * torch.cos(lat_down) * torch.sin(lon_down)
        z_down = equatorial_radius * torch.cos(lat_down) * torch.cos(lon_down)
        y_down = -equatorial_radius * torch.sin(lat_down)

        points_down = torch.stack((x_down, y_down, z_down), -1)
        points_flat_down = points_down.reshape(-1, 3)
        new_points_3d_down = points_flat_down.to(self.device)

        image_latest_down = ToTensor()(img_down).unsqueeze(0).to(self.device)
        colors_down = rearrange(image_latest_down, "b c h w -> (h w b) c")

        mask_above_ground = new_points_3d_down[:, 1] >= ground_threshold
        new_points_3d_down = new_points_3d_down[mask_above_ground]
        colors_down = colors_down[mask_above_ground]
        self.sky_pc_downsampled = {"xyz": new_points_3d_down, "rgb": colors_down}
        
        self.generate_sky_cameras()
        print('No using sky top for efficiency.')
        return
        
    @torch.no_grad()
    def get_camera_by_js_view_matrix(self, view_matrix, xyz_scale=1.0, big_view=False):
        """
        args:
            view_matrix: list of 16 elements, representing the view matrix of the camera
            xyz_scale: This was used to scale the x, y, z coordinates of the camera when converting to 3DGS.
                Need to convert it back.
        return:
            camera: PyTorch3D camera object
        """
        view_matrix = torch.tensor(view_matrix, device=self.device, dtype=torch.float).reshape(4, 4)
        xy_negate_matrix = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=self.device, dtype=torch.float)
        view_matrix_negate_xy = view_matrix @ xy_negate_matrix
        R = view_matrix_negate_xy[:3, :3].unsqueeze(0)
        T = view_matrix_negate_xy[3, :3].unsqueeze(0)
        camera = self.get_camera_at_origin(big_view=big_view)
        camera.R = R
        camera.T = T / xyz_scale
        return camera

    @torch.no_grad()
    def update_sky_mask(self):
        sky_mask_latest, sem_seg = self.generate_sky_mask(self.image_latest, return_sem_seg=True)
        self.sky_mask_latest = sky_mask_latest[None, None, :]
        return sem_seg

    @torch.no_grad()
    def generate_sky_mask(self, input_image=None, return_sem_seg=False):
        if input_image is not None:
            image = ToPILImage()(input_image.squeeze())
        else:
            image = ToPILImage()(self.image_latest.squeeze())
            
        segmenter_input = self.segment_processor(image, ["semantic"], return_tensors="pt")
        segmenter_input = {name: tensor.to("cuda") for name, tensor in segmenter_input.items()}
        segment_output = self.segment_model(**segmenter_input)
        pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
                                segment_output, target_sizes=[image.size[::-1]])[0]
        sky_mask = pred_semantic_map == 2  # 2 for ade20k, 119 for coco
        if self.sky_erode_kernel_size > 0:
            sky_mask = erosion(sky_mask.float()[None, None], 
                            kernel=torch.ones(self.sky_erode_kernel_size, self.sky_erode_kernel_size).to(self.device)
                            ).squeeze() > 0.5
        if return_sem_seg:
            return sky_mask, pred_semantic_map
        else:
            return sky_mask
    
    @torch.no_grad()
    def generate_ground_mask(self, sem_map=None, input_image=None):
        if sem_map is None:
            if input_image is not None:
                image = ToPILImage()(input_image.squeeze())
            else:
                image = ToPILImage()(self.image_latest.squeeze())
                
            segmenter_input = self.segment_processor(image, ["semantic"], return_tensors="pt")
            segmenter_input = {name: tensor.to("cuda") for name, tensor in segmenter_input.items()}
            segment_output = self.segment_model(**segmenter_input)
            pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
                                    segment_output, target_sizes=[image.size[::-1]])[0]
            sem_map = pred_semantic_map
        # 3: floor; 6: road; 9: grass; 11: pavement; 13: earth; 26: sea; 29: field; 46: sand; 128: lake
        ground_mask = (sem_map == 3) | (sem_map == 6) | (sem_map == 9) | (sem_map == 11) | (sem_map == 13) | (sem_map == 26) | (sem_map == 29) | (sem_map == 46) | (sem_map == 128)
        if self.config['ground_erode_kernel_size'] > 0:
            ground_mask = erosion(ground_mask.float()[None, None], 
                            kernel=torch.ones(self.config['ground_erode_kernel_size'], self.config['ground_erode_kernel_size']).to(self.device)
                            ).squeeze() > 0.5
        return ground_mask

    @torch.no_grad()
    def generate_grad_magnitude(self, disparity):
        vmin, vmax = disparity.min(), disparity.max()
        normalized_disparity = (disparity - vmin) / (vmax - vmin)
        cmap = plt.get_cmap('viridis')
        rgb_image = cmap(normalized_disparity)
        rgb_image = rgb_image[...,1]
        disparity = np.uint8(rgb_image * 255)

        ToPILImage()(disparity).save(self.run_dir / 'images' / 'disparity_gradient' / f'{self.kf_idx}_normalized_disparity.png')

        # Compute gradients along the x and y axis
        grad_x = cv2.Sobel(disparity, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(disparity, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        grad_magnitude = cv2.magnitude(grad_x, grad_y)
        grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        threshold = 10
        mask = torch.from_numpy(grad_magnitude > threshold)
        return mask
        
    @torch.no_grad()
    def generate_layer(self, pred_semantic_map=None, scene_name=None):
        self.image_latest_init = copy.deepcopy(self.image_latest)
        self.depth_latest_init = copy.deepcopy(self.depth_latest)
        self.disparity_latest_init = copy.deepcopy(self.disparity_latest)
        if pred_semantic_map is None:
            image = ToPILImage()(self.image_latest.squeeze())
            
            segmenter_input = self.segment_processor(image, ["semantic"], return_tensors="pt")
            segmenter_input = {name: tensor.to("cuda") for name, tensor in segmenter_input.items()}
            segment_output = self.segment_model(**segmenter_input)
            pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
                                    segment_output, target_sizes=[image.size[::-1]])[0]

        unique_elements = torch.unique(pred_semantic_map)
        masks = {str(element.item()): (pred_semantic_map == element) for element in unique_elements}
        
        # erosion the mask to avoid margin effect
        disparity_np = self.disparity_latest.squeeze().cpu().numpy()
        grad_magnitude_mask = self.generate_grad_magnitude(disparity_np)
        mask_disocclusion = np.full((512, 512), False, dtype=bool)

        dilation_kernel=torch.ones(9, 9).to(self.device)
        for id, mask in masks.items():
            # exclude 3: floor; 6: road; 9: grass; 11: pavement; 13: earth; 26: sea; 29: field; 46: sand; 52: path, 128: lake
            if id in ['3', '6', '9', '11', '13', '26', '29', '46', '52', '128']:
                continue
            #-- 1. Dilate each segment --#
            mask = dilation((mask).float()[None, None], 
                            kernel=dilation_kernel).squeeze().cpu() > 0.5
            # 4: tree; 6: boat; 83: truck; 87: street lamp
            if id in ['4', '76', '83', '87']:
                mask_disocclusion |= mask.numpy()
                continue
            labeled_array, num_features = label(mask)
            for i in range(1, num_features+1):
                #-- 2. Fetch all disparity values, within the segment --#
                mask_i = labeled_array==i
                disp_pixels = disparity_np[mask_i]
                disparity_mean = disp_pixels.mean()
                #-- 3. [Remove distant segments] --#
                if disparity_mean < np.percentile(disparity_np, 60):
                    continue
                #-- 4. [Find the disparity boundary] --#
                grad_magnitude_segment = grad_magnitude_mask[mask_i]
                #-- 5. [Remove segments without disparity boundaries] --#
                if grad_magnitude_segment.float().mean() < 0.02:
                    continue
                #-- 6. [Find boundary pixels] --#
                segment_boundary = np.where(mask_i, grad_magnitude_mask, 0) 
                if disparity_np[segment_boundary!=0].mean() > np.percentile(disp_pixels, 70):
                    continue
                #-- 7. [Find big-enough region] --#
                if mask_i.mean() < 0.001:
                    continue
                #-- 8. [Find non-road region] --#
                mask_i_erosion = erosion(torch.from_numpy(mask_i).float()[None, None], 
                            kernel=dilation_kernel.cpu()).squeeze() > 0.5
                disp_pixels = disparity_np[mask_i_erosion]
                p20 = np.percentile(disp_pixels, 20)
                p80 = np.percentile(disp_pixels, 80)
                if 1/p20 - 1/p80 > 0.0003 and mask_i.mean() > 0.05: # indicates it is a road
                    continue
                
                save_prompt = False
                # print(i, "disparity_mean:", disparity_mean, "segment_disparity_mean:", disparity_np[segment_boundary!=0].mean())
                mask_disocclusion |= mask_i

        inpainting_prompt = scene_name if scene_name is not None else 'road, building'
        print("Base layer inpainting_prompt: ", inpainting_prompt)
        mask_disocclusion = torch.from_numpy(mask_disocclusion)[None, None]
        
        """Outside of this function, mask_disocclusion will be used to update point cloud and compute depth; 
        # For depth, we want the mask to be accurate, because we want to align correctly;
        # For point cloud, we also want the mask accurate, because we don't want to attach sky edges to the trees."""

        self.mask_disocclusion = erosion(mask_disocclusion.float().to(self.device),
                                         kernel=dilation_kernel)
        inpaint_mask = self.mask_disocclusion > 0.5  # Erode a bit to prevent over-inpaint
        self.inpaint(self.image_latest, inpaint_mask=inpaint_mask, inpainting_prompt=inpainting_prompt, negative_prompt='tree, plant', mask_strategy=np.max, diffusion_steps=50)
        inpainter_output = self.image_latest
        
        stitch_mask = erosion(mask_disocclusion.float().to(self.device),
                            kernel=torch.ones(5, 5).to(self.device))  # keep it slightly dilated to prevent dirty artifacts
        self.image_latest = soft_stitching(inpainter_output, self.image_latest_init, stitch_mask, sigma=1, blur_size=3)
        ToPILImage()(grad_magnitude_mask.float()).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_grad_magnitude_mask.png')
        ToPILImage()((self.image_latest.cpu() * (~mask_disocclusion).float())[0]).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_mask_disocclusion.png')
        ToPILImage()((self.image_latest_init * inpaint_mask.float())[0]).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_inpaint_mask.png')
        ToPILImage()(self.image_latest_init[0]).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_image_init.png')
        ToPILImage()(self.image_latest[0]).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_remove_disocclusion.png')

        
    @torch.no_grad()
    def transform_all_cam_to_current_cam(self, center=False):
        """Transform all self.cameras such that the current camera is at the origin."""

        if self.cameras != []:
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
    def set_current_camera(self, camera, archive_camera=False):
        self.current_camera = camera
        if archive_camera:
            self.cameras_archive.append(copy.deepcopy(camera))
    
    @torch.no_grad()
    def generate_9_cameras(self, center_camera, distance = 0.00001):
        # center_camera is the original camera, assumed to be an object with attributes for position (T) and rotation (R).
        cameras = []
        cameras.append(center_camera)  # The center camera

        # Generate positions around the center camera
        offsets = [
            (-1, -1), (0, -1), (1, -1),  # Top row
            (-1,  0),          (1,  0),  # Middle row (excluding the center)
            (-1,  1), (0,  1), (1,  1)   # Bottom row
        ]

        for offset in offsets:
            # Create a new camera based on the center camera
            new_camera = copy.deepcopy(center_camera)
            
            # Compute the new translation
            right = new_camera.R[0, :, 0]  # Camera's right direction
            forward = new_camera.R[0, :, 1]  # Camera's forward direction
            delta_position = offset[0] * right + offset[1] * forward
            delta_position = delta_position / torch.norm(delta_position) * distance

            # Update camera position
            new_camera.T[0] = new_camera.T[0] + delta_position
            
            cameras.append(new_camera)
        
        return cameras

    @torch.no_grad()
    def set_cameras(self, rotation_path):
        move_left_count = 0
        move_right_count = 0
        for rotation in rotation_path:
            new_camera = copy.deepcopy(self.cameras[-1])

            if rotation == 0:
                forward_speed_multiplier = -1.0
                right_multiplier = 0
                camera_speed = self.camera_speed
                
                # If the camera is not centered, rotate the camera by rotation matrix
                if move_left_count != 0 or move_right_count != 0:
                    # moving backward and previous motion is moving right/left
                    new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                    move_left_count = 0
                    move_right_count = 0
                    
            elif abs(rotation) == 2:
                # If the camera is not centered, rotate the camera by rotation matrix
                if rotation > 0:
                    move_left_count += 1
                    # moving left and previous motion is moving right
                    if move_right_count != 0:
                        new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                        move_right_count = 0
                else:
                    move_right_count += 1
                    # moving right and previous motion is moving left
                    if move_left_count != 0:
                        new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                        move_left_count = 0
                    
                forward_speed_multiplier = 0
                right_multiplier = 0
                camera_speed = 0
                theta = torch.tensor(self.rotation_range_theta * rotation / 2)
                rotation_matrix = torch.tensor(
                    [[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]],
                    device=self.device,
                )
                new_camera.R[0] = rotation_matrix @ new_camera.R[0]
                    
            elif abs(rotation) == 1:  # Pre-compute camera movement, accounting for a set of kfinterp rotations
                # If the camera is not centered, rotate the camera by rotation matrix
                if move_left_count != 0 or move_right_count != 0:
                    # moving backward and previous motion is moving right/left
                    new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                    move_left_count = 0
                    move_right_count = 0
                    
                theta_frame = torch.tensor(self.rotation_range_theta / (self.interp_frames + 1)) * rotation
                sin = torch.sum(torch.stack([torch.sin(i*theta_frame) for i in range(1, self.interp_frames+2)]))
                cos = torch.sum(torch.stack([torch.cos(i*theta_frame) for i in range(1, self.interp_frames+2)]))
                forward_speed_multiplier = -1.0 / (self.interp_frames + 1) * cos.item()
                right_multiplier = -1.0 / (self.interp_frames + 1) * sin.item()
                camera_speed = self.camera_speed * self.camera_speed_multiplier_rotation

                theta = torch.tensor(self.rotation_range_theta * rotation)
                rotation_matrix = torch.tensor(
                    [[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]],
                    device=self.device,
                )
                new_camera.R[0] = rotation_matrix @ new_camera.R[0]

            elif rotation == 3:
                continue
                
            move_dir = torch.tensor([[-right_multiplier, 0.0, -forward_speed_multiplier]], device=self.device)

            # move camera backwards
            new_camera.T += camera_speed * move_dir
            self.cameras.append(copy.deepcopy(new_camera))

        return new_camera
    
    @torch.no_grad()
    def generate_cameras(self, rotation_path):
        print("-- generating 360-degree cameras...")
        # Generate init camera for each scene
        camera = self.get_camera_at_origin()
        self.cameras.append(copy.deepcopy(camera))
        self.scene_cameras_idx.append(len(self.cameras) - 1)
        self.transform_all_cam_to_current_cam()
        # Generate camera sequence based on rotation_path
        self.set_cameras(rotation_path)
        self.center_camera_idx = 0
        self.transform_all_cam_to_current_cam(True)
        print("-- generated 360-degree cameras!")
    
    @torch.no_grad()
    def generate_sky_cameras(self):
        print("-- generating sky cameras...")
        cameras_cache = copy.deepcopy(self.cameras)
        init_len = len(self.cameras)
        
        # Generate cameras for sky generation
        for i in tqdm(range(1)):
            delta = -torch.tensor(torch.pi) / (8) * (i + 1)
            for camera_id in range(init_len):    
                self.center_camera_idx = camera_id
                self.transform_all_cam_to_current_cam(True)
                new_camera = copy.deepcopy(self.cameras[camera_id])                
                
                rotation_matrix = torch.tensor(
                    [[1, 0, 0], [0, torch.cos(delta), -torch.sin(delta)], [0, torch.sin(delta), torch.cos(delta)]],
                    device=self.device,
                )
                new_camera.R[0] = rotation_matrix @ new_camera.R[0]
                
                self.cameras.append(copy.deepcopy(new_camera))
        self.center_camera_idx = 0
        self.transform_all_cam_to_current_cam(True)
        self.sky_cameras = copy.deepcopy(self.cameras)
        self.cameras = cameras_cache
        print("-- generated sky cameras!")
        
    @torch.no_grad()
    def set_kf_param(self, inpainting_resolution, inpainting_prompt, adaptive_negative_prompt):
        super().set_frame_param(inpainting_resolution=inpainting_resolution, 
                                inpainting_prompt=inpainting_prompt, adaptive_negative_prompt=adaptive_negative_prompt)

    @torch.no_grad()
    def refine_disp_with_segments(self, save_intermediates=False, keep_threshold_disp_range=10, no_refine_mask=None, existing_mask=None, existing_disp=None):
        """
        args:
            no_refine_mask: basically it is ground mask, if not None. Then, if a SAM segment has significant intersection with the ground, then we discard it.
            existing_mask: if not None, then we are refining layer-inpainted disp. In this case, most regions have existing fixed disp. 
                            For a SAM segment, if it intersects with existing_mask significantly, 
                            then we should only use the values from existing_disp, not from the current estimate, to minimize the gaps.
        """
        print('Refining disparity with segments...')
        if save_intermediates:
            (self.run_dir / 'refine_intermediates').mkdir(parents=True, exist_ok=True)
        image = ToPILImage()(self.image_latest.squeeze())
        image_np = np.array(image)
        masks = self.mask_generator.generate(image_np)
        sorted_mask = sorted(masks, key=(lambda x: x['area']), reverse=False)  # Iterate from small to large, finally large will have higher priority.
        min_mask_area = 100
        sorted_mask = [m for m in sorted_mask if m['area'] > min_mask_area]

        if save_intermediates:
            save_sam_anns(masks, self.run_dir / 'refine_intermediates' / f"kf{self.kf_idx:02}_SAM.png")
        
        disparity_np = self.disparity_latest.squeeze().cpu().numpy()

        refined_disparity = refine_disp_with_segments_2(disparity_np, sorted_mask, keep_threshold=keep_threshold_disp_range, no_refine_mask=no_refine_mask,
                                                        existing_mask=existing_mask, existing_disp=existing_disp)

        if save_intermediates:
            save_depth_map(1/refined_disparity, self.run_dir / 'refine_intermediates' / f"kf{self.kf_idx:02}_p1_SAM")

        refined_depth = 1 / refined_disparity

        refined_depth = torch.from_numpy(refined_depth).to(self.device)
        refined_disparity = torch.from_numpy(refined_disparity).to(self.device)

        self.depth_latest[0, 0] = refined_depth
        self.disparity_latest[0, 0] = refined_disparity

        print('Refining done!')
        return refined_depth, refined_disparity

    @torch.no_grad()
    def generate_visible_pc(self):        
        camera = self.current_camera
        raster_settings = PointsRasterizationSettings(
            image_size = 512,
            radius = 0.003,
            points_per_pixel = 8,
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
        )
        points, colors = self.get_combined_pc()["xyz"], self.get_combined_pc()["rgb"]
        point_cloud = Pointclouds(points=[points], features=[colors])
        images, fragment_idx = renderer(point_cloud, return_fragment_idx=True)
        fragment_idx = fragment_idx[..., :1]
        
        n_kf1_points = points.shape[0]
        fragment_idx = fragment_idx.reshape(-1)
        visible_points_idx = (fragment_idx < n_kf1_points) & (fragment_idx >= 0)
        fragment_idx = fragment_idx[visible_points_idx]
        
        if self.current_visible_pc is None:
            self.current_visible_pc = {"xyz": points[fragment_idx], "rgb": colors[fragment_idx]}
        else:
            self.current_visible_pc = {"xyz": torch.cat([self.current_visible_pc["xyz"], points[fragment_idx]], dim=0), "rgb": torch.cat([self.current_visible_pc["rgb"], colors[fragment_idx]], dim=0)}
        
    @torch.no_grad()
    def render(self, archive_output=False, camera=None, render_visible=False, render_sky=False, big_view=False, render_fg=False):
        camera = self.current_camera if camera is None else camera
        raster_settings = PointsRasterizationSettings(
            image_size = 1536 if big_view else 512,
            radius = 0.003,
            points_per_pixel = 8,
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
        )
        if render_visible:
            points, colors = self.current_visible_pc["xyz"], self.current_visible_pc["rgb"]
        elif render_sky:
            points, colors = self.current_pc_sky["xyz"], self.current_pc_sky["rgb"]
        elif render_fg:
            points, colors = self.current_pc["xyz"], self.current_pc["rgb"]
        else:
            points, colors = self.get_combined_pc()["xyz"], self.get_combined_pc()["rgb"]
            
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

        return {
            "rendered_image": rendered_image,
            "rendered_depth": rendered_depth,
            "inpaint_mask": inpaint_mask,
        }
        
    @torch.no_grad()
    def archive_latest(self, idx=None):
        if idx is None:
            idx = self.kf_idx
        vmax = 0.006
        super().archive_latest(idx=idx, vmax=vmax)
        self.rendered_images.append(self.rendered_image_latest)
        self.rendered_depths.append(self.rendered_depth_latest)
        self.sky_mask_list.append(~self.sky_mask_latest.bool())


def get_extrinsics(camera):
    extrinsics = torch.cat([camera.R[0], camera.T.T], dim=1)
    padding = torch.tensor([[0, 0, 0, 1]], device=extrinsics.device)
    extrinsics = torch.cat([extrinsics, padding], dim=0)
    return extrinsics

def save_point_cloud_as_ply(points, filename="output.ply", colors=None):
    """
    Save a PyTorch tensor of shape [N, 3] as a PLY file. Optionally with colors.
    
    Parameters:
    - points (torch.Tensor): The point cloud tensor of shape [N, 3].
    - filename (str): The name of the output PLY file.
    - colors (torch.Tensor, optional): The color tensor of shape [N, 3] with values in [0, 1]. Default is None.
    """
    
    assert points.dim() == 2 and points.size(1) == 3, "Input tensor should be of shape [N, 3]."
    
    if colors is not None:
        assert colors.dim() == 2 and colors.size(1) == 3, "Color tensor should be of shape [N, 3]."
        assert points.size(0) == colors.size(0), "Points and colors tensors should have the same number of entries."
    
    # Header for the PLY file
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.size(0)}",
        "property float x",
        "property float y",
        "property float z"
    ]
    
    # Add color properties to header if colors are provided
    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue"
        ])
    
    header.append("end_header")
    
    # Write to file
    with open(filename, "w") as f:
        for line in header:
            f.write(line + "\n")
        
        for i in range(points.size(0)):
            line = f"{points[i, 0].item()} {points[i, 1].item()} {points[i, 2].item()}"
            
            # Add color data to the line if colors are provided
            if colors is not None:
                # Scale color values from [0, 1] to [0, 255] and convert to integers
                r, g, b = (colors[i] * 255).clamp(0, 255).int().tolist()
                line += f" {r} {g} {b}"
            
            f.write(line + "\n")

def convert_pytorch3d_kornia(camera, focal_length, size=512):
    transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
    transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)

    pt3d_to_kornia = torch.diag(torch.tensor([-1., -1, 1, 1], device=camera.device))
    transform_matrix_w2c_kornia = pt3d_to_kornia @ transform_matrix_w2c_pt3d
    
    extrinsics = transform_matrix_w2c_kornia.unsqueeze(0)
    h = torch.tensor([size], device="cuda")
    w = torch.tensor([size], device="cuda")
    K = torch.eye(4)[None].to("cuda")
    K[0, 0, 2] = size // 2
    K[0, 1, 2] = size // 2
    K[0, 0, 0] = focal_length
    K[0, 1, 1] = focal_length
    return PinholeCamera(K, extrinsics, h, w)


def inpaint_cv2(rendered_image, mask_diff):
    """
    Performs inpainting on a single image using a corresponding mask, both provided as PyTorch tensors.
    
    Args:
    - rendered_image (torch.Tensor): A tensor of shape [batch_size, channels, height, width].
      This function uses only the first image in the batch for inpainting.
    - mask_diff (torch.Tensor): A tensor of shape [batch_size, 1, height, width] representing the inpainting mask.
      This function uses only the first mask in the batch.
    
    Returns:
    - torch.Tensor: Inpainted image in a tensor of shape [1, channels, height, width], where 'channels'
      corresponds to the same number of channels as the input image, typically 3 (RGB).
    
    """
    image_cv2 = rendered_image[0].permute(1, 2, 0).cpu().numpy()
    image_cv2 = (image_cv2 * 255).astype(np.uint8)
    mask_cv2 = mask_diff[0, 0].cpu().numpy()
    mask_cv2 = (mask_cv2 * 255).astype(np.uint8)
    inpainting = cv2.inpaint(image_cv2, mask_cv2, 3, cv2.INPAINT_TELEA)
    inpainting = torch.from_numpy(inpainting).permute(2, 0, 1).float() / 255
    return inpainting.unsqueeze(0)