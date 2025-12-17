import os
import copy
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
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
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
# HuggingFace hub download
from huggingface_hub import snapshot_download
# 获取当前文件所在目录 (sceneflow_test)
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "wonder_journey"))
from util.midas_utils import dpt_transform, dpt_512_transform
from util.utils import functbl, save_depth_map
from util.segment_utils import refine_disp_with_segments, save_sam_anns
from midas.model_loader import load_model

from typing import List, Optional, Tuple, Union
from kornia.morphology import erosion

BG_COLOR=(1, 0, 0)

# ----------------- Models from models.py start -----------------
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
    def __init__(
        self, background_color: Optional[Union[Tuple, List, torch.Tensor]] = None, softmax_scale=1.0,
    ) -> None:
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


class FrameSyn(torch.nn.Module):
    def __init__(self, config, inpainter_pipeline, depth_model, vae, rotation,
                 image, inpainting_prompt, adaptive_negative_prompt):
        super().__init__()

        self.device = config["device"]
        self.config = config
        self.background_hard_depth = config['depth_shift'] + config['fg_depth_range']
        self.is_upper_mask_aggressive = False
        self.use_noprompt = False
        self.total_frames = config['frames']

        self.inpainting_prompt = inpainting_prompt
        self.adaptive_negative_prompt = adaptive_negative_prompt
        self.inpainting_pipeline = inpainter_pipeline

        # resize image to 512x512
        image = image.resize((512, 512))
        self.image_tensor = ToTensor()(image).unsqueeze(0).to(self.device)

        self.depth_model = depth_model
        with torch.no_grad():
                self.depth, self.disparity = self.get_depth(self.image_tensor)

        self.current_camera = self.get_init_camera()
        if self.config["motion"] == "rotations":
            self.current_camera.rotating = rotation != 0
            self.current_camera.no_rotations_count = 0
            self.current_camera.rotations_count = 0
            self.current_camera.rotating_right = rotation
            self.current_camera.move_dir = torch.tensor([[-config['right_multiplier'], 0.0, -config['forward_speed_multiplier']]], device=self.device)
        elif self.config["motion"] == "predefined":
            intrinsics = np.load(self.config["intrinsics"]).astype(np.float32)
            extrinsics = np.load(self.config["extrinsics"]).astype(np.float32)

            intrinsics = torch.from_numpy(intrinsics).to(self.device)
            extrinsics = torch.from_numpy(extrinsics).to(self.device)

            Ks = F.pad(intrinsics, (0, 1, 0, 1), value=0)
            Ks[:, 2, 3] = Ks[:, 3, 2] = 1

            Rs, ts = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
            Rs = Rs.movedim(1, 2)

            self.predefined_cameras = [
                PerspectiveCameras(K=K.unsqueeze(0), R=R.T.unsqueeze(0), T=t.unsqueeze(0), device=self.device)
                for K, R, t in zip(Ks, Rs, ts)
            ]
            self.current_camera = self.predefined_cameras[0]

        self.images = [self.image_tensor]
        self.inpaint_input_image = [image]
        self.disparities = [self.disparity]
        self.depths = [self.depth]
        self.masks = [torch.ones_like(self.depth)]
        self.post_masks = [torch.ones_like(self.depth)]
        self.post_mask_tmp = None
        self.rendered_images = [self.image_tensor]
        self.rendered_depths = [self.depth]

        self.vae = vae
        self.decoder_copy = copy.deepcopy(self.vae.decoder)

        self.camera_speed = self.config["camera_speed"] if rotation == 0 else self.config["camera_speed"] * self.config["camera_speed_multiplier_rotation"]
        self.cameras = [self.current_camera]

        self.border_mask = torch.ones(
            (1, 1, self.config["inpainting_resolution"], self.config["inpainting_resolution"])
        ).to(self.device)
        self.border_size = (self.config["inpainting_resolution"] - 512) // 2
        self.border_mask[:, :, self.border_size : -self.border_size, self.border_size : -self.border_size] = 0
        self.border_image = torch.zeros(
            1, 3, self.config["inpainting_resolution"], self.config["inpainting_resolution"]
        ).to(self.device)
        self.images_orig_decoder = [
            Resize((self.config["inpainting_resolution"], self.config["inpainting_resolution"]))(self.image_tensor)
        ]

        x = torch.arange(512).float() + 0.5
        y = torch.arange(512).float() + 0.5
        self.points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        self.points = rearrange(self.points, "h w c -> (h w) c").to(self.device)

        self.kf_delta_t = self.camera_speed

    def get_depth(self, image):
        if self.depth_model is None:
            depth = torch.zeros_like(image[:, 0:1])
            disparity = torch.zeros_like(image[:, 0:1])
            return depth, disparity
        if self.config['depth_model'].lower() == "midas":
            disparity = self.depth_model(dpt_transform(image))
            disparity = torch.nn.functional.interpolate(
                disparity.unsqueeze(1),
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            disparity = disparity.clip(1e-6, max=None)
            depth = 1 / disparity
        if self.config['depth_model'].lower() == "midas_v3.1":
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
        elif self.config['depth_model'].lower() == "zoedepth":
            depth = self.depth_model(image)['metric_depth']
        depth = depth + self.config['depth_shift']
        disparity = 1 / depth
        return depth, disparity

    def get_init_camera(self):
        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = self.config["init_focal_length"]
        K[0, 1, 1] = self.config["init_focal_length"]
        K[0, 0, 2] = 256
        K[0, 1, 2] = 256
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device)
        return camera

    def finetune_depth_model_step(self, target_depth, inpainted_image, mask_align=None, mask_cutoff=None, cutoff_depth=None):
        next_depth, _ = self.get_depth(inpainted_image.detach().cuda())
        loss_align = F.l1_loss(target_depth.detach(), next_depth, reduction="none")
        if mask_align is not None and torch.any(mask_align):
            mask_align = mask_align.detach()
            loss_align = (loss_align * mask_align)[mask_align > 0].mean()
        else:
            loss_align = torch.zeros(1).to(self.device)

        if mask_cutoff is not None and cutoff_depth is not None and torch.any(mask_cutoff):
            hinge_loss = (cutoff_depth - next_depth).clamp(min=0)
            hinge_loss = F.l1_loss(hinge_loss, torch.zeros_like(hinge_loss), reduction="none")
            mask_cutoff = mask_cutoff.detach()
            hinge_loss = (hinge_loss * mask_cutoff)[mask_cutoff > 0].mean()
        else:
            hinge_loss = torch.zeros(1).to(self.device)

        total_loss = loss_align + hinge_loss
        if torch.isnan(total_loss):
            raise ValueError("Depth FT loss is NaN")
        print(f"(1000x) loss_align: {loss_align.item()*1000:.4f}, hinge_loss: {hinge_loss.item()*1000:.4f}, total_loss: {total_loss.item()*1000:.4f}")

        return total_loss

    def finetune_decoder_step(self, inpainted_image, inpainted_image_latent, rendered_image, inpaint_mask, inpaint_mask_dilated):
        reconstruction = self.decode_latents(inpainted_image_latent)
        new_content_loss = F.mse_loss(inpainted_image * inpaint_mask, reconstruction * inpaint_mask)
        preservation_loss = F.mse_loss(rendered_image * (1 - inpaint_mask_dilated), reconstruction * (1 - inpaint_mask_dilated)) * self.config["preservation_weight"]
        loss = new_content_loss + preservation_loss
        return loss

    @torch.no_grad()
    def inpaint(self, rendered_image, inpaint_mask, fill_mask=None, fill_mode = 'cv2_telea'):
        process_width, process_height = self.config["inpainting_resolution"], self.config["inpainting_resolution"]

        img = (rendered_image[0].cpu().permute([1, 2, 0]).numpy() * 255).astype(np.uint8)
        fill_mask = inpaint_mask if fill_mask is None else fill_mask
        fill_mask_ = (fill_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        mask = (inpaint_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        for _ in range(3):
            img, _ = functbl[fill_mode](img, fill_mask_)

        if self.config['use_postmask']:
            mask_block_size = 8
            mask_boundary = mask.shape[0] // 2
            mask_upper = skimage.measure.block_reduce(mask[:mask_boundary, :], (mask_block_size, mask_block_size), np.max if self.is_upper_mask_aggressive else np.min)
            mask_upper = mask_upper.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
            mask_lower = skimage.measure.block_reduce(mask[mask_boundary:, :], (mask_block_size, mask_block_size), np.min)
            mask_lower = mask_lower.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
            mask = np.concatenate([mask_upper, mask_lower], axis=0)

        init_image = Image.fromarray(img)
        mask_image = Image.fromarray(mask)

        inpainted_image_latents = self.inpainting_pipeline(
            prompt='' if self.use_noprompt else self.inpainting_prompt,
            negative_prompt=self.adaptive_negative_prompt + self.config["negative_inpainting_prompt"],
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=25,
            guidance_scale=0 if self.use_noprompt else 7.5,
            height=process_height,
            width=process_width,
            output_type='latent',
        ).images

        if isinstance(inpainted_image_latents, np.ndarray):
            inpainted_image_latents = torch.from_numpy(inpainted_image_latents).to(
                device=self.inpainting_pipeline.device, 
                dtype=self.inpainting_pipeline.vae.dtype
            )
        
        if inpainted_image_latents.ndim == 4 and inpainted_image_latents.shape[-1] == 3:
            x = inpainted_image_latents.permute(0, 3, 1, 2)
            x = 2.0 * x - 1.0
            dist = self.inpainting_pipeline.vae.encode(x).latent_dist
            inpainted_image_latents = dist.sample() * self.inpainting_pipeline.vae.config.scaling_factor
        
        inpainted_image = self.inpainting_pipeline.vae.decode(inpainted_image_latents / self.inpainting_pipeline.vae.config.scaling_factor, return_dict=False)[0]
        inpainted_image = (inpainted_image / 2 + 0.5).clamp(0, 1).to(torch.float32)
        post_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() * 255
        self.post_mask_tmp = post_mask
        self.inpaint_input_image.append(init_image)

        return {"inpainted_image": inpainted_image, "latent": inpainted_image_latents.float()}

    @torch.no_grad()
    def update_images_and_masks(self, latent, inpaint_mask):
        decoded_image = self.decode_latents(latent).detach()
        post_mask = inpaint_mask if self.post_mask_tmp is None else self.post_mask_tmp
        if self.config["inpainting_resolution"] > 512:
            decoded_image = decoded_image[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ]
            inpaint_mask = inpaint_mask[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ]
            post_mask = post_mask[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ]
        else:
            decoded_image = decoded_image
            inpaint_mask = inpaint_mask
            post_mask = post_mask

        self.images.append(decoded_image)
        self.masks.append(inpaint_mask)
        self.post_masks.append(post_mask)

    def decode_latents(self, latents):
        images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1)

        return images

    def get_next_camera_rotation(self):
        next_camera = copy.deepcopy(self.current_camera)
        
        if next_camera.rotating:
            next_camera.rotating_right = self.current_camera.rotating_right
            theta = torch.tensor(self.config["rotation_range_theta"] * next_camera.rotating_right)
            rotation_matrix = torch.tensor(
                [[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]],
                device=self.device,
            )
            next_camera.R[0] = rotation_matrix @ next_camera.R[0]
            
            if self.current_camera.rotations_count != 0: 
                theta_current = theta * (self.config['frames'] + 2 - self.current_camera.rotations_count)
                next_camera.move_dir = torch.tensor([-self.config['forward_speed_multiplier'] * torch.sin(theta_current).item(), 0.0, self.config['forward_speed_multiplier'] * torch.cos(theta_current).item()], device=self.device)
                next_camera.rotations_count = self.current_camera.rotations_count + 1
        else:
            if self.current_camera.rotations_count != 0:
                v = self.config['forward_speed_multiplier']
                rc = self.current_camera.rotations_count
                k = self.config['camera_speed_multiplier_rotation']
                acceleration_frames = self.config["frames"] // 2
                if self.speed_up and rc <= acceleration_frames:
                    next_camera.move_dir = torch.tensor([0.0, 0.0, v * (k + (1-k) * (rc/acceleration_frames))], device=self.device)
                elif self.speed_down and rc > self.total_frames - acceleration_frames:
                    next_camera.move_dir = torch.tensor([0.0, 0.0, v * (k + (1-k) * ((self.total_frames-rc+1)/acceleration_frames))], device=self.device)
                else:
                    next_camera.move_dir = torch.tensor([0.0, 0.0, v], device=self.device)

                theta = torch.tensor(2 * torch.pi * self.current_camera.rotations_count / (self.total_frames + 1))
                next_camera.move_dir[1] = -self.random_walk_scale_vertical * 0.01 * torch.sin(theta).item()

                next_camera.rotations_count = self.current_camera.rotations_count + 1

        speed = self.camera_speed
        next_camera.T += speed * next_camera.move_dir

        return next_camera


class KeyframeGen(FrameSyn):
    def __init__(self, config, inpainter_pipeline, mask_generator, depth_model, vae, rotation, 
                 image, inpainting_prompt, adaptive_negative_prompt="", 
                 segment_model=None, segment_processor=None):
        
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        run_dir_root = Path(config["runs_dir"])
        self.run_dir = run_dir_root / f"Gen-{dt_string}_{inpainting_prompt.replace(' ', '_')[:40]}"
        (self.run_dir / 'images').mkdir(parents=True, exist_ok=True)
        config['rotation_range_theta'] = config['rotation_range']

        if rotation == 0:
            config['forward_speed_multiplier'] = -1.0
            config['right_multiplier'] = 0
        else:
            theta = torch.tensor(config['rotation_range_theta'] / (config['frames'] + 1)) * rotation
            sin = torch.sum(torch.stack([torch.sin(i*theta) for i in range(1, config['frames']+2)]))
            cos = torch.sum(torch.stack([torch.cos(i*theta) for i in range(1, config['frames']+2)]))
            config['forward_speed_multiplier'] = -1.0 / (config['frames'] + 1) * cos.item()
            config['right_multiplier'] = -1.0 / (config['frames'] + 1) * sin.item()
        config['inpainting_resolution'] = config['inpainting_resolution_gen']
        super().__init__(config, inpainter_pipeline, depth_model, vae, rotation,
                         image, inpainting_prompt, adaptive_negative_prompt)
        
        self.mask_generator = mask_generator
        self.segment_model = segment_model
        self.segment_processor = segment_processor
        self.is_upper_mask_aggressive = True

    @torch.no_grad()
    def refine_disp_with_segments(self, kf_idx, background_depth_cutoff=1./7.):
        image = ToPILImage()(self.images[kf_idx].squeeze())
        segmenter_input = self.segment_processor(image, ["semantic"], return_tensors="pt")
        segmenter_input = {name: tensor.to("cuda") for name, tensor in segmenter_input.items()}
        segment_output = self.segment_model(**segmenter_input)
        pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
                                segment_output, target_sizes=[image.size[::-1]])[0]
        sky_mask = pred_semantic_map.cpu() == 119
        sky_mask = erosion(sky_mask.float()[None, None], 
                           kernel=torch.ones(self.config['sky_erode_kernel_size'], self.config['sky_erode_kernel_size'])
                           ).squeeze() > 0.5
        sky_mask = sky_mask.cpu()
        ToPILImage()(sky_mask.float()).save(self.run_dir / 'images' / f"kf{kf_idx+1}_sky_mask.png")

        image_np = np.array(image)
        masks = self.mask_generator.generate(image_np)
        sorted_mask = sorted(masks, key=(lambda x: x['area']), reverse=False)
        min_mask_area = 30
        sorted_mask = [m for m in sorted_mask if m['area'] > min_mask_area]

        save_sam_anns(masks, self.run_dir / 'images' / f"SAM_kf{kf_idx+1}.png")
        disparity_np = self.disparities[kf_idx].squeeze().cpu().numpy()
        keep_threshold_ratio = 0.3
        refined_disparity = refine_disp_with_segments(disparity_np, sorted_mask, keep_threshold=1 / background_depth_cutoff * keep_threshold_ratio)

        save_depth_map(1/refined_disparity, self.run_dir / 'images' / f"kf{kf_idx+1}_p1_SAM", vmax=self.config['sky_hard_depth'])

        sky_hard_disp = 1. / self.config['sky_hard_depth']
        bg_hard_disp = 1. / (background_depth_cutoff)
        refined_disparity[sky_mask] = sky_hard_disp

        save_depth_map(1/refined_disparity, self.run_dir / 'images' / f"kf{kf_idx+1}_p2_sky", vmax=self.config['sky_hard_depth'])

        background_cutoff = 1./background_depth_cutoff
        background_mask = refined_disparity < background_cutoff
        background_but_not_sky_mask = np.logical_and(background_mask, np.logical_not(sky_mask.numpy()))
        refined_disparity[background_but_not_sky_mask] = bg_hard_disp

        save_depth_map(1/refined_disparity, self.run_dir / 'images' / f"kf{kf_idx+1}_p3_cutoff", vmax=self.config['sky_hard_depth'])

        refined_disparity = refine_disp_with_segments(refined_disparity, sorted_mask, keep_threshold=1 / background_depth_cutoff * keep_threshold_ratio)
        
        save_depth_map(1/refined_disparity, self.run_dir / 'images' / f"kf{kf_idx+1}_p4_SAM", vmax=self.config['sky_hard_depth'])

        refined_depth = 1 / refined_disparity

        refined_depth = torch.from_numpy(refined_depth).to(self.device)
        refined_disparity = torch.from_numpy(refined_disparity).to(self.device)

        self.depths[kf_idx][0, 0] = refined_depth
        self.disparities[kf_idx][0, 0] = refined_disparity

        return refined_depth, refined_disparity, sky_mask, background_but_not_sky_mask

    @torch.no_grad()
    def render(self, epoch):
        if self.config["motion"] == "rotations":
            camera = self.get_next_camera_rotation()
        elif self.config["motion"] == "predefined":
            camera = self.predefined_cameras[epoch]
        else:
            raise NotImplementedError
        current_camera = convert_pytorch3d_kornia(self.current_camera, self.config["init_focal_length"])
        point_depth = rearrange(self.depths[epoch - 1], "b c h w -> (w h b) c")
        points_3d = current_camera.unproject(self.points, point_depth)

        colors = rearrange(self.images[epoch - 1], "b c h w -> (w h b) c")
        depth_normalizer = self.background_hard_depth
        min_ratio = self.config['point_size_min_ratio']
        radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
        radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier'])
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius = radius,
            points_per_pixel = 8,
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
        )
        points_3d[..., :2] = - points_3d[..., :2]
        point_cloud = Pointclouds(points=[points_3d], features=[colors])
        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)

        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=(0,), softmax_scale=1.0)
        )
        orig_depth, _ = self.get_depth(self.images[epoch - 1])
        orig_depth = rearrange(orig_depth, "b c h w -> (w h b) c")
        point_cloud_orig_depth = Pointclouds(points=[points_3d], features=[orig_depth])
        rendered_depth = renderer(point_cloud_orig_depth) + self.kf_delta_t
        rendered_depth = rearrange(rendered_depth, "b h w c -> b c h w")

        rendered_image = rearrange(images, "b h w c -> b c h w")
        inpaint_mask = bg_mask.float()[:, None, ...]
        rendered_depth = rearrange(zbuf[..., 0:1], "b h w c -> b c h w")
        rendered_depth[rendered_depth < 0] = 0

        self.current_camera = copy.deepcopy(camera)
        self.cameras.append(self.current_camera)
        self.rendered_images.append(rendered_image)
        self.rendered_depths.append(rendered_depth)

        if self.config["inpainting_resolution"] > 512:
            padded_inpainting_mask = self.border_mask.clone()
            padded_inpainting_mask[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ] = inpaint_mask
            padded_image = self.border_image.clone()
            padded_image[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ] = rendered_image
        else:
            padded_inpainting_mask = inpaint_mask
            padded_image = rendered_image

        return {
            "rendered_image": padded_image,
            "rendered_depth": rendered_depth,
            "inpaint_mask": padded_inpainting_mask,
            "inpaint_mask_512": inpaint_mask,
            "rendered_depth_original": rendered_depth,
        }
    

class KeyframeInterp(FrameSyn):
    def __init__(self, config, inpainter_pipeline, depth_model, vae, rotation, 
                 image, inpainting_prompt, adaptive_negative_prompt, kf2_upsample_coef=1,
                 kf1_image=None, kf2_image=None, kf1_depth=None, kf2_depth=None, kf1_camera=None, kf2_camera=None, kf2_mask=None,
                 speed_up=False, speed_down=False, total_frames=None):
        
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        run_dir_root = Path(config["runs_dir"])
        self.run_dir = run_dir_root / f"Interp-{dt_string}_{inpainting_prompt.replace(' ', '_')[:40]}"
        (self.run_dir / 'images').mkdir(parents=True, exist_ok=True)

        self.speed_up = speed_up
        self.speed_down = speed_down

        self.random_walk_scale_vertical = np.random.uniform(0.1, 0.3)

        config['forward_speed_multiplier'] = -1. / (config['frames'] + 1)
        config['inpainting_resolution'] = config['inpainting_resolution_interp']
        config['right_multiplier'] = 0
        config['rotation_range_theta'] = config['rotation_range'] / (config['frames'] + 1)
        super().__init__(config, inpainter_pipeline, depth_model, vae, rotation,  
                         image, inpainting_prompt, adaptive_negative_prompt)
        self.total_frames = config['frames'] if total_frames is None else total_frames

        self.additional_points_3d = torch.tensor([]).cuda()
        self.additional_colors = torch.tensor([]).cuda()

        self.kf2_upsample_coef = kf2_upsample_coef
        x = torch.arange(512 * kf2_upsample_coef)
        y = torch.arange(512 * kf2_upsample_coef)
        self.points_kf2 = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        self.points_kf2 = rearrange(self.points_kf2, "h w c -> (h w) c").to(self.device)
        self.use_noprompt = True
        self.kf1_colors = rearrange(kf1_image, "b c h w -> (w h b) c")
        self.kf1_image = kf1_image
        self.kf2_image = kf2_image

        self.kf1_depth = kf1_depth
        self.kf2_depth = kf2_depth
        self.kf1_camera = kf1_camera
        self.kf2_camera = kf2_camera
        self.kf2_mask = kf2_mask

        self.point_depth = rearrange(kf1_depth, "b c h w -> (w h b) c")
        kf1_camera = convert_pytorch3d_kornia(kf1_camera, self.config["init_focal_length"])
        self.points_3d = kf1_camera.unproject(self.points, self.point_depth)
        self.points_3d[..., :2] = - self.points_3d[..., :2]

        self.reinit()

    @torch.no_grad()
    def reinit(self):
        self.images = [self.kf2_image]
        self.inpaint_input_image = [self.inpaint_input_image[-1]]
        self.depths = [self.kf2_depth]
        self.masks = [self.masks[-1]]
        self.post_masks = [self.post_masks[-1]]
        self.post_mask_tmp = None
        self.rendered_images = [self.kf2_image]
        self.rendered_depths = [self.kf2_depth]

        self.current_camera = copy.deepcopy(self.kf2_camera)
        if self.config["motion"] == "rotations":
            self.current_camera.no_rotations_count = 0
            self.current_camera.rotations_count = 1
            self.current_camera.rotating_right = -self.current_camera.rotating_right
            self.current_camera.move_dir = torch.tensor([[0.0, 0.0, self.config['forward_speed_multiplier']]], device=self.device)
        else:
            raise NotImplementedError
        
    @torch.no_grad()
    def upsample_kf2(self):
        kf2_size = 512 * self.kf2_upsample_coef
        kf2_focal = self.config["init_focal_length"] * self.kf2_upsample_coef
        kf2_camera_upsample = convert_pytorch3d_kornia(self.kf2_camera, kf2_focal, size=kf2_size)
        kf2_depth_upsample = F.interpolate(self.kf2_depth, size=(kf2_size, kf2_size), mode="nearest")
        kf2_mask_upsample = F.interpolate(self.kf2_mask, size=(kf2_size, kf2_size), mode="nearest")
        kf2_pil_upsample = ToPILImage()(self.kf2_image[0]).resize((kf2_size, kf2_size), resample=Image.LANCZOS)
        kf2_image_upsample = ToTensor()(kf2_pil_upsample).unsqueeze(0).to(self.config['device'])
        return kf2_camera_upsample, kf2_depth_upsample, kf2_mask_upsample, kf2_image_upsample
    
    @torch.no_grad()
    def render_kf1(self, epoch):
        if self.config["motion"] == "rotations":
            camera = self.get_next_camera_rotation()
        elif self.config["motion"] == "predefined":
            camera = self.predefined_cameras[epoch]
        else:
            raise NotImplementedError

        point_depth_aug = torch.cat([self.point_depth, self.additional_points_3d[..., -1:]], dim=0)

        depth_normalizer = self.background_hard_depth
        min_ratio = self.config['point_size_min_ratio']
        radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth_aug.permute([1, 0]) / depth_normalizer))
        radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier'])
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius = radius,
            points_per_pixel = 8,
        )
        renderer = PointsRenderer(
                    rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
                    compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                )
        points_3d_aug = torch.cat([self.points_3d, self.additional_points_3d], dim=0)
        colors_aug = torch.cat([self.kf1_colors, self.additional_colors], dim=0)
        point_cloud = Pointclouds(points=[points_3d_aug], features=[colors_aug])
        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)

        rendered_image = rearrange(images, "b h w c -> b c h w")
        inpaint_mask = bg_mask.float()[:, None, ...]
        rendered_depth = rearrange(zbuf[..., 0:1], "b h w c -> b c h w")
        rendered_depth[rendered_depth < 0] = 0

        self.rendered_images.append(rendered_image)
        self.rendered_depths.append(rendered_depth)
        self.current_camera = copy.deepcopy(camera)
        self.cameras.append(self.current_camera)

        if self.config["inpainting_resolution"] > 512:
            padded_inpainting_mask = self.border_mask.clone()
            padded_inpainting_mask[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ] = inpaint_mask
            padded_image = self.border_image.clone()
            padded_image[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ] = rendered_image
        else:
            padded_inpainting_mask = inpaint_mask
            padded_image = rendered_image

        return {
            "rendered_image": padded_image,
            "rendered_depth": rendered_depth,
            "inpaint_mask": padded_inpainting_mask,
        }

    @torch.no_grad()
    def visibility_check(self):
        radius = self.config['point_size']
        K = 32
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius = radius,
            points_per_pixel = K,
        )
        renderer = PointsRenderer(
                    rasterizer=PointsRasterizer(cameras=self.kf1_camera, raster_settings=raster_settings),
                    compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                )
        points_3d = self.points_3d
        n_kf1_points = points_3d.shape[0]
        colors = self.kf1_colors
        point_cloud = Pointclouds(points=[points_3d], features=[colors])
        images = renderer(point_cloud)

        re_rendered = rearrange(images, "b h w c -> b c h w")

        points_3d_aug = torch.cat([self.points_3d, self.additional_points_3d], dim=0)
        colors_aug = torch.cat([self.kf1_colors, self.additional_colors], dim=0)
        point_cloud_aug = Pointclouds(points=[points_3d_aug], features=[colors_aug])
        images_aug, fragment_idx = renderer(point_cloud_aug, return_fragment_idx=True)  # fragment_idx: [B, H, W, K]
        re_rendered_aug = rearrange(images_aug, "b h w c -> b c h w")

        difference_image = torch.abs(re_rendered - re_rendered_aug).sum(dim=1)  # [B, H, W]
        inconsistent_px = difference_image > 0
        inconsistent_px_point_idx = fragment_idx[inconsistent_px]  # [N, K]
        inconsistent_px_point_from_kf1 = (inconsistent_px_point_idx < n_kf1_points) & (inconsistent_px_point_idx >= 0)  # [N, K], only one True in each

        def find_nearer_points(x):
            x_int = x.int()
            true_indices = torch.argmax(x_int, dim=1)
            indices = torch.arange(x.shape[1]).unsqueeze(0).expand_as(x).to(x.device)
            y_vectorized = indices < true_indices.unsqueeze(1)
            return y_vectorized
        
        inconsistent_px_point_from_addi = find_nearer_points(inconsistent_px_point_from_kf1)  # [N, K]
        inconsistent_px_point_from_addi_ = inconsistent_px_point_idx[inconsistent_px_point_from_addi]
        inconsistent_px_point_from_addi_ = inconsistent_px_point_from_addi_.unique()  # [T]
        inconsistent_addi_point_idx = inconsistent_px_point_from_addi_ - n_kf1_points  # [T]

        return inconsistent_addi_point_idx

    @torch.no_grad()
    def update_additional_point_cloud(self, rendered_depth, image, valid_mask=None, camera=None, points_2d=None, append_depth=False):
        inpaint_mask = rendered_depth == 0
        rendered_depth_filled = rendered_depth.clone()
        inpaint_mask_onthefly = inpaint_mask.clone()

        def nearest_neighbor_inpainting(inpaint_mask, rendered_depth, window_size=20):
            invalid_coords = torch.nonzero(inpaint_mask.squeeze(), as_tuple=False)
            valid_coords = torch.nonzero(~inpaint_mask.squeeze(), as_tuple=False)
            rendered_depth_copy = rendered_depth.clone()
            hw = window_size // 2
            for idx in range(invalid_coords.size(0)):
                x, y = invalid_coords[idx, 0], invalid_coords[idx, 1]
                x_start, x_end = max(0, x - hw), min(rendered_depth.size(2), x + hw + 1)
                y_start, y_end = max(0, y - hw), min(rendered_depth.size(3), y + hw + 1)
                local_valid_coords = valid_coords[(valid_coords[:, 0] >= x_start) & (valid_coords[:, 0] < x_end) & 
                                                (valid_coords[:, 1] >= y_start) & (valid_coords[:, 1] < y_end)]
                if local_valid_coords.size(0) > 0:
                    dists = torch.cdist(invalid_coords[idx, :].unsqueeze(0).float(), local_valid_coords.float())
                    min_idx = torch.argmin(dists)
                    rendered_depth_copy[0, 0, x, y] = rendered_depth[0, 0, local_valid_coords[min_idx, 0], local_valid_coords[min_idx, 1]]

            return rendered_depth_copy

        while inpaint_mask_onthefly.sum() > 0:  # iteratively inpaint depth until all depth holes are filled
            rendered_depth_filled = nearest_neighbor_inpainting(inpaint_mask_onthefly, rendered_depth_filled, window_size=50)
            inpaint_mask_onthefly = rendered_depth_filled == 0

        current_camera = convert_pytorch3d_kornia(self.current_camera, self.config["init_focal_length"]) if camera is None else camera
        points_2d = self.points if points_2d is None else points_2d
        points_3d = current_camera.unproject(points_2d, rearrange(rendered_depth_filled, "b c h w -> (w h b) c"))
        points_3d[..., :2] = - points_3d[..., :2]
        inpaint_mask = rearrange(inpaint_mask, "b c h w -> (w h b) c")
        colors = rearrange(image, "b c h w -> (w h b) c")
        if valid_mask is None:
            extract_mask = inpaint_mask[:, 0].bool()
        else:
            extract_mask = rearrange(valid_mask, "b c h w -> (w h b) c")[:, 0].bool()
        additional_points_3d = points_3d[extract_mask]

        additional_colors = colors[extract_mask]
        backward_points = (- additional_points_3d[..., 2]) > current_camera.tz
        additional_points_3d = additional_points_3d[~backward_points]
        additional_colors = additional_colors[~backward_points]

        self.additional_points_3d = torch.cat([self.additional_points_3d, additional_points_3d], dim=0)
        self.additional_colors = torch.cat([self.additional_colors, additional_colors], dim=0)

        if append_depth:
            self.depths.append(rendered_depth_filled.cpu())


    @torch.no_grad()
    def update_additional_point_depth(self, inconsistent_point_index, depth, mask):
        h, w = depth.shape[2:]
        depth = rearrange(depth.clone(), "b c h w -> (w h b) c")
        extract_mask = rearrange(mask, "b c h w -> (w h b) c")[:, 0].bool()
        depth_extracted = depth[extract_mask]
        if inconsistent_point_index.shape[0] > 0:
            assert depth_extracted.shape[0] >= inconsistent_point_index.max() + 1
        depth_extracted[inconsistent_point_index] = self.config['sky_hard_depth'] * 2
        depth[extract_mask] = depth_extracted
        depth = rearrange(depth, "(w h b) c -> b c h w", w=w, h=h)
        return depth

    @torch.no_grad()
    def reset_additional_point_cloud(self):
        self.additional_colors = torch.tensor([]).cuda()
        self.additional_points_3d = torch.tensor([]).cuda()

def convert_pytorch3d_kornia(camera, focal_length, size=512):
    R = torch.clone(camera.R)
    T = torch.clone(camera.T)
    T[0, 0] = -T[0, 0]
    extrinsics = torch.eye(4, device=R.device).unsqueeze(0)
    extrinsics[:, :3, :3] = R
    extrinsics[:, :3, 3] = T
    h = torch.tensor([size], device="cuda")
    w = torch.tensor([size], device="cuda")
    K = torch.eye(4)[None].to("cuda")
    K[0, 0, 2] = size // 2
    K[0, 1, 2] = size // 2
    K[0, 0, 0] = focal_length
    K[0, 1, 1] = focal_length
    return PinholeCamera(K, extrinsics, h, w)
# ----------------- Models from models.py end -----------------

class VisualRepresentation(object):
    def __init__(self, oneformer_model, oneformer_processor, depth_model):
        """
        初始化representation模型
        """
        self.oneformer_model = oneformer_model
        self.oneformer_processor = oneformer_processor
        self.depth_model = depth_model

    @classmethod
    def from_pretrained(cls, oneformer_path, depth_model_path="dpt_beit_large_512.pt", device=None, **kwargs):
        """
        input: 用户输入的huggingface权重链接
        output: 对应的Representation类
        """
        # Load OneFormer
        if os.path.isdir(oneformer_path):
            model_root = oneformer_path
        else:
            print(f"Downloading OneFormer weights from HuggingFace repo: {oneformer_path}")
            try:
                model_root = snapshot_download(oneformer_path)
            except:
                # Fallback or strict requirement
                model_root = oneformer_path
            print(f"Model downloaded to: {model_root}")
            
        segment_processor = OneFormerProcessor.from_pretrained(model_root)
        segment_model = OneFormerForUniversalSegmentation.from_pretrained(model_root)
        
        # Load Depth Model
        # Using the logic from run.py: depth_model, _, _, _ = load_model(...)
        # We assume load_model is available
        depth_model, _, _, _ = load_model(torch.device("cuda" if device is None else device), depth_model_path, 'dpt_beit_large_512', optimize=False)

        return cls(segment_model, segment_processor, depth_model)

    def api_init(self, api_key, endpoint):
        pass

    def get_representation(self, data):
        """
        Placeholder for general representation extraction if needed
        """
        pass
    
    # ======== 以下为点云save内容，后续可以放到memory中 ========

def save_point_cloud_as_ply(points, filename="output.ply", colors=None):
    """
    Save a PyTorch tensor of shape [N, 3] as a PLY file. Optionally with colors.
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