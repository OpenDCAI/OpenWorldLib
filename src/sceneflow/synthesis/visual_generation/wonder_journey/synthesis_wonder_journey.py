import torch
import torch.nn.functional as F
import numpy as np
import cv2
import copy
import skimage.measure
from PIL import Image
from einops import rearrange
from typing import List, Optional, Tuple, Union

from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
)
from pytorch3d.renderer.points.compositor import _add_background_color_to_images
from pytorch3d.structures import Pointclouds

from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

BG_COLOR = (1, 0, 0) 

def inpaint_cv2(rendered_image, mask_diff):
    """
    util/utils.py，用于预处理填补微小空洞
    """
    image_cv2 = rendered_image[0].permute(1, 2, 0).cpu().numpy()
    image_cv2 = (image_cv2 * 255).astype(np.uint8)
    mask_cv2 = mask_diff[0, 0].cpu().numpy()
    mask_cv2 = (mask_cv2 * 255).astype(np.uint8)
    inpainting = cv2.inpaint(image_cv2, mask_cv2, 3, cv2.INPAINT_TELEA)
    inpainting = torch.from_numpy(inpainting).permute(2, 0, 1).float() / 255
    return inpainting.unsqueeze(0)

# 渲染组件：从 models.py 复制
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

# Synthesis 主类
class WonderJourneySynthesis:
    def __init__(self, inpainting_pipeline, vae, device='cuda', **kwargs):
        """
        inpainting_pipeline: Stable Diffusion Inpaint Pipeline
        vae: AutoencoderKL
        kwargs: 对应原本的 config 字典
        """
        self.device = device
        self.inpainting_pipeline = inpainting_pipeline
        self.vae = vae
        
        self.config = {
            'inpainting_resolution': 512,
            'point_size': 0.01,
            'point_size_min_ratio': 0.5,
            'sky_point_size_multiplier': 5.0,
            'depth_shift': 0.0,
            'fg_depth_range': 0.1,
            'use_postmask': True,
            'negative_inpainting_prompt': ", blur, low quality",
            'preservation_weight': 0.5, 
        }
        self.config.update(kwargs)

        #  FrameSyn.__init__
        self.background_hard_depth = self.config['depth_shift'] + self.config['fg_depth_range']
        self.is_upper_mask_aggressive = True # KeyframeGen 默认为 True
        self.use_noprompt = False
        self.border_mask = torch.ones(
            (1, 1, self.config["inpainting_resolution"], self.config["inpainting_resolution"])
        ).to(self.device)
        self.border_size = (self.config["inpainting_resolution"] - 512) // 2
        
        if self.border_size > 0:
            self.border_mask[:, :, self.border_size : -self.border_size, self.border_size : -self.border_size] = 0
        
        self.border_image = torch.zeros(
            1, 3, self.config["inpainting_resolution"], self.config["inpainting_resolution"]
        ).to(self.device)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, device='cuda', **kwargs):
        print(f"Loading Synthesis Models from {pretrained_model_path}...")
        
        inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_path,
            safety_checker=None,
            torch_dtype=torch.float16,
        ).to(device)
        inpainter_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(inpainter_pipeline.scheduler.config)
        
        vae = inpainter_pipeline.vae
        
        return cls(inpainter_pipeline, vae, device=device, **kwargs)

    def predict(self, data):
        """这里还要做调整"""
        pass

    def render_scene(self, camera, points_3d, colors):
        """
        models.py KeyframeGen.render
        Input: 
            camera: Pytorch3D camera 对象
            points_3d, colors: 点云数据
        Output:
            rendered_image, rendered_depth, inpaint_mask
        """
        # 注意：PyTorch3D 与 Kornia 坐标系差异，必须保留这个负号
        points_3d_render = points_3d.clone()
        points_3d_render[..., :2] = - points_3d_render[..., :2]

        point_depth = points_3d_render[..., 2:3]

        depth_normalizer = self.background_hard_depth
        min_ratio = self.config['point_size_min_ratio']
        
        # 计算点大小
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
        
        point_cloud = Pointclouds(points=[points_3d_render], features=[colors])
        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)

        rendered_image = rearrange(images, "b h w c -> b c h w")
        inpaint_mask = bg_mask.float()[:, None, ...]
        rendered_depth = rearrange(zbuf[..., 0:1], "b h w c -> b c h w")
        rendered_depth[rendered_depth < 0] = 0

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

        return padded_image, rendered_depth, padded_inpainting_mask

    @torch.no_grad()
    def inpaint(self, rendered_image, inpaint_mask, prompt, negative_prompt=""):
        """
        [保留原代码] 搬运自 models.py FrameSyn.inpaint
        """
        process_width, process_height = self.config["inpainting_resolution"], self.config["inpainting_resolution"]

        # cv2.inpaint
        img_filled = inpaint_cv2(rendered_image, inpaint_mask)
        img = (img_filled[0].permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8)
        
        mask = (inpaint_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)

        # block_reduce
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

        # [IO] self.inpaint_input_image.append(init_image)

        inpainted_image_latents = self.inpainting_pipeline(
            prompt='' if self.use_noprompt else prompt,
            negative_prompt=negative_prompt + self.config["negative_inpainting_prompt"],
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=25,
            guidance_scale=0 if self.use_noprompt else 7.5,
            height=process_height,
            width=process_width,
            output_type='latent',
        ).images

        # 3. Decode Latents
        inpainted_image = self.decode_latents(inpainted_image_latents)
        
        # 4. Crop back to 512
        if self.config["inpainting_resolution"] > 512:
            inpainted_image = inpainted_image[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ]

        # 返回生成的图片 (Tensor [1, 3, H, W]) 以及 Latent (供微调使用)
        return inpainted_image, inpainted_image_latents

    # FrameSyn.finetune_decoder_step
    def finetune_decoder(self, inpainted_image, inpainted_image_latent, rendered_image, inpaint_mask, steps=10):
        """
        微调 VAE Decoder，使其在保留区域与原渲染图一致，在补全区域与生成图一致。
        """
        self.vae.decoder.train()
        
        for param in self.vae.encoder.parameters():
            param.requires_grad = False
            
        optimizer = torch.optim.Adam(self.vae.decoder.parameters(), lr=1e-4) # 原项目未指定LR，假定值
        
        inpaint_mask_dilated = inpaint_mask
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            # Reconstruction
            reconstruction = self.decode_latents(inpainted_image_latent)
            
            # Loss Calculation 
            new_content_loss = F.mse_loss(inpainted_image * inpaint_mask, reconstruction * inpaint_mask)
            preservation_loss = F.mse_loss(rendered_image * (1 - inpaint_mask_dilated), reconstruction * (1 - inpaint_mask_dilated)) * self.config["preservation_weight"]
            
            loss = new_content_loss + preservation_loss
            
            loss.backward()
            optimizer.step()
            
        self.vae.decoder.eval()

    # KeyframeInterp.visibility_check 的渲染部分
    def get_fragment_indices(self, camera, points_3d, colors):
        """
        渲染场景并返回 fragment_idx，供 Representation 计算遮挡关系。
        完全还原原项目逻辑：使用固定半径，K=32。
        """
        # 1. 坐标系转换 
        points_3d_render = points_3d.clone()
        points_3d_render[..., :2] = - points_3d_render[..., :2]
        
        # 2. 设置光栅化参数
        # 原项目逻辑：在 visibility_check 中，作者使用的是【固定半径】，而不是 render 函数中的动态半径。这是为了获得准确的几何遮挡关系。
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius = self.config['point_size'], # 固定半径
            points_per_pixel = 32, # K=32 (原项目写死的值)
        )
        
        # 3. 初始化渲染器
        # BG_COLOR 需要在文件头部定义 (通常是 (1, 0, 0))
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
        )
        
        point_cloud = Pointclouds(points=[points_3d_render], features=[colors])
        
        # 4. 执行渲染
        # 原项目逻辑只设置了 return_fragment_idx=True，其他默认为 False。
        # 因此 renderer 只返回 [images, fragment_idx] 两个值。
        # 如果用 _, _, _, fragment_idx 会报错 "not enough values to unpack"。
        _, fragment_idx = renderer(point_cloud, return_fragment_idx=True)
        
        # fragment_idx: [B, H, W, K]
        return fragment_idx

    def decode_latents(self, latents):
        """
        FrameSyn.decode_latents
        """
        images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1)
        return images