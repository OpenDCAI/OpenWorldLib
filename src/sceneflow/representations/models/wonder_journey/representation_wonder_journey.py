import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from einops import rearrange
from torchvision.transforms import ToPILImage, ToTensor
from kornia.morphology import erosion
from util.midas_utils import dpt_transform, dpt_512_transform
from util.segment_utils import refine_disp_with_segments as refine_disp_algo 
from util.segment_utils import save_sam_anns
from util.segment_utils import create_mask_generator
from util.utils import save_depth_map, save_point_cloud_as_ply
from midas.model_loader import load_model 
from midas.transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

class WonderJourneyRepresentation:
    def __init__(self, depth_model, segment_model, segment_processor, mask_generator, device='cuda', **kwargs):
        self.device = device
        self.depth_model = depth_model
        self.segment_model = segment_model
        self.segment_processor = segment_processor
        self.mask_generator = mask_generator
        
        self.config = {
            'depth_model': 'midas_v3.1',
            'depth_shift': 0.0,
            'sky_erode_kernel_size': 5,
            'sky_hard_depth': 0.025, # 1/40
            'init_focal_length': 512,
            'finetune_depth_model': False,
            'finetune_lr': 0.00001,
        }
        self.config.update(kwargs)
        # 预留IO 路径
        self.run_dir = Path(kwargs.get('run_dir', './output_debug'))

        # 为了memory，删除以下两行
        # self.points_3d = torch.tensor([], device=device)
        # self.colors = torch.tensor([], device=device)
        
        x = torch.arange(512).float() + 0.5
        y = torch.arange(512).float() + 0.5
        self.points_grid = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        self.points_grid = rearrange(self.points_grid, "h w c -> (h w) c").to(device)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, device='cuda', **kwargs):
        """
        加载 MiDaS, OneFormer, SAM。
        pretrained_model_path: 本地文件夹，包含 dpt_beit_large_512.pt
        """
        
        print("Loading OneFormer...")
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
        segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(device)

        print("Loading MiDaS...")
        midas_path = os.path.join(pretrained_model_path, "dpt_beit_large_512.pt") if os.path.isdir(pretrained_model_path) else "dpt_beit_large_512.pt"
        if not os.path.exists(midas_path):
            print(f"Warning: MiDaS weights not found at {midas_path}. Please download.")
        
        depth_model, _, _, _ = load_model(device, midas_path, 'dpt_beit_large_512', optimize=False)

        print("Loading SAM...")
        mask_generator = create_mask_generator() # 内部会加载 sam_vit_h_4b8939.pth

        return cls(depth_model, segment_model, processor, mask_generator, device=device, **kwargs)

    def get_representation(self, data):
        return self.get_depth(data)

    def get_depth(self, image):
        """
        models.py FrameSyn.get_depth
        Input: image tensor [1, 3, H, W]
        """
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
        elif self.config['depth_model'].lower() == "midas_v3.1":
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

    def finetune_depth_model(self, target_depth, inpainted_image, mask_align=None, mask_cutoff=None, cutoff_depth=None, steps=10):
        """
        在推理阶段微调 MiDaS，以强制几何一致性。
        """
        if self.depth_model is None: return
        
        print("Finetuning depth model...")
        self.depth_model.train()
        optimizer = torch.optim.Adam(self.depth_model.parameters(), lr=self.config['finetune_lr'])
        
        for i in range(steps):
            optimizer.zero_grad()
            
            if self.config['depth_model'].lower() == "midas_v3.1":
                img_transformed = dpt_512_transform(inpainted_image)
                disparity = self.depth_model(img_transformed)
                # 插值需要保留梯度
                disparity = torch.nn.functional.interpolate(
                    disparity.unsqueeze(1),
                    size=inpainted_image.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                disparity = disparity.clip(1e-6, max=None)
                next_depth = 1 / disparity + self.config['depth_shift']
            else:
                continue

            # 计算 Loss 
            # L1 loss for the mask_align region
            loss_align = F.l1_loss(target_depth.detach(), next_depth, reduction="none")
            if mask_align is not None and torch.any(mask_align):
                mask_align = mask_align.detach()
                loss_align = (loss_align * mask_align)[mask_align > 0].mean()
            else:
                loss_align = torch.zeros(1).to(self.device)

            # Hinge loss for the mask_cutoff region
            if mask_cutoff is not None and cutoff_depth is not None and torch.any(mask_cutoff):
                hinge_loss = (cutoff_depth - next_depth).clamp(min=0)
                hinge_loss = F.l1_loss(hinge_loss, torch.zeros_like(hinge_loss), reduction="none")
                mask_cutoff = mask_cutoff.detach()
                hinge_loss = (hinge_loss * mask_cutoff)[mask_cutoff > 0].mean()
            else:
                hinge_loss = torch.zeros(1).to(self.device)

            total_loss = loss_align + hinge_loss
            
            if torch.isnan(total_loss):
                print("Depth FT loss is NaN, skipping.")
                break
                
            total_loss.backward()
            optimizer.step()
            
        self.depth_model.eval()
    def upsample_data(self, image_tensor, depth_tensor, mask_tensor, coef=2):
        """
        在反投影前放大图像和深度图，增加点云密度。
        """
        if coef == 1:
            return image_tensor, depth_tensor, mask_tensor
            
        target_size = (512 * coef, 512 * coef)
        
        depth_up = F.interpolate(depth_tensor, size=target_size, mode="nearest")
        mask_up = F.interpolate(mask_tensor.float(), size=target_size, mode="nearest")
        image_up = F.interpolate(image_tensor, size=target_size, mode="bilinear", align_corners=False)
        
        # 更新 self.points_grid 以匹配新的分辨率 
        x = torch.arange(target_size[0]).float() + 0.5
        y = torch.arange(target_size[1]).float() + 0.5
        grid_up = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        grid_up = rearrange(grid_up, "h w c -> (h w) c").to(self.device)
        
        return image_up, depth_up, mask_up, grid_up

    @torch.no_grad()
    def refine_depth_logic(self, image_tensor, disparity_tensor, kf_idx=0, background_depth_cutoff=1./7.):
        """
        models.py KeyframeGen.refine_disp_with_segments
        """
        image_pil = ToPILImage()(image_tensor.squeeze())
        
        # OneFormer 语义分割
        segmenter_input = self.segment_processor(image_pil, ["semantic"], return_tensors="pt")
        segmenter_input = {name: tensor.to(self.device) for name, tensor in segmenter_input.items()}
        segment_output = self.segment_model(**segmenter_input)
        pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
                                segment_output, target_sizes=[image_pil.size[::-1]])[0]
        
        sky_mask = pred_semantic_map.cpu() == 119
        sky_mask = erosion(sky_mask.float()[None, None], 
                           kernel=torch.ones(self.config['sky_erode_kernel_size'], self.config['sky_erode_kernel_size'])
                           ).squeeze() > 0.5
        sky_mask = sky_mask.cpu()
        
        # [IO Preserved]
        # ToPILImage()(sky_mask.float()).save(self.run_dir / 'images' / f"kf{kf_idx+1}_sky_mask.png")

        # SAM 分割
        image_np = np.array(image_pil)
        masks = self.mask_generator.generate(image_np)
        sorted_mask = sorted(masks, key=(lambda x: x['area']), reverse=False)
        min_mask_area = 30
        sorted_mask = [m for m in sorted_mask if m['area'] > min_mask_area]

        # [IO Preserved]
        # save_sam_anns(masks, self.run_dir / 'images' / f"SAM_kf{kf_idx+1}.png")

        # 深度优化
        disparity_np = disparity_tensor.squeeze().cpu().numpy()
        keep_threshold_ratio = 0.3
        refined_disparity = refine_disp_algo(disparity_np, sorted_mask, keep_threshold=1 / background_depth_cutoff * keep_threshold_ratio)

        # [IO Preserved]
        # save_depth_map(1/refined_disparity, self.run_dir / 'images' / f"kf{kf_idx+1}_p1_SAM", vmax=self.config['sky_hard_depth'])

        sky_hard_disp = 1. / self.config['sky_hard_depth']
        bg_hard_disp = 1. / (background_depth_cutoff)
        refined_disparity[sky_mask] = sky_hard_disp

        # [IO Preserved]
        # save_depth_map(1/refined_disparity, self.run_dir / 'images' / f"kf{kf_idx+1}_p2_sky", vmax=self.config['sky_hard_depth'])

        background_cutoff = 1./background_depth_cutoff
        background_mask = refined_disparity < background_cutoff
        background_but_not_sky_mask = np.logical_and(background_mask, np.logical_not(sky_mask.numpy()))
        refined_disparity[background_but_not_sky_mask] = bg_hard_disp

        # [IO Preserved]
        # save_depth_map(1/refined_disparity, self.run_dir / 'images' / f"kf{kf_idx+1}_p3_cutoff", vmax=self.config['sky_hard_depth'])

        refined_disparity = refine_disp_algo(refined_disparity, sorted_mask, keep_threshold=1 / background_depth_cutoff * keep_threshold_ratio)
        
        # [IO Preserved]
        # save_depth_map(1/refined_disparity, self.run_dir / 'images' / f"kf{kf_idx+1}_p4_SAM", vmax=self.config['sky_hard_depth'])

        refined_depth = 1 / refined_disparity
        refined_depth = torch.from_numpy(refined_depth).to(self.device).unsqueeze(0).unsqueeze(0)
        refined_disparity = torch.from_numpy(refined_disparity).to(self.device).unsqueeze(0).unsqueeze(0)

        return refined_depth, refined_disparity

    @torch.no_grad()
    def compute_new_points(self, rendered_depth, image, valid_mask=None, camera=None, points_2d=None):
        """
        计算需要新增的点云数据 (不再直接修改 self.points_3d)
        Returns:
            new_points (Tensor), new_colors (Tensor)
        """
        """
         models.py KeyframeInterp.update_additional_point_cloud+convert_pytorch3d_kornia
        """
        inpaint_mask = rendered_depth == 0
        rendered_depth_filled = rendered_depth.clone()
        inpaint_mask_onthefly = inpaint_mask.clone()
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

        while inpaint_mask_onthefly.sum() > 0:
            rendered_depth_filled = nearest_neighbor_inpainting(inpaint_mask_onthefly, rendered_depth_filled, window_size=50)
            inpaint_mask_onthefly = rendered_depth_filled == 0

        current_camera_kornia = convert_pytorch3d_kornia(camera, self.config["init_focal_length"])

        grid = points_2d if points_2d is not None else self.points_grid
        
        points_3d = current_camera_kornia.unproject(grid, rearrange(rendered_depth_filled, "b c h w -> (w h b) c"))
        points_3d[..., :2] = - points_3d[..., :2]
        
        inpaint_mask = rearrange(inpaint_mask, "b c h w -> (w h b) c")
        colors = rearrange(image, "b c h w -> (w h b) c")
        
        if valid_mask is None:
            extract_mask = inpaint_mask[:, 0].bool()
        else:
            extract_mask = rearrange(valid_mask, "b c h w -> (w h b) c")[:, 0].bool()
        
        additional_points_3d = points_3d[extract_mask]

        # IO Preserved
        # original_points_3d = points_3d[~inpaint_mask[:, 0]]
        # save_point_cloud_as_ply(original_points_3d, "tmp/original_points_3d.ply", colors[~inpaint_mask[:, 0]])
        # save_point_cloud_as_ply(additional_points_3d, "tmp/additional_points_3d.ply", colors[inpaint_mask[:, 0]])

        additional_colors = colors[extract_mask]

        backward_points = (- additional_points_3d[..., 2]) > current_camera_kornia.tz
        additional_points_3d = additional_points_3d[~backward_points]
        additional_colors = additional_colors[~backward_points]

        # self.points_3d = torch.cat([self.points_3d, additional_points_3d], dim=0)
        # self.colors = torch.cat([self.colors, additional_colors], dim=0)
        return additional_points_3d, additional_colors

    def remove_occluded_points(self, inconsistent_indices):
        """
        根据 Synthesis 提供的索引，从点云中删除被遮挡的点。
        """
        if inconsistent_indices is None or len(inconsistent_indices) == 0:
            return

        total_points = self.points_3d.shape[0]

        keep_mask = torch.ones(total_points, dtype=torch.bool, device=self.device)
        valid_indices = inconsistent_indices[inconsistent_indices < total_points]
        keep_mask[valid_indices] = False
        
        self.points_3d = self.points_3d[keep_mask]
        self.colors = self.colors[keep_mask]

    # 为了memory删除
    # def reset_cloud(self):
    #     self.points_3d = torch.tensor([], device=self.device)
    #     self.colors = torch.tensor([], device=self.device)
