"""
input image and output 3D reconstruction (depth, normal, point cloud, gaussians, colmap)
load operators and WorldMirror representation model
"""
import torch
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image
from typing import Optional, Any, Dict, List
import cv2
import onnxruntime

from ...representations.point_clouds_generation.hunyuan_world.hunyuan_world_mirror_representation import HunyuanWorldMirrorRepresentation
from ...representations.point_clouds_generation.hunyuan_world.hunyuan_mirror.mirror_src.utils.inference_utils import prepare_images_to_tensor
from ...representations.point_clouds_generation.hunyuan_world.hunyuan_mirror.mirror_src.utils.save_utils import save_depth_png, save_depth_npy, save_normal_png
from ...representations.point_clouds_generation.hunyuan_world.hunyuan_mirror.mirror_src.utils.save_utils import save_scene_ply, save_gs_ply
from ...representations.point_clouds_generation.hunyuan_world.hunyuan_mirror.mirror_src.utils.render_utils import render_interpolated_video
from ...representations.point_clouds_generation.hunyuan_world.hunyuan_mirror.mirror_src.utils.geometry import depth_edge, normals_edge
from ...representations.point_clouds_generation.hunyuan_world.hunyuan_mirror.mirror_src.utils.visual_util import segment_sky, download_file_from_url


class HunyuanMirrorPipeline:
    def __init__(self,
                 operators: Optional[Any] = None,
                 represent_model: Optional[HunyuanWorldMirrorRepresentation] = None,
                 output_path: str = "./output/hunyuan_mirror",
                 device: str = 'cuda'):
        self.operators = operators
        self.represent_model = represent_model
        self.output_path = Path(output_path)
        self.device = device
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_pretrained(cls,
                        model_path: str = "tencent/HunyuanWorld-Mirror",
                        local_model_path: Optional[str] = None,
                        output_path: str = "./output/hunyuan_mirror",
                        device: str = "cuda",
                        **kwargs) -> 'HunyuanMirrorPipeline':
        """
        Load pipeline from pretrained model
        
        Args:
            model_path: HuggingFace model path
            local_model_path: Local model path (priority if provided)
            output_path: Output directory path
            device: Device
            **kwargs: Additional arguments for model
            
        Returns:
            HunyuanMirrorPipeline: Initialized pipeline instance
        """
        # Set model path
        if local_model_path:
            actual_model_path = local_model_path
        else:
            actual_model_path = model_path
        
        # Load representation model
        print(f"Loading HunyuanWorld-Mirror model from {actual_model_path}")
        device_torch = torch.device(device if torch.cuda.is_available() else "cpu")
        
        represent_model = HunyuanWorldMirrorRepresentation.from_pretrained(actual_model_path, local_files_only=True).to(device_torch)
        represent_model.eval()
        
        # Create pipeline instance
        pipeline = cls(
            operators=None,  # HunyuanMirror doesn't need specific operators
            represent_model=represent_model,
            output_path=output_path,
            device=device
        )
        
        return pipeline
    
    def create_filter_mask(self,
                          pts3d_conf: np.ndarray,
                          depth_preds: np.ndarray, 
                          normal_preds: np.ndarray,
                          sky_mask: np.ndarray,
                          confidence_percentile: float = 10.0,
                          edge_normal_threshold: float = 5.0,
                          edge_depth_threshold: float = 0.03,
                          apply_confidence_mask: bool = True,
                          apply_edge_mask: bool = True,
                          apply_sky_mask: bool = False) -> np.ndarray:
        """
        Create comprehensive filter mask based on confidence, edges, and sky segmentation
        
        Args:
            pts3d_conf: Point confidence scores [S, H, W]
            depth_preds: Depth predictions [S, H, W, 1]
            normal_preds: Normal predictions [S, H, W, 3]
            sky_mask: Sky segmentation mask [S, H, W]
            confidence_percentile: Percentile threshold for confidence filtering (0-100)
            edge_normal_threshold: Normal angle threshold in degrees for edge detection
            edge_depth_threshold: Relative depth threshold for edge detection
            apply_confidence_mask: Whether to apply confidence-based filtering
            apply_edge_mask: Whether to apply edge-based filtering
            apply_sky_mask: Whether to apply sky mask filtering
        
        Returns:
            final_mask: Boolean mask array [S, H, W] for filtering points
        """
        S, H, W = pts3d_conf.shape[:3]
        final_mask_list = []
        
        for i in range(S):
            final_mask = None
            
            if apply_confidence_mask:
                # Compute confidence mask based on the pointmap confidence
                confidences = pts3d_conf[i, :, :]  # [H, W]
                percentile_threshold = np.quantile(confidences, confidence_percentile / 100.0)
                conf_mask = confidences >= percentile_threshold
                if final_mask is None:
                    final_mask = conf_mask
                else:
                    final_mask = final_mask & conf_mask
            
            if apply_edge_mask:
                # Compute edge mask based on the normalmap
                normal_pred = normal_preds[i]  # [H, W, 3]
                normal_edges = normals_edge(
                    normal_pred, tol=edge_normal_threshold, mask=final_mask
                )
                # Compute depth mask based on the depthmap
                depth_pred = depth_preds[i, :, :, 0]  # [H, W]
                depth_edges = depth_edge(
                    depth_pred, rtol=edge_depth_threshold, mask=final_mask
                )
                edge_mask = ~(depth_edges & normal_edges)
                if final_mask is None:
                    final_mask = edge_mask
                else:
                    final_mask = final_mask & edge_mask
            
            if apply_sky_mask:
                # Apply sky mask filtering (sky_mask is already inverted: True = non-sky)
                sky_mask_frame = sky_mask[i]  # [H, W]
                if final_mask is None:
                    final_mask = sky_mask_frame
                else:
                    final_mask = final_mask & sky_mask_frame
            
            final_mask_list.append(final_mask)
        
        # Stack all frame masks
        if final_mask_list[0] is not None:
            final_mask = np.stack(final_mask_list, axis=0)  # [S, H, W]
        else:
            final_mask = np.ones(pts3d_conf.shape[:3], dtype=bool)  # [S, H, W]
        
        return final_mask
    
    def process_images(self, 
                      image_paths: List[str],
                      confidence_percentile: float = 10.0,
                      edge_normal_threshold: float = 5.0,
                      edge_depth_threshold: float = 0.03,
                      apply_confidence_mask: bool = True,
                      apply_edge_mask: bool = True,
                      apply_sky_mask: bool = False,
                      cond_flags: List[int] = [0, 0, 0]) -> Dict[str, Any]:
        """
        Process input images and generate 3D reconstruction results
        
        Args:
            image_paths: Original image paths
            confidence_percentile: Confidence filtering percentile
            edge_normal_threshold: Normal edge threshold
            edge_depth_threshold: Depth edge threshold
            apply_confidence_mask: Whether to apply confidence mask
            apply_edge_mask: Whether to apply edge mask
            apply_sky_mask: Whether to apply sky mask
            cond_flags: Conditioning flags [pose, intrinsics, depth]
            
        Returns:
            Dictionary containing all results
        """
        # Preprocess images
        imgs = prepare_images_to_tensor(image_paths, target_size=518, resize_strategy="crop").to(self.device)
        B, S, C, H, W = imgs.shape
        
        print(f"📸 Loaded {S} images with shape {imgs.shape}")
        
        # Inference
        print("\n🚀 Starting inference pipeline...")
        start_time = time.time()
        
        use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=bool(use_amp), dtype=amp_dtype):
                views = {"img": imgs}
                predictions = self.represent_model(views=views, cond_flags=cond_flags)
        
        print(f"🕒 Inference time: {time.time() - start_time:.3f} seconds")
        
        # Sky mask segmentation (if needed)
        sky_mask = None
        if apply_sky_mask:
            print("\n🌤️  Computing sky masks...")
            if not os.path.exists("skyseg.onnx"):
                print("Downloading skyseg.onnx...")
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
                )
            skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
            sky_mask_list = []
            for i, img_path in enumerate(image_paths or [f"image_{i}" for i in range(S)]):
                sky_mask_frame = segment_sky(img_path, skyseg_session)
                # Resize mask to match H×W if needed
                if sky_mask_frame.shape[0] != H or sky_mask_frame.shape[1] != W:
                    sky_mask_frame = cv2.resize(sky_mask_frame, (W, H))
                sky_mask_list.append(sky_mask_frame)
            sky_mask = np.stack(sky_mask_list, axis=0)  # [S, H, W]
            sky_mask = sky_mask > 0  # Binary mask: True = non-sky, False = sky
            print(f"✅ Sky masks computed for {S} frames")
        else:
            # Create dummy sky mask (all True = keep all points)
            sky_mask = np.ones((S, H, W), dtype=bool)
        
        # Prepare image data for saving
        processed_image_names = []
        images_data = {
            'processed_images': [],
            'original_images': [],
            'image_paths': image_paths,
            'H': H,
            'W': W
        }
        
        for i in range(S):
            im = (imgs[0, i].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            fname = f"image_{i+1:04d}.png"
            processed_image_names.append(fname)
            
            # Collect image data
            images_data['processed_images'].append({
                'data': im,
                'filename': fname
            })
            
            if image_paths and i < len(image_paths):
                pil_img = Image.open(image_paths[i]).convert("RGB")
                processed_height, processed_width = imgs[0, i].shape[1], imgs[0, i].shape[2]
                processed_aspect_ratio = processed_width / processed_height
                orig_width, orig_height = pil_img.size
                new_height = int(orig_width / processed_aspect_ratio)
                new_width = orig_width
                
                images_data['original_images'].append({
                    'data': pil_img,
                    'filename': fname,
                    'resize_params': {
                        'new_width': new_width,
                        'new_height': new_height
                    }
                })
        
        # Prepare pointmap data for filtering and saving
        pointmap_data = None
        if "pts3d" in predictions:
            print("Computing filter mask for pointmap...")
            
            # Prepare data for mask computation
            pts3d_conf_np = predictions["pts3d_conf"][0].detach().cpu().numpy()  # [S, H, W]
            depth_preds_np = predictions["depth"][0].detach().cpu().numpy()  # [S, H, W, 1]
            normal_preds_np = predictions["normals"][0].detach().cpu().numpy()  # [S, H, W, 3]
            
            # Compute comprehensive filter mask
            final_mask = self.create_filter_mask(
                pts3d_conf=pts3d_conf_np,
                depth_preds=depth_preds_np,
                normal_preds=normal_preds_np,
                sky_mask=sky_mask,
                confidence_percentile=confidence_percentile,
                edge_normal_threshold=edge_normal_threshold,
                edge_depth_threshold=edge_depth_threshold,
                apply_confidence_mask=apply_confidence_mask,
                apply_edge_mask=apply_edge_mask,
                apply_sky_mask=apply_sky_mask,
            )  # [S, H, W]
            
            # Collect points and colors
            pts_list = []
            pts_colors_list = []
            
            for i in range(S):
                pts = predictions["pts3d"][0, i]  # [H,W,3]
                img_colors = imgs[0, i].permute(1, 2, 0)  # [H, W, 3]
                img_colors = (img_colors * 255).to(torch.uint8)
                
                pts_list.append(pts.reshape(-1, 3))
                pts_colors_list.append(img_colors.reshape(-1, 3))
            
            all_pts = torch.cat(pts_list, dim=0)
            all_colors = torch.cat(pts_colors_list, dim=0)
            
            # Apply filter mask
            final_mask_flat = final_mask.reshape(-1)  # Flatten to [S*H*W]
            final_mask_torch = torch.from_numpy(final_mask_flat).to(all_pts.device)
            
            filtered_pts = all_pts[final_mask_torch]
            filtered_colors = all_colors[final_mask_torch]
            
            pointmap_data = {
                'filtered_pts': filtered_pts,
                'filtered_colors': filtered_colors
            }
        
        # Return all results for saving later
        return {
            'predictions': predictions,
            'images_data': images_data,
            'pointmap_data': pointmap_data,
            'H': H,
            'W': W,
            'S': S
        }
    
    def save_results(self, 
                    results: Dict[str, Any],
                    save_pointmap: bool = True,
                    save_depth: bool = True,
                    save_normal: bool = True,
                    save_gs: bool = True,
                    save_rendered: bool = True,
                    save_colmap: bool = True) -> Dict[str, Any]:
        """
        Save the results from process_images to files
        
        Args:
            results: The output from process_images method
            save_pointmap: Whether to save point cloud
            save_depth: Whether to save depth maps
            save_normal: Whether to save normal maps
            save_gs: Whether to save gaussians
            save_rendered: Whether to save rendered video
            save_colmap: Whether to save COLMAP reconstruction
            
        Returns:
            Dictionary containing paths to saved files
        """
        predictions = results['predictions']
        images_data = results['images_data']
        pointmap_data = results['pointmap_data']
        H = results['H']
        W = results['W']
        S = results['S']
        
        # Create output directories
        images_dir = self.output_path / "images"
        images_dir.mkdir(exist_ok=True)
        images_resized_dir = self.output_path / "images_resized"
        images_resized_dir.mkdir(exist_ok=True)
        
        if save_depth:
            depth_dir = self.output_path / "depth"
            depth_dir.mkdir(exist_ok=True)
        if save_normal:
            normal_dir = self.output_path / "normal"
            normal_dir.mkdir(exist_ok=True)
        if save_colmap:
            sparse_dir = self.output_path / "sparse" / "0"
            sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images
        for processed_image in images_data['processed_images']:
            Image.fromarray(processed_image['data']).save(str(images_resized_dir / processed_image['filename']))
        
        for original_image in images_data['original_images']:
            pil_img = original_image['data']
            resize_params = original_image['resize_params']
            pil_img = pil_img.resize((resize_params['new_width'], resize_params['new_height']), Image.Resampling.BICUBIC)
            pil_img.save(str(images_dir / original_image['filename']))
        
        save_results = {}
        
        # Save pointmap with filtering
        if pointmap_data is not None and save_pointmap:
            ply_path = self.output_path / "pts_from_pointmap.ply"
            save_scene_ply(ply_path, pointmap_data['filtered_pts'], pointmap_data['filtered_colors'])
            print(f"  - Saved {len(pointmap_data['filtered_pts'])} filtered points to {ply_path}")
            save_results['pointmap_path'] = str(ply_path)
        
        # Save depthmap
        if "depth" in predictions and save_depth:
            for i in range(S):
                # Save both PNG (for visualization) and NPY (for actual depth values)
                save_depth_png(depth_dir / f"depth_{i:04d}.png", predictions["depth"][0, i, :, :, 0])
                save_depth_npy(depth_dir / f"depth_{i:04d}.npy", predictions["depth"][0, i, :, :, 0])
            print(f"  - Saved {S} depth maps to {depth_dir} (both PNG and NPY formats)")
            save_results['depth_dir'] = str(depth_dir)
        
        # Save normalmap
        if "normals" in predictions and save_normal:
            for i in range(S):
                save_normal_png(normal_dir / f"normal_{i:04d}.png", predictions["normals"][0, i])
            print(f"  - Saved {S} normal maps to {normal_dir}")
            save_results['normal_dir'] = str(normal_dir)
        
        # Save Gaussians PLY and render video
        if "splats" in predictions and save_gs:
            # Get Gaussian parameters (already filtered by GaussianSplatRenderer)
            means = predictions["splats"]["means"][0].reshape(-1, 3)
            scales = predictions["splats"]["scales"][0].reshape(-1, 3)
            quats = predictions["splats"]["quats"][0].reshape(-1, 4)
            colors = (predictions["splats"]["sh"][0] if "sh" in predictions["splats"] else predictions["splats"]["colors"][0]).reshape(-1, 3)
            opacities = predictions["splats"]["opacities"][0].reshape(-1)
            
            # Save Gaussian PLY
            ply_path = self.output_path / "gaussians.ply"
            save_gs_ply(
                ply_path,
                means,
                scales,
                quats,
                colors,
                opacities,
            )
            
            # Render video using the same filtered splats from predictions
            num_views = S
            if save_rendered:
                e4x4 = predictions['camera_poses']
                k3x3 = predictions['camera_intrs']
                render_interpolated_video(self.represent_model.gs_renderer, predictions["splats"], e4x4, k3x3, (H, W), self.output_path / "rendered", interp_per_pair=15, loop_reverse=num_views==1)
                print(f"  - Saved rendered.mp4 to {self.output_path}")
            else:
                print(f"⚠️  Not set save_rendered flag, skipping video rendering")
            
            save_results['gaussians_path'] = str(ply_path)
            save_results['rendered_video_path'] = str(self.output_path / "rendered") if save_rendered else None
        
        return save_results
    
    def __call__(self,
                 input_paths: List[str],
                 output_path: Optional[str] = None,
                 confidence_percentile: float = 10.0,
                 edge_normal_threshold: float = 5.0,
                 edge_depth_threshold: float = 0.03,
                 apply_confidence_mask: bool = True,
                 apply_edge_mask: bool = True,
                 apply_sky_mask: bool = False,
                 cond_flags: List[int] = [0, 0, 0],
                 **kwargs):
        """
        调用接口，支持额外参数
        
        Args:
            input_paths: 输入图片路径列表
            output_path: 输出路径（如果提供则覆盖默认路径）
            confidence_percentile: 置信度过滤百分位
            edge_normal_threshold: 法线边缘阈值
            edge_depth_threshold: 深度边缘阈值
            apply_confidence_mask: 是否应用置信度掩码
            apply_edge_mask: 是否应用边缘掩码
            apply_sky_mask: 是否应用天空掩码
            cond_flags: 条件标志 [pose, intrinsics, depth]
            **kwargs: 其他参数
            
        Returns:
            包含所有处理结果的字典（用于外部调用save_results）
        """

        # 如果提供了输出路径，更新输出路径
        if output_path:
            self.output_path = Path(output_path)
            self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 处理图像并返回结果，不进行保存
        return self.process_images(
            image_paths=input_paths,
            confidence_percentile=confidence_percentile,
            edge_normal_threshold=edge_normal_threshold,
            edge_depth_threshold=edge_depth_threshold,
            apply_confidence_mask=apply_confidence_mask,
            apply_edge_mask=apply_edge_mask,
            apply_sky_mask=apply_sky_mask,
            cond_flags=cond_flags
        )
    
    def save_pretrained(self, save_directory: str):
        """保存模型到指定目录"""
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存表示模型
        if self.represent_model:
            represent_dir = os.path.join(save_directory, "representation_model")
            self.represent_model.save_pretrained(represent_dir)
        
        # 保存pipeline配置
        config = {
            'output_path': str(self.output_path),
            'device': self.device
        }
        
        torch.save(config, os.path.join(save_directory, "pipeline_config.pt"))
        print(f"Pipeline saved to {save_directory}")