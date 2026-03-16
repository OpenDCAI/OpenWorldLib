import os
import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from PIL import Image

from huggingface_hub import snapshot_download

from ...base_representation import BaseRepresentation
from ....base_models.three_dimensions.point_clouds.gaussian_splatting.scene.dataset_readers import (
    fetchPly,
)
from ....representations.point_clouds_generation.flash_world.flash_world.render import (
    gaussian_render,
)

# Try to import gdown for Google Drive downloads
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False
    print("Warning: gdown not installed. Install with 'pip install gdown' to enable Google Drive downloads.")

# Import CUT3R modules using relative imports
from .cut3r import (
    ARCroco3DStereo,
    inference,
    load_images,
    pose_encoding_to_camera,
    estimate_focal_knowing_depth,
    geotrf,
)


# CUT3R model registry mapping model names to Google Drive file IDs
CUT3R_MODEL_REGISTRY = {
    "cut3r_224_linear_4": {
        "file_id": "11dAgFkWHpaOHsR6iuitlB_v4NFFBrWjy",
        "filename": "cut3r_224_linear_4.pth",
        "size": 224,
    },
    "cut3r_512_dpt_4_64": {
        "file_id": "1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD",
        "filename": "cut3r_512_dpt_4_64.pth",
        "size": 512,
    },
}


class CUT3RRepresentation(BaseRepresentation):
    """
    Representation for CUT3R 3D scene reconstruction.
    """
    
    def __init__(self, model: Optional[ARCroco3DStereo] = None, device: Optional[str] = None):
        """
        Initialize CUT3R representation model.
        
        Args:
            model: Pre-loaded ARCroco3DStereo model (optional)
            device: Device to run on ('cuda' or 'cpu')
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        
        if self.model is not None:
            self.model = self.model.to(self.device).eval()
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[str] = None,
        size: Optional[int] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> 'CUT3RRepresentation':
        """
        Create representation instance from pretrained model.
        
        Args:
            pretrained_model_path: Model identifier - can be:
                - CUT3R model name: "cut3r_224_linear_4" or "cut3r_512_dpt_4_64"
                - HuggingFace repo ID (e.g., "username/repo")
                - Local path to model checkpoint or directory
            device: Device to run on
            size: Input image size (auto-detected from model name if not specified)
            cache_dir: Directory to cache downloaded models (default: ~/.cache/cut3r)
            **kwargs: Additional arguments
            
        Returns:
            CUT3RRepresentation instance
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if it's a registered CUT3R model name
        if pretrained_model_path in CUT3R_MODEL_REGISTRY:
            model_info = CUT3R_MODEL_REGISTRY[pretrained_model_path]
            model_path = cls._download_from_google_drive(
                model_info["file_id"],
                model_info["filename"],
                cache_dir=cache_dir
            )
            # Auto-detect size from model name if not specified
            if size is None:
                size = model_info["size"]
        elif os.path.isdir(pretrained_model_path):
            # Local directory
            model_path = pretrained_model_path
            # Look for .pth file in the directory
            pth_files = list(Path(model_path).glob("*.pth"))
            if pth_files:
                model_path = str(pth_files[0])
        elif os.path.isfile(pretrained_model_path):
            # Local file path
            model_path = pretrained_model_path
        else:
            # Try to download from HuggingFace
            print(f"Downloading weights from HuggingFace repo: {pretrained_model_path}")
            try:
                downloaded_path = snapshot_download(pretrained_model_path, cache_dir=cache_dir)
                # Look for .pth file in the downloaded directory
                pth_files = list(Path(downloaded_path).glob("*.pth"))
                if pth_files:
                    model_path = str(pth_files[0])
                else:
                    # If no .pth file, check if the directory itself contains model files
                    model_path = downloaded_path
                print(f"Model downloaded to: {model_path}")
            except Exception as e:
                print(f"Warning: Could not download from HuggingFace: {e}")
                print(f"Trying to use as local path: {pretrained_model_path}")
                model_path = pretrained_model_path
        
        # Load model
        try:
            # ARCroco3DStereo.from_pretrained expects a file path, not a directory
            if os.path.isdir(model_path):
                # If it's a directory, look for checkpoint file
                pth_files = list(Path(model_path).glob("*.pth"))
                if pth_files:
                    model_path = str(pth_files[0])
                else:
                    raise ValueError(f"No .pth file found in {model_path}")
            
            model = ARCroco3DStereo.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load CUT3R model from {model_path}: {e}")
        
        instance = cls(model=model, device=device)
        # Use auto-detected size or default
        instance.size = size if size is not None else 224
        return instance
    
    @staticmethod
    def _download_from_google_drive(
        file_id: str,
        filename: str,
        cache_dir: Optional[str] = None
    ) -> str:
        """
        Download model from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            filename: Output filename
            cache_dir: Cache directory (default: ~/.cache/cut3r)
            
        Returns:
            Path to downloaded model file
        """
        if not HAS_GDOWN:
            raise RuntimeError(
                "gdown is required for Google Drive downloads. "
                "Install with: pip install gdown"
            )
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "cut3r")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if file already exists
        output_path = os.path.join(cache_dir, filename)
        if os.path.exists(output_path):
            print(f"Using cached model: {output_path}")
            return output_path
        
        # Download from Google Drive
        print(f"Downloading CUT3R model from Google Drive: {filename}")
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, output_path, quiet=False)
            print(f"Model downloaded to: {output_path}")
            return output_path
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model from Google Drive (file_id: {file_id}): {e}\n"
                f"Please check your internet connection and try again."
            )
    
    def api_init(self, api_key: str, endpoint: str):
        """Initialize API connection if needed."""
        pass
    
    def _prepare_views(
        self,
        images: Union[np.ndarray, List[np.ndarray], List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare input views for CUT3R inference.
        
        Args:
            images: List of image paths, numpy arrays, or single numpy array
            
        Returns:
            List of view dictionaries
        """
        # Convert to list if single image
        if isinstance(images, np.ndarray):
            images = [images]
        
        # Convert numpy arrays to file paths (temporary) if needed
        # CUT3R's load_images expects file paths
        import tempfile
        temp_files = []
        image_paths = []
        
        for img in images:
            if isinstance(img, str):
                image_paths.append(img)
            elif isinstance(img, np.ndarray):
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                # Convert to uint8 and save
                img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                Image.fromarray(img_uint8).save(temp_file.name)
                image_paths.append(temp_file.name)
                temp_files.append(temp_file.name)
        
        # CUT3R recurrent inference expects at least two views.
        # If only one image is provided, duplicate it as a minimal fallback
        # to avoid invalid position/index errors inside RoPE attention.
        if len(image_paths) == 1:
            image_paths.append(image_paths[0])
        
        # Load images using CUT3R's loader
        loaded_images = load_images(image_paths, size=self.size, verbose=False)
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        # Convert to views format
        views = []
        for i, img_data in enumerate(loaded_images):
            # Ensure true_shape is in the correct format: (batch_size, 2)
            # load_images returns (1, 2), we need to expand to (batch_size, 2)
            batch_size = img_data["img"].shape[0]
            true_shape_np = img_data["true_shape"]  # Shape: (1, 2)
            true_shape_tensor = torch.from_numpy(true_shape_np)  # Shape: (1, 2)
            # Expand to (batch_size, 2) if needed
            if true_shape_tensor.shape[0] == 1 and batch_size > 1:
                true_shape_tensor = true_shape_tensor.repeat(batch_size, 1)
            elif true_shape_tensor.shape[0] != batch_size:
                # If shape doesn't match, use the first row and repeat
                true_shape_tensor = true_shape_tensor[0:1].repeat(batch_size, 1)
            
            view = {
                "img": img_data["img"],
                "ray_map": torch.full(
                    (
                        img_data["img"].shape[0],
                        6,
                        img_data["img"].shape[-2],
                        img_data["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": true_shape_tensor,
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            views.append(view)
        
        return views
    
    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get 3D scene representation from input data.
        
        Args:
            data: Dictionary containing:
                - 'images': List of image paths, numpy arrays, or single numpy array
                - 'output_type': str, "point_cloud", "depth_map", "camera_pose", or "all"
                - Optional: 'size': int, input image size (default: self.size)
                - Optional: 'vis_threshold': float, confidence threshold for filtering point clouds (default: 1.0)
                
        Returns:
            Dictionary containing:
                - 'point_cloud': List of point clouds (if output_type includes "point_cloud" or "all")
                - 'depth_map': List of depth maps (if output_type includes "depth_map" or "all")
                - 'camera_pose': List of camera poses (if output_type includes "camera_pose" or "all")
                - 'colors': List of color maps for point clouds
                - 'confidence': List of confidence maps
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() first.")
        
        images = data['images']
        output_type = data.get('output_type', 'all')
        size = data.get('size', self.size)
        vis_threshold = data.get('vis_threshold', 1.0)
        
        # Prepare views
        views = self._prepare_views(images)
        
        # Run inference
        with torch.no_grad():
            outputs, state_args = inference(views, self.model, self.device, verbose=False)
        
        # Process outputs
        results = {}
        
        # Extract predictions
        pts3ds_self = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
        conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
        colors = [
            0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0).cpu() 
            for output in outputs["views"]
        ]
        
        # Recover camera poses
        pr_poses = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in outputs["pred"]
        ]
        
        # Transform points to world coordinates
        pts3ds_world = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            pts3d_world = geotrf(pose, pself.unsqueeze(0))
            pts3ds_world.append(pts3d_world)
        
        # Estimate focal length
        B, H, W, _ = pts3ds_self[0].shape
        pp = torch.tensor([W // 2, H // 2], device=pts3ds_self[0].device).float().repeat(B, 1)
        focal = estimate_focal_knowing_depth(pts3ds_self[0], pp, focal_mode="weiszfeld")
        
        # Convert to numpy and apply vis_threshold filtering
        if output_type in ["point_cloud", "all"]:
            filtered_pcs = []
            filtered_colors = []
            for pc_world, color, conf in zip(pts3ds_world, colors, conf_self):
                pc_np = pc_world.numpy()
                color_np = color.numpy()
                conf_np = conf.numpy()
                
                # Apply vis_threshold filtering if confidence is available
                if vis_threshold > 0:
                    pc_flat = pc_np.reshape(-1, 3)
                    color_flat = color_np.reshape(-1, 3)
                    conf_flat = conf_np.reshape(-1)
                    
                    # Filter points with confidence > vis_threshold
                    mask = conf_flat > vis_threshold
                    if mask.sum() > 0:
                        pc_filtered = pc_flat[mask]
                        color_filtered = color_flat[mask]
                        # Reshape back to original shape if possible, otherwise keep as flat
                        # For visualization purposes, we keep it flat
                        filtered_pcs.append(pc_filtered)
                        filtered_colors.append(color_filtered)
                    else:
                        # If no points pass threshold, use all points
                        filtered_pcs.append(pc_np)
                        filtered_colors.append(color_np)
                else:
                    filtered_pcs.append(pc_np)
                    filtered_colors.append(color_np)
            
            results['point_cloud'] = filtered_pcs
            results['colors'] = filtered_colors
        
        if output_type in ["depth_map", "all"]:
            results['depth_map'] = [p[..., 2].numpy() for p in pts3ds_self]  # Z component is depth
            results['confidence'] = [c.numpy() for c in conf_self]
        
        if output_type in ["camera_pose", "all"]:
            results['camera_pose'] = [p.numpy() for p in pr_poses]
            results['focal'] = focal.cpu().numpy()
            results['principal_point'] = pp.cpu().numpy()
        
        # Store state_args for potential future use
        results['state_args'] = state_args
        
        return results

    @staticmethod
    def _preprocess_point_cloud_for_render(
        points: np.ndarray,
        colors: np.ndarray,
        scene_center: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Light-weight cleanup to make rendering closer to CUT3R visualization:
        1) remove invalid rows
        2) trim far outliers
        3) voxel downsample to reduce overdraw blur
        """
        valid_mask = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
        points = points[valid_mask]
        colors = colors[valid_mask]
        if len(points) == 0:
            return points, colors

        # Trim extreme outliers by distance-to-center (keeps dense core).
        d = np.linalg.norm(points - scene_center[None, :], axis=1)
        d_thr = np.quantile(d, 0.995)
        keep = d <= d_thr
        points = points[keep]
        colors = colors[keep]
        if len(points) == 0:
            return points, colors

        scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
        voxel_size = max(scene_radius / 512.0, 1e-4)

        # Voxel downsample (first-point per voxel, deterministic).
        voxel_coords = np.floor(points / voxel_size).astype(np.int64)
        _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        points = points[unique_idx]
        colors = colors[unique_idx]

        return points, colors

    @staticmethod
    def _estimate_gaussian_scale(points: np.ndarray, scene_center: np.ndarray) -> float:
        """
        Estimate a conservative Gaussian scale from local spacing.
        Large scales are the main reason for "foggy/blurry" outputs.
        """
        if len(points) < 4:
            scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
            return max(scene_radius / 2000.0, 1e-4)

        sample_n = min(len(points), 2048)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=sample_n, replace=False)
        sample = torch.from_numpy(points[idx]).float()
        # Pairwise distances on a small sample for robust nearest-neighbor spacing.
        dist = torch.cdist(sample, sample, p=2)
        dist.fill_diagonal_(1e9)
        nn = dist.min(dim=1).values
        nn_med = float(nn.median().item())

        scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
        min_scale = max(scene_radius / 5000.0, 1e-4)
        max_scale = max(scene_radius / 300.0, min_scale)
        return float(np.clip(nn_med * 0.6, min_scale, max_scale))

    def render_with_3dgs(
        self,
        ply_path: str,
        camera_config: Dict[str, Any],
        image_width: int = 704,
        image_height: int = 480,
        device: Optional[str] = None,
        near_plane: float = 0.01,
        far_plane: float = 1000.0,
    ) -> Image.Image:
        """
        Render a single frame from a CUT3R reconstruction using 3D Gaussian Splatting.
        Follow the same robust rendering strategy as VGGT:
        - normalize scene for scale invariance
        - dynamic near/far planes
        - multiple camera convention probing
        - deterministic point-projection fallback when gsplat fails
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        pcd = fetchPly(ply_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32)

        if points.size == 0:
            raise RuntimeError(f"No points loaded from PLY: {ply_path}")

        # Estimate global scene scale and normalize to stabilize rendering.
        scene_center = points.mean(axis=0)
        scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
        scene_radius = max(scene_radius, 1e-6)

        points_norm = (points - scene_center[None, :]) / scene_radius
        center = np.asarray(camera_config.get("center", scene_center.tolist()), dtype=np.float32)
        center_norm = (center - scene_center) / scene_radius

        radius_raw = float(camera_config.get("radius", 1.0 * scene_radius))
        radius_norm = max(radius_raw / scene_radius, 1e-3)
        # +180° yaw so camera is on the opposite side (scene faces camera, not back).
        yaw_deg = float(camera_config.get("yaw", 0.0)) + 180.0
        pitch_deg = float(camera_config.get("pitch", 0.0))

        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        cam_x = center_norm[0] + radius_norm * np.cos(pitch) * np.sin(yaw)
        cam_y = center_norm[1] + radius_norm * np.sin(pitch)
        cam_z = center_norm[2] + radius_norm * np.cos(pitch) * np.cos(yaw)
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        def build_c2w(
            look_at: np.ndarray,
            eye: np.ndarray,
            reverse_forward: bool = False,
            basis_layout: str = "row",
        ) -> np.ndarray:
            forward = (eye - look_at) if reverse_forward else (look_at - eye)
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, up)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                right = np.cross(forward, up)
                right_norm = np.linalg.norm(right)
            right = right / (right_norm + 1e-8)
            up = np.cross(right, forward)
            up = up / (np.linalg.norm(up) + 1e-8)

            c2w_local = np.eye(4, dtype=np.float32)
            if basis_layout == "row":
                c2w_local[0, :3] = right
                c2w_local[1, :3] = up
                c2w_local[2, :3] = forward
            else:
                c2w_local[:3, 0] = right
                c2w_local[:3, 1] = up
                c2w_local[:3, 2] = forward
            c2w_local[:3, 3] = eye
            return c2w_local

        fx = 0.5 * image_width / np.tan(np.deg2rad(60.0) / 2.0)
        fy = 0.5 * image_height / np.tan(np.deg2rad(45.0) / 2.0)
        cx = image_width / 2.0
        cy = image_height / 2.0

        xyz = torch.from_numpy(points_norm).to(device=device, dtype=torch.float32)
        scale_value = self._estimate_gaussian_scale(points_norm, center_norm)
        scale = torch.full((xyz.shape[0], 3), scale_value, device=device, dtype=torch.float32)
        rotation = torch.zeros((xyz.shape[0], 4), device=device, dtype=torch.float32)
        rotation[:, 0] = 1.0
        opacity = torch.full((xyz.shape[0], 1), 0.95, device=device, dtype=torch.float32)
        color_tensor = torch.from_numpy(np.clip(colors, 0.0, 1.0)).to(device=device, dtype=torch.float32)

        gaussian_params = torch.cat([xyz, opacity, scale, rotation, color_tensor], dim=-1).unsqueeze(0)
        intr = torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32, device=device).unsqueeze(0)

        # Dynamic planes are more robust for arbitrary world scales.
        near_dynamic = max(near_plane, radius_norm * 0.01)
        far_dynamic = max(far_plane, radius_norm * 20.0)

        if not hasattr(self, "_render_variant_cache"):
            self._render_variant_cache = {}

        def render_candidate(reverse_forward: bool, basis_layout: str):
            c2w_local = build_c2w(
                look_at=center_norm,
                eye=cam_pos,
                reverse_forward=reverse_forward,
                basis_layout=basis_layout,
            )
            test_c2ws_local = torch.from_numpy(c2w_local).unsqueeze(0).unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            rgb_local, _ = gaussian_render(
                gaussian_params,
                test_c2ws_local,
                intr,
                image_width,
                image_height,
                near_plane=near_dynamic,
                far_plane=far_dynamic,
                use_checkpoint=False,
                sh_degree=0,
                bg_mode="black",
            )
            rgb_img_local = rgb_local[0, 0].clamp(-1.0, 1.0).add(1.0).div(2.0)
            gray = rgb_img_local.mean(dim=0)
            non_bg_ratio = float((gray > 0.03).float().mean().item())
            std_v = float(rgb_img_local.std().item())
            score = non_bg_ratio + 0.5 * std_v
            return rgb_img_local, score, non_bg_ratio

        cached_variant = self._render_variant_cache.get(
            ply_path,
            {"reverse_forward": False, "basis_layout": "row"},
        )
        rgb_img, best_score, best_non_bg_ratio = render_candidate(
            reverse_forward=bool(cached_variant["reverse_forward"]),
            basis_layout=str(cached_variant["basis_layout"]),
        )

        # If the cached/default pose is too empty, probe multiple camera conventions once.
        if best_score < 0.03 or best_non_bg_ratio < 0.001:
            candidates = [
                {"reverse_forward": False, "basis_layout": "row"},
                {"reverse_forward": True, "basis_layout": "row"},
                {"reverse_forward": False, "basis_layout": "col"},
                {"reverse_forward": True, "basis_layout": "col"},
            ]
            best_variant = cached_variant
            for cand in candidates:
                rgb_try, score_try, non_bg_try = render_candidate(
                    reverse_forward=bool(cand["reverse_forward"]),
                    basis_layout=str(cand["basis_layout"]),
                )
                if score_try > best_score:
                    rgb_img = rgb_try
                    best_score = score_try
                    best_non_bg_ratio = non_bg_try
                    best_variant = cand
            self._render_variant_cache[ply_path] = best_variant

        # If gsplat still fails (near-empty), fallback to deterministic point projection.
        if best_score < 0.03 or best_non_bg_ratio < 0.001:
            best_variant = self._render_variant_cache.get(
                ply_path,
                {"reverse_forward": False, "basis_layout": "row"},
            )
            c2w_best = build_c2w(
                look_at=center_norm,
                eye=cam_pos,
                reverse_forward=bool(best_variant["reverse_forward"]),
                basis_layout=str(best_variant["basis_layout"]),
            )

            img_fallback = np.zeros((image_height, image_width, 3), dtype=np.float32)
            depth_buf = np.full((image_height, image_width), np.inf, dtype=np.float32)

            max_points = 300000
            if points_norm.shape[0] > max_points:
                rng = np.random.default_rng(42)
                keep_idx = rng.choice(points_norm.shape[0], size=max_points, replace=False)
                proj_points = points_norm[keep_idx]
                proj_colors = colors[keep_idx]
            else:
                proj_points = points_norm
                proj_colors = colors

            w2c = np.linalg.inv(c2w_best).astype(np.float32)
            pts_h = np.concatenate(
                [proj_points, np.ones((proj_points.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            cam_pts = (w2c @ pts_h.T).T[:, :3]

            best_proj_count = -1
            best_proj_payload = None
            for depth_sign in [1.0, -1.0]:
                z = cam_pts[:, 2] * depth_sign
                valid_z = z > 1e-4
                cam_pts_s = cam_pts[valid_z]
                z_s = z[valid_z]
                c_s = proj_colors[valid_z]
                if cam_pts_s.shape[0] == 0:
                    continue

                u = (fx * (cam_pts_s[:, 0] / z_s) + cx).astype(np.int32)
                v = (fy * (cam_pts_s[:, 1] / z_s) + cy).astype(np.int32)
                in_view = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
                view_count = int(in_view.sum())
                if view_count > best_proj_count:
                    best_proj_count = view_count
                    best_proj_payload = (u[in_view], v[in_view], z_s[in_view], c_s[in_view])

            if best_proj_payload is not None and best_proj_count > 0:
                u, v, z, c_proj = best_proj_payload
                order = np.argsort(z)
                u = u[order]
                v = v[order]
                z = z[order]
                c_proj = c_proj[order]

                for uu, vv, zz, cc in zip(u, v, z, c_proj):
                    if zz < depth_buf[vv, uu]:
                        depth_buf[vv, uu] = zz
                        img_fallback[vv, uu] = np.clip(cc, 0.0, 1.0)

                valid_mask = np.isfinite(depth_buf).astype(np.uint8)
                if valid_mask.any():
                    kernel = np.ones((3, 3), np.uint8)
                    dilated = cv2.dilate((img_fallback * 255).astype(np.uint8), kernel, iterations=1)
                    filled = cv2.dilate(valid_mask, kernel, iterations=1)
                    img_fallback[filled > 0] = dilated[filled > 0] / 255.0

                rgb_img = torch.from_numpy(img_fallback).permute(2, 0, 1).to(torch.float32)

        rgb_np = (
            rgb_img.mul(255.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        # Orientation fix: vertical flip only (correct upside-down).
        rgb_np = np.flipud(rgb_np)
        return Image.fromarray(rgb_np)

