import os
from typing import Optional, List, Union, Dict, Any, Generator
import numpy as np
from PIL import Image
import cv2
import torch
from diffusers.utils import export_to_video

from ...operators.cut3r_operator import CUT3ROperator
from ...representations.point_clouds_generation.cut3r.cut3r_representation import (
    CUT3RRepresentation,
)
from ...base_models.three_dimensions.point_clouds.gaussian_splatting.scene.dataset_readers import (
    storePly,
)
from ...representations.point_clouds_generation.flash_world.flash_world.render import (
    gaussian_render,
)


class CUT3RResult:
    """Container class for CUT3R results."""
    
    def __init__(
        self, 
        images: List[Image.Image],
        point_clouds: Optional[List[np.ndarray]] = None,
        depth_maps: Optional[List[np.ndarray]] = None,
        camera_poses: Optional[List[np.ndarray]] = None,
        data_type: str = "image"
    ):
        """
        Initialize CUT3R result container.
        
        Args:
            images: List of PIL Images (rendered point clouds or depth visualizations)
            point_clouds: List of point cloud arrays (optional)
            depth_maps: List of depth map arrays (optional)
            camera_poses: List of camera pose arrays (optional)
            data_type: Type of data ('image' or 'video')
        """
        self.images = images
        self.point_clouds = point_clouds
        self.depth_maps = depth_maps
        self.camera_poses = camera_poses
        self.data_type = data_type
    
    def save(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Save results to files.
        
        Args:
            output_dir: Output directory. If None, uses default.
            
        Returns:
            List of saved file paths
        """
        if output_dir is None:
            output_dir = "./cut3r_output"
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files: List[str] = []
        
        # Save images
        for i, img in enumerate(self.images):
            output_path = os.path.join(output_dir, f"frame_{i:06d}.png")
            img.save(output_path)
            saved_files.append(output_path)
        
        # Save point clouds if available
        if self.point_clouds is not None:
            pc_dir = os.path.join(output_dir, "point_clouds")
            os.makedirs(pc_dir, exist_ok=True)
            for i, pc in enumerate(self.point_clouds):
                output_path = os.path.join(pc_dir, f"pc_{i:06d}.npy")
                np.save(output_path, pc)
                saved_files.append(output_path)
        
        # Save depth maps if available
        if self.depth_maps is not None:
            depth_dir = os.path.join(output_dir, "depth_maps")
            os.makedirs(depth_dir, exist_ok=True)
            for i, depth in enumerate(self.depth_maps):
                # Ensure depth is 2D array (H, W) - follow CUT3R original code style
                if depth.ndim > 2:
                    # If batch dimension exists, take first item
                    depth = depth[0] if depth.ndim == 3 else depth.squeeze()
                elif depth.ndim < 2:
                    # If 1D, skip this depth map
                    print(f"Warning: Skipping depth map {i} with unexpected shape: {depth.shape}")
                    continue
                
                # Normalize depth values (follow CUT3R style: normalize to [0, 1] then scale to [0, 255])
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max > depth_min:
                    depth_norm = (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth_norm = np.zeros_like(depth)
                
                # Convert to uint8 and ensure it's 2D (H, W) - required by cv2.applyColorMap
                depth_uint8 = (depth_norm * 255).astype(np.uint8)
                if depth_uint8.ndim != 2:
                    depth_uint8 = depth_uint8.squeeze()
                    if depth_uint8.ndim != 2:
                        print(f"Warning: Skipping depth map {i} - cannot convert to 2D, shape: {depth_uint8.shape}")
                        continue
                
                # Apply colormap (requires 2D uint8 array) - follow CUT3R original code
                depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
                output_path = os.path.join(depth_dir, f"depth_{i:06d}.png")
                cv2.imwrite(output_path, depth_colored)
                saved_files.append(output_path)
        
        return saved_files
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]


class CUT3RPipeline:
    """Pipeline for CUT3R 3D scene reconstruction."""
    
    def __init__(
        self,
        representation_model: Optional[CUT3RRepresentation] = None,
        reasoning_model: Optional[Any] = None,
        synthesis_model: Optional[Any] = None,
        operator: Optional[CUT3ROperator] = None,
    ):
        """
        Initialize CUT3R pipeline.
        
        Args:
            representation_model: Pre-loaded CUT3RRepresentation instance (optional)
            reasoning_model: Reasoning model (not used for CUT3R, kept for compatibility)
            synthesis_model: Synthesis model (not used for CUT3R, kept for compatibility)
            operator: CUT3ROperator instance (optional)
        """
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.operator = operator or CUT3ROperator()
    
    @classmethod
    def from_pretrained(
        cls,
        representation_path: str,
        reasoning_path: Optional[str] = None,
        synthesis_path: Optional[str] = None,
        **kwargs
    ) -> 'CUT3RPipeline':
        """
        Create pipeline instance from pretrained models.
        
        Args:
            representation_path: HuggingFace repo ID for representation model
            reasoning_path: Not used for CUT3R (kept for compatibility)
            synthesis_path: Not used for CUT3R (kept for compatibility)
            **kwargs: Additional arguments passed to representation.from_pretrained()
            
        Returns:
            CUT3RPipeline instance
        """
        representation_model = CUT3RRepresentation.from_pretrained(
            pretrained_model_path=representation_path,
            **kwargs
        )
        
        # CUT3R doesn't use reasoning or synthesis models, but keep for compatibility
        reasoning_model = None
        synthesis_model = None
        
        return cls(
            representation_model=representation_model,
            reasoning_model=reasoning_model,
            synthesis_model=synthesis_model,
        )
    
    def process(
        self,
        input_: Union[str, Image.Image, np.ndarray, List[str], List[Image.Image], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> CUT3RResult:
        """
        Process input and generate 3D scene representation.
        
        Args:
            input_: Input image(s) - can be:
                - Image file path (str)
                - List of image file paths
                - PIL Image
                - List of PIL Images
                - Numpy array (H, W, 3)
                - List of numpy arrays
            interaction: Interaction string or dictionary
            **kwargs: Additional arguments:
                - output_type: "point_cloud", "depth_map", "camera_pose", or "all" (default: "all")
                - size: Input image size (default: auto-detected from model or 224)
                - vis_threshold: Confidence threshold for filtering point clouds (default: 1.0)
                - return_point_clouds: If True, include point clouds in result (default: True)
                - return_depth_maps: If True, include depth maps in result (default: True)
                - return_camera_poses: If True, include camera poses in result (default: True)
                
        Returns:
            CUT3RResult object containing processed results
        """
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")
        
        # Process input using operator's process_perception
        images_data = self.operator.process_perception(input_)
        if not isinstance(images_data, list):
            images_data = [images_data]
        
        # Process interaction
        if interaction is None:
            interaction_dict = {
                "data_type": "image",
                "output_type": "all"
            }
        elif isinstance(interaction, str):
            self.operator.get_interaction(interaction)
            interaction_dict = self.operator.process_interaction()
        else:
            interaction_dict = interaction
        
        # Prepare data for representation
        # Get size from kwargs or use representation model's default size
        size = kwargs.get('size', None)
        if size is None and self.representation_model is not None:
            size = getattr(self.representation_model, 'size', 224)
        elif size is None:
            size = 224
        
        data = {
            'images': images_data,
            'output_type': interaction_dict.get('output_type', kwargs.get('output_type', 'all')),
            'size': size,
            'vis_threshold': kwargs.get('vis_threshold', 1.0),
        }
        
        # Get representation
        results = self.representation_model.get_representation(data)
        
        # Convert results to PIL Images for visualization
        output_images = []
        
        # Prefer depth map visualization (more reliable for filtered point clouds)
        if 'depth_map' in results and results['depth_map']:
            # Use depth map visualization
            for depth in results['depth_map']:
                # Ensure depth is 2D array (H, W)
                if depth.ndim > 2:
                    # If batch dimension exists, take first item
                    depth = depth[0] if depth.ndim == 3 else depth.squeeze()
                elif depth.ndim < 2:
                    # If 1D, try to reshape (shouldn't happen normally)
                    raise ValueError(f"Unexpected depth shape: {depth.shape}")
                
                # Normalize depth values
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max > depth_min:
                    depth_norm = (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth_norm = np.zeros_like(depth)
                
                # Convert to uint8 and ensure it's 2D (H, W)
                depth_uint8 = (depth_norm * 255).astype(np.uint8)
                if depth_uint8.ndim != 2:
                    depth_uint8 = depth_uint8.squeeze()
                    if depth_uint8.ndim != 2:
                        raise ValueError(f"Depth array must be 2D after processing, got shape: {depth_uint8.shape}")
                
                # Apply colormap (requires 2D uint8 array)
                depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
                output_images.append(Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)))
        elif 'point_cloud' in results and results['point_cloud']:
            # Use point cloud visualization as fallback
            for pc, color in zip(results['point_cloud'], results.get('colors', [])):
                # Ensure point cloud is flattened (N, 3) format
                if pc.ndim == 3:
                    pc_2d = pc.reshape(-1, 3)
                else:
                    pc_2d = pc
                
                # Normalize and create image
                # For now, create a simple depth visualization
                depth = pc_2d[:, 2]
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_img = (depth_norm * 255).astype(np.uint8)
                
                # Try to reshape to square image if possible
                num_points = len(pc_2d)
                h = int(np.sqrt(num_points))
                w = num_points // h
                if h * w == num_points and h > 0 and w > 0:
                    depth_img = depth_img.reshape(h, w)
                    depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_VIRIDIS)
                    output_images.append(Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)))
                else:
                    # Fallback: use original input images if point cloud can't be visualized
                    pass
        
        # If no visualization was created, use input images as fallback
        if len(output_images) == 0:
            for img_data in images_data:
                if isinstance(img_data, np.ndarray):
                    img_uint8 = (img_data * 255).astype(np.uint8) if img_data.max() <= 1.0 else img_data.astype(np.uint8)
                    output_images.append(Image.fromarray(img_uint8))
        
        # Determine data type
        data_type = "image" if len(images_data) == 1 else "video"
        
        # Create result object
        result = CUT3RResult(
            images=output_images,
            point_clouds=results.get('point_cloud') if kwargs.get('return_point_clouds', True) else None,
            depth_maps=results.get('depth_map') if kwargs.get('return_depth_maps', True) else None,
            camera_poses=results.get('camera_pose') if kwargs.get('return_camera_poses', True) else None,
            data_type=data_type
        )
        
        return result

    def reconstruct_ply(
        self,
        input_: Union[str, Image.Image, np.ndarray, List[str], List[Image.Image], List[np.ndarray]],
        ply_path: Optional[str] = None,
        size: Optional[int] = None,
        vis_threshold: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Stage 1: Run CUT3R to reconstruct a point cloud and export it as a PLY file,
        together with a simple camera parameter range for downstream 3DGS rendering.

        Args:
            input_: Input image(s), same formats as ``process``.
            ply_path: Optional output path for the reconstructed PLY. If it is a
                directory, ``pointcloud.ply`` will be created inside it. If None,
                ``./cut3r_output/pointcloud.ply`` is used.
            size: Optional input image size. If None, falls back to the representation
                model's default.
            vis_threshold: Confidence threshold used when generating the point cloud.

        Returns:
            Dictionary with:
                - 'ply_path': str, path to the saved PLY file
                - 'camera_range': dict with basic camera parameter ranges
                - 'default_camera': dict describing a default viewpoint
        """
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")

        images_data = self.operator.process_perception(input_)
        if not isinstance(images_data, list):
            images_data = [images_data]

        if size is None:
            if self.representation_model is not None:
                size = getattr(self.representation_model, 'size', 224)
            else:
                size = 224

        data = {
            'images': images_data,
            'output_type': 'all',
            'size': size,
            'vis_threshold': vis_threshold,
        }

        results = self.representation_model.get_representation(data)

        point_clouds = results.get('point_cloud', None)
        colors = results.get('colors', None)
        if not point_clouds or not colors:
            raise RuntimeError("CUT3R representation did not return point clouds and colors.")

        pcs_flat = []
        colors_flat = []
        for pc, color in zip(point_clouds, colors):
            pc_arr = pc.reshape(-1, 3)
            color_arr = color.reshape(-1, 3)
            if pc_arr.shape[0] != color_arr.shape[0]:
                n = min(pc_arr.shape[0], color_arr.shape[0])
                pc_arr = pc_arr[:n]
                color_arr = color_arr[:n]
            pcs_flat.append(pc_arr)
            colors_flat.append(color_arr)

        all_points = np.concatenate(pcs_flat, axis=0)
        all_colors = np.concatenate(colors_flat, axis=0)

        if all_points.size == 0:
            raise RuntimeError("Empty point cloud reconstructed from CUT3R.")

        if ply_path is None:
            output_dir = "./cut3r_output"
            os.makedirs(output_dir, exist_ok=True)
            ply_path = os.path.join(output_dir, "pointcloud.ply")
        else:
            if not ply_path.endswith(".ply"):
                os.makedirs(ply_path, exist_ok=True)
                ply_path = os.path.join(ply_path, "pointcloud.ply")
            else:
                os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)

        rgb_uint8 = (np.clip(all_colors, 0.0, 1.0) * 255).astype(np.uint8)
        storePly(ply_path, all_points.astype(np.float32), rgb_uint8)

        center = all_points.mean(axis=0)
        dists = np.linalg.norm(all_points - center[None, :], axis=1)
        radius = float(dists.max() + 1e-6)

        radius_min = max(radius * 0.2, 1e-3)
        radius_max = radius * 4.0

        camera_range = {
            "center": center.tolist(),
            "radius_min": radius_min,
            "radius_max": radius_max,
            "yaw_min": -180.0,
            "yaw_max": 180.0,
            "pitch_min": -75.0,
            "pitch_max": 75.0,
        }

        default_camera = {
            "center": center.tolist(),
            "radius": radius * 1.5,
            "yaw": 0.0,
            "pitch": 0.0,
        }

        return {
            "ply_path": ply_path,
            "camera_range": camera_range,
            "default_camera": default_camera,
        }

    @staticmethod
    def _normalize_interaction_sequence(
        interaction: Optional[Union[str, List[str]]]
    ) -> List[str]:
        """
        Normalize interaction input to a flat list of strings.
        Supports None, single string, or list of strings.
        """
        if interaction is None:
            return []
        if isinstance(interaction, str):
            return [interaction]
        return [str(sig) for sig in interaction if str(sig).strip()]

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
        image_width: int = 640,
        image_height: int = 352,
        device: Optional[str] = None,
        near_plane: float = 0.01,
        far_plane: float = 1000.0,
    ) -> Image.Image:
        """
        Thin wrapper: delegate 3DGS rendering to CUT3RRepresentation.
        """
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")
        return self.representation_model.render_with_3dgs(
            ply_path=ply_path,
            camera_config=camera_config,
            image_width=image_width,
            image_height=image_height,
            device=device,
            near_plane=near_plane,
            far_plane=far_plane,
        )
    
    def render_orbit_video_with_3dgs(
        self,
        ply_path: str,
        base_camera_config: Dict[str, Any],
        num_frames: int = 16,
        yaw_step: float = 5.0,
        image_width: int = 640,
        image_height: int = 352,
        fps: int = 12,
        output_path: Optional[str] = None,
    ) -> List[Image.Image]:
        """
        Convenience helper: render a short orbit video around the scene using 3DGS.

        Args:
            ply_path: Path to the reconstructed point cloud PLY.
            base_camera_config: Base camera configuration dict with keys:
                - 'center': list of 3 floats, scene center
                - 'radius': float, camera distance to center
                - 'yaw': float, starting yaw angle in degrees
                - 'pitch': float, pitch angle in degrees
            num_frames: Number of frames to render.
            yaw_step: Delta yaw (in degrees) added per frame.
            image_width: Output frame width.
            image_height: Output frame height.
            fps: Frames per second for the exported video.
            output_path: If provided, export an MP4 video to this path.

        Returns:
            List of PIL.Image frames.
        """
        frames: List[Image.Image] = []

        center = base_camera_config.get("center")
        radius = float(base_camera_config.get("radius", 4.0))
        base_yaw = float(base_camera_config.get("yaw", 0.0))
        pitch = float(base_camera_config.get("pitch", 0.0))

        for i in range(num_frames):
            camera_config = {
                "center": center,
                "radius": radius,
                "yaw": base_yaw + i * yaw_step,
                "pitch": pitch,
            }
            img = self.render_with_3dgs(
                ply_path=ply_path,
                camera_config=camera_config,
                image_width=image_width,
                image_height=image_height,
            )
            frames.append(img)

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            export_to_video([np.array(f) for f in frames], output_path, fps=fps)

        return frames

    @staticmethod
    def _apply_interaction_to_camera(
        camera_cfg: Dict[str, Any],
        interaction: str,
        camera_range: Dict[str, Any],
        yaw_step: float = 22.0,
        pitch_step: float = 15.0,
        zoom_factor: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Update a simple (radius, yaw, pitch) camera configuration according to a
        high-level interaction signal, clamped by camera_range.
        Only supports the unified 3D interaction schema
        (forward/backward/left/right, forward_left, camera_l, camera_zoom_in, ...).
        """
        yaw = float(camera_cfg.get("yaw", 0.0))
        pitch = float(camera_cfg.get("pitch", 0.0))
        radius = float(camera_cfg.get("radius", 4.0))
        sig = interaction.strip().lower()

        # Yaw (left/right)
        if sig in ["left", "camera_l"]:
            yaw -= yaw_step
        elif sig in ["right", "camera_r"]:
            yaw += yaw_step
        elif sig == "camera_ul":
            yaw -= yaw_step
            pitch += pitch_step
        elif sig == "camera_ur":
            yaw += yaw_step
            pitch += pitch_step
        elif sig == "camera_dl":
            yaw -= yaw_step
            pitch -= pitch_step
        elif sig == "camera_dr":
            yaw += yaw_step
            pitch -= pitch_step
        # Pitch (up/down)
        elif sig == "camera_up":
            pitch += pitch_step
        elif sig == "camera_down":
            pitch -= pitch_step
        # Radius (forward/backward, zoom)
        elif sig in ["forward", "camera_zoom_in"]:
            radius *= zoom_factor
        elif sig in ["backward", "camera_zoom_out"]:
            radius /= zoom_factor
        elif sig == "forward_left":
            yaw -= yaw_step
            radius *= zoom_factor
        elif sig == "forward_right":
            yaw += yaw_step
            radius *= zoom_factor
        elif sig == "backward_left":
            yaw -= yaw_step
            radius /= zoom_factor
        elif sig == "backward_right":
            yaw += yaw_step
            radius /= zoom_factor

        yaw = max(camera_range["yaw_min"], min(camera_range["yaw_max"], yaw))
        pitch = max(camera_range["pitch_min"], min(camera_range["pitch_max"], pitch))
        radius = max(camera_range["radius_min"], min(camera_range["radius_max"], radius))

        camera_cfg["yaw"] = yaw
        camera_cfg["pitch"] = pitch
        camera_cfg["radius"] = radius

        return camera_cfg

    def render_interaction_video_with_3dgs(
        self,
        ply_path: str,
        camera_range: Dict[str, Any],
        base_camera_config: Dict[str, Any],
        interaction_sequence: List[str],
        image_width: int = 640,
        image_height: int = 352,
        fps: int = 12,
        output_path: Optional[str] = None,
    ) -> List[Image.Image]:
        """
        Render a 3DGS video by applying a sequence of high-level interaction
        signals (e.g. ['move_left', 'zoom_in']) to the camera.

        This is the natural two-stage workflow:
        1) Reconstruct PLY and camera_range with ``reconstruct_ply``.
        2) Call this method with the resulting ``camera_range`` and a base
           camera configuration.
        """
        frames: List[Image.Image] = []

        camera_cfg: Dict[str, Any] = {
            "center": base_camera_config.get("center", camera_range["center"]),
            "radius": float(base_camera_config.get("radius", 4.0)),
            "yaw": float(base_camera_config.get("yaw", 0.0)),
            "pitch": float(base_camera_config.get("pitch", 0.0)),
        }

        for sig in interaction_sequence:
            camera_cfg = self._apply_interaction_to_camera(
                camera_cfg,
                sig,
                camera_range,
            )
            img = self.render_with_3dgs(
                ply_path=ply_path,
                camera_config=camera_cfg,
                image_width=image_width,
                image_height=image_height,
            )
            frames.append(img)

        if output_path is not None and len(frames) > 0:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            export_to_video([np.array(f) for f in frames], output_path, fps=fps)

        return frames

    def run_two_stage_3dgs_video(
        self,
        image_path: Union[str, Image.Image, np.ndarray, List[str], List[Image.Image], List[np.ndarray]],
        interactions: Optional[Union[str, List[str]]] = None,
        frames_per_interaction: int = 10,
        size: Optional[int] = None,
        vis_threshold: float = 1.5,
        output_dir: str = "./cut3r_output",
        camera_radius: float = 4.0,
        camera_yaw: float = 0.0,
        camera_pitch: float = 0.0,
        image_width: int = 704,
        image_height: int = 480,
        output_name: str = "cut3r_3dgs_demo.mp4",
    ) -> str:
        """
        High-level helper for the complete two-stage workflow:

        1) Reconstruct point cloud and camera range from input data.
        2) Use either a default orbit or an interaction sequence to render
           a 3DGS video.

        Args:
            data_path: Input image(s) or path(s), same formats as ``process``.
            interaction: None for default orbit, or a list of interaction
                strings such as ['move_left', 'zoom_in'].
            size: Optional CUT3R input size.
            vis_threshold: Confidence threshold for point cloud filtering.
            output_dir: Directory to store PLY and video.
            camera_radius: Initial camera radius.
            camera_yaw: Initial camera yaw angle (degrees).
            camera_pitch: Initial camera pitch angle (degrees).
            image_width: Rendered frame width.
            image_height: Rendered frame height.
            output_name: Name of the output MP4 file.

        Returns:
            Absolute or relative path to the rendered MP4 video.
        """
        os.makedirs(output_dir, exist_ok=True)

        recon_info = self.reconstruct_ply(
            image_path,
            ply_path=output_dir,
            size=size,
            vis_threshold=vis_threshold,
        )

        ply_path = recon_info["ply_path"]
        camera_range = recon_info["camera_range"]

        base_camera_config: Dict[str, Any] = {
            "center": camera_range["center"],
            "radius": camera_radius,
            "yaw": camera_yaw,
            "pitch": camera_pitch,
        }

        output_video_path = os.path.join(output_dir, output_name)

        interaction_sequence = self._normalize_interaction_sequence(interactions)
        if interaction_sequence and frames_per_interaction > 1:
            interaction_sequence = [
                a for a in interaction_sequence for _ in range(frames_per_interaction)
            ]

        if interaction_sequence:
            self.render_interaction_video_with_3dgs(
                ply_path=ply_path,
                camera_range=camera_range,
                base_camera_config=base_camera_config,
                interaction_sequence=interaction_sequence,
                image_width=image_width,
                image_height=image_height,
                output_path=output_video_path,
            )
        else:
            self.render_orbit_video_with_3dgs(
                ply_path=ply_path,
                base_camera_config=base_camera_config,
                image_width=image_width,
                image_height=image_height,
                output_path=output_video_path,
            )

        return output_video_path
    
    def __call__(
        self,
        image_path: Optional[Union[str, List[str]]] = None,
        images: Any = None,
        interactions: Optional[Union[str, List[str]]] = None,
        task_type: Optional[str] = None,
        **kwargs,
    ) -> Union[CUT3RResult, str]:
        """
        Main call interface.
        - Base mode (task_type is None or "cut3r_base"): direct CUT3R representation/process.
        - "cut3r_two_stage_3dgs": two-stage reconstruction + 3DGS video (returns output_video_path: str).
        """
        data = images if images is not None else image_path
        if data is None:
            raise ValueError("Provide image_path or images.")

        if task_type == "cut3r_two_stage_3dgs":
            return self.run_two_stage_3dgs_video(
                image_path=data,
                interactions=interactions,
                **kwargs,
            )

        return self.process(
            input_=data,
            interaction=interactions,
            **kwargs,
        )
    
    def stream(
        self,
        input_: Union[str, Image.Image, np.ndarray, List[str], List[Image.Image], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[Union[torch.Tensor, List[str]], None, None]:
        """
        Stream processing interface for real-time interactive updates.
        
        Args:
            input_: Input image(s)
            interaction: Interaction string or dictionary
            **kwargs: Additional arguments
            
        Yields:
            Processed results as torch.Tensor or List[str] (for compatibility with diffusers-style streaming)
        """
        # For CUT3R, streaming is equivalent to regular processing
        # since inference is typically fast and not iterative
        result = self.process(input_, interaction, **kwargs)
        
        # Yield images as tensors for streaming compatibility
        for img in result.images:
            # Convert PIL Image to tensor
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            yield img_tensor


__all__ = ["CUT3RPipeline", "CUT3RResult"]

