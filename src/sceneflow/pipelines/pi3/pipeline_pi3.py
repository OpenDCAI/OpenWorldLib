import os
import json
import math
from typing import List, Optional, Union, Dict, Any, Generator

import numpy as np
from PIL import Image

from ...operators.pi3_operator import Pi3Operator
from ...representations.point_clouds_generation.pi3.pi3_representation import (
    Pi3Representation,
)
from ...representations.point_clouds_generation.pi3.pi3x_representation import (
    Pi3XRepresentation,
)


def _apply_camera_delta(c2w: np.ndarray, delta: List[float]) -> np.ndarray:
    """Apply a [dx,dy,dz,theta_x,theta_z] delta to a 4x4 camera-to-world matrix."""
    dx, dy, dz, theta_x, theta_z = delta
    result = c2w.copy()

    rx = np.eye(4)
    cx, sx = math.cos(theta_x), math.sin(theta_x)
    rx[1, 1], rx[1, 2] = cx, -sx
    rx[2, 1], rx[2, 2] = sx, cx

    rz = np.eye(4)
    cz, sz = math.cos(theta_z), math.sin(theta_z)
    rz[0, 0], rz[0, 1] = cz, -sz
    rz[1, 0], rz[1, 1] = sz, cz

    result = result @ rx @ rz
    result[0, 3] += dx
    result[1, 3] += dy
    result[2, 3] += dz
    return result


def render_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    camera_to_world: np.ndarray,
    height: int,
    width: int,
    focal_scale: float = 1.0,
    splat_radius: int = 3,
) -> Image.Image:
    """Render a point cloud with strict z-buffer and front-to-back splatting.

    Points are sorted front-to-back. Each point is splatted as a disk.
    Only the closest point at each pixel is kept (strict z-buffer),
    which eliminates ghosting/layering artifacts.
    """
    c2w = camera_to_world.astype(np.float64)
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]

    pts_cam = (R @ points.T).T + t
    valid = pts_cam[:, 2] > 1e-4
    pts_cam = pts_cam[valid]
    cols = colors[valid]
    if cols.dtype == np.float64 or cols.dtype == np.float32:
        if cols.max() <= 1.0:
            cols = (cols * 255).clip(0, 255).astype(np.uint8)
        else:
            cols = cols.clip(0, 255).astype(np.uint8)

    fx = fy = focal_scale * max(height, width)
    cx_img, cy_img = width / 2.0, height / 2.0

    u = np.round(fx * pts_cam[:, 0] / pts_cam[:, 2] + cx_img).astype(np.int32)
    v = np.round(fy * pts_cam[:, 1] / pts_cam[:, 2] + cy_img).astype(np.int32)
    z = pts_cam[:, 2].astype(np.float32)

    sort_idx = np.argsort(z)
    u, v, z, cols = u[sort_idx], v[sort_idx], z[sort_idx], cols[sort_idx]

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    z_buf = np.full((height, width), np.inf, dtype=np.float32)

    r = splat_radius
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            py = v + dy
            px = u + dx
            mask = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            px_m, py_m, z_m, cols_m = px[mask], py[mask], z[mask], cols[mask]
            closer = z_m < z_buf[py_m, px_m]
            px_c, py_c = px_m[closer], py_m[closer]
            z_buf[py_c, px_c] = z_m[closer]
            canvas[py_c, px_c] = cols_m[closer]

    return Image.fromarray(canvas)


class Pi3Result:

    def __init__(
        self,
        depth_images: List[Image.Image],
        numpy_data: Dict[str, np.ndarray],
        camera_params: List[Dict[str, Any]],
        camera_range: Dict[str, Any],
        input_images: Optional[List[np.ndarray]] = None,
        data_type: str = "image",
    ):
        self.depth_images = depth_images
        self.numpy_data = numpy_data
        self.camera_params = camera_params
        self.camera_range = camera_range
        self.input_images = input_images
        self.data_type = data_type

    def __len__(self):
        return len(self.depth_images)

    def __getitem__(self, idx):
        return {
            "depth_image": self.depth_images[idx] if idx < len(self.depth_images) else None,
            "camera_params": self.camera_params[idx] if idx < len(self.camera_params) else None,
        }

    def save(self, output_dir: Optional[str] = None) -> List[str]:
        if output_dir is None:
            output_dir = "./pi3_output"

        os.makedirs(output_dir, exist_ok=True)
        saved_files: List[str] = []

        # Point cloud (PLY)
        ply_dir = os.path.join(output_dir, "point_cloud")
        os.makedirs(ply_dir, exist_ok=True)
        if "points" in self.numpy_data and "masks" in self.numpy_data and self.input_images is not None:
            try:
                from plyfile import PlyData, PlyElement

                points_b0 = self.numpy_data["points"][0]
                masks_b0 = self.numpy_data["masks"][0].astype(bool)
                colors = np.stack(self.input_images, axis=0)

                pts = points_b0[masks_b0].astype(np.float32)
                col = (colors[masks_b0] * 255).clip(0, 255).astype(np.uint8)

                vertices = np.zeros(
                    pts.shape[0],
                    dtype=[
                        ("x", "f4"), ("y", "f4"), ("z", "f4"),
                        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                    ],
                )
                vertices["x"], vertices["y"], vertices["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
                vertices["nx"], vertices["ny"], vertices["nz"] = 0.0, 0.0, 0.0
                vertices["red"], vertices["green"], vertices["blue"] = col[:, 0], col[:, 1], col[:, 2]

                ply_path = os.path.join(ply_dir, "result.ply")
                PlyData([PlyElement.describe(vertices, "vertex")]).write(ply_path)
                saved_files.append(ply_path)
            except ImportError:
                pass

        # Raw numpy data
        raw_dir = os.path.join(output_dir, "raw_data")
        os.makedirs(raw_dir, exist_ok=True)
        for key, value in self.numpy_data.items():
            if isinstance(value, np.ndarray):
                npy_path = os.path.join(raw_dir, f"{key}.npy")
                np.save(npy_path, value)
                saved_files.append(npy_path)

        # Depth map visualizations
        depth_dir = os.path.join(output_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        for i, img in enumerate(self.depth_images):
            depth_path = os.path.join(depth_dir, f"depth_{i:04d}.png")
            img.save(depth_path)
            saved_files.append(depth_path)

        # Input RGB frames
        if self.input_images is not None and len(self.input_images) > 0:
            rgb_dir = os.path.join(output_dir, "rgb")
            os.makedirs(rgb_dir, exist_ok=True)
            for i, img_arr in enumerate(self.input_images):
                img_uint8 = (img_arr * 255).clip(0, 255).astype(np.uint8)
                rgb_path = os.path.join(rgb_dir, f"frame_{i:04d}.png")
                Image.fromarray(img_uint8).save(rgb_path)
                saved_files.append(rgb_path)

        # Camera poses
        poses_dir = os.path.join(output_dir, "camera_poses")
        os.makedirs(poses_dir, exist_ok=True)
        for i, cam in enumerate(self.camera_params):
            pose_path = os.path.join(poses_dir, f"pose_{i:04d}.json")
            with open(pose_path, "w") as f:
                json.dump(cam, f, indent=2)
            saved_files.append(pose_path)

        # meta.json with camera_range
        meta_path = os.path.join(output_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump({"camera_range": self.camera_range}, f, indent=2)
        saved_files.append(meta_path)

        return saved_files


def _build_camera_range(camera_params: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute camera parameter range from a list of camera_to_world matrices."""
    n = len(camera_params)
    if n == 0:
        return {}

    translations = np.array([np.array(c["camera_to_world"])[:3, 3] for c in camera_params])
    return {
        "available_view_indices": list(range(n)),
        "default_view_index": 0,
        "num_views": n,
        "translation_min": translations.min(axis=0).tolist(),
        "translation_max": translations.max(axis=0).tolist(),
    }


class Pi3Pipeline:

    def __init__(
        self,
        representation_model=None,
        reasoning_model: Optional[Any] = None,
        synthesis_model: Optional[Any] = None,
        operator: Optional[Pi3Operator] = None,
    ) -> None:
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.operator = operator or Pi3Operator()
        self._cached_result: Optional[Pi3Result] = None
        self._current_camera: Optional[np.ndarray] = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        required_components: Optional[Dict[str, str]] = None,
        mode: str = "pi3x",
        device: Optional[str] = None,
        weight_dtype: Optional[str] = None,
        representation_path: Optional[str] = None,
        model_type: Optional[str] = None,
        **kwargs,
    ) -> "Pi3Pipeline":
        path = model_path or representation_path
        if path is None:
            raise ValueError("model_path is required.")
        m = model_type or mode

        if m == "pi3x":
            representation_model = Pi3XRepresentation.from_pretrained(
                pretrained_model_path=path, device=device, **kwargs,
            )
        elif m == "pi3":
            representation_model = Pi3Representation.from_pretrained(
                pretrained_model_path=path, device=device, **kwargs,
            )
        else:
            raise ValueError(f"Unknown mode: {m}. Choose 'pi3x' or 'pi3'.")
        return cls(representation_model=representation_model)

    def _run_inference(
        self,
        images: Union[str, np.ndarray, List[str], List[np.ndarray]],
        **kwargs,
    ) -> Pi3Result:
        """Run Pi3 model inference (single-shot, produces all outputs)."""
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")

        interval = kwargs.get("interval", -1)
        images_data = self.operator.process_perception(images, interval=interval)
        if not isinstance(images_data, list):
            images_data = [images_data]

        device = self.representation_model.device
        imgs_tensor = self.operator.images_to_tensor(images_data, device=device)

        resized_images = [
            imgs_tensor[0, i].permute(1, 2, 0).cpu().numpy()
            for i in range(imgs_tensor.shape[1])
        ]

        data = {
            "images": imgs_tensor,
            "conf_threshold": kwargs.get("conf_threshold", 0.1),
            "edge_rtol": kwargs.get("edge_rtol", 0.03),
        }

        conditions_path = kwargs.get("conditions_path")
        if conditions_path is not None and os.path.exists(conditions_path):
            import torch as _torch
            cond_data = np.load(conditions_path, allow_pickle=True)
            if "poses" in cond_data:
                data["poses"] = _torch.from_numpy(cond_data["poses"]).float().unsqueeze(0)
            if "depths" in cond_data:
                data["depths"] = _torch.from_numpy(cond_data["depths"]).float().unsqueeze(0)
            if "intrinsics" in cond_data:
                data["intrinsics"] = _torch.from_numpy(cond_data["intrinsics"]).float().unsqueeze(0)

        results = self.representation_model.get_representation(data)

        depth_images = []
        depth_maps = results.get("depth_map")
        if depth_maps is not None:
            depth_b0 = depth_maps[0]
            if depth_b0.ndim == 2:
                depth_b0 = depth_b0[np.newaxis, ...]
            for i in range(depth_b0.shape[0]):
                d = depth_b0[i].astype(np.float64)
                d_min, d_max = d.min(), d.max()
                d_norm = (d - d_min) / (d_max - d_min + 1e-8)
                d_uint8 = (d_norm * 255).astype(np.uint8)
                depth_images.append(Image.fromarray(d_uint8, mode="L"))

        camera_params = []
        cam_poses = results.get("camera_poses")
        if cam_poses is not None:
            for i in range(cam_poses[0].shape[0]):
                camera_params.append({
                    "camera_to_world": cam_poses[0][i].tolist(),
                })

        camera_range = _build_camera_range(camera_params)

        result = Pi3Result(
            depth_images=depth_images,
            numpy_data=results,
            camera_params=camera_params,
            camera_range=camera_range,
            input_images=resized_images,
            data_type="image",
        )
        self._cached_result = result
        if camera_params:
            self._current_camera = np.array(camera_params[0]["camera_to_world"])
        return result

    def render_view(
        self,
        result: Optional["Pi3Result"] = None,
        view_index: Optional[int] = None,
        camera_to_world: Optional[np.ndarray] = None,
        camera_view: Optional[List[float]] = None,
    ) -> Image.Image:
        """Render a view from cached point cloud (no model inference).

        Specify one of:
          - view_index: index into result.camera_params
          - camera_to_world: explicit 4x4 matrix
          - camera_view: [dx,dy,dz,theta_x,theta_z] delta from default camera
        """
        res = result or self._cached_result
        if res is None:
            raise RuntimeError("No result available. Run inference first via __call__().")

        if "points" not in res.numpy_data or "masks" not in res.numpy_data:
            raise RuntimeError("Result does not contain point cloud data.")

        pts_all = res.numpy_data["points"][0]
        masks = res.numpy_data["masks"][0].astype(bool)
        colors_all = np.stack(res.input_images, axis=0) if res.input_images else None
        if colors_all is None:
            raise RuntimeError("No input images in result for coloring.")

        pts = pts_all[masks].astype(np.float64)
        cols = (colors_all[masks] * 255).clip(0, 255).astype(np.uint8)

        if camera_to_world is not None:
            c2w = np.array(camera_to_world, dtype=np.float64)
        elif view_index is not None:
            c2w = np.array(res.camera_params[view_index]["camera_to_world"], dtype=np.float64)
        elif camera_view is not None:
            base = np.array(res.camera_params[0]["camera_to_world"], dtype=np.float64)
            c2w = _apply_camera_delta(base, camera_view)
        else:
            c2w = np.array(res.camera_params[0]["camera_to_world"], dtype=np.float64)

        h = pts_all.shape[1] if pts_all.ndim >= 3 else 480
        w = pts_all.shape[2] if pts_all.ndim >= 4 else 640
        if pts_all.ndim >= 4:
            h, w = pts_all.shape[1], pts_all.shape[2]

        return render_point_cloud(pts, cols, c2w, h, w)

    def _render_trajectory(self, n_interp: int = 15, fps: int = 15, **kwargs) -> List[Image.Image]:
        """Render a trajectory video by interpolating between original camera poses.
        Returns a list of PIL.Image frames.
        """
        res = self._cached_result
        if res is None:
            raise RuntimeError("No result available. Run reconstruction first.")

        pts_all = res.numpy_data["points"][0]
        masks = res.numpy_data["masks"][0].astype(bool)
        colors_all = np.stack(res.input_images, axis=0)
        pts = pts_all[masks].astype(np.float64)
        cols = (colors_all[masks] * 255).clip(0, 255).astype(np.uint8)
        h, w = pts_all.shape[1], pts_all.shape[2]

        c2ws = [np.array(c["camera_to_world"], dtype=np.float64) for c in res.camera_params]
        frames = []
        for vi in range(len(c2ws) - 1):
            for j in range(n_interp):
                t = j / n_interp
                c2w = c2ws[vi] * (1 - t) + c2ws[vi + 1] * t
                frames.append(render_point_cloud(pts, cols, c2w, h, w))
        frames.append(render_point_cloud(pts, cols, c2ws[-1], h, w))
        return frames

    def __call__(
        self,
        images: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
        videos: Optional[Union[str, List[str]]] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        task_type: str = "reconstruction",
        interactions: Optional[List[str]] = None,
        camera_view: Optional[List[float]] = None,
        view_index: Optional[int] = None,
        visualize_ops: bool = True,
        **kwargs,
    ):
        """Unified call interface.

        Args:
            images: Image input path/list/tensor/array.
            videos: Video input path/list. If provided, takes precedence over images.
            image_path: Alias for a single image path.
            video_path: Alias for a single video path.
            task_type: One of "reconstruction", "render_view", "render_trajectory".
            interactions: Navigation signals like ["forward", "left", "camera_r"].
                When provided with task_type="render_view", applies interactions
                then renders the resulting viewpoint.
            camera_view: [dx,dy,dz,theta_x,theta_z] camera delta for render_view.
            view_index: Camera index for render_view.
            visualize_ops: Whether to generate visualizations.

        Returns:
            - task_type="reconstruction": Pi3Result
            - task_type="render_view": PIL.Image
            - task_type="render_trajectory": List[PIL.Image]
        """
        visual_input = videos or video_path or images or image_path

        if task_type == "reconstruction" and visual_input is None and interactions is not None:
            task_type = "render_view"

        if task_type == "reconstruction":
            if visual_input is None:
                raise ValueError("images is required for task_type='reconstruction'.")
            return self._run_inference(visual_input, **kwargs)

        elif task_type == "render_view":
            if interactions is not None:
                for sig in interactions:
                    self.stream(interaction_signal=sig)
                return self.render_view(camera_to_world=self._current_camera)
            return self.render_view(
                view_index=view_index, camera_view=camera_view, **kwargs,
            )

        elif task_type == "render_trajectory":
            return self._render_trajectory(**kwargs)

        else:
            raise ValueError(
                f"Unknown task_type: {task_type}. "
                "Choose 'reconstruction', 'render_view', or 'render_trajectory'."
            )

    def stream(
        self,
        interaction_signal: Union[str, List[str]],
        result: Optional[Pi3Result] = None,
        **kwargs,
    ) -> Image.Image:
        """Interactive rendering: apply navigation interaction to current camera and render.

        This does NOT run model inference. It uses the cached point cloud from a
        previous __call__() and applies the interaction delta to move the camera.
        """
        res = result or self._cached_result
        if res is None:
            raise RuntimeError("No result available. Run reconstruction first via __call__().")

        if self._current_camera is None:
            if res.camera_params:
                self._current_camera = np.array(res.camera_params[0]["camera_to_world"])
            else:
                raise RuntimeError("No camera parameters available.")

        if isinstance(interaction_signal, str):
            interaction_signal = [interaction_signal]

        self.operator.get_interaction(interaction_signal)
        delta = self.operator.process_interaction()

        self._current_camera = _apply_camera_delta(self._current_camera, delta)

        return self.render_view(result=res, camera_to_world=self._current_camera)


__all__ = ["Pi3Pipeline", "Pi3Result"]
