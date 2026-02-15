import os
import json
from typing import List, Optional, Union, Dict, Any

import numpy as np
from PIL import Image

from ...operators.pi3_operator import Pi3Operator
from ...representations.point_clouds_generation.pi3.pi3_representation import (
    Pi3Representation,
)
from ...representations.point_clouds_generation.pi3x.pi3x_representation import (
    Pi3XRepresentation,
)


class Pi3Result:

    def __init__(
        self,
        depth_images: List[Image.Image],
        numpy_data: Dict[str, np.ndarray],
        camera_params: List[Dict[str, Any]],
        input_images: Optional[List[np.ndarray]] = None,
        data_type: str = "image",
    ):
        self.depth_images = depth_images
        self.numpy_data = numpy_data
        self.camera_params = camera_params
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

                points_b0 = self.numpy_data["points"][0]        # (N, H, W, 3)
                masks_b0 = self.numpy_data["masks"][0].astype(bool)
                colors = np.stack(self.input_images, axis=0)     # (N, H, W, 3)

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

        return saved_files


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

    @classmethod
    def from_pretrained(
        cls,
        representation_path: str,
        reasoning_path: Optional[str] = None,
        synthesis_path: Optional[str] = None,
        model_type: str = "pi3x",
        **kwargs,
    ) -> "Pi3Pipeline":

        if model_type == "pi3x":
            representation_model = Pi3XRepresentation.from_pretrained(
                pretrained_model_path=representation_path,
                **kwargs,
            )
        elif model_type == "pi3":
            representation_model = Pi3Representation.from_pretrained(
                pretrained_model_path=representation_path,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'pi3x' or 'pi3'.")
        return cls(
            representation_model=representation_model,
            reasoning_model=None,
            synthesis_model=None,
        )

    def process(
        self,
        input_: Union[str, np.ndarray, List[str], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> Pi3Result:

        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")

        # Step 1: Load and preprocess input
        interval = kwargs.get("interval", -1)
        images_data = self.operator.process_perception(input_, interval=interval)
        if not isinstance(images_data, list):
            images_data = [images_data]

        # Step 2: Process interaction flags
        if interaction is None:
            interaction_dict = {
                "predict_points": True,
                "predict_cameras": True,
                "predict_depth": True,
                "predict_conf": True,
                "use_conditions": False,
            }
        elif isinstance(interaction, str):
            self.operator.get_interaction(interaction)
            interaction_dict = self.operator.process_interaction()
        else:
            interaction_dict = interaction

        # Step 3: Build model input tensor
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

        # Optional conditions (Pi3X only)
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

        # Step 4: Run model
        results = self.representation_model.get_representation(data)

        # Step 5: Build depth visualizations (per-frame min-max normalization)
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

        # Step 6: Extract camera parameters
        camera_params = []
        cam_poses = results.get("camera_poses")
        if cam_poses is not None:
            for i in range(cam_poses[0].shape[0]):
                camera_params.append({
                    "camera_to_world": cam_poses[0][i].tolist(),
                })

        return Pi3Result(
            depth_images=depth_images,
            numpy_data=results,
            camera_params=camera_params,
            input_images=resized_images,
            data_type="image",
        )

    def __call__(
        self,
        input_: Union[str, np.ndarray, List[str], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> Pi3Result:
        """Main call interface for the pipeline."""
        return self.process(input_=input_, interaction=interaction, **kwargs)


__all__ = ["Pi3Pipeline", "Pi3Result"]
