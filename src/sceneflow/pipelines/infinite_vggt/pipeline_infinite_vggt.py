"""
InfiniteVGGT pipeline: operators + representation / reasoning / synthesis.
Output is saveable (PIL.Image, video frame list); pipeline does not save — user saves in test.
"""

import os
from typing import List, Optional, Union, Dict, Any, Generator, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import numpy as np
import cv2
from PIL import Image

from ...operators.infinite_vggt_operator import InfiniteVGGTOperator
from ...representations.point_clouds_generation.vggt.infinite_vggt_representation import (
    InfiniteVGGTRepresentation,
)


class InfiniteVGGTResult:
    """
    Pipeline output: saveable objects (PIL.Image, video frame list, point cloud data).
    User can result.depth_images[i].save(...), or result.save(output_dir); pipeline never saves.
    """

    def __init__(
        self,
        point_map: np.ndarray,
        depth_map: np.ndarray,
        colors: List[np.ndarray],
        extrinsic: np.ndarray,
        intrinsic: Optional[np.ndarray],
        point_conf: np.ndarray,
        output_format: str = "ply",
        depth_images: Optional[List[Image.Image]] = None,
        video_frames: Optional[List[Union[np.ndarray, Image.Image]]] = None,
    ):
        self.point_map = point_map
        self.depth_map = depth_map
        self.colors = colors
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.point_conf = point_conf
        self.output_format = output_format
        self.depth_images = depth_images or []
        self.video_frames = video_frames or []

    def save(self, output_dir: Optional[str] = None) -> List[str]:
        """Export to ply, glb, or depth images. User calls this; pipeline does not save."""
        if output_dir is None:
            output_dir = "./infinite_vggt_output"
        os.makedirs(output_dir, exist_ok=True)
        saved: List[str] = []

        fmt = self.output_format.lower()
        world_points = self.point_map
        conf = self.point_conf
        N, H, W, _ = world_points.shape
        pts_flat = world_points.reshape(-1, 3)
        rgb_flat = np.concatenate([self.colors[i].reshape(-1, 3) for i in range(N)], axis=0)
        conf_flat = conf.reshape(-1)
        valid = np.isfinite(pts_flat).all(axis=1) & (conf_flat >= 0)
        pts_flat = pts_flat[valid]
        rgb_flat = np.clip(rgb_flat[valid], 0, 1)

        if fmt == "ply":
            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_flat.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(rgb_flat.astype(np.float64))
                out_path = os.path.join(output_dir, "pointcloud.ply")
                o3d.io.write_point_cloud(out_path, pcd)
                saved.append(out_path)
            except ImportError:
                out_path = os.path.join(output_dir, "pointcloud.ply")
                with open(out_path, "w") as f:
                    f.write("ply\nformat ascii 1.0\n")
                    f.write(f"element vertex {len(pts_flat)}\n")
                    f.write("property float x\nproperty float y\nproperty float z\n")
                    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                    f.write("end_header\n")
                    for i in range(len(pts_flat)):
                        r, g, b = (np.clip(rgb_flat[i], 0, 1) * 255).astype(np.uint8)
                        f.write(f"{pts_flat[i, 0]:f} {pts_flat[i, 1]:f} {pts_flat[i, 2]:f} {r} {g} {b}\n")
                saved.append(out_path)
        elif fmt == "glb":
            try:
                import trimesh
                pcd = trimesh.PointCloud(pts_flat, colors=np.clip(rgb_flat, 0, 1))
                out_path = os.path.join(output_dir, "scene.glb")
                pcd.export(out_path)
                saved.append(out_path)
            except ImportError:
                out_path = os.path.join(output_dir, "pointcloud.ply")
                with open(out_path, "w") as f:
                    f.write("ply\nformat ascii 1.0\n")
                    f.write(f"element vertex {len(pts_flat)}\n")
                    f.write("property float x\nproperty float y\nproperty float z\n")
                    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                    f.write("end_header\n")
                    for i in range(len(pts_flat)):
                        r, g, b = (np.clip(rgb_flat[i], 0, 1) * 255).astype(np.uint8)
                        f.write(f"{pts_flat[i, 0]:f} {pts_flat[i, 1]:f} {pts_flat[i, 2]:f} {r} {g} {b}\n")
                saved.append(out_path)
        elif fmt == "depth" or fmt == "depth_map":
            for i in range(N):
                d = self.depth_map[i] if self.depth_map.ndim == 3 else self.depth_map[i].squeeze()
                d_vis = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-8)
                d_vis = (d_vis * 255).astype(np.uint8)
                d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
                out_path = os.path.join(output_dir, f"depth_{i:04d}.png")
                cv2.imwrite(out_path, d_color)
                saved.append(out_path)

        return saved


class InfiniteVGGTPipeline:
    """
    Pipeline: operators + representation / reasoning / synthesis.
    Output is PIL.Image or video frame list (saveable); pipeline does not save.
    """

    def __init__(
        self,
        representation_model: Optional[InfiniteVGGTRepresentation] = None,
        reasoning_model: Optional[Any] = None,
        synthesis_model: Optional[Any] = None,
        operator: Optional[InfiniteVGGTOperator] = None,
    ):
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.operator = operator or InfiniteVGGTOperator()

    @classmethod
    def from_pretrained(
        cls,
        representation_path: Optional[str] = None,
        reasoning_path: Optional[str] = None,
        synthesis_path: Optional[str] = None,
        pretrained_model_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "InfiniteVGGTPipeline":
        """
        在这里对 representation, reasoning, synthesis 模型进行权重加载。
        Returns cls(representation_model, reasoning_model, synthesis_model).
        """
        rep_path = representation_path or pretrained_model_path
        representation_model = None
        if rep_path:
            representation_model = InfiniteVGGTRepresentation.from_pretrained(
                pretrained_model_path=rep_path,
                device=device,
                **kwargs,
            )
        reasoning_model = None
        if reasoning_path:
            # TODO: load reasoning model when implemented
            pass
        synthesis_model = None
        if synthesis_path:
            # TODO: load synthesis model when implemented
            pass
        return cls(
            representation_model=representation_model,
            reasoning_model=reasoning_model,
            synthesis_model=synthesis_model,
        )

    def process(
        self,
        input_: Union[str, np.ndarray, List[str]],
        interaction: Optional[Dict[str, Any]] = None,
    ) -> InfiniteVGGTResult:
        """
        input_: 图片、视频等输入 (image path, video path, image dir, or list of paths).
        interaction: from operator.process_interaction().
        Returns result with PIL.Image / video frame list (no save).
        """
        if self.representation_model is None:
            raise RuntimeError("Representation not loaded. Use from_pretrained() first.")

        perceived = self.operator.process_perception(input_)
        if isinstance(perceived, list) and perceived and isinstance(perceived[0], str):
            image_list = perceived
        elif isinstance(perceived, np.ndarray):
            image_list = [perceived]
        else:
            image_list = list(perceived)

        data = {"images": image_list}
        rep = self.representation_model.get_representation(data)

        output_format = "ply"
        if interaction:
            output_format = interaction.get("output_format", "ply")

        # Build saveable outputs: PIL.Image list for depth, frame list for video
        depth_images: List[Image.Image] = []
        depth_map = rep["depth_map"]
        N = depth_map.shape[0] if getattr(depth_map, "shape", None) is not None else len(rep["colors"])
        for i in range(N):
            d = depth_map[i] if depth_map.ndim >= 3 else depth_map
            d = np.asarray(d).squeeze()
            d_vis = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-8)
            d_vis = (np.clip(d_vis, 0, 1) * 255).astype(np.uint8)
            d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
            depth_images.append(Image.fromarray(cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)))
        video_frames = [Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8)) for c in rep["colors"]]

        return InfiniteVGGTResult(
            point_map=rep["point_map"],
            depth_map=rep["depth_map"],
            colors=rep["colors"],
            extrinsic=rep["extrinsic"],
            intrinsic=rep.get("intrinsic"),
            point_conf=rep["point_conf"],
            output_format=output_format,
            depth_images=depth_images,
            video_frames=video_frames,
        )

    def __call__(
        self,
        data_path: str,
        interaction: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> InfiniteVGGTResult:
        """
        Pipeline 调用入口。不在此处 save；返回可保存对象 (PIL.Image, video frame list)，
        由用户在 test 中保存或后续处理。
        """
        if interaction is None and self.operator.current_interaction:
            interaction = self.operator.process_interaction()
        elif interaction is None:
            interaction = {"output_format": "ply"}
        return self.process(data_path, interaction=interaction)

    def stream(
        self,
        *args,
        **kwds,
    ) -> "Generator[Any, List[str], None]":
        """
        Stream pipeline outputs: yields (tensor, frame_ids) for incremental processing.
        Yield type: tensor (e.g. point_map or depth); Send type: List[str] (frame ids).
        """
        import torch
        input_ = args[0] if args else kwds.get("input_") or kwds.get("data_path")
        interaction = kwds.get("interaction")
        if not input_:
            return
        result = self.process(input_, interaction=interaction)
        frame_ids = [str(i) for i in range(len(result.video_frames))]
        yield torch.from_numpy(result.point_map), frame_ids


__all__ = ["InfiniteVGGTPipeline", "InfiniteVGGTResult"]
