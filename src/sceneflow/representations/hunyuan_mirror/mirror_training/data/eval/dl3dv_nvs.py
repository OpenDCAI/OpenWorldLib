import cv2
import numpy as np
from mirror_training.data.multiview_dataset import MultiViewDataset
import os
import json
import torch
from pathlib import Path
from PIL import Image
import PIL
import numpy as np
from einops import rearrange
from PIL import Image


def rescale(
    image: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[
    torch.Tensor,  # updated images
    torch.Tensor,  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[
    torch.Tensor,  # updated images
    torch.Tensor,  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)


def rescale(
    image: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[
    torch.Tensor,  # updated images
    torch.Tensor,  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[
    torch.Tensor,  # updated images
    torch.Tensor,  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    # assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)


def load_from_json(filename):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def load_metadata(example_path: Path) -> dict:
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    key = str(example_path).split("/")[-2]
    with open(example_path, "r") as f:
        meta_data = json.load(f)

    store_h, store_w = meta_data["h"], meta_data["w"]
    fx, fy, cx, cy = (
        meta_data["fl_x"],
        meta_data["fl_y"],
        meta_data["cx"],
        meta_data["cy"],
    )
    saved_fx = float(fx) / float(store_w)
    saved_fy = float(fy) / float(store_h)
    saved_cx = float(cx) / float(store_w)
    saved_cy = float(cy) / float(store_h)

    timestamps = []
    cameras = []
    opencv_c2ws = []  # will be used to calculate camera distance

    for frame in meta_data["frames"]:
        timestamps.append(
            int(os.path.basename(frame["file_path"]).split(".")[0].split("_")[-1])
        )
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]
        # transform_matrix is in blender c2w, while we need to store opencv w2c matrix here
        opencv_c2w = np.array(frame["transform_matrix"]) @ blender2opencv
        opencv_c2ws.append(opencv_c2w)
        camera.extend(np.linalg.inv(opencv_c2w)[:3].flatten().tolist())
        cameras.append(np.array(camera))

    # timestamp should be the one that match the above images keys, use for indexing
    timestamps = np.array(timestamps, dtype=np.int64)
    cameras = np.stack(cameras)

    return {"key": key, "timestamps": timestamps, "cameras": cameras}


class DL3DV_NVStest(MultiViewDataset):
    def __init__(
        self,
        *args,
        split,
        ROOT,
        meta="training/assets/nvs-index-maps/DL3DV.json",
        num_views=8,
        **kwargs
    ):
        self.ROOT = ROOT
        ROOT = Path(ROOT)
        self.chunks = []
        self.index_json = load_from_json(meta)
        self.root = ROOT / "10K"
        self.available_scenes = []
        for current_scene, chunk_gt in self.index_json.items():
            if chunk_gt is None:
                continue
            assert os.path.isdir(self.root / current_scene)
            self.available_scenes.append(current_scene)

        super().__init__(*args, **kwargs)
        self.split = split
        self.rendering = True
        self.num_views = num_views

    def __len__(self):
        return len(self.available_scenes)

    @staticmethod
    def image_read(image_file):
        img = cv2.imread(image_file)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _fetch_views(self, idx, resolution, *args, **kwargs):
        idx = idx % len(self.available_scenes)
        current_scene = self.available_scenes[idx]
        chunk_gt = self.index_json[current_scene]
        file_path = current_scene
        chunk_path = self.root / file_path

        meta_path = chunk_path / "transforms.json"
        metadata = load_metadata(meta_path)
        imageroot = chunk_path / "images_8"
        imagepaths = {
            int(os.path.basename(path).split(".")[0].split("_")[-1]): imageroot / path
            for path in os.listdir(imageroot)
        }
        metadata["imagepaths"] = [
            imagepaths[timestamp.item()] for timestamp in metadata["timestamps"]
        ]

        index_list = list(chunk_gt["input"])
        index_target = list(chunk_gt["target"])
        is_target = [False] * len(index_list) + [True] * len(index_target)

        translate_content = []
        for index in index_list:
            poses = metadata["cameras"][index]
            w2c = np.eye(4)
            w2c[:3] = poses[6:].reshape(3, 4)
            camera_pose = np.linalg.inv(w2c)
            translate_content.append(np.linalg.norm(camera_pose[:3, 3]))
        scale = np.stack(translate_content).mean()

        views = []
        for i, index in enumerate(index_list + index_target):
            poses = metadata["cameras"][index]
            intrinsics = np.eye(3)
            fx, fy, cx, cy = poses[:4]
            intrinsics[0, 0] = fx
            intrinsics[1, 1] = fy
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy
            # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
            w2c = np.eye(4)
            w2c[:3] = poses[6:].reshape(3, 4)
            camera_pose = np.linalg.inv(w2c)
            camera_pose[:3, 3] = camera_pose[:3, 3] / scale

            scene = metadata["key"]
            framepath = metadata["imagepaths"][index]
            frame = self.image_read(framepath)
            frame = np.asarray(frame)
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            intrinsics = torch.tensor(intrinsics)
            images, intrinsics = rescale_and_crop(
                frame, intrinsics, (resolution[1], resolution[0])
            )
            images = images.permute(1, 2, 0).numpy() * 255
            H, W = images.shape[:2]
            images = PIL.Image.fromarray(images.astype(np.uint8))
            intrinsics[0, 0] = intrinsics[0, 0] * W
            intrinsics[1, 1] = intrinsics[1, 1] * H
            intrinsics[0, 2] = intrinsics[0, 2] * W
            intrinsics[1, 2] = intrinsics[1, 2] * H
            intrinsics = intrinsics.numpy()
            views.append(
                dict(
                    img=images,
                    camera_poses=camera_pose.astype(np.float32),
                    camera_intrs=intrinsics.astype(np.float32),
                    dataset="dl3dv_nvs_test",
                    label=file_path,
                    instance=framepath.name,
                    is_target=is_target[i],
                    nvs_sample=False,
                    scale_norm=False,
                )
            )

        return views
