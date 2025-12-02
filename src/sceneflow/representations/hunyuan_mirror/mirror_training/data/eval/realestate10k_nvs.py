import cv2
import numpy as np
from mirror_training.data.multiview_dataset import MultiViewDataset
import json
import torch
from io import BytesIO
from pathlib import Path
from PIL import Image
import PIL
import numpy as np
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor

def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")

def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
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
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
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
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")

def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
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
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
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

class Re10K_NVStest(MultiViewDataset):
    def __init__(self, *args, split, ROOT, meta='training/assets/nvs-index-maps/evaluation_index_re10k.json', testset_json='training/assets/nvs-index-maps/index.json', num_views=2, **kwargs):
        self.ROOT = ROOT
        ROOT = Path(ROOT)
        self.chunks = []
        self.index_json = load_from_json(meta)
        self.root =  ROOT / split
        self.test_path = load_from_json(testset_json)
        self.available_scenes = []
        for current_scene, chunk_gt in self.index_json.items():
            if chunk_gt is None:
                continue
            if 'overlap_tag' in chunk_gt:
                if current_scene in self.test_path.keys() and chunk_gt is not None and chunk_gt['overlap_tag'] != 'large':
                    self.available_scenes.append(current_scene)
            else:
                if current_scene in self.test_path.keys() and chunk_gt is not None:
                    self.available_scenes.append(current_scene)
        self.available_scenes = self.available_scenes[:200]

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
        file_path = self.test_path[current_scene]
        chunk_path = self.root/ file_path
        if 'overlap_tag' not in chunk_gt:
            index_list = list(chunk_gt['context'])
            index_target = list(chunk_gt['target'])
        else:
            index_list = list(chunk_gt['context_index'])
            index_target = list(chunk_gt['target_index'])

        chunk = torch.load(chunk_path)
        final_i = None
        for i in range(len(chunk)):
            if chunk[i]['key']==current_scene:
                final_i = i
        chunk = chunk[final_i]
        
        poses_right = chunk["cameras"][index_list[0]]
        w2c_right = np.eye(4)
        w2c_right[:3] = poses_right[6:].reshape(3, 4)
        camera_pose_right =  np.linalg.inv(w2c_right)
        poses_left = chunk["cameras"][index_list[1]]
        w2c_left = np.eye(4)
        w2c_left[:3] = poses_left[6:].reshape(3, 4)
        camera_pose_left =  np.linalg.inv(w2c_left)
        a, b = camera_pose_right[:3, 3], camera_pose_left[:3, 3]
        scale = np.linalg.norm(a - b)
        
        is_target = [False] * len(index_list) + [True] * len(index_target)
        index_list.extend(index_target)
        views = []
        for i,index in enumerate(index_list):
            poses = chunk["cameras"][index]
            intrinsics = np.eye(3)
            fx, fy, cx, cy = poses[:4]
            intrinsics[0, 0] = fx
            intrinsics[1, 1] = fy
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy
            # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
            w2c = np.eye(4)
            w2c[:3] = poses[6:].reshape(3, 4)
            camera_pose =  np.linalg.inv(w2c)
            camera_pose[:3, 3] = camera_pose[:3, 3] / scale
            
            scene = chunk["key"]
            frame = chunk["images"][index] 
            frame = Image.open(BytesIO(frame.numpy().tobytes())).convert('RGB')
            frame = np.asarray(frame)
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            intrinsics = torch.tensor(intrinsics)
            images, intrinsics = rescale_and_crop(frame, intrinsics, (resolution[1], resolution[0]))
            images = images.permute(1, 2, 0).numpy() * 255
            H, W = images.shape[:2]
            images = PIL.Image.fromarray(images.astype(np.uint8))
            intrinsics[0, 0] = intrinsics[0, 0] * W
            intrinsics[1, 1] = intrinsics[1, 1] * H
            intrinsics[0, 2] = intrinsics[0, 2] * W
            intrinsics[1, 2] = intrinsics[1, 2] * H
            intrinsics = intrinsics.numpy()
            views.append(dict(
                img=images,
                camera_poses=camera_pose.astype(np.float32),
                camera_intrs=intrinsics.astype(np.float32),
                dataset='re10k_nvs_test',
                label=scene,
                instance=f"{index:0>3}",
                nvs_sample=False,
                scale_norm=False,
                is_target=is_target[i]
            ))

        return views