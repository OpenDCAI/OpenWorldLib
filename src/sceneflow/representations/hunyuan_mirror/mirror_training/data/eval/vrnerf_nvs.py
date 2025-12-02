import os.path as osp
import numpy as np
from mirror_training.data.multiview_dataset import MultiViewDataset
import os
import torch
from PIL import Image
import PIL
import numpy as np
from einops import rearrange
from PIL import Image
import imageio


def _rescale_single(image_chw: torch.Tensor, shape_hw: tuple[int, int]) -> torch.Tensor:
    h, w = shape_hw
    image_new = (image_chw * 255).clip(min=0, max=255).to(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255.0
    image_new = torch.tensor(image_new, dtype=image_chw.dtype, device=image_chw.device)
    return rearrange(image_new, "h w c -> c h w")

def _center_crop(images: torch.Tensor, intrinsics: torch.Tensor, shape_hw: tuple[int, int]):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape_hw
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2
    images = images[..., :, row:row + h_out, col:col + w_out]
    K = intrinsics.clone()
    K[..., 0, 0] *= w_in / w_out  # fx
    K[..., 1, 1] *= h_in / h_out  # fy
    return images, K

def _rescale_and_crop(images: torch.Tensor, intrinsics: torch.Tensor, shape_hw: tuple[int, int]):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape_hw
    scale = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale)
    w_scaled = round(w_in * scale)
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([_rescale_single(img, (h_scaled, w_scaled)) for img in images], dim=0)
    images = images.reshape(*batch, c, h_scaled, w_scaled)
    return _center_crop(images, intrinsics, shape_hw)
# -----------------------------------------------------------

context_target_dict = {32: 4, 48: 5, 64: 6}


class VRNeRF_NVStest(MultiViewDataset):
    def __init__(
        self,
        *args,
        ROOT: str,
        num_views: int = 32,
        llffhold: int = 8,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.num_views = int(num_views)
        self.llffhold = int(llffhold)

        self.folder_path = os.path.join(self.ROOT, f"{num_views}")
        all_scenes = sorted([x for x in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, x))])
        self.available_scenes = all_scenes

        self.scene_meta = {}
        for scene in self.available_scenes:
            scene_path = osp.join(self.folder_path, scene)
            image_names = sorted([os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            ctx_indices = [idx for idx, name in enumerate(image_names) if idx % llffhold != 0]
            tgt_indices = [idx for idx, name in enumerate(image_names) if idx % llffhold == 0]
            if len(ctx_indices) < num_views:
                ctx_indices += tgt_indices[:num_views - len(ctx_indices)]
                tgt_indices = tgt_indices[num_views - len(ctx_indices): num_views - len(ctx_indices) + context_target_dict[num_views]]
            else:
                ctx_indices = ctx_indices[:num_views]
                tgt_indices = tgt_indices[:context_target_dict[num_views]]

            i_test = np.array(tgt_indices)
            i_train = np.array(ctx_indices)

            self.scene_meta[scene] = dict(
                rgb_files=np.array(image_names),
                i_test=i_test,
                i_train=i_train,
            )

    def __len__(self):
        return len(self.available_scenes)

    @staticmethod
    def _read_image(fp: str):
        img = imageio.imread(fp).astype(np.float32) / 255.0
        return img  # H,W,3 in [0,1]

    def _fetch_views(self, idx, resolution, *args, **kwargs):
        idx = idx % len(self.available_scenes)
        scene = self.available_scenes[idx]
        meta = self.scene_meta[scene]

        W_out, H_out = resolution
        assert W_out > 0 and H_out > 0

        i_test = meta["i_test"]
        i_train = meta["i_train"]

        all_ids = list(i_train) + list(i_test)
        is_target = [False] * len(i_train) + [True] * len(i_test)

        views = []
        for i, ids in enumerate(all_ids):
            # virtual camera, not be used
            K = np.eye(3)                # (3,3)

            rgb_path = meta["rgb_files"][ids]
            img = self._read_image(rgb_path)                     # H,W,3 in [0,1]
            img_t = torch.tensor(img).permute(2, 0, 1).float()   # 3,H,W

            K_t = torch.tensor(K, dtype=torch.float32)
            img_t, _ = _rescale_and_crop(img_t.unsqueeze(0), K_t.unsqueeze(0), (H_out, W_out))
            img_t = img_t.squeeze(0)

            img_np = (img_t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            img_out = PIL.Image.fromarray(img_np)

            views.append(dict(
                img=img_out,
                dataset=f'vrnerf_nvs_test_{self.num_views}',
                label=scene,
                instance=os.path.basename(rgb_path),
                is_target=is_target[i],
                nvs_sample=False,
                scale_norm=False,
            ))

        return views