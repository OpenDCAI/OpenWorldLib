import os
import cv2
import numpy as np
from glob import glob
from PIL import Image

from mirror_training.data.multiview_dataset import MultiViewDataset
from mirror_training.utils.eval.depthmap_eval import EVAL_DEPTH_METADATA


def load_images_wo_crop(img_path: str, new_width: int):
    try:
        img_pil = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Could not load image {img_path}: {e}")
    W_orig, H_orig = img_pil.size
    TARGET_W, TARGET_H = new_width, round(H_orig * (new_width / W_orig) / 14) * 14
    try:
        # Resize to the uniform target size
        resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error processing an image: {e}")
    return resized_img

class Nyuv2_MonoDepth(MultiViewDataset):
    def __init__(self, *args, ROOT, target_width, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.dataset_name = "nyuv2_monodepth"
        self.target_width = target_width
        self.img_dir = os.path.join(self.ROOT, "nyu_images")
        self.depth_dir = os.path.join(self.ROOT, "nyu_depths")
        assert os.path.exists(self.img_dir) and os.path.exists(self.depth_dir)
        self.imagenames = sorted(glob(f"{self.img_dir}/*.png"))

    def __len__(self):
        return len(self.imagenames)
    
    def _fetch_views(self, idx, *args, **kwargs):
        image_path = self.imagenames[idx]
        scene_name = os.path.basename(image_path).split(".")[0]
        depth_path = os.path.join(self.depth_dir, scene_name + ".npy")

        assert os.path.exists(image_path)
        assert os.path.exists(depth_path)

        img = load_images_wo_crop(image_path, self.target_width)
        depthmap = EVAL_DEPTH_METADATA[self.dataset_name]["depth_read_func"](depth_path)
        
        sample = [dict(
            label=scene_name,
            img=img,
            depthmap=depthmap,
            dataset=self.dataset_name,
            img_name=os.path.basename(image_path),
            instance=scene_name,
            single_view=True,
            nvs_sample=False,
            scale_norm=False,
        )]

        return sample