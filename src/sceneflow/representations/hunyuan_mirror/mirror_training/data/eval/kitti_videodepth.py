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

class Kitti_VideoDepth(MultiViewDataset):
    def __init__(self, *args, ROOT, target_width, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.dataset_name = f"kitti_videodepth"
        self.target_width = target_width
        self.img_dir = os.path.join(self.ROOT, "image_gathered")
        self.depth_dir = os.path.join(self.ROOT, "groundtruth_depth_gathered")
        self.seq_names = os.listdir(self.img_dir)
        self.all_seqs = []
        for seq_name in self.seq_names:
            self.all_seqs.append(os.path.join(self.img_dir, seq_name))

    def __len__(self):
        return len(self.all_seqs)
    
    def _fetch_views(self, idx, *args, **kwargs):
        imagepaths = sorted(glob(os.path.join(self.all_seqs[idx], "*.png")))
        seq_name = os.path.basename(self.all_seqs[idx])
        views = []
        for i, image_path in enumerate(imagepaths):
            image_name = os.path.basename(image_path).split(".")[0]
            depth_path = os.path.join(self.depth_dir, seq_name, image_name + ".png")
            img = load_images_wo_crop(image_path, self.target_width)
            depthmap = EVAL_DEPTH_METADATA[self.dataset_name]["depth_read_func"](depth_path)
            views.append(dict(
                label=os.path.join(seq_name, image_name),
                img=img,
                depthmap=depthmap,
                instance=os.path.join(seq_name, image_name),
                dataset=self.dataset_name,
                img_name=os.path.basename(image_path),
                nvs_sample=False,
                scale_norm=False,
            ))
        
        return views