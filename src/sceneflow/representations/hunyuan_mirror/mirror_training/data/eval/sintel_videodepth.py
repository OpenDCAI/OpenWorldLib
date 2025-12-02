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

class Sintel_VideoDepth(MultiViewDataset):
    def __init__(self, *args, ROOT, target_width, type="video", **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.dataset_name = f"sintel_{type}depth"
        self.type = type
        self.target_width = target_width
        self.img_dir = os.path.join(self.ROOT, "final")
        self.depth_dir = os.path.join(self.ROOT, "depth")
        self.seq_names = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2",
                "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]
        self.all_seqs = []
        for seq_name in self.seq_names:
            if type == "video":
                self.all_seqs.append(os.path.join(self.img_dir, seq_name))
            elif type == "mono":
                imagepaths = sorted(glob(f"{os.path.join(self.img_dir, seq_name)}/*.png"))
                for imagepath in imagepaths:
                    self.all_seqs.append(imagepath)
            else:
                raise ValueError(f"Unknown type: {type}, type must be video or mono!")

    def __len__(self):
        return len(self.all_seqs)
    
    def _fetch_views(self, idx, *args, **kwargs):
        if self.type == "video":
            imagepaths = sorted(glob(os.path.join(self.all_seqs[idx], "*.png")))
            seq_name = os.path.basename(self.all_seqs[idx])
            views = []
            for i, image_path in enumerate(imagepaths):
                image_name = os.path.basename(image_path).split(".")[0]
                depth_path = os.path.join(self.depth_dir, seq_name, image_name + ".dpt")
                img = load_images_wo_crop(image_path, self.target_width)
                depthmap = EVAL_DEPTH_METADATA[self.dataset_name]["depth_read_func"](depth_path)
                views.append(dict(
                    label=os.path.join(seq_name, image_name),
                    img=img,
                    depthmap=depthmap,
                    dataset=self.dataset_name,
                    img_name=os.path.basename(image_path),
                    instance=os.path.join(seq_name, image_name),
                    nvs_sample=False,
                    scale_norm=False,
                ))
        
        elif self.type == "mono":
            image_path = self.all_seqs[idx]
            image_name = os.path.basename(image_path).split(".")[0]
            seq_name = os.path.basename(os.path.dirname(image_path))
            depth_path = os.path.join(self.depth_dir, seq_name, image_name + ".dpt")
            img = load_images_wo_crop(image_path, self.target_width)
            depthmap = EVAL_DEPTH_METADATA[self.dataset_name]["depth_read_func"](depth_path)
            views = [dict(
                    label=os.path.join(seq_name, image_name),
                    img=img,
                    depthmap=depthmap,
                    dataset=self.dataset_name,
                    img_name=os.path.basename(image_path),
                    instance=os.path.join(seq_name, image_name),
                    single_view=True,
                    nvs_sample=False,
                    scale_norm=False,
                )]
        
        else:
            raise ValueError(f"Unknown type: {self.type}, type must be video or mono!")
        
        return views