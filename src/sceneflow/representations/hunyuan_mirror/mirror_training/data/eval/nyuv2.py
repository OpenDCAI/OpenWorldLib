""" Get samples from NYUv2 (https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)
    NOTE: GT surface normals are from GeoNet (CVPR 2018) - https://github.com/xjqi/GeoNet
"""
import os
import cv2
import numpy as np

from mirror_training.data.multiview_dataset import MultiViewDataset
import mirror_src.utils.cropping as cropping
import PIL

class Nyuv2(MultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.dataset_name = 'nyuv2'
        split_path = os.path.join(self.ROOT, 'split', self.split+'.txt')
        assert os.path.exists(split_path)
        with open(split_path, 'r') as f:
            self.filenames = [i.strip() for i in f.readlines()]

    def __len__(self):
        return len(self.filenames)
    
    def _fetch_views(self, idx, *args, **kwargs):
        # e.g. sample_path = "test/000000_img.png"
        sample_path = self.filenames[idx]
        scene_name = sample_path
        img_name, img_ext = sample_path.split('/')[1].split('_img')

        img_path = '%s/%s' % (self.ROOT, sample_path)
        normal_path = img_path.replace('_img'+img_ext, '_normal.png')
        intrins_path = img_path.replace('_img'+img_ext, '_intrins.npy')
        assert os.path.exists(img_path)
        assert os.path.exists(normal_path)
        assert os.path.exists(intrins_path)

        # read image (H, W, 3)
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

        # read normal (H, W, 3)
        normalmap = cv2.cvtColor(cv2.imread(normal_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        normal_mask = np.sum(normalmap, axis=2, keepdims=True) > 0
        normalmap = (normalmap.astype(np.float32) / 255.0) * 2.0 - 1.0

        # read intrins (3, 3)
        intrins = np.load(intrins_path)
        
        # resize
        img, normalmap, intrins, normal_mask = self._apply_crop_and_resize(
            img, normalmap, intrins, normal_mask, self._resolution)
        
        sample = [dict(
            label=scene_name,
            img=img,
            normals=normalmap,
            valid_mask=normal_mask,
            camera_intrs=intrins,
            dataset='nyuv2',
            img_name=img_name,
            single_view=True,
            nvs_sample=False,
            scale_norm=False,
        )]

        return sample
    
    def _apply_crop_and_resize(
        self, image, normalmap, intrinsics, normal_mask, resolution, rng=None, info=None
    ):
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, normalmap, intrinsics, normal_mask = cropping.crop_image_normalmap(
            image, normalmap, intrinsics, normal_mask, crop_bbox
        )

        # transpose the resolution if necessary
        W, H = image.size  # new size
        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        elif 0 < self.aug_crop < 1:
            delta_target_ratio = rng.random() * (1. / self.aug_crop - 1.)
            delta_target_resolution = (np.array(resolution) * delta_target_ratio).astype("int")
            target_resolution += (
                delta_target_resolution
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )

        image, normalmap, intrinsics, normal_mask = cropping.rescale_image_normalmap(
            image, normalmap, intrinsics, normal_mask, target_resolution
        )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )
        image, normalmap, intrinsics2, normal_mask = cropping.crop_image_normalmap(
            image, normalmap, intrinsics2, normal_mask, crop_bbox
        )

        return image, normalmap, intrinsics2, normal_mask