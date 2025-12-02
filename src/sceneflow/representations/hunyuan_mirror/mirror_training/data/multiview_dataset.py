import PIL
import numpy as np
import torch
import itertools

from mirror_training.data.base_dataset import BaseDataset
from mirror_training.data.sampler.novelview_sampler import ViewSamplingStrategy

from mirror_training.utils.image import ToTensor, ImageAugmentation
from mirror_training.utils.geometry import depthmap_to_absolute_camera_coordinates
import mirror_src.utils.cropping as cropping


class MultiViewDataset(BaseDataset):
    def __init__(
        self,
        *,
        num_views=None,
        split=None,
        resolution=None,
        transform=ImageAugmentation(apply_aug=False),
        aug_crop=False,
        seed=None,
        allow_repeat=False,
        seq_aug_crop=False,
        patch_size=14,
        video_aug=False,
    ):
        self.num_views = num_views
        self.split = split
        self._configure_resolution(resolution)

        if isinstance(transform, str):
            transform = eval(transform)
        self.transform = transform
        self.patch_size = patch_size

        self.aug_crop = aug_crop
        self.seed = seed
        self.allow_repeat = allow_repeat
        self.seq_aug_crop = seq_aug_crop
        self.video_aug = video_aug
        self.viewsampler = ViewSamplingStrategy(seed=seed, video_aug=video_aug)

    def __len__(self):
        return len(self.scenes)

    @staticmethod
    def shuffle_in_blocks(x, rng, block_shuffle):
        """Perform shuffling within blocks of specified size or full permutation if block_shuffle is None."""
        if block_shuffle is None:
            return rng.permutation(x).tolist()
        else:
            assert block_shuffle > 0
            blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
            shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
            shuffled_list = [item for block in shuffled_blocks for item in block]
            return shuffled_list

    def extract_view_sequence(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        """Generate sequence of view positions from a starting reference id with configurable sampling strategies.
        
        Parameters:
            num_views: Total number of views required in output sequence
            id_ref: Starting reference identifier for sequence generation
            ids_all: Complete list of available identifiers
            rng: Random number generator instance for stochastic operations
            min_interval: Minimum spacing between consecutive views
            max_interval: Maximum spacing between consecutive views
            video_prob: Probability of generating temporal video-like sequences
            fix_interval_prob: Probability of using fixed uniform intervals
            block_shuffle: Size of blocks for shuffle operation, None for full shuffle
            
        Returns:
            Tuple of (position_list, is_video_flag) where position_list contains indices
            into ids_all and is_video_flag indicates temporal ordering
        """
        assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
        assert (
            min_interval <= max_interval
        ), f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
        assert id_ref in ids_all
        pos_ref = ids_all.index(id_ref)
        if num_views == 1:
            return [pos_ref], False
        all_possible_pos = np.arange(pos_ref, len(ids_all))

        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
            intervals = [
                rng.choice(range(min_interval, max_interval + 1))
                for _ in range(num_views - 1)
            ]

            if rng.random() < video_prob:
                if rng.random() < fix_interval_prob:
                    fixed_interval = rng.choice(
                        range(
                            1,
                            min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                        )
                    )
                    intervals = [fixed_interval for _ in range(num_views - 1)]
                is_video = True
            else:
                is_video = False

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = (
                pos
                + rng.choice(
                    pos_candidates, num_views - len(pos), replace=False
                ).tolist()
            )

            pos = (
                sorted(pos)
                if is_video
                else self.shuffle_in_blocks(pos, rng, block_shuffle)
            )
        else:
            uniq_num = remaining_sum
            new_pos_ref = rng.choice(np.arange(pos_ref + 1))
            new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
            new_intervals = [
                rng.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)
            ]

            revisit_random = rng.random()
            video_random = rng.random()

            if rng.random() < fix_interval_prob and video_random < video_prob:
                fixed_interval = rng.choice(range(1, new_max_interval + 1))
                new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:
                is_video = video_random < video_prob
                pos = (
                    self.shuffle_in_blocks(pos, rng, block_shuffle)
                    if not is_video
                    else pos
                )
                num_full_repeat = num_views // uniq_num
                pos = (
                    pos * num_full_repeat
                    + pos[: num_views - len(pos) * num_full_repeat]
                )
            elif revisit_random < 0.9:
                pos = rng.choice(pos, num_views, replace=True)
            else:
                pos = sorted(rng.choice(pos, num_views, replace=True))
        assert len(pos) == num_views
        return pos, is_video
    
    def apply_pointcloud_normalization(self, views, norm_mode="avg_dis"):
        """Apply normalization to 3D point clouds and camera parameters across multiple views.
        
        This method normalizes spatial data by computing a scale factor from valid point distances,
        excluding invalid regions from the calculation. The normalization ensures consistent scale
        across different scenes and views.
        
        Parameters:
            views: List of view dictionaries containing pts3d, valid_mask, depthmap, camera_poses
            norm_mode: Normalization strategy string formatted as 'stat_disttype' where stat is
                      the statistical measure (avg, median) and disttype is distance transform
                      (dis, log1p, warp-log1p)
        
        Returns:
            Modified views list with normalized pts3d, depthmap, camera_poses and added norm_factor
        """
        pts_list = np.concatenate([views[i]["pts3d"] for i in range(len(views))], axis=0)
        valid_list = np.concatenate([views[i]["valid_mask"] for i in range(len(views))], axis=0)
        assert pts_list.ndim >= 3 and pts_list.shape[-1] == 3
        
        norm_mode, dis_mode = norm_mode.split("_")
        all_pts = pts_list.reshape(-1, 3)
        all_valid = valid_list.reshape(-1, )
        all_pts[all_valid == 0] = float('nan')

        valid_pts = all_pts

        dis = np.linalg.norm(valid_pts, ord=2, axis=-1)

        if dis_mode == "dis":
            pass
        elif dis_mode == "log1p":
            dis = np.log1p(dis)
        elif dis_mode == "warp-log1p":
            log_dis = np.log1p(dis)
            warp_factor = log_dis / dis.clip(min=1e-8)
            all_pts = all_pts * warp_factor.reshape(-1, 1)
            dis = log_dis
        else:
            raise ValueError(f"Unsupported distance mode: {dis_mode}")

        if norm_mode == "avg":
            norm_factor = np.nanmean(dis, axis=-1)
        elif norm_mode == "median":
            norm_factor = np.nanmedian(dis, axis=-1)
        else:
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")
        
        bad_case = False
        if np.isnan(norm_factor):
            norm_factor = 1.0 
            bad_case = True

        norm_factor = np.nan_to_num(norm_factor, nan=1.0).clip(min=1e-8)

        for i in range(len(views)):
            views[i]["pts3d"] /= norm_factor
            views[i]["depthmap"] /= norm_factor
            views[i]["camera_poses"][:3, 3] /= norm_factor
            views[i]["norm_factor"] = norm_factor
            views[i]["bad_case"] = bad_case

        return views
    
    def generate_statistics(self):
        """Generate string summary of dataset statistics."""
        return f"{len(self)} groups of views"

    def __repr__(self):
        return (
            f"""{type(self).__name__}({self.generate_statistics()},
            {self.split=},
            {self.seed=},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _fetch_views(self, idx, *args, **kwargs):
        """Abstract method to be implemented by subclasses for retrieving view data."""
        raise NotImplementedError()

    def __getitem__(self, idx):
        """Retrieve and process a batch of multi-view data with optional augmentation and normalization.
        
        This method handles flexible indexing, random number generation setup, view sampling,
        image transformation, coordinate frame alignment, and scale normalization.
        
        Parameters:
            idx: Either integer index or tuple of (index, aspect_ratio, view_counts, pixel_count)
            
        Returns:
            List of processed view dictionaries with images, poses, depth, and metadata
        """
        if isinstance(idx, (tuple, list, np.ndarray)):
            idx, aspect_ratio, tuple_views, npixels = idx
            nview, nview_source = tuple_views
        else:
            aspect_ratio = 1.0
            nview = self.num_views
            nview_source = max(nview - 1, 1) if nview is not None else None
            npixels = None
        if nview is not None:
            assert nview >= 1 and nview <= self.num_views

        if self.seed:
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=seed)
        self.viewsampler.rng = self._rng

        resolution = compute_adjusted_resolution(self._resolution, aspect_ratio, self.patch_size, npixels)
        views = self._fetch_views(idx, resolution, self._rng, nview)

        if self.seq_aug_crop:
            delta_target_ratio = self._rng.random() * (1. / self.aug_crop - 1.)
            self.delta_target_resolution = (np.array(resolution) * delta_target_ratio).astype("int")

        nvs_sample = views[0].get("nvs_sample", False)
        scale_norm = views[0].get("scale_norm", False)
        cam_align = views[0].get("cam_align", True)
        width, height = views[0]['img'].size

        for v, view in enumerate(views):
            view["img"] = ToTensor(view["img"])
            if "depthmap" in view.keys() and "camera_intrs" in view.keys() and "camera_poses" in view.keys():
                assert np.isfinite(view["depthmap"]).all(), f"NaN in depthmap for view {construct_view_identifier(view)}"
                pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
                view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)

        if nvs_sample:
            context_views, target_views = self.viewsampler.sample_views(views, nview_source)
            views = context_views + target_views
        
        first_view_camera_pose = views[0]["camera_poses"] if "camera_poses" in views[0] else np.eye(4, dtype=np.float32)
        for v, view in enumerate(views):
            if cam_align and "camera_poses" in view:
                view["camera_poses"] = np.linalg.inv(first_view_camera_pose) @ view["camera_poses"]

            if "depthmap" in view.keys() and "camera_intrs" in view.keys() and "camera_poses" in view.keys():
                pts3d, _ = depthmap_to_absolute_camera_coordinates(**view)
                view["pts3d"] = pts3d

            for key, val in view.items():
                res, err_msg = validate_data_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {construct_view_identifier(view)}"

        allview_imgs = torch.stack([v["img"] for v in views])
        aug_allview_imgs = self.transform(allview_imgs)
        for v, view in enumerate(views):
            view["img"] = aug_allview_imgs[v]

        if scale_norm:
            views = self.apply_pointcloud_normalization(views)
        
        for view in views:
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")

        return views

    def _configure_resolution(self, resolution):
        """Parse and validate resolution specification, storing as width-height tuple.
        
        Parameters:
            resolution: Either integer for square resolution or tuple of (width, height)
        """
        if resolution is None:
            self._resolution = None
            return
        if isinstance(resolution, int):
            width = height = resolution
        else:
            width, height = resolution
        assert isinstance(
            width, int
        ), f"Bad type for {width=} {type(width)=}, should be int"
        assert isinstance(
            height, int
        ), f"Bad type for {height=} {type(height)=}, should be int"
        self._resolution = (width, height)

    def _apply_crop_and_resize(
        self, image, depthmap, intrinsics, resolution, rng=None, info=None
    ):
        """Process image through center-crop and resize pipeline with optional augmentation.
        
        This method performs multi-stage image processing: converts to PIL format, crops around
        principal point, applies optional random augmentation, rescales with high-quality
        interpolation, and performs final precision cropping.
        
        Parameters:
            image: Input image as PIL Image or numpy array
            depthmap: Corresponding depth map aligned with image
            intrinsics: 3x3 camera intrinsic matrix
            resolution: Target output resolution tuple
            rng: Random generator for augmentation randomness
            info: Debug information string for error messages
            
        Returns:
            Tuple of (processed_image, processed_depthmap, adjusted_intrinsics)
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        W, H = image.size
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

        image, depthmap, intrinsics = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        return image, depthmap, intrinsics2
    
    
def validate_data_type(key, v):
    """Check if value has acceptable data type for dataset operations.
    
    Parameters:
        key: Dictionary key name for context
        v: Value to validate
        
    Returns:
        Tuple of (is_valid_bool, error_message_or_none)
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def construct_view_identifier(view, batch_index=None):
    """Build hierarchical identifier string from view metadata fields.
    
    Parameters:
        view: View dictionary containing dataset, label, instance fields
        batch_index: Optional index for selecting from batched data
        
    Returns:
        Formatted identifier string as 'dataset/label/instance'
    """
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"


def compute_adjusted_resolution(resolution, aspect_ratio, patch_size, npixels):
    """Calculate final resolution from aspect ratio and pixel count constraints.
    
    When npixels is specified, computes resolution satisfying both pixel count and aspect ratio,
    then aligns to patch_size grid. Otherwise returns original resolution unchanged.
    
    Parameters:
        resolution: Base resolution tuple (width, height)
        aspect_ratio: Target width to height ratio
        patch_size: Grid alignment size for dimension snapping
        npixels: Target total pixel count, or None to skip adjustment
        
    Returns:
        Tuple of (adjusted_width, adjusted_height)
    """
    if npixels is not None:
        W_d_H = aspect_ratio
        H = int((npixels / W_d_H) ** 0.5)
        W = int(W_d_H * H)
        H = H // patch_size * patch_size
        W = W // patch_size * patch_size
        return (W, H)
    else:
        return resolution