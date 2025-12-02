import os.path as osp
import os
import cv2
import numpy as np

from mirror_training.data.multiview_dataset import MultiViewDataset
from mirror_training.utils.image import imread_cv2
import h5py
import torch.distributed as dist


class HyperSim_Multi(MultiViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 4
        super().__init__(*args, **kwargs)
        self.split = split

        self.loaded_data = self._load_data()

    def _load_data(self):
        cache_file = osp.join(self.ROOT, f"cached_metadata_hypersim_{self.split}.h5")
        rank = dist.get_rank() if dist.is_initialized() else 0

        # ---------- build from filesystem ----------
        if not osp.exists(cache_file) and rank == 0:
            self.all_scenes = sorted(
                [f for f in os.listdir(self.ROOT) if os.path.isdir(osp.join(self.ROOT, f))]
            )
            subscenes = []
            for scene in self.all_scenes:
                subscenes.extend(
                    [
                        osp.join(scene, f)
                        for f in os.listdir(osp.join(self.ROOT, scene))
                        if os.path.isdir(osp.join(self.ROOT, scene, f))
                        and len(os.listdir(osp.join(self.ROOT, scene, f))) > 0
                    ]
                )

            offset = 0
            scenes = []
            sceneids = []
            images = []
            start_img_ids = []
            scene_img_list = []
            j = 0

            for scene_idx, scene in enumerate(subscenes):
                scene_dir = osp.join(self.ROOT, scene)
                rgb_paths = sorted([f for f in os.listdir(scene_dir) if f.endswith(".png")])
                assert len(rgb_paths) > 0, f"{scene_dir} is empty."
                num_imgs = len(rgb_paths)

                cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
                if num_imgs < cut_off:
                    print(f"[HyperSim] Skipping {scene} (imgs={num_imgs} < cut_off={cut_off})")
                    continue

                img_ids = list(np.arange(num_imgs, dtype=np.int64) + offset)
                start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

                scenes.append(scene) 
                scene_img_list.append(np.array(img_ids, dtype=np.int64))
                sceneids.extend([j] * num_imgs)
                images.extend(rgb_paths)
                start_img_ids.extend(start_img_ids_)
                offset += num_imgs
                j += 1

            self.scenes = scenes
            self.sceneids = np.array(sceneids, dtype=np.int64)
            self.images = images
            self.scene_img_list = [np.array(x, dtype=np.int64) for x in scene_img_list]
            self.start_img_ids = np.array(start_img_ids, dtype=np.int64)

            # ---------- write cache ----------
            vlen_i64 = h5py.vlen_dtype(np.dtype("int64"))
            with h5py.File(cache_file, "w") as hf:
                hf.create_dataset(
                    "scenes",
                    data=np.array(self.scenes, dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    compression="lzf",
                    chunks=True,
                )
                hf.create_dataset("sceneids", data=self.sceneids, compression="lzf", chunks=True)
                hf.create_dataset(
                    "images",
                    data=np.array(self.images, dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    compression="lzf",
                    chunks=True,
                )
                hf.create_dataset("start_img_ids", data=self.start_img_ids, compression="lzf", chunks=True)
                ds = hf.create_dataset("scene_img_list", (len(self.scene_img_list),), dtype=vlen_i64)
                for i, arr in enumerate(self.scene_img_list):
                    ds[i] = arr
            print(f"[HyperSim] Cached metadata written to {cache_file}")

        if dist.is_initialized():
            dist.barrier()
        
        # ---------- try load from cache ----------
        if osp.exists(cache_file):
            print(f"[HyperSim] Loading cached metadata from {cache_file}")
            with h5py.File(cache_file, "r") as hf:
                self.scenes = [s.decode("utf-8") for s in hf["scenes"][:]]
                self.sceneids = hf["sceneids"][:].astype(np.int64)
                self.images = [s.decode("utf-8") for s in hf["images"][:]]
                self.start_img_ids = hf["start_img_ids"][:].astype(np.int64)
                self.scene_img_list = [np.array(x, dtype=np.int64) for x in hf["scene_img_list"]]
            return

    def __len__(self):
        return len(self.start_img_ids) * 10

    def get_image_num(self):
        return len(self.images)

    def _fetch_views(self, idx, resolution, rng, num_views, *args, **kwargs):
        idx = idx // 10
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        pos, ordered_video = self.extract_view_sequence(
            num_views,
            start_id,
            all_image_ids.tolist(),
            rng,
            max_interval=self.max_interval,
            block_shuffle=16,
        )
        image_idxs = np.array(all_image_ids)[pos]
        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            rgb_path = self.images[view_idx]
            depth_path = rgb_path.replace("rgb.png", "depth.npy")
            cam_path = rgb_path.replace("rgb.png", "cam.npz")

            rgb_image = imread_cv2(osp.join(scene_dir, rgb_path), cv2.IMREAD_COLOR)
            depthmap = np.load(osp.join(scene_dir, depth_path)).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            cam_file = np.load(osp.join(scene_dir, cam_path))
            intrinsics = cam_file["intrinsics"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

            rgb_image, depthmap, intrinsics = self._apply_crop_and_resize(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_poses=camera_pose.astype(np.float32),
                    camera_intrs=intrinsics.astype(np.float32),
                    dataset="hypersim",
                    label=self.scenes[scene_id] + "_" + rgb_path,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    nvs_sample=True,
                    scale_norm=True,
                    cam_align=True
                )
            )
        assert len(views) == num_views
        return views
