import os
import random
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import numpy as np
import glob
import json
from PIL import Image

from mirror_training.data.multiview_dataset import MultiViewDataset
from mirror_training.utils.image import imread_cv2
import mirror_src.utils.cropping as cropping


class Re10K_Pose(MultiViewDataset):
    def __init__(self, *args, map_path, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.dataset_name = 're10k_test'
        self.video_root = os.path.join(self.ROOT, 'test')
        self.txt_root = os.path.join(self.ROOT, 'trajectories', 'test')
        self.subset_file = os.path.join(self.ROOT, 're10k_test_1800.txt')
        self.map_path = map_path

        self._load_data(self.video_root, self.txt_root)
    
    def __len__(self):
        return len(self.frame_files_list)

    def _load_data(self, video_root, txt_root):
        all_folders = sorted(os.listdir(video_root))
        all_folders = [f for f in all_folders if os.path.isdir(os.path.join(video_root, f)) and '.cache' not in f]

        frame_files_list = []
        lines_map_list = []
        ids_list = []
        
        num_seq = len(all_folders)
        
        seq_id_map = None
        if os.path.exists(self.map_path):
            with open(self.map_path, "r") as f:
                seq_id_map = json.load(f)
            all_folders = [fd for fd in all_folders if fd in seq_id_map]
        elif os.path.exists(self.subset_file):
            with open(self.subset_file, "r") as f:
                subset_scenes = set(line.strip() for line in f if line.strip())
            all_folders = [fd for fd in all_folders if fd in subset_scenes]
        else:
            all_folders = all_folders[:num_seq]

        for vid_folder in all_folders:
            if seq_id_map is not None:
                ids = seq_id_map[vid_folder]
                ids_list.append(ids)    

            folder_path = os.path.join(video_root, vid_folder)
            if not os.path.isdir(folder_path):
                print(f"{folder_path} video not exist, please check...")
                continue

            txt_path = os.path.join(txt_root, vid_folder + ".txt")
            if not os.path.exists(txt_path):
                print(f"{folder_path} txt not exist, please check...")
                continue

            with open(txt_path, "r") as f:
                txt_lines = f.read().strip().split("\n")
            if len(txt_lines) <= 1:
                continue
            txt_lines = txt_lines[1:]  # skip first line (URL)

            lines_map = {}
            for line in txt_lines:
                parts = line.strip().split()
                # Expect at least 19 columns
                if len(parts) < 19:
                    continue
                frame_id = parts[0]
                lines_map[frame_id] = parts

            frame_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
            if len(frame_files) < 2:
                continue
        
            frame_files_list.append(frame_files)
            lines_map_list.append(lines_map)
        
        self.frame_files_list = frame_files_list
        self.lines_map_list = lines_map_list
        self.all_folders = all_folders
        self.ids_list = ids_list
    
    def _fetch_views(self, idx, resolution, *args, **kwargs):
        frame_files = self.frame_files_list[idx]
        lines_map = self.lines_map_list[idx]

        # Sample up to 10 frames per folder
        if len(self.ids_list) > 0:
            ids = self.ids_list[idx]
            sampled_frames = [frame_files[id] for id in ids]
            sampled_frames.sort()
        else:
            n_to_sample = min(10, len(frame_files))
            random.seed(self.seed + idx)
            sampled_frames = random.sample(frame_files, n_to_sample)
            sampled_frames.sort()

        selected_views = []
        for frame_path in sampled_frames:
            basename = os.path.splitext(os.path.basename(frame_path))[0]
            if basename not in lines_map:
                continue

            columns = lines_map[basename]
            # parse fx, fy, cx, cy
            fx = float(columns[1])
            fy = float(columns[2])
            cx = float(columns[3])
            cy = float(columns[4])

            # parse extrinsic row-major 3×4 => build 4×4, then invert to get c2w
            extrinsic_val = [float(v) for v in columns[7:19]]  # 12 floats
            extrinsic = np.array(extrinsic_val, dtype=np.float64).reshape(3, 4)

            # Build 4x4
            pose_4x4 = np.eye(4, dtype=np.float32)
            pose_4x4[:3, :3] = extrinsic[:3, :3]
            pose_4x4[:3, 3]  = extrinsic[:3, 3]  # translation in last column
            # poses_c2w_gt = np.linalg.inv(pose_4x4).astype(np.float32)
            poses_w2c_gt = pose_4x4.astype(np.float32)

            # Load image
            img_rgb = imread_cv2(frame_path)
            if img_rgb is None:
                continue

            H_org, W_org = img_rgb.shape[:2]
            # Build K
            K_3x3 = np.array([
                [fx * W_org, 0.0,       cx * W_org],
                [0.0,        fy * H_org, cy * H_org],
                [0.0,        0.0,       1.0]
            ], dtype=np.float32)

            # Crop+resize => final_img_pil, final_intrinsics
            pil_img = Image.fromarray(img_rgb)
            final_img_pil, final_intrinsics_3x3 = self._apply_crop_and_resize(
                pil_img, K_3x3, target_resolution=resolution
            )

            # Put data on GPU
            view_dict = {
                "label": self.all_folders[idx],
                "video_folder": self.all_folders[idx],
                "img": final_img_pil,              # (1,3,H,W)
                "camera_poses": poses_w2c_gt.astype(np.float32),  # (1,4,4)
                "camera_intrs": final_intrinsics_3x3.astype(np.float32),  # (1,3,3)
                "dataset": "Re10K_Pose",
                "nvs_sample": False,
                "scale_norm": False,
                "cam_align": False
            }
            selected_views.append(view_dict)

        return selected_views

    def _apply_crop_and_resize(
            self,
            image,
            intrinsics_3x3,
            target_resolution=(512, 288)
        ):
            """Crop around principal point + downscale => (512×288) or (288×512)."""
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            W_org, H_org = image.size
            cx, cy = int(round(intrinsics_3x3[0, 2])), int(round(intrinsics_3x3[1, 2]))

            min_margin_x = min(cx, W_org - cx)
            min_margin_y = min(cy, H_org - cy)

            left = cx - min_margin_x
            top = cy - min_margin_y
            right = cx + min_margin_x
            bottom = cy + min_margin_y
            crop_bbox = (left, top, right, bottom)

            image_c, _, intrinsics_c = cropping.crop_image_depthmap(
                image,
                None,
                intrinsics_3x3,
                crop_bbox
            )

            # Check orientation
            W_c, H_c = image_c.size
            if H_c > W_c:
                # swap if portrait
                target_resolution = (target_resolution[1], target_resolution[0])

            # Downscale
            image_rs, _, intrinsics_rs = cropping.rescale_image_depthmap(
                image_c, None, intrinsics_c, np.array(target_resolution)
            )
            intrinsics2 = cropping.camera_matrix_of_crop(
                intrinsics_rs, image_rs.size, target_resolution, offset_factor=0.5
            )
            final_bbox = cropping.bbox_from_intrinsics_in_out(
                intrinsics_rs, intrinsics2, target_resolution
            )
            image_out, _, intrinsics_out = cropping.crop_image_depthmap(
                image_rs, None, intrinsics_rs, final_bbox
            )

            return image_out, intrinsics_out