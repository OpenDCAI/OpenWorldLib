import os
import cv2
import json
import numpy as np
import os.path as osp
from collections import deque
from mirror_training.data.multiview_dataset import MultiViewDataset
from mirror_training.utils.image import imread_cv2



class SevenScenes(MultiViewDataset):
    def __init__(self,
                 num_seq=1,
                 test_id=None,
                 map_path=None,
                 kf_every=1,
                 *args,
                 ROOT,
                 **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.test_id = test_id
        self.kf_every = kf_every
    
        # load all scenes
        self.load_sequence_map_ids(map_path)
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        if self.seq_names_sel is not None:
            return len(self.seq_names_sel)
        return len(self.scene_list) * self.num_seq
    
    def load_sequence_map_ids(self, map_path):
        if map_path is not None:
            seq_names_sel = []
            seq_ids_sel = []
            with open(map_path, "r") as f:
                seq_id_map: dict = json.load(f)
                for seq_name, ids in seq_id_map.items():
                    seq_names_sel.append(seq_name)
                    seq_ids_sel.append(ids)
                self.seq_names_sel = seq_names_sel
                self.seq_ids_sel = seq_ids_sel
        else:
            self.seq_names_sel = None
            self.seq_ids_sel = None

    def load_all_scenes(self, base_dir):
        
        if self.seq_names_sel is not None:
            return 
            
        scenes = os.listdir(base_dir)
        
        file_split = {'train': 'TrainSplit.txt', 'test': 'TestSplit.txt'}[self.split]
        
        self.scene_list = []
        for scene in scenes:
            if self.test_id is not None and scene != self.test_id:
                continue
            # read file split
            with open(osp.join(base_dir, scene, file_split)) as f:
                seq_ids = f.read().splitlines()
                for seq_id in seq_ids:
                    num_part = ''.join(filter(str.isdigit, seq_id))
                    seq_id = f'seq-{num_part.zfill(2)}'
                    if self.seq_id is not None and seq_id != self.seq_id:
                        continue
                    self.scene_list.append(f"{scene}/{seq_id}")
         
    def _fetch_views(self, idx, resolution, rng: np.random.Generator, *args, **kwargs): 
        if self.seq_names_sel is not None:
            scene_id = self.seq_names_sel[idx]
            img_idxs = self.seq_ids_sel[idx]
        else:
            scene_id = self.scene_list[idx // self.num_seq]
            data_path = osp.join(self.ROOT, scene_id)
            num_files = len([name for name in os.listdir(data_path) if 'color' in name])
            img_idxs = [f'{i:06d}' for i in range(num_files)]
            img_idxs = self.sample_frame_idx(img_idxs, rng, full_video=self.full_video)
        
        # Intrinsics used in SimpleRecon
        fx, fy, cx, cy = 525, 525, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        
        views = []
        imgs_idxs = deque(img_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            if self.seq_names_sel is not None:
                impath = osp.join(self.ROOT, scene_id, f'frame-{str(im_idx).zfill(6)}.color.png')
                depthpath = osp.join(self.ROOT, scene_id, f'frame-{str(im_idx).zfill(6)}.depth.proj.png')
                posepath = osp.join(self.ROOT, scene_id, f'frame-{str(im_idx).zfill(6)}.pose.txt')
            else:
                impath = osp.join(self.ROOT, scene_id, f'frame-{im_idx}.color.png')
                depthpath = osp.join(self.ROOT, scene_id, f'frame-{im_idx}.depth.proj.png')
                posepath = osp.join(self.ROOT, scene_id, f'frame-{im_idx}.pose.txt')

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap[depthmap==65535] = 0
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap>10] = 0
            depthmap[depthmap<1e-3] = 0

            camera_pose = np.loadtxt(posepath).astype(np.float32)

            rgb_image, depthmap, intrinsics = self._apply_crop_and_resize(
                rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath)
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_poses=camera_pose,
                camera_intrs=intrinsics,
                dataset='7scenes',
                label=osp.join(scene_id, str(im_idx)),
                instance=osp.split(impath)[1],
                nvs_sample=False,
                scale_norm=False,
                cam_align=True
            ))
        return views

    def sample_frame_idx(self, img_idxs):
        img_idxs = img_idxs[::self.kf_every]
        
        return img_idxs