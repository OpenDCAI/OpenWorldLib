import torch
import numpy as np
from PIL import Image
from .cameras import Camera
from ......base_models.three_dimensions.point_clouds.gaussian_splatting.utils.graphics_utils import fov2focal, focal2fov
from ......base_models.three_dimensions.point_clouds.gaussian_splatting.scene.dataset_readers import (
    getNerfppNorm, BasicPointCloud, NamedTuple
)


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    preset_cameras: list
    nerf_normalization: dict
    ply_path: str


def loadCamerasFromData(traindata, white_background):
    cameras = []

    fovx = traindata["camera_angle_x"]
    frames = traindata["frames"]
    for idx, frame in enumerate(frames):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image = frame["image"] if "image" in frame else None
        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")
        loaded_mask = np.ones_like(norm_data[:, :, 3:4])

        fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
        FovY = fovy 
        FovX = fovx

        image = torch.Tensor(arr).permute(2,0,1)
        loaded_mask = None #torch.Tensor(loaded_mask).permute(2,0,1)

        no_loss_mask = frame['no_loss_mask']
        
        ### torch
        cameras.append(Camera(colmap_id=idx, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, no_loss_mask=no_loss_mask,
                                gt_alpha_mask=loaded_mask, image_name='', uid=idx, data_device='cuda'))
            
    return cameras


def readDataInfo(traindata, white_background):
    print("Reading Training Transforms")

    train_cameras = loadCamerasFromData(traindata, white_background)

    nerf_normalization = getNerfppNorm(train_cameras)

    try:
        pcd = BasicPointCloud(points=traindata['pcd_points'].T, colors=traindata['pcd_colors'], normals=traindata['pcd_normals'])
    except:
        pcd = BasicPointCloud(points=traindata['pcd_points'].T, colors=traindata['pcd_colors'], normals=None)

    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cameras,
                           test_cameras=[],
                           preset_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path='')
    return scene_info