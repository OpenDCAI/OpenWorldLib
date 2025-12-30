import torch
from torch import nn
import numpy as np
from ......base_models.three_dimensions.point_clouds.gaussian_splatting.utils.graphics_utils import (
    getWorld2View2, getProjectionMatrix)
# from ......base_models.three_dimensions.point_clouds.gaussian_splatting.scene.cameras import Camera as CameraBase


## the gt image is generated online in wonderworld
class Camera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, 
                 image=torch.zeros([3, 512, 512]),
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 no_loss_mask=None,
                 gt_alpha_mask=None,
                 image_name=None,
                 colmap_id=None,
                 uid=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        if no_loss_mask is None:
            no_loss_mask = torch.zeros_like(self.original_image).bool()  # Do not remove image
        self.no_loss_mask = no_loss_mask.to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        self.focal_y = self.image_height / (2.0 * tan_fovy)
        self.focal_x = self.image_width / (2.0 * tan_fovx)
