from .container import BaseLoss
from .utils import check_and_fix_inf_nan

from mirror_src.models.utils.geometry import closed_form_inverse_se3
from mirror_src.models.utils.camera_utils import camera_params_to_vector


class CameraLoss(BaseLoss):
    """Camera pose loss"""
    
    def __init__(self, weight_T=1.0, weight_R=1.0, weight_fl=0.5):
        super().__init__()
        self.weight_T = weight_T
        self.weight_R = weight_R
        self.weight_fl = weight_fl
    
    def compute_loss(self, preds, gts):
        B, S, _, H, W = gts['img'].shape
    
        # Convert ground truth camera matrices to compact vector representation
        # Extrinsics: world-to-camera -> camera-to-world transformation
        gt_extrinsics = gts['camera_poses']
        gt_intrinsics = gts['camera_intrs']
        gt_extrinsics = closed_form_inverse_se3(gt_extrinsics.flatten(0, 1)).reshape(B, S, 4, 4)
        
        # Encode ground truth as [t(3), q(4), fov_v(1), fov_u(1)] vector
        gt_camera_params = camera_params_to_vector(gt_extrinsics, gt_intrinsics, (H, W))
        
        # Extract predicted camera parameters (B, S, 9)
        pred_camera_params = preds['camera_params']
        
        # Check if frames have valid points (at least 100 valid pixels)
        point_masks = gts['valid_mask']
        valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
        
        # If no valid frames, return zero loss
        if valid_frame_mask.sum() == 0:
            loss_dict = {
                "loss_camera": (pred_camera_params * 0).mean(),
                "loss_T": (pred_camera_params * 0).mean(),
                "loss_R": (pred_camera_params * 0).mean(),
                "loss_FL": (pred_camera_params * 0).mean(),
            }
        else:
            # Compute L1 loss for each component
            loss_T = (pred_camera_params[..., :3] - gt_camera_params[..., :3]).abs()      # Translation (3D)
            loss_R = (pred_camera_params[..., 3:7] - gt_camera_params[..., 3:7]).abs()    # Rotation quaternion (4D)
            loss_FL = (pred_camera_params[..., 7:] - gt_camera_params[..., 7:]).abs()     # Focal length / FoV (2D)
            
            # Check and fix numerical issues (NaN/Inf) in loss components
            loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
            loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
            loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")
            
            # Clamp translation loss to prevent gradient explosion, then average
            loss_T = loss_T.clamp(max=100).mean()
            loss_R = loss_R.mean()
            loss_FL = loss_FL.mean()
            
            loss_dict = {
                "loss_T": loss_T,
                "loss_R": loss_R,
                "loss_FL": loss_FL,
            }
        
        # Compute total camera loss
        loss = loss_dict["loss_T"] * self.weight_T + loss_dict["loss_R"] * self.weight_R + loss_dict["loss_FL"] * self.weight_fl
        return loss, loss_dict
    
    @property
    def name(self):
        return f"CameraLoss"
    