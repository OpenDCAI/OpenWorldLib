from .container import BaseLoss
from .utils import check_and_fix_inf_nan, regression_loss


class PointLoss(BaseLoss):
    """3D point cloud loss with confidence weighting and regularization"""
    
    def __init__(self, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=-1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.gradient_loss_fn = gradient_loss_fn
        self.valid_range = valid_range
    
    def compute_loss(self, preds, gts):
        # Extract predictions and ground truth
        pred_points = preds['pts3d']
        pred_points_conf = preds['pts3d_conf']
        gt_points = gts['pts3d']
        gt_points_mask = gts['valid_mask']
        
        # Check and fix numerical issues in ground truth
        gt_points = check_and_fix_inf_nan(gt_points, "gt_points")
        
        # If insufficient valid points, return zero loss
        if gt_points_mask.sum() < 100:
            loss_dict = {
                "loss_conf_point": (pred_points * 0).mean(),
                "loss_reg_point": (pred_points * 0).mean(),
                "loss_grad_point": (pred_points * 0).mean(),
            }
        else:
            # Compute confidence-weighted regression loss with optional gradient loss
            loss_conf, loss_grad, loss_reg = regression_loss(
                pred_points, gt_points, gt_points_mask, 
                conf=pred_points_conf,
                gradient_loss_fn=self.gradient_loss_fn, 
                gamma=self.gamma, 
                alpha=self.alpha, 
                valid_range=self.valid_range
            )
            
            loss_dict = {
                "loss_conf_point": loss_conf,
                "loss_reg_point": loss_reg,
                "loss_grad_point": loss_grad,
            }
        
        # Compute total point loss
        loss = loss_dict["loss_conf_point"] + loss_dict["loss_reg_point"] + loss_dict["loss_grad_point"]
        return loss, loss_dict
    
    @property
    def name(self):
        return "PointLoss"
