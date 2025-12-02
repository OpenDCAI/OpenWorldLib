from .container import BaseLoss
from .utils import check_and_fix_inf_nan, regression_loss


class DepthLoss(BaseLoss):
    """Depth loss"""
    
    def __init__(self, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=0.98):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.gradient_loss_fn = gradient_loss_fn
        self.valid_range = valid_range

    def compute_loss(self, preds, gts):
        # Extract predictions and ground truth
        pred_depth = preds['depth']
        pred_depth_conf = preds['depth_conf']
        gt_depth = gts['depthmap']
        gt_depth_mask = gts['valid_mask']
        
        # Check and fix numerical issues in ground truth
        gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
        gt_depth = gt_depth[..., None]              # (B, H, W, 1)
        
        # If insufficient valid points, return zero loss
        if gt_depth_mask.sum() < 100:
            loss_dict = {
                "loss_conf_depth": (pred_depth * 0).mean(),
                "loss_reg_depth": (pred_depth * 0).mean(),
                "loss_grad_depth": (pred_depth * 0).mean(),
            }
        else:
            # Compute confidence-weighted regression loss with optional gradient loss
            loss_conf, loss_grad, loss_reg = regression_loss(
                pred_depth, gt_depth, gt_depth_mask, 
                conf=pred_depth_conf,
                gradient_loss_fn=self.gradient_loss_fn, 
                gamma=self.gamma, 
                alpha=self.alpha, 
                valid_range=self.valid_range
            )
            
            loss_dict = {
                "loss_conf_depth": loss_conf,
                "loss_reg_depth": loss_reg,
                "loss_grad_depth": loss_grad,
            }
        
        # Compute total depth loss
        loss = loss_dict["loss_conf_depth"] + loss_dict["loss_reg_depth"] + loss_dict["loss_grad_depth"]
        return loss, loss_dict
    
    @property
    def name(self):
        return f"DepthLoss"