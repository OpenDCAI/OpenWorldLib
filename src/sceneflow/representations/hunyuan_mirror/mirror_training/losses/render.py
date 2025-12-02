import torch
import torch.nn as nn

from .container import BaseLoss
from .utils import regression_loss, check_and_fix_inf_nan
from lpips import LPIPS


class RenderMSELoss(BaseLoss):
    """Rendering MSE Loss"""
    
    def __init__(self, reduction='mean', mask_weight=1.0, use_l1=False, mask_mode=None, 
                 norender_datatsets=["MegaDepth", "VirtualKITTI2", "PointOdyssey"]):
        super().__init__()
        self.reduction = reduction
        self.mask_weight = mask_weight
        self.use_l1 = use_l1
        self.mask_mode = mask_mode
        self.norender_datatsets = norender_datatsets
    
    def compute_loss(self, preds, context_view_nums, dataset_name=None):
        """Compute MSE loss between rendered and ground truth colors
        
        Args:
            pred_colors: Predicted rendered colors, shape [B, V, H, W, 3]
            gt_colors: Ground truth colors, shape [B, V, H, W, 3]
            valid_mask: Ground truth mask, shape [B, V, H, W]
            context_view_nums: Number of context views, type: int
            
        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with detailed loss terms
        """
        pred_colors = check_and_fix_inf_nan(preds["rendered_colors"], "render_colors")
        gt_colors = check_and_fix_inf_nan(preds["gt_colors"], "gt_colors")
        valid_mask = check_and_fix_inf_nan(preds["valid_masks"].clone(), "valid_masks")
        B, V, H, W, C = gt_colors.shape
        
        mask = torch.full_like(valid_mask, fill_value=True)
        if self.mask_mode == "context_mask + target_mask":
            valid_mask = valid_mask
        elif self.mask_mode == "context_mask":
            mask[:, :context_view_nums, :, :] = valid_mask[:, :context_view_nums, :, :]
            valid_mask = mask
        elif self.mask_mode == "target_mask":
            mask[:, context_view_nums:, :, :] = valid_mask[:, context_view_nums:, :, :]
            valid_mask = mask
        else:
            valid_mask = mask

        if V==1 or valid_mask.sum() == 0:
            if self.use_l1:
                loss_name = "l1"
            else:
                loss_name = "mse"
            loss_value = torch.tensor(0.).to(pred_colors.device)
            loss_dict = {
                f"render_{loss_name}": loss_value.item()
            }
            return loss_value, loss_dict
        pred_colors = pred_colors * valid_mask[..., None]
        gt_colors = gt_colors * valid_mask[..., None]

        # Ensure input shapes match
        assert pred_colors.shape == gt_colors.shape, f"Prediction shape {pred_colors.shape} doesn't match ground truth shape {gt_colors.shape}"
        if self.use_l1:
            pixel_loss = torch.abs(pred_colors - gt_colors).sum(dim=-1)
            loss_name = "l1"
        else:
            pixel_loss = ((pred_colors - gt_colors) ** 2).sum(dim=-1)
            loss_name = "mse"
        
        # set zero for norendered datasets
        weight = torch.ones(B, 1, 1, 1, device=pred_colors.device)
        for i, name in enumerate(dataset_name):
            if name in self.norender_datatsets:
                weight[i] *= 0.
        pixel_loss *= weight
                
        # Calculate final loss based on reduction method
        if self.reduction == 'mean':
            loss_value = pixel_loss.sum() / (valid_mask * weight).sum().clamp_min(1)
        elif self.reduction == 'sum':
            loss_value = (pixel_loss * valid_mask).sum()
        else:
            loss_value = pixel_loss
        
        # Create loss dictionary
        loss_dict = {
            f"render_{loss_name}": loss_value.item()
        }
        return loss_value, loss_dict
    
    @property
    def name(self):
        suffix = "_l1" if self.use_l1 else ""
        return f"RenderMSELoss{suffix}"


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


class RenderPerceptualLoss(BaseLoss):
    """Rendering Perceptual Loss"""
    
    def __init__(self, spatial=True, mask_mode=None, norender_datatsets=["MegaDepth", "VirtualKITTI2", "PointOdyssey"]):
        super().__init__()
        self.lpips = LPIPS(net="vgg", spatial=spatial)
        convert_to_buffer(self.lpips, persistent=False)
        self.spatial = spatial
        self.mask_mode = mask_mode
        self.norender_datatsets = norender_datatsets
    
    def compute_loss(self, preds, context_view_nums, dataset_name=None):
        pred_colors = check_and_fix_inf_nan(preds["rendered_colors"], "render_colors")
        gt_colors = check_and_fix_inf_nan(preds["gt_colors"], "gt_colors")
        valid_mask = check_and_fix_inf_nan(preds["valid_masks"].clone(), "valid_masks")

        B, V, H, W, C = gt_colors.shape

        mask = torch.full_like(valid_mask, fill_value=True)
        if self.mask_mode == "context_mask + target_mask":
            valid_mask = valid_mask
        elif self.mask_mode == "context_mask":
            mask[:, :context_view_nums, :, :] = valid_mask[:, :context_view_nums, :, :]
            valid_mask = mask
        elif self.mask_mode == "target_mask":
            mask[:, context_view_nums:, :, :] = valid_mask[:, context_view_nums:, :, :]
            valid_mask = mask
        else:
            valid_mask = mask

        if V == 1 or valid_mask.sum() == 0:
            loss_value = torch.tensor(0.).to(pred_colors.device)
            loss_dict = {
                f"render_lpips": loss_value.item()
            }
            return loss_value, loss_dict
        pred_colors = pred_colors * valid_mask[..., None]
        gt_colors = gt_colors * valid_mask[..., None]

        loss_value = []
        if not self.spatial:
            for b in range(B):
                loss_value_perbatch = self.lpips.forward(
                    pred_colors[b].permute(0, 3, 1, 2),
                    gt_colors[b].permute(0, 3, 1, 2),
                    normalize=True
                    )
                loss_value.append(loss_value_perbatch)
        else:
            for b in range(B):
                loss_value_perbatch = self.lpips.forward(
                    pred_colors[b].permute(0, 3, 1, 2),
                    gt_colors[b].permute(0, 3, 1, 2),
                    normalize=True
                ) * valid_mask[b][:, None]
                loss_value.append(loss_value_perbatch)
        loss_value = torch.stack(loss_value)
    
        # set zero for norendered datasets
        weight = torch.ones(B, V, 1, 1, 1, device=pred_colors.device)
        for i, name in enumerate(dataset_name):
            if name in self.norender_datatsets:
                weight[i] *= 0.
        loss_value *= weight
        if not self.spatial:
            loss_value = loss_value.sum() / weight.sum().clamp_min(1)
        else:
            loss_value = loss_value.sum() / (valid_mask[..., None] * weight).sum().clamp_min(1)
        loss_dict = {
            f"render_lpips": loss_value.item()
        }
        return loss_value, loss_dict
    
    @property
    def name(self):
        return "RenderPerceptualLoss"


class RenderDepthLoss(BaseLoss):
    """GSRender Depth Loss"""
    
    def __init__(self, depth_head_sup=True, depth_conf_mask=True, gradient_loss_fn=None,
                 gamma=1.0, alpha=0.2, valid_range=-1, mask_mode=None, eps=1e-6, 
                 norender_datatsets=["MegaDepth", "VirtualKITTI2", "PointOdyssey"]):
        super().__init__()
        self.depth_head_sup = depth_head_sup
        self.depth_conf_mask = depth_conf_mask
        self.gradient_loss_fn = gradient_loss_fn
        self.gamma = gamma
        self.alpha = alpha
        self.valid_range = valid_range
        self.mask_mode = mask_mode
        self.eps = eps
        self.norender_datatsets = norender_datatsets
    
    def compute_loss(self, preds, context_view_nums, dataset_name=None):
        depth = check_and_fix_inf_nan(preds["rendered_depths"], "render_depths")
        if self.depth_head_sup:
            gt_depth = preds["depth"].detach()
        else:
            gt_depth = preds["gt_depths"].unsqueeze(-1)
        if self.depth_conf_mask:
            valid_mask = preds["depth_conf"].detach()
        else:
            valid_mask = preds["valid_masks"].detach()

        gt_depth = check_and_fix_inf_nan(gt_depth, "render_gt_depth")
        valid_mask = valid_mask.clone()
        mask = torch.full_like(valid_mask, fill_value=True, dtype=bool)
        if self.mask_mode == "context_mask + target_mask":
            valid_mask = valid_mask
        elif self.mask_mode == "context_mask":
            mask[:, :context_view_nums, :, :] = valid_mask[:, :context_view_nums, :, :]
            valid_mask = mask
        elif self.mask_mode == "target_mask":
            mask[:, context_view_nums:, :, :] = valid_mask[:, context_view_nums:, :, :]
            valid_mask = mask
        else:
            valid_mask = mask
        if depth.shape[1] == 1 or valid_mask.sum() < 100:
            # If there are less than 100 valid points, skip this batch
            dummy_loss = (0.0 * depth).mean()
            loss_dict = {f"loss_conf_renderdepth": dummy_loss,
                        f"loss_reg_renderdepth": dummy_loss,
                        f"loss_grad_renderdepth": dummy_loss,}
        else:
            gt_depth, valid_mask = gt_depth[:, :depth.shape[1]], valid_mask[:, :depth.shape[1]]
            
            weight = torch.ones(gt_depth.shape[0], 1, 1, 1, device=gt_depth.device)
            for i, name in enumerate(dataset_name):
                if name in self.norender_datatsets:
                    weight[i] *= 0.
            dataset_mask = weight > 0
            valid_mask = valid_mask & dataset_mask.expand_as(valid_mask)
            
            loss_conf, loss_grad, loss_reg = regression_loss(
                depth, gt_depth, valid_mask,
                conf=None,
                gradient_loss_fn=self.gradient_loss_fn, 
                gamma=self.gamma, 
                alpha=self.alpha, 
                valid_range=self.valid_range,
            )
            
            loss_dict = {
                f"loss_conf_renderdepth": loss_conf,
                f"loss_reg_renderdepth": loss_reg,    
                f"loss_grad_renderdepth": loss_grad,
            }
        if "only" in self.gradient_loss_fn:
            loss_value = loss_dict["loss_grad_renderdepth"] 
        else:
            loss_value = loss_dict["loss_conf_renderdepth"] + loss_dict["loss_reg_renderdepth"] +loss_dict["loss_grad_renderdepth"] 

        return loss_value, loss_dict
    
    @property
    def name(self):
        name = f"RenderDepthLoss"
        return name

class GSDepthLoss(BaseLoss):
    """Gaussian Splatting depth loss"""
    
    def __init__(self, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=0.98):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.gradient_loss_fn = gradient_loss_fn
        self.valid_range = valid_range

    def compute_loss(self, predictions, batch):
        loss_dict = compute_gsdepth_loss(
            predictions, 
            batch,
            gamma=self.gamma,
            alpha=self.alpha,
            gradient_loss_fn=self.gradient_loss_fn,
            valid_range=self.valid_range
        )
        loss = loss_dict["loss_conf_gsdepth"] + loss_dict["loss_reg_gsdepth"] + loss_dict["loss_grad_gsdepth"]

        return loss, loss_dict
    
    @property
    def name(self):
        return f"GSDepthLoss"


def compute_gsdepth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute gsdepth loss.
    
    Args:
        predictions: Dict containing 'gs_depth' and 'gs_depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['gs_depth']
    pred_depth_conf = predictions['gs_depth_conf']

    gt_depth = batch['depthmap'][:, :pred_depth.shape[1]]
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)
    gt_depth_mask = batch['valid_mask'].clone()[:, :pred_depth.shape[1]]   # 3D points derived from depth map, so we use the same mask

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_gsdepth": dummy_loss,
                    f"loss_reg_gsdepth": dummy_loss,
                    f"loss_grad_gsdepth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_gsdepth": loss_conf,
        f"loss_reg_gsdepth": loss_reg,    
        f"loss_grad_gsdepth": loss_grad,
    }

    return loss_dict
