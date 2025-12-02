import torch
import torch.nn as nn
import math
from typing import List


class BaseLoss(nn.Module):
    """
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    def __repr__(self):
        name = self.name
        return name

    def forward(self, *args, **kwargs):
        loss, details = self.compute_loss(*args, **kwargs)
        return loss, details


class LossContainer(nn.Module):
    """
    Container class for combining multiple loss functions
    """
    def __init__(
        self, 
        enable_cam: bool = True, 
        enable_pts: bool = True, 
        enable_depth: bool = True, 
        enable_norm: bool = False, 
        enable_gs: bool = False,
        losses: List[nn.Module] = None, 
        weights: List[float] = None
    ):
        super().__init__()
        filtered_losses = []
        filtered_weights = []

        loss_enable_map = {
            'CameraLoss': enable_cam,
            'PointLoss': enable_pts,
            'DepthLoss': enable_depth,
            'NormalLoss': enable_norm,
            'RenderMSELoss': enable_gs,
            'RenderPerceptualLoss': enable_gs,
            'RenderDepthLoss': enable_gs,
            'GSDepthLoss': enable_gs
        }
        
        for (loss, weight) in zip(losses, weights):
            if loss_enable_map.get(loss.name, False):
                filtered_losses.append(loss)
                filtered_weights.append(weight)

        self.losses = nn.ModuleList(filtered_losses)
        self.weights = filtered_weights
        
        if len(self.weights) != len(self.losses):
            raise ValueError(f"Number of weights ({len(self.weights)}) must equal number of losses ({len(self.losses)})")
    
    def forward(self, gts, preds, *args, **kwargs):
        total_loss = 0
        combined_dict = {}
        
        for i, (loss_fn, weight) in enumerate(zip(self.losses, self.weights)):
            if loss_fn.name == 'NormalLoss':
                loss, loss_dict = loss_fn(preds, gts, dataset_name=gts['dataset'])
            elif loss_fn.name in ['RenderMSELoss', 'RenderPerceptualLoss', 'RenderDepthLoss']:
                context_views = (gts["is_target"][0] == False).sum().item()
                loss, loss_dict = loss_fn(preds, context_views, dataset_name=gts['dataset'])
            else:
                loss, loss_dict = loss_fn(preds, gts, *args, **kwargs)
            total_loss = total_loss + weight * loss
            for key, value in loss_dict.items():
                combined_dict[f"{key}"] = value
        combined_dict["total_loss"] = total_loss

        if "bad_case" in gts and gts["bad_case"].any():
            total_loss, combined_dict = self._skip_bad_case(total_loss, combined_dict)
        
        return total_loss, combined_dict

    def _skip_bad_case(self, total_loss, loss_dict):
        """Skip bad cases by zeroing out losses with NaN/Inf values"""
        # Zero out total loss if it contains NaN or Inf
        total_loss = torch.zeros_like(total_loss)
        
        # Zero out all loss dict values
        for key, val in list(loss_dict.items()):
            if torch.is_tensor(val):
                loss_dict[key] = torch.zeros_like(val)
            elif isinstance(val, (float, int)):
                loss_dict[key] = 0.0
        
        return total_loss, loss_dict