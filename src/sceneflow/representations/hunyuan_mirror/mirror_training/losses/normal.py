import torch
import numpy as np
from einops import rearrange

from .container import BaseLoss
from .utils import check_and_fix_inf_nan, Depth2Normal


class NormalLoss(BaseLoss):
    def __init__(self, loss_type='AL', real_weight=1.0, pseudo_weight=1.0, ignore_datasets=[], real_datasets=[], **kwargs):
        """loss_fn can be one of following:
            - L1            - L1 loss (no uncertainty)
            - AL            - Angular loss (no uncertainty)
            - NLL_vMF       - NLL of vonMF distribution
        """
        super().__init__()
        self.loss_type = loss_type
        self.real_weight = real_weight
        self.pseudo_weight = pseudo_weight
        self.ignore_datasets = ignore_datasets
        self.real_datasets = real_datasets
        
        self.depth2normal = Depth2Normal()
        
    def compute_loss(self, preds, gts, dataset_name):
        # gt normals
        if 'normals' in gts:
            gt_norm = gts['normals']
            gt_norm_mask = gts['valid_mask']
        else:
            B, S, _, H, W = gts['img'].shape
            depthmap = gts['depthmap'].reshape(B*S, 1, H, W)
            intrinsics = gts['camera_intrs'].reshape(B*S, 3, 3)
            masks = gts['valid_mask'].reshape(B*S, 1, H, W)
            gt_norm, norm_mask = self.depth2normal(depthmap, intrinsics, masks, scale=1.0)
            gt_norm = gt_norm.reshape(B, S, 3, H, W)
            gt_norm_mask = norm_mask.reshape(B, S, 1, H, W)

        gt_norm = check_and_fix_inf_nan(gt_norm, "gt_norm")
        gt_norm_mask = gt_norm_mask & gts['valid_mask'].reshape(B, S, 1, H, W)
          
        if not gt_norm_mask.any():
            return torch.tensor(0.0, device=gt_norm_mask.device, requires_grad=True)
        
        # pred normals
        pred_norm = preds['normals'].permute(0, 1, 4, 2, 3)
        pred_norm_conf = preds['normals_conf'].unsqueeze(2)
        
        if pred_norm.ndim == 5:
            pred_norm = rearrange(pred_norm, 'b s c h w -> (b s) c h w')
        if gt_norm.ndim == 5:
            gt_norm = rearrange(gt_norm, 'b s c h w -> (b s) c h w')
        if gt_norm_mask.ndim == 5:
            gt_norm_mask = rearrange(gt_norm_mask, 'b s c h w -> (b s) c h w')
        if pred_norm_conf.ndim == 5:
            pred_norm_conf = rearrange(pred_norm_conf, 'b s c h w -> (b s) c h w')

        if torch.isnan(gt_norm).any() or torch.isinf(gt_norm).any():
            return torch.tensor(0.0, device=gt_norm.device), {"loss_normal": torch.tensor(0.0, device=gt_norm.device)}
        
        weight = torch.ones((B, S), device=gt_norm.device)
        for bi in range(B):
            for i, name in enumerate(dataset_name[0]):
                if name in self.ignore_datasets:
                    weight[bi, :] *= 0.0
                elif name in self.real_datasets:
                    weight[bi, :] *= self.real_weight
                else:
                    weight[bi, :] *= self.pseudo_weight
        weight = weight.reshape(B*S)
                
        if 'NLL' in self.loss_type:
            pred_norm, pred_kappa = pred_norm[:, 0:3, :, :], pred_norm_conf
        else:
            pred_norm = pred_norm

        if self.loss_type == 'L1':
            l1 = torch.sum(torch.abs(gt_norm - pred_norm), dim=1, keepdim=True)
            l1 = weight[:, None, None] * l1
            loss = check_and_fix_inf_nan(torch.mean(l1[gt_norm_mask]), "loss_normal")

        elif self.loss_type == 'AL':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask *= weight[:, None, None]
            valid_mask = valid_mask > 0.0
            if valid_mask.sum() == 0:
                loss = check_and_fix_inf_nan(dot.mean() * 0.,  "loss_normal")
            else:
                al = torch.acos(dot[valid_mask])
                al *= weight[:, None, None].expand_as(valid_mask)[valid_mask]
                loss = check_and_fix_inf_nan(torch.mean(al), "loss_normal")

        elif self.loss_type == 'NLL_vMF':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            dot = dot[valid_mask]
            kappa = pred_kappa[:, 0, :, :][valid_mask]

            loss_pixelwise = - torch.log(kappa) \
                             - (kappa * (dot - 1)) \
                             + torch.log(1 - torch.exp(- 2 * kappa))
            loss_pixelwise *= weight[:, None, None].expand_as(valid_mask)[valid_mask]
            loss = check_and_fix_inf_nan(torch.mean(loss_pixelwise), "loss_normal")
        else:
            raise Exception('invalid loss type')

        loss_dict = {
            "loss_normal": loss
        }
        return loss_dict["loss_normal"], loss_dict

    @property
    def name(self):
        name = f"NormalLoss"
        return name