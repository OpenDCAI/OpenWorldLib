import torch
import torch.nn.functional as F
import lpips


class RenderingMetrics:
    """Class for computing rendering quality metrics (PSNR, SSIM, LPIPS) on batched images"""
    
    def __init__(self):
        """Initialize rendering metrics calculator
        
        Args:
            device: The torch device to use for calculations
        """
        self.lpips_fn = lpips.LPIPS(net='alex', spatial=True)
    
    def compute_psnr(self, pred, target, mask=None):
        """Compute PSNR (Peak Signal-to-Noise Ratio)
        
        Args:
            pred: Predicted images (B, H, W, 3)
            target: Ground truth images (B, H, W, 3)
            mask: (B, H, W)

        Returns:
            PSNR values (B,)
        """
        if mask is None:
            mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        else:
            mask = mask[..., None].repeat(1, 1, 1, 3)
            mse = ((pred - target) ** 2 * mask).sum(dim=(1, 2, 3))/(mask.sum(dim=(1, 2, 3)) + 1e-6)
        return -10 * torch.log10(mse + 1e-8)
    
    def compute_ssim(self, pred, target, window_size=11, mask=None):
        """Compute SSIM (Structural Similarity Index)
        
        Args:
            pred: Predicted images (B, H, W, 3)
            target: Ground truth images (B, H, W, 3)
            window_size: Size of the Gaussian window
            mask: (B, H, W)
            
        Returns:
            SSIM values (B,)
        """
        if mask is not None:
            mask = mask[:, None].repeat(1, 3, 1, 1)
        # Convert to format [B, C, H, W]
        img1 = pred.permute(0, 3, 1, 2)
        img2 = target.permute(0, 3, 1, 2)
        
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Gaussian kernel
        kernel_size = window_size
        sigma = 1.5
        
        # Create 1D kernels
        x_kernel = torch.arange(kernel_size, device=img1.device) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-0.5 * (x_kernel / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D kernels (separable convolution)
        kernel = kernel_1d.view(1, 1, kernel_size, 1) * kernel_1d.view(1, 1, 1, kernel_size)
        kernel = kernel.expand(3, 1, kernel_size, kernel_size).contiguous()
        
        # Pad images for convolution
        padding = (window_size - 1) // 2
        
        batch_size = img1.shape[0]
        ssim_values = []
        
        for i in range(batch_size):
            # Process each image in the batch individually
            img1_i = img1[i:i+1]
            img2_i = img2[i:i+1]
            
            mu1 = F.conv2d(img1_i, kernel.to(img1.device), padding=padding, groups=3)
            mu2 = F.conv2d(img2_i, kernel.to(img2.device), padding=padding, groups=3)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.conv2d(img1_i ** 2, kernel.to(img1.device), padding=padding, groups=3) - mu1_sq
            sigma2_sq = F.conv2d(img2_i ** 2, kernel.to(img2.device), padding=padding, groups=3) - mu2_sq
            sigma12 = F.conv2d(img1_i * img2_i, kernel.to(img1.device), padding=padding, groups=3) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            if mask is None:
                ssim_values.append(ssim_map.mean())
            else:
                ssim_values.append((ssim_map * mask[i:i+1]).sum()/ (mask[i:i+1].sum() + 1e-6))
        
        return torch.stack(ssim_values)
    
    def compute_lpips(self, pred, target, mask=None):
        """Compute LPIPS (Learned Perceptual Image Patch Similarity)
        
        Args:
            pred: Predicted images (B, H, W, 3)
            target: Ground truth images (B, H, W, 3)
            mask: (B, H, W)
            
        Returns:
            LPIPS values (B,) or None if lpips not available
        """
        if mask is not None:
            mask = mask[:, None].repeat(1, 3, 1, 1)
            
        # Convert to range [-1, 1] for LPIPS
        pred_norm = pred.permute(0, 3, 1, 2) * 2 - 1
        target_norm = target.permute(0, 3, 1, 2) * 2 - 1
        
        batch_size = pred.shape[0]
        lpips_values = []
        
        with torch.no_grad():
            for i in range(batch_size):
                lpips_value = self.lpips_fn(pred_norm[i:i+1], target_norm[i:i+1])
                if mask is None:
                    lpips_values.append(lpips_value.mean().item())
                else:
                    lpips_values.append((lpips_value * mask[i:i+1]).sum()/(mask[i:i+1].sum() + 1e-6).item())
            
        return torch.tensor(lpips_values)
    
    def __call__(self, pred, target, mask=None):
        """Compute all metrics between predicted and target images
        
        Args:
            pred: Predicted images (B, H, W, 3)
            target: Ground truth images (B, H, W, 3)
            mask: (B, H, W)
            
        Returns:
            Dictionary with PSNR, SSIM, and LPIPS values (batch averages)
        """
        # Ensure tensors are on the correct device
        pred = pred
        target = target
        if mask is not None:
            pred = pred * mask[..., None]
            target = target * mask[..., None]
        
        # Compute metrics
        psnr_values = self.compute_psnr(pred, target, mask=mask)
        ssim_values = self.compute_ssim(pred, target, mask=mask)
        
        lpips_values = self.compute_lpips(pred, target, mask=mask)
        
        # Return batch averages
        return {
            "psnr": psnr_values.mean().item(),
            "ssim": ssim_values.mean().item(),
            "lpips": lpips_values.mean().item()
        }
    
    def compute_metrics_per_sample(self, pred, target, mask=None):
        """Compute metrics for each sample in the batch
        
        Args:
            pred: Predicted images (B, H, W, 3)
            target: Ground truth images (B, H, W, 3)
            mask: (B, H, W)
            
        Returns:
            Dictionary with per-sample PSNR, SSIM, and LPIPS values
        """
        # Ensure tensors are on the correct device
        pred = pred
        target = target
        pred = pred * mask[..., None]
        target = target * mask[..., None]
        
        # Compute metrics
        psnr_values = self.compute_psnr(pred, target, mask=mask)
        ssim_values = self.compute_ssim(pred, target, mask=mask)
        
        if self.has_lpips:
            lpips_values = self.compute_lpips(pred, target, mask=mask)
        else:
            lpips_values = torch.zeros_like(psnr_values)
        
        # Return per-sample metrics
        return {
            "psnr": psnr_values.cpu().tolist(),
            "ssim": ssim_values.cpu().tolist(),
            "lpips": lpips_values.cpu().tolist() if self.has_lpips else [0.0] * len(psnr_values)
        }