import torch
import torch.nn.functional as F
from typing import Dict

from .dpt import DepthAnything
from .util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose


class DepthAnythingAdapter:
    """Adapter class to make DepthAnything compatible with the depth model interface."""

    def __init__(self, model: DepthAnything, device: str = "cuda"):
        """
        Initialize DepthAnything adapter.

        Args:
            model: DepthAnything model instance
            device: Device string ('cuda' or 'cpu')
        """
        self.model = model.to(device).eval()
        self.device = device
        self.transform = Compose(
            [
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=3,  # cv2.INTER_CUBIC
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def infer(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inference method compatible with MoGeModel interface.

        Args:
            image_tensor: Input image tensor of shape (B, 3, H, W) or (3, H, W)

        Returns:
            Dictionary with 'depth' key containing depth map of shape (H, W)
        """
        # Handle batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Get original image dimensions
        original_h, original_w = image_tensor.shape[-2:]

        # Convert tensor to numpy for DepthAnything processing
        # DepthAnything expects RGB image in range [0, 1]
        if image_tensor.max() > 1.0:
            image_np = image_tensor[0].permute(1, 2, 0).cpu().numpy() / 255.0
        else:
            image_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()

        # Ensure RGB format (DepthAnything expects RGB)
        # Check if it's BGR by comparing first and last channel means
        if image_np.shape[2] == 3:
            # Simple heuristic: if first channel mean > last channel mean, likely BGR
            if image_np[..., 0].mean() > image_np[..., 2].mean():
                image_np = image_np[..., ::-1]  # BGR to RGB

        # Prepare tensor for DepthAnything
        image = self.transform({"image": image_np})["image"]
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            depth = self.model(tensor)

        # Interpolate to original size
        depth = F.interpolate(
            depth[None], (original_h, original_w), mode="bilinear", align_corners=False
        )[0, 0]

        # DepthAnything outputs relative depth values (larger = closer to camera)
        # For point cloud generation, we typically want absolute depth where larger = farther
        # Convert relative depth to a reasonable depth range
        # The depth values from DepthAnything are typically in a small range
        # We'll normalize and scale to a reasonable metric depth range

        # Invert so larger values = farther (standard depth map convention)
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max > depth_min:
            # Normalize to [0, 1] then invert
            depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
            # Invert: closer objects get larger depth values
            depth = (1.0 - depth_normalized) * 100.0  # Scale to 0-100 meters range
        else:
            depth = torch.ones_like(depth) * 50.0  # Default depth if all values are same

        if squeeze_batch:
            depth = depth.squeeze(0)

        return {"depth": depth}


__all__ = ["DepthAnythingAdapter"]


