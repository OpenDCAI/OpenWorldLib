import os
import torch
import numpy as np
from typing import Dict, Any, Optional

from huggingface_hub import snapshot_download

from .pi3x.pi3x import Pi3X
from .pi3.utils.geometry import depth_edge
from ...base_representation import BaseRepresentation


class Pi3XRepresentation(BaseRepresentation):
    """
    Pi3X representation model for 3D point cloud reconstruction
    with multimodal conditioning support.
    """

    def __init__(
        self,
        model=None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model_type = "pi3x"

        if self.model is not None:
            self.model = self.model.to(self.device).eval()

        if self.device == "cuda" and torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if compute_capability >= 8 else torch.float16
        else:
            self.dtype = torch.float32

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[str] = None,
        **kwargs,
    ) -> "Pi3XRepresentation":
        """
        Load a pretrained Pi3X model.
        Returns: Pi3XRepresentation instance with loaded model.
        """
        try:
            model = Pi3X.from_pretrained(pretrained_model_path)
        except Exception:
            if os.path.isdir(pretrained_model_path):
                model_root = pretrained_model_path
            else:
                print(f"Downloading weights from HuggingFace: {pretrained_model_path}")
                model_root = snapshot_download(pretrained_model_path)
                print(f"Model downloaded to: {model_root}")

            ckpt_path = os.path.join(model_root, "model.safetensors")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"model.safetensors not found in {model_root}")

            from safetensors.torch import load_file
            model = Pi3X()
            model.load_state_dict(load_file(ckpt_path), strict=False)

        return cls(model=model, device=device)

    def api_init(self, api_key: str, endpoint: str):
        """Placeholder for future API-based models."""
        pass

    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Pi3X inference and return all outputs.
        Returns: Dict with numpy arrays for all model outputs.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() first.")

        imgs = data["images"]
        if not isinstance(imgs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for 'images', got {type(imgs)}")
        imgs = imgs.to(self.device)

        conf_threshold = data.get("conf_threshold", 0.1)
        edge_rtol = data.get("edge_rtol", 0.03)

        results = {}

        with torch.no_grad():
            autocast_enabled = self.device == "cuda"
            with torch.amp.autocast("cuda", dtype=self.dtype, enabled=autocast_enabled):
                condition_keys = [
                    "poses", "depths", "intrinsics", "rays",
                    "mask_add_depth", "mask_add_ray", "mask_add_pose",
                ]
                conditions = {}
                for key in condition_keys:
                    val = data.get(key)
                    if val is not None:
                        if isinstance(val, np.ndarray):
                            val = torch.from_numpy(val).float()
                        if isinstance(val, torch.Tensor):
                            val = val.to(self.device)
                        conditions[key] = val
                res = self.model(imgs=imgs, **conditions)

        results["points"] = res["points"].cpu().numpy()              # (B, N, H, W, 3)
        results["local_points"] = res["local_points"].cpu().numpy()  # (B, N, H, W, 3)
        results["camera_poses"] = res["camera_poses"].cpu().numpy()  # (B, N, 4, 4)
        results["conf"] = res["conf"].cpu().numpy()                  # (B, N, H, W, 1)

        if "rays" in res:
            results["rays"] = res["rays"].cpu().numpy()              # (B, N, H, W, 3)
        if "metric" in res:
            results["metric"] = res["metric"].cpu().numpy()          # (B,)

        # Quality masks: confidence + depth-edge filtering
        conf_tensor = res["conf"]
        masks = torch.sigmoid(conf_tensor[..., 0]) > conf_threshold
        non_edge = ~depth_edge(res["local_points"][..., 2], rtol=edge_rtol)
        masks = torch.logical_and(masks, non_edge)
        results["masks"] = masks.cpu().numpy()                       # (B, N, H, W)

        results["depth_map"] = res["local_points"][..., 2].cpu().numpy()  # (B, N, H, W)

        return results
