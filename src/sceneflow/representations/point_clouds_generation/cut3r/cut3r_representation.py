import os
import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from PIL import Image

from huggingface_hub import snapshot_download

# Try to import gdown for Google Drive downloads
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False
    print("Warning: gdown not installed. Install with 'pip install gdown' to enable Google Drive downloads.")

# Import CUT3R modules using relative imports
from .cut3r import (
    ARCroco3DStereo,
    inference,
    load_images,
    pose_encoding_to_camera,
    estimate_focal_knowing_depth,
    geotrf,
)


# CUT3R model registry mapping model names to Google Drive file IDs
CUT3R_MODEL_REGISTRY = {
    "cut3r_224_linear_4": {
        "file_id": "11dAgFkWHpaOHsR6iuitlB_v4NFFBrWjy",
        "filename": "cut3r_224_linear_4.pth",
        "size": 224,
    },
    "cut3r_512_dpt_4_64": {
        "file_id": "1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD",
        "filename": "cut3r_512_dpt_4_64.pth",
        "size": 512,
    },
}


class CUT3RRepresentation:
    """
    Representation for CUT3R 3D scene reconstruction.
    """
    
    def __init__(self, model: Optional[ARCroco3DStereo] = None, device: Optional[str] = None):
        """
        Initialize CUT3R representation model.
        
        Args:
            model: Pre-loaded ARCroco3DStereo model (optional)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        
        if self.model is not None:
            self.model = self.model.to(self.device).eval()
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[str] = None,
        size: Optional[int] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> 'CUT3RRepresentation':
        """
        Create representation instance from pretrained model.
        
        Args:
            pretrained_model_path: Model identifier - can be:
                - CUT3R model name: "cut3r_224_linear_4" or "cut3r_512_dpt_4_64"
                - HuggingFace repo ID (e.g., "username/repo")
                - Local path to model checkpoint or directory
            device: Device to run on
            size: Input image size (auto-detected from model name if not specified)
            cache_dir: Directory to cache downloaded models (default: ~/.cache/cut3r)
            **kwargs: Additional arguments
            
        Returns:
            CUT3RRepresentation instance
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if it's a registered CUT3R model name
        if pretrained_model_path in CUT3R_MODEL_REGISTRY:
            model_info = CUT3R_MODEL_REGISTRY[pretrained_model_path]
            model_path = cls._download_from_google_drive(
                model_info["file_id"],
                model_info["filename"],
                cache_dir=cache_dir
            )
            # Auto-detect size from model name if not specified
            if size is None:
                size = model_info["size"]
        elif os.path.isdir(pretrained_model_path):
            # Local directory
            model_path = pretrained_model_path
            # Look for .pth file in the directory
            pth_files = list(Path(model_path).glob("*.pth"))
            if pth_files:
                model_path = str(pth_files[0])
        elif os.path.isfile(pretrained_model_path):
            # Local file path
            model_path = pretrained_model_path
        else:
            # Try to download from HuggingFace
            print(f"Downloading weights from HuggingFace repo: {pretrained_model_path}")
            try:
                downloaded_path = snapshot_download(pretrained_model_path, cache_dir=cache_dir)
                # Look for .pth file in the downloaded directory
                pth_files = list(Path(downloaded_path).glob("*.pth"))
                if pth_files:
                    model_path = str(pth_files[0])
                else:
                    # If no .pth file, check if the directory itself contains model files
                    model_path = downloaded_path
                print(f"Model downloaded to: {model_path}")
            except Exception as e:
                print(f"Warning: Could not download from HuggingFace: {e}")
                print(f"Trying to use as local path: {pretrained_model_path}")
                model_path = pretrained_model_path
        
        # Load model
        try:
            # ARCroco3DStereo.from_pretrained expects a file path, not a directory
            if os.path.isdir(model_path):
                # If it's a directory, look for checkpoint file
                pth_files = list(Path(model_path).glob("*.pth"))
                if pth_files:
                    model_path = str(pth_files[0])
                else:
                    raise ValueError(f"No .pth file found in {model_path}")
            
            model = ARCroco3DStereo.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load CUT3R model from {model_path}: {e}")
        
        instance = cls(model=model, device=device)
        # Use auto-detected size or default
        instance.size = size if size is not None else 224
        return instance
    
    @staticmethod
    def _download_from_google_drive(
        file_id: str,
        filename: str,
        cache_dir: Optional[str] = None
    ) -> str:
        """
        Download model from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            filename: Output filename
            cache_dir: Cache directory (default: ~/.cache/cut3r)
            
        Returns:
            Path to downloaded model file
        """
        if not HAS_GDOWN:
            raise RuntimeError(
                "gdown is required for Google Drive downloads. "
                "Install with: pip install gdown"
            )
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "cut3r")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if file already exists
        output_path = os.path.join(cache_dir, filename)
        if os.path.exists(output_path):
            print(f"Using cached model: {output_path}")
            return output_path
        
        # Download from Google Drive
        print(f"Downloading CUT3R model from Google Drive: {filename}")
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, output_path, quiet=False)
            print(f"Model downloaded to: {output_path}")
            return output_path
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model from Google Drive (file_id: {file_id}): {e}\n"
                f"Please check your internet connection and try again."
            )
    
    def api_init(self, api_key: str, endpoint: str):
        """Initialize API connection if needed."""
        pass
    
    def _prepare_views(
        self,
        images: Union[np.ndarray, List[np.ndarray], List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare input views for CUT3R inference.
        
        Args:
            images: List of image paths, numpy arrays, or single numpy array
            
        Returns:
            List of view dictionaries
        """
        # Convert to list if single image
        if isinstance(images, np.ndarray):
            images = [images]
        
        # Convert numpy arrays to file paths (temporary) if needed
        # CUT3R's load_images expects file paths
        import tempfile
        temp_files = []
        image_paths = []
        
        for img in images:
            if isinstance(img, str):
                image_paths.append(img)
            elif isinstance(img, np.ndarray):
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                # Convert to uint8 and save
                img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                Image.fromarray(img_uint8).save(temp_file.name)
                image_paths.append(temp_file.name)
                temp_files.append(temp_file.name)
        
        # Load images using CUT3R's loader
        loaded_images = load_images(image_paths, size=self.size, verbose=False)
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        # Convert to views format
        views = []
        for i, img_data in enumerate(loaded_images):
            view = {
                "img": img_data["img"],
                "ray_map": torch.full(
                    (
                        img_data["img"].shape[0],
                        6,
                        img_data["img"].shape[-2],
                        img_data["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(img_data["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            views.append(view)
        
        return views
    
    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get 3D scene representation from input data.
        
        Args:
            data: Dictionary containing:
                - 'images': List of image paths, numpy arrays, or single numpy array
                - 'output_type': str, "point_cloud", "depth_map", "camera_pose", or "all"
                - Optional: 'size': int, input image size (default: self.size)
                - Optional: 'vis_threshold': float, confidence threshold for filtering point clouds (default: 1.0)
                
        Returns:
            Dictionary containing:
                - 'point_cloud': List of point clouds (if output_type includes "point_cloud" or "all")
                - 'depth_map': List of depth maps (if output_type includes "depth_map" or "all")
                - 'camera_pose': List of camera poses (if output_type includes "camera_pose" or "all")
                - 'colors': List of color maps for point clouds
                - 'confidence': List of confidence maps
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() first.")
        
        images = data['images']
        output_type = data.get('output_type', 'all')
        size = data.get('size', self.size)
        vis_threshold = data.get('vis_threshold', 1.0)
        
        # Prepare views
        views = self._prepare_views(images)
        
        # Run inference
        with torch.no_grad():
            outputs, state_args = inference(views, self.model, self.device, verbose=False)
        
        # Process outputs
        results = {}
        
        # Extract predictions
        pts3ds_self = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
        conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
        colors = [
            0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0).cpu() 
            for output in outputs["views"]
        ]
        
        # Recover camera poses
        pr_poses = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in outputs["pred"]
        ]
        
        # Transform points to world coordinates
        pts3ds_world = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            pts3d_world = geotrf(pose, pself.unsqueeze(0))
            pts3ds_world.append(pts3d_world)
        
        # Estimate focal length
        B, H, W, _ = pts3ds_self[0].shape
        pp = torch.tensor([W // 2, H // 2], device=pts3ds_self[0].device).float().repeat(B, 1)
        focal = estimate_focal_knowing_depth(pts3ds_self[0], pp, focal_mode="weiszfeld")
        
        # Convert to numpy and apply vis_threshold filtering
        if output_type in ["point_cloud", "all"]:
            filtered_pcs = []
            filtered_colors = []
            for pc_world, color, conf in zip(pts3ds_world, colors, conf_self):
                pc_np = pc_world.numpy()
                color_np = color.numpy()
                conf_np = conf.numpy()
                
                # Apply vis_threshold filtering if confidence is available
                if vis_threshold > 0:
                    pc_flat = pc_np.reshape(-1, 3)
                    color_flat = color_np.reshape(-1, 3)
                    conf_flat = conf_np.reshape(-1)
                    
                    # Filter points with confidence > vis_threshold
                    mask = conf_flat > vis_threshold
                    if mask.sum() > 0:
                        pc_filtered = pc_flat[mask]
                        color_filtered = color_flat[mask]
                        # Reshape back to original shape if possible, otherwise keep as flat
                        # For visualization purposes, we keep it flat
                        filtered_pcs.append(pc_filtered)
                        filtered_colors.append(color_filtered)
                    else:
                        # If no points pass threshold, use all points
                        filtered_pcs.append(pc_np)
                        filtered_colors.append(color_np)
                else:
                    filtered_pcs.append(pc_np)
                    filtered_colors.append(color_np)
            
            results['point_cloud'] = filtered_pcs
            results['colors'] = filtered_colors
        
        if output_type in ["depth_map", "all"]:
            results['depth_map'] = [p[..., 2].numpy() for p in pts3ds_self]  # Z component is depth
            results['confidence'] = [c.numpy() for c in conf_self]
        
        if output_type in ["camera_pose", "all"]:
            results['camera_pose'] = [p.numpy() for p in pr_poses]
            results['focal'] = focal.cpu().numpy()
            results['principal_point'] = pp.cpu().numpy()
        
        # Store state_args for potential future use
        results['state_args'] = state_args
        
        return results

