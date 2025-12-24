import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Union

from huggingface_hub import snapshot_download

# Import VGGT model components from SceneFlow's vggt directory
from ....base_models.three_dimensions.point_clouds.vggt.vggt.models.vggt import VGGT
from ....base_models.three_dimensions.point_clouds.vggt.vggt.utils.load_fn import load_and_preprocess_images, load_and_preprocess_images_square
from ....base_models.three_dimensions.point_clouds.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from ....base_models.three_dimensions.point_clouds.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map



class VGGTRepresentation:
    """VGGT Representation model for 3D scene reconstruction."""
    
    def __init__(self, model: Optional[VGGT] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        
        if self.model is not None:
            self.model = self.model.to(self.device).eval()
            
            if self.device == "cuda" and torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_capability()[0]
                self.dtype = torch.bfloat16 if compute_capability >= 8 else torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = torch.float32
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> 'VGGTRepresentation':
        try:
            model = VGGT.from_pretrained(pretrained_model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load VGGT model: {e}")
        
        instance = cls(model=model, device=device)
        instance.preprocess_mode = kwargs.get('preprocess_mode', 'crop')
        instance.resolution = kwargs.get('resolution', 518)
        return instance
    
    def api_init(self, api_key: str, endpoint: str):
        """Initialize API access (placeholder for future API-based models)."""
        pass
    
    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get representation from input data using VGGT model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() first.")
        
        images_input = data['images']
        predict_cameras = data.get('predict_cameras', True)
        predict_depth = data.get('predict_depth', True)
        predict_points = data.get('predict_points', True)
        predict_tracks = data.get('predict_tracks', False)
        query_points = data.get('query_points', None)
        preprocess_mode = data.get('preprocess_mode', self.preprocess_mode)
        resolution = data.get('resolution', self.resolution)
        if isinstance(images_input, list):
            image_list = images_input
        elif isinstance(images_input, np.ndarray):
            if images_input.ndim == 3:
                image_list = [images_input]
            elif images_input.ndim == 4:
                image_list = [images_input[i] for i in range(images_input.shape[0])]
            else:
                image_list = [images_input]
        else:
            if isinstance(images_input, str):
                image_list = [images_input]
            else:
                image_list = images_input if isinstance(images_input, list) else [images_input]
        
        has_paths = any(isinstance(img, str) for img in image_list)
        
        if has_paths:
            if preprocess_mode == "square":
                images, _ = load_and_preprocess_images_square(image_list, target_size=resolution)
            else:
                images = load_and_preprocess_images(image_list, mode=preprocess_mode)
        else:
            image_tensors = []
            for img_array in image_list:
                if isinstance(img_array, np.ndarray):
                    if img_array.max() > 1.0:
                        img_array = img_array / 255.0
                    
                    if img_array.ndim == 3 and img_array.shape[2] == 3:
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
                    elif img_array.ndim == 2:
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()
                        img_tensor = img_tensor.repeat(3, 1, 1)
                    else:
                        raise ValueError(f"Unsupported image array shape: {img_array.shape}")
                    image_tensors.append(img_tensor)
                else:
                    raise ValueError(f"Unsupported image type: {type(img_array)}")
            
            images = torch.stack(image_tensors)
            images = F.interpolate(
                images,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False
            )
        
        images = images.to(self.device)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        query_points_tensor = None
        if predict_tracks and query_points is not None:
            if isinstance(query_points, np.ndarray):
                query_points_tensor = torch.FloatTensor(query_points).to(self.device)
            elif isinstance(query_points, torch.Tensor):
                query_points_tensor = query_points.to(self.device)
        
        results = {}
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype, enabled=(self.device == "cuda")):
                images_batch = images[None]
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_batch)
                
                if predict_cameras:
                    pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        pose_enc, images_batch.shape[-2:]
                    )
                    results['extrinsic'] = extrinsic.squeeze(0).cpu().numpy()
                    results['intrinsic'] = intrinsic.squeeze(0).cpu().numpy()
                
                if predict_depth:
                    depth_map, depth_conf = self.model.depth_head(
                        aggregated_tokens_list, images_batch, ps_idx
                    )
                    results['depth_map'] = depth_map.squeeze(0).cpu().numpy()
                    results['depth_conf'] = depth_conf.squeeze(0).cpu().numpy()
                
                if predict_points:
                    point_map, point_conf = self.model.point_head(
                        aggregated_tokens_list, images_batch, ps_idx
                    )
                    results['point_map'] = point_map.squeeze(0).cpu().numpy()
                    results['point_conf'] = point_conf.squeeze(0).cpu().numpy()
                
                if predict_tracks and query_points_tensor is not None:
                    if query_points_tensor.dim() == 2:
                        query_points_tensor = query_points_tensor.unsqueeze(0)
                    track, vis_score, conf_score = self.model.track_head(
                        aggregated_tokens_list, images_batch, ps_idx,
                        query_points=query_points_tensor
                    )
                    results['tracks'] = track.squeeze(0).cpu().numpy()
                    results['track_vis_score'] = vis_score.squeeze(0).cpu().numpy()
                    results['track_conf_score'] = conf_score.squeeze(0).cpu().numpy()
                
                if predict_depth and predict_cameras and predict_points:
                    point_map_from_depth = unproject_depth_map_to_point_map(
                        results['depth_map'],
                        results['extrinsic'],
                        results['intrinsic']
                    )
                    results['point_map_from_depth'] = point_map_from_depth
        
        return results

