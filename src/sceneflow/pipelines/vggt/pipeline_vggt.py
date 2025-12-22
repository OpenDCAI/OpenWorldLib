import os
from typing import List, Optional, Union, Dict, Any

import numpy as np
from PIL import Image
import json

from ...operators.vggt_operator import VGGTOperator
from ...base_models.three_dimensions.point_clouds.vggt.vggt_representation import (
    VGGTRepresentation,
)


class VGGTResult:
    """Container class for VGGT inference results."""
    
    def __init__(
        self,
        images: List[Image.Image],
        numpy_data: Dict[str, np.ndarray],
        camera_params: List[Dict[str, Any]],
        data_type: str = "image"
    ):
        self.images = images
        self.numpy_data = numpy_data
        self.camera_params = camera_params
        self.data_type = data_type
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'camera_params': self.camera_params[idx] if idx < len(self.camera_params) else None,
            'numpy_data': {k: v[idx] if isinstance(v, np.ndarray) and v.ndim > len(self.images) else v 
                          for k, v in self.numpy_data.items()}
        }
    
    def save(self, output_dir: Optional[str] = None) -> List[str]:
        """Save VGGT results to files."""
        if output_dir is None:
            output_dir = "./vggt_output"
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files: List[str] = []
        
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        for i, img in enumerate(self.images):
            img_path = os.path.join(vis_dir, f"result_{i:04d}.png")
            img.save(img_path)
            saved_files.append(img_path)
        
        np_dir = os.path.join(output_dir, "numpy")
        os.makedirs(np_dir, exist_ok=True)
        for key, value in self.numpy_data.items():
            if isinstance(value, np.ndarray):
                np_path = os.path.join(np_dir, f"{key}.npy")
                np.save(np_path, value)
                saved_files.append(np_path)
        
        json_dir = os.path.join(output_dir, "json")
        os.makedirs(json_dir, exist_ok=True)
        for i, camera_param in enumerate(self.camera_params):
            json_path = os.path.join(json_dir, f"camera_{i:04d}.json")
            with open(json_path, 'w') as f:
                json.dump(camera_param, f, indent=2)
            saved_files.append(json_path)
        
        return saved_files


class VGGTPipeline:
    """Pipeline for VGGT 3D scene reconstruction."""
    
    def __init__(
        self,
        representation_model: Optional[VGGTRepresentation] = None,
        reasoning_model: Optional[Any] = None,
        synthesis_model: Optional[Any] = None,
        operator: Optional[VGGTOperator] = None,
    ) -> None:
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.operator = operator or VGGTOperator()
    
    @classmethod
    def from_pretrained(
        cls,
        representation_path: str,
        reasoning_path: Optional[str] = None,
        synthesis_path: Optional[str] = None,
        **kwargs
    ) -> 'VGGTPipeline':
        representation_model = VGGTRepresentation.from_pretrained(
            pretrained_model_path=representation_path,
            **kwargs
        )
        reasoning_model = None
        synthesis_model = None
        return cls(
            representation_model=representation_model,
            reasoning_model=reasoning_model,
            synthesis_model=synthesis_model,
        )
    
    def process(
        self,
        input_: Union[str, np.ndarray, List[str], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> VGGTResult:
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")
        
        images_data = self.operator.process_perception(input_)
        if not isinstance(images_data, list):
            images_data = [images_data]
        
        if interaction is None:
            interaction_dict = {
                'predict_cameras': True,
                'predict_depth': True,
                'predict_points': True,
                'predict_tracks': False,
            }
        elif isinstance(interaction, str):
            self.operator.get_interaction(interaction)
            interaction_dict = self.operator.process_interaction()
        else:
            interaction_dict = interaction
        
        data = {
            'images': images_data,
            'predict_cameras': interaction_dict.get('predict_cameras', True),
            'predict_depth': interaction_dict.get('predict_depth', True),
            'predict_points': interaction_dict.get('predict_points', True),
            'predict_tracks': interaction_dict.get('predict_tracks', False),
            'query_points': kwargs.get('query_points', None),
            'preprocess_mode': kwargs.get('preprocess_mode', 'crop'),
            'resolution': kwargs.get('resolution', 518),
        }
        
        results = self.representation_model.get_representation(data)
        
        numpy_data = {}
        for key in ['extrinsic', 'intrinsic', 'depth_map', 'depth_conf', 
                   'point_map', 'point_conf', 'point_map_from_depth',
                   'tracks', 'track_vis_score', 'track_conf_score']:
            if key in results:
                numpy_data[key] = results[key]
        
        camera_params = []
        if 'extrinsic' in results and 'intrinsic' in results:
            num_images = results['extrinsic'].shape[0] if results['extrinsic'].ndim > 2 else 1
            for i in range(num_images):
                if results['extrinsic'].ndim > 2:
                    extrinsic = results['extrinsic'][i].tolist()
                    intrinsic = results['intrinsic'][i].tolist()
                else:
                    extrinsic = results['extrinsic'].tolist()
                    intrinsic = results['intrinsic'].tolist()
                camera_params.append({
                    'extrinsic': extrinsic,
                    'intrinsic': intrinsic,
                })
        
        return_visualization = kwargs.get('return_visualization', True)
        images = []
        
        if return_visualization and 'depth_map' in results:
            depth_maps = results['depth_map']
            if depth_maps.ndim == 2:
                depth_maps = depth_maps[np.newaxis, ...]
            for i in range(depth_maps.shape[0]):
                depth = depth_maps[i]
                if depth.ndim > 2:
                    depth = depth.squeeze()
                if depth.ndim != 2:
                    raise ValueError(f"Expected 2D depth map, got shape {depth.shape}")
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                depth_img = Image.fromarray(depth_uint8, mode='L')
                images.append(depth_img)
        else:
            for img_data in images_data:
                if isinstance(img_data, np.ndarray):
                    img_uint8 = (img_data * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_uint8)
                    images.append(img_pil)
        
        return VGGTResult(
            images=images,
            numpy_data=numpy_data,
            camera_params=camera_params,
            data_type="image"
        )
    
    def __call__(
        self,
        input_: Union[str, np.ndarray, List[str], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> VGGTResult:
        """Main call interface for the pipeline."""
        return self.process(
            input_=input_,
            interaction=interaction,
            **kwargs
        )


__all__ = ["VGGTPipeline", "VGGTResult"]

