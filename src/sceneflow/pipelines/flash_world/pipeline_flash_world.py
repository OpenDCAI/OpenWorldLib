from typing import Optional, List, Union, Dict, Any
import os
import shutil
import json
import base64
import tempfile
import warnings
import torch
import numpy as np
from PIL import Image
import imageio

from ...operators.flash_world_operator import FlashWorldOperator
from ...representations.point_clouds_generation.flash_world.flash_world_representation import (
    FlashWorldRepresentation,
)
from ...representations.point_clouds_generation.flash_world.flash_world.utils import (
    export_gaussians,
)


class FlashWorldPipeline:
    """Pipeline for FlashWorld 3D scene generation."""
    
    def __init__(
        self,
        representation_model: Optional[FlashWorldRepresentation] = None,
        operator: Optional[FlashWorldOperator] = None,
    ):
        """
        Initialize FlashWorld pipeline.
        
        Args:
            representation_model: Pre-loaded FlashWorldRepresentation instance (optional)
            operator: FlashWorldOperator instance (optional)
        """
        self.representation_model = representation_model
        self.operator = operator or FlashWorldOperator()
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        required_components: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> 'FlashWorldPipeline':
        """
        Create pipeline instance from pretrained models.
        
        Args:
            model_path: HuggingFace repo ID for representation model
            required_components: Optional required component paths. Supports:
                - 'wan_model_path': Wan model repo ID/path for internal modules
            **kwargs: Additional arguments passed to representation.from_pretrained()
            
        Returns:
            FlashWorldPipeline instance
        """
        if isinstance(required_components, dict) and "wan_model_path" in required_components.keys():
            kwargs["wan_model_path"] = required_components.get(
                "wan_model_path",
                "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            )
        elif "wan_model_path" not in kwargs and "model_id" not in kwargs:
            kwargs["wan_model_path"] = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

        representation_model = FlashWorldRepresentation.from_pretrained(
            pretrained_model_path=model_path,
            **kwargs
        )
        
        return cls(representation_model=representation_model)
    
    @staticmethod
    def load_config_from_json(
        json_path: str,
        output_dir: Optional[str] = None,
        default_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load configuration from JSON file (FlashWorld format).
        
        Args:
            json_path: Path to JSON configuration file
            output_dir: Directory for saving temporary files (if needed)
            default_config: Default configuration values (optional)
            
        Returns:
            Dictionary containing:
                - 'input_': Image path or None
                - 'text_prompt': str
                - 'cameras': List[Dict] or None
                - 'num_frames': int
                - 'image_height': int
                - 'image_width': int
                - 'image_index': int
                - 'return_video': bool
                - 'video_fps': int
        """
        if default_config is None:
            default_config = {
                'text_prompt': "",
                'image_prompt': None,
                'resolution': [16, 480, 704],
                'image_index': 0,
                'cameras': None,
                'return_video': True,
                'video_fps': 15,
            }
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON config file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # Check for missing keys and issue warnings
        expected_keys = ['text_prompt', 'image_prompt', 'resolution', 'image_index', 
                        'cameras', 'return_video', 'video_fps']
        missing_keys = [key for key in expected_keys if key not in config]
        if missing_keys:
            warnings.warn(
                f"Config file '{json_path}' is missing the following keys: {', '.join(missing_keys)}. "
                f"Using default values for missing keys.",
                UserWarning,
                stacklevel=2
            )
        
        # Extract values from config
        text_prompt = config.get('text_prompt', default_config['text_prompt'])
        image_prompt = config.get('image_prompt', default_config['image_prompt'])
        resolution = config.get('resolution', default_config['resolution'])
        image_index = config.get('image_index', default_config['image_index'])
        cameras = config.get('cameras', default_config['cameras'])
        return_video = config.get('return_video', default_config.get('return_video', True))
        video_fps = config.get('video_fps', default_config.get('video_fps', 15))
        
        num_frames, image_height, image_width = resolution
        
        # Handle image_prompt (can be base64 or path)
        input_image_path = None
        if image_prompt:
            if os.path.exists(image_prompt):
                # It's a file path
                input_image_path = image_prompt
            else:
                # It might be base64 encoded
                base64_str = image_prompt
                if ',' in base64_str:
                    base64_str = base64_str.split(',', 1)[1]
                
                try:
                    image_bytes = base64.b64decode(base64_str)
                    # Save to temp file
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        temp_image_path = os.path.join(output_dir, 'temp_input_image.jpg')
                    else:
                        temp_dir = tempfile.gettempdir()
                        temp_image_path = os.path.join(temp_dir, f'temp_flashworld_input_{os.getpid()}.jpg')
                    
                    with open(temp_image_path, 'wb') as f:
                        f.write(image_bytes)
                    input_image_path = temp_image_path
                except Exception:
                    # If decoding fails, treat as regular path
                    input_image_path = image_prompt
        
        return {
            'input_': input_image_path,
            'text_prompt': text_prompt,
            'cameras': cameras,
            'num_frames': num_frames,
            'image_height': image_height,
            'image_width': image_width,
            'image_index': image_index,
            'return_video': return_video,
            'video_fps': video_fps,
        }
    
    def process(
        self,
        input_: Union[str, Image.Image, np.ndarray, torch.Tensor],
        interaction: Dict[str, Any],
        num_frames: int = 16,
        image_height: int = 480,
        image_width: int = 704,
        image_index: int = 0,
        return_video: bool = False,
        video_fps: int = 15,
    ) -> Dict[str, Any]:
        """
        Process input and generate 3D scene representation.
        
        Args:
            input_: Input image (path, PIL Image, numpy array, or tensor)
            interaction: Dictionary containing:
                - 'text_prompt': str, text description
                - 'cameras': torch.Tensor or List[Dict], camera parameters
            num_frames: Number of frames for generation
            image_height: Output image height
            image_width: Output image width
            image_index: Frame index for reference image
            return_video: If True, return video frames
            video_fps: FPS for video rendering
            
        Returns:
            Dictionary containing:
                - 'scene_params': torch.Tensor, 3D Gaussian Splatting parameters
                - 'ref_w2c': torch.Tensor, reference world-to-camera transform
                - 'T_norm': torch.Tensor, normalization transform
                - 'video_frames': List[PIL.Image], rendered video frames (if return_video=True)
        """
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")
        
        # Process input image using operator's process_perception
        image = None
        if input_ is not None:
            image = self.operator.process_perception(input_)
        
        # Process interaction
        text_prompt = interaction.get('text_prompt', "")
        cameras = interaction.get('cameras')
        
        # Convert cameras to tensor if needed
        if isinstance(cameras, list):
            # Convert list of camera dicts to tensor
            cameras_tensor = self._cameras_list_to_tensor(cameras, image_width, image_height)
        elif isinstance(cameras, torch.Tensor):
            cameras_tensor = cameras
        else:
            raise ValueError(f"Unsupported cameras type: {type(cameras)}")
        
        # Prepare data for representation
        data = {
            'text_prompt': text_prompt,
            'cameras': cameras_tensor,
            'image': image,
            'image_index': image_index,
            'image_height': image_height,
            'image_width': image_width,
            'num_frames': num_frames,
            'video_fps': video_fps,
            'return_video': return_video,
        }
        
        # Get representation
        result = self.representation_model.get_representation(data)
        
        # Store video_fps in result for save_results
        result['video_fps'] = video_fps
        
        return result
    
    def _cameras_list_to_tensor(
        self,
        cameras: List[Dict[str, Any]],
        image_width: int,
        image_height: int
    ) -> torch.Tensor:
        """
        Convert list of camera dictionaries to tensor format.
        
        Args:
            cameras: List of camera dicts with keys:
                - 'position': [x, y, z]
                - 'quaternion': [w, x, y, z]
                - 'fx', 'fy', 'cx', 'cy': camera intrinsics
            image_width: Image width
            image_height: Image height
            
        Returns:
            Camera tensor of shape (N, 11)
        """
        camera_tensors = []
        for camera in cameras:
            quat = camera.get('quaternion', [1, 0, 0, 0])
            pos = camera.get('position', [0, 0, 0])
            fx = camera.get('fx', image_width * 0.5)
            fy = camera.get('fy', image_height * 0.5)
            cx = camera.get('cx', image_width * 0.5)
            cy = camera.get('cy', image_height * 0.5)
            
            # Format: [quat_w, quat_x, quat_y, quat_z, pos_x, pos_y, pos_z, fx/width, fy/height, cx/width, cy/height]
            camera_tensor = torch.tensor([
                quat[0], quat[1], quat[2], quat[3],
                pos[0], pos[1], pos[2],
                fx / image_width, fy / image_height,
                cx / image_width, cy / image_height
            ], dtype=torch.float32)
            camera_tensors.append(camera_tensor)
        
        return torch.stack(camera_tensors, dim=0)
    
    def __call__(
        self,
        images: Union[str, Image.Image, np.ndarray, torch.Tensor, None],
        prompt: str = "",
        interactions: Optional[List[str]] = None,
        camera_view: Union[torch.Tensor, List[Dict[str, Any]]] = None,
        num_frames: int = 16,
        fps: int = 15,
        image_height: int = 480,
        image_width: int = 704,
        image_index: int = 0,
        return_video: bool = False,
        **kwargs
    ) -> Union[List[Image.Image], Dict[str, Any]]:
        """
        Main call interface for the pipeline.
        
        Args:
            input_: Input image (path, PIL Image, numpy array, tensor, or None)
            text_prompt: Text description for scene generation
            cameras: Camera parameters (tensor or list of dicts). Ignored if interactions is provided.
            interactions: List of interaction strings (e.g., ["camera_rotate_left", "camera_forward"]).
                          If provided, cameras will be generated from these interactions.
            num_frames: Number of frames
            image_height: Output image height
            image_width: Output image width
            return_video: If True, return video frames as List[PIL.Image]
            video_fps: FPS for video rendering
            **kwargs: Additional arguments
            
        Returns:
            If return_video=True: List[PIL.Image] of video frames
            Otherwise: Dict with scene_params, ref_w2c, T_norm
        """
        # Process interactions if provided
        if interactions is not None:
            # Clear previous interactions
            self.operator.current_interaction = []
            # Add new interactions
            for interaction in interactions:
                self.operator.get_interaction(interaction)
            # Process interactions to get camera parameters
            interaction_result = self.operator.process_interaction(
                num_frames=num_frames,
                image_width=image_width,
                image_height=image_height
            )
            camera_view = interaction_result['cameras']
        elif camera_view is not None:
            # Create default cameras if not provided
            camera_view = self._create_default_cameras(num_frames, image_width, image_height)
        
        interaction = {
            'prompt': prompt,
            'cameras': camera_view,
        }
        
        result = self.process(
            input_=images,
            interaction=interaction,
            num_frames=num_frames,
            image_height=image_height,
            image_width=image_width,
            image_index=image_index,
            return_video=return_video,
            video_fps=fps,
        )
        
        # Always return full result dict (don't filter video_frames)
        return result
    
    def _create_default_cameras(
        self,
        num_frames: int,
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        """
        Create default camera trajectory (circular path).
        
        Args:
            num_frames: Number of frames
            image_width: Image width
            image_height: Image height
            
        Returns:
            List of camera dictionaries
        """
        cameras = []
        radius = 2.0
        
        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            
            # Circular camera path
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0.5
            
            # Look at origin
            direction = np.array([-x, -y, -z])
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Simple quaternion (simplified, should use proper rotation)
            quat = [1.0, 0.0, 0.0, 0.0]  # Identity rotation
            
            camera = {
                'position': [float(x), float(y), float(z)],
                'quaternion': quat,
                'fx': image_width * 0.7,
                'fy': image_height * 0.7,
                'cx': image_width * 0.5,
                'cy': image_height * 0.5,
            }
            cameras.append(camera)
        
        return cameras
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str,
        save_ply: bool = True,
        save_spz: bool = True,
        save_video: bool = True,
        opacity_threshold: float = 0.000,
        ply_path: Optional[str] = None,
        spz_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save pipeline results to files (gaussians.ply, gaussians.spz, video.mp4).
        
        Args:
            results: Dictionary returned from pipeline.__call__() containing:
                - 'scene_params': torch.Tensor, 3D Gaussian Splatting parameters
                - 'ref_w2c': torch.Tensor, reference world-to-camera transform
                - 'T_norm': torch.Tensor, normalization transform
                - 'video_frames': List[PIL.Image], rendered video frames (optional)
                - 'video_path': str, path to temporary video file (optional)
            output_dir: Directory to save output files (used as default if custom paths not provided)
            save_ply: If True, save gaussians.ply
            save_spz: If True, save gaussians.spz
            save_video: If True, save video.mp4
            opacity_threshold: Opacity threshold for Gaussian pruning
            ply_path: Custom path for PLY file (can be a directory or full file path). 
                      If directory, uses gaussians.ply in that directory. If None, uses output_dir/gaussians.ply
            spz_path: Custom path for SPZ file (can be a directory or full file path).
                      If directory, uses gaussians.spz in that directory. If None, uses output_dir/gaussians.spz
            video_path: Custom path for video file (can be a directory or full file path).
                       If directory, uses video.mp4 in that directory. If None, uses output_dir/video.mp4
            
        Returns:
            Dictionary with paths to saved files:
                - 'ply_path': str or None
                - 'spz_path': str or None
                - 'video_path': str or None
        """
        os.makedirs(output_dir, exist_ok=True)
        
        scene_params = results['scene_params']
        ref_w2c = results['ref_w2c']
        T_norm = results['T_norm']
        
        saved_paths = {}
        
        # Export gaussians.ply and gaussians.spz
        # Use custom paths if provided, otherwise use default paths in output_dir
        if save_ply:
            if ply_path is None:
                ply_path = os.path.join(output_dir, 'gaussians.ply')
            else:
                # Check if ply_path is a directory (doesn't end with .ply) or a file path
                if not ply_path.endswith('.ply'):
                    # It's a directory, use default filename
                    ply_path = os.path.join(ply_path, 'gaussians.ply')
                # Ensure directory exists for custom path
                os.makedirs(os.path.dirname(ply_path) if os.path.dirname(ply_path) else '.', exist_ok=True)
        else:
            ply_path = None
            
        if save_spz:
            if spz_path is None:
                spz_path = os.path.join(output_dir, 'gaussians.spz')
            else:
                # Check if spz_path is a directory (doesn't end with .spz) or a file path
                if not spz_path.endswith('.spz'):
                    # It's a directory, use default filename
                    spz_path = os.path.join(spz_path, 'gaussians.spz')
                # Ensure directory exists for custom path
                os.makedirs(os.path.dirname(spz_path) if os.path.dirname(spz_path) else '.', exist_ok=True)
        else:
            spz_path = None
        
        if save_ply or save_spz:
            export_gaussians(
                scene_params,
                opacity_threshold=opacity_threshold,
                T_norm=T_norm,
                ply_path=ply_path,
                spz_path=spz_path,
            )
            if save_ply:
                saved_paths['ply_path'] = ply_path
            if save_spz:
                saved_paths['spz_path'] = spz_path
        
        # Save video if generated
        if save_video:
            # Use custom path if provided, otherwise use default path in output_dir
            if video_path is None:
                video_path = os.path.join(output_dir, 'video.mp4')
            else:
                # Check if video_path is a directory (doesn't end with video extension) or a file path
                video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
                if not video_path.endswith(video_extensions):
                    # It's a directory, use default filename
                    video_path = os.path.join(video_path, 'video.mp4')
                # Ensure directory exists for custom path
                os.makedirs(os.path.dirname(video_path) if os.path.dirname(video_path) else '.', exist_ok=True)
            
            if 'video_path' in results and os.path.exists(results['video_path']):
                # If video was already saved, copy it to output directory
                temp_video_path = results['video_path']
                shutil.copy2(temp_video_path, video_path)
                os.remove(temp_video_path)  # Clean up temp file
                saved_paths['video_path'] = video_path
            elif 'video_frames' in results:
                # Convert PIL Images to numpy arrays and save
                video_frames = results['video_frames']
                frames = [np.array(frame) for frame in video_frames]
                # Get fps from results or use default
                fps = results.get('video_fps', 15)
                imageio.mimsave(video_path, frames, fps=fps)
                saved_paths['video_path'] = video_path
        
        return saved_paths


__all__ = ["FlashWorldPipeline"]

