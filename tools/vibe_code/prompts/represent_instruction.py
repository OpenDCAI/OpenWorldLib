representation_prompt = """
The world model requires the implementation of 3D scene representation capabilities, such as extracting depth maps, point clouds, camera poses, and 3D scene structures from images or videos. Our framework needs to possess 3D representation learning capabilities; therefore, a Representation class must be defined.

The Representation class is invoked within the Pipeline class. It accepts processed inputs from the Operator and performs inference using the underlying model to generate 3D scene representations (point clouds, depth maps, camera poses, etc.).
It should follow the structure below:
```python
class BaseRepresentation(object):
    def __init__(self):
        ## Initialize the model used by the Representation class

    @classmethod
    def from_pretrained(cls, pretrained_model_path, device=None, **kwargs):
        ## Load the model weights required by the Representation class
        ## Supports both local paths and HuggingFace repo IDs

    def api_init(self, api_key, endpoint):
        ## If calling an online model, initialize the API key or API URL

    @torch.no_grad()
    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ## Accept external inputs (images, videos, etc.) and output 3D scene representations
        ## Returns dictionary containing point clouds, depth maps, camera poses, etc.
```
"""

example_represent_code = """
Here are the organized code results for FlashWorld: https://github.com/imlixinyang/FlashWorld".
The Operator implementation is as follows:
```python
from .base_operator import BaseOperator
import os
import numpy as np
import torch
from typing import List, Optional, Union, Dict, Any
from PIL import Image
import base64
import io

class FlashWorldOperator(BaseOperator):
    # Operator for FlashWorld pipeline utilities.
    
    def __init__(
        self,
        operation_types=["textual_instruction", "action_instruction", "visual_instruction"],
        interaction_template=[
            "text_prompt",
            "camera_forward", "camera_backward", "camera_left", "camera_right",
            "camera_up", "camera_down", "camera_rotate_left", "camera_rotate_right",
            "camera_zoom_in", "camera_zoom_out"
        ]
    ):
        # Initialize FlashWorld operator.
        # Args:
        #     operation_types: List of operation types
        #     interaction_template: List of valid interaction types
        #         - "text_prompt": Text description for scene generation
        #         - "camera_forward/backward/left/right/up/down": Camera movement
        #         - "camera_rotate_left/right": Camera rotation
        #         - "camera_zoom_in/out": Camera zoom
        super(FlashWorldOperator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()
    
    def check_interaction(self, interaction):
        # Check if interaction is in the interaction template.
        # Args:
        #     interaction: Interaction string to check
        # Returns:
        #     True if interaction is valid
        # Raises:
        #     ValueError: If interaction is not in template
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template. Available: {self.interaction_template}")
        return True
    
    def get_interaction(self, interaction):
        # Add interaction to current_interaction list after validation.
        # Args:
        #     interaction: Interaction string to add
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)
    
    def process_interaction(
        self, 
        num_frames: Optional[int] = None,
        image_width: int = 704,
        image_height: int = 480
    ) -> Dict[str, Any]:
        # Process current interactions and convert to features for representation/synthesis.
        # Converts camera actions to actual camera parameters that can be used by representation.
        # Args:
        #     num_frames: Number of frames for video generation (optional)
        #     image_width: Image width for camera intrinsics
        #     image_height: Image height for camera intrinsics
        # Returns:
        #     Dictionary containing processed interaction features:
        #         - text_prompt: str, text description (if provided)
        #         - cameras: List[Dict], camera parameters for each frame
        #         - num_frames: int, number of frames
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")
        
        # Get the latest interaction
        latest_interaction = self.current_interaction[-1]
        self.interaction_history.append(latest_interaction)
        
        num_frames = num_frames or 16
        
        # Extract text prompts
        text_prompt = ""
        camera_actions = []
        for interaction in self.current_interaction:
            if interaction == "text_prompt":
                # Text prompt should be passed separately via data
                pass
            elif interaction.startswith("camera_"):
                camera_actions.append(interaction)
        
        # Convert camera actions to camera parameters
        cameras = self._camera_actions_to_cameras(
            camera_actions=camera_actions,
            num_frames=num_frames,
            image_width=image_width,
            image_height=image_height
        )
        
        result = {
            "text_prompt": text_prompt,
            "cameras": cameras,
            "num_frames": num_frames,
        }
        
        return result
    
    def _camera_actions_to_cameras(
        self,
        camera_actions: List[str],
        num_frames: int,
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        # Convert camera action strings to camera parameter dictionaries.
        # Args:
        #     camera_actions: List of camera action strings
        #     num_frames: Number of frames
        #     image_width: Image width
        #     image_height: Image height
        # Returns:
        #     List of camera dictionaries with position, quaternion, and intrinsics
        if not camera_actions:
            # Default circular camera path
            return self._create_default_cameras(num_frames, image_width, image_height)
        
        # Process camera actions to generate camera trajectory
        cameras = []
        radius = 2.0
        base_position = np.array([0.0, 0.5, 2.0])  # Default position
        
        for i in range(num_frames):
            # Apply camera actions sequentially
            position = base_position.copy()
            angle = 2 * np.pi * i / num_frames
            
            # Process each camera action
            for action in camera_actions:
                if action == "camera_forward":
                    position[2] -= 0.1 * (i / num_frames)
                elif action == "camera_backward":
                    position[2] += 0.1 * (i / num_frames)
                elif action == "camera_left":
                    position[0] -= 0.1 * (i / num_frames)
                elif action == "camera_right":
                    position[0] += 0.1 * (i / num_frames)
                elif action == "camera_up":
                    position[1] += 0.1 * (i / num_frames)
                elif action == "camera_down":
                    position[1] -= 0.1 * (i / num_frames)
                elif action == "camera_rotate_left":
                    angle -= np.pi / 4 * (i / num_frames)
                elif action == "camera_rotate_right":
                    angle += np.pi / 4 * (i / num_frames)
                # zoom_in/out affects intrinsics, handled separately
            
            # Calculate position based on angle (circular path with modifications)
            x = radius * np.cos(angle) + position[0]
            z = radius * np.sin(angle) + position[2]
            y = position[1]
            
            # Calculate quaternion (look at origin)
            direction = np.array([-x, -y, -z])
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Simple quaternion (identity for now, can be improved)
            quat = [1.0, 0.0, 0.0, 0.0]
            
            # Handle zoom
            zoom_factor = 1.0
            for action in camera_actions:
                if action == "camera_zoom_in":
                    zoom_factor *= 1.1
                elif action == "camera_zoom_out":
                    zoom_factor *= 0.9
            
            camera = {
                'position': [float(x), float(y), float(z)],
                'quaternion': quat,
                'fx': image_width * 0.7 * zoom_factor,
                'fy': image_height * 0.7 * zoom_factor,
                'cx': image_width * 0.5,
                'cy': image_height * 0.5,
            }
            cameras.append(camera)
        
        return cameras
    
    def _create_default_cameras(
        self,
        num_frames: int,
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        # Create default camera trajectory (circular path).
        # Args:
        #     num_frames: Number of frames
        #     image_width: Image width
        #     image_height: Image height
        # Returns:
        #     List of camera dictionaries
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
            
            # Simple quaternion (identity rotation)
            quat = [1.0, 0.0, 0.0, 0.0]
            
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
    
    def process_perception(
        self,
        input_signal: Union[str, np.ndarray, torch.Tensor, Image.Image, bytes]
    ) -> Union[Image.Image, torch.Tensor]:
        # Process visual signal (image) for real-time interactive updates.
        # Args:
        #     input_signal: Visual input signal - can be:
        #         - Image file path (str)
        #         - Numpy array (H, W, 3) in RGB format
        #         - Torch tensor (C, H, W) or (1, C, H, W) in CHW format
        #         - PIL Image
        #         - Base64 encoded image string
        #         - Bytes of image data
        # Returns:
        #     PIL Image in RGB format
        # Raises:
        #     ValueError: If image cannot be loaded or processed
        if isinstance(input_signal, Image.Image):
            # Already a PIL Image, convert to RGB
            return input_signal.convert('RGB')
        
        elif isinstance(input_signal, str):
            # Check if it's a file path or base64
            if os.path.exists(input_signal):
                # File path
                image = Image.open(input_signal)
                return image.convert('RGB')
            elif input_signal.startswith('data:image'):
                # Base64 encoded image
                if ',' in input_signal:
                    image_data = input_signal.split(',', 1)[1]
                else:
                    image_data = input_signal
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                return image.convert('RGB')
            else:
                raise ValueError(f"Invalid input: {input_signal}")
        
        elif isinstance(input_signal, bytes):
            # Bytes data
            image = Image.open(io.BytesIO(input_signal))
            return image.convert('RGB')
        
        elif isinstance(input_signal, np.ndarray):
            # Numpy array
            if input_signal.max() <= 1.0:
                input_signal = (input_signal * 255).astype(np.uint8)
            else:
                input_signal = input_signal.astype(np.uint8)
            
            # Convert BGR to RGB if needed
            if len(input_signal.shape) == 3 and input_signal.shape[2] == 3:
                if input_signal[..., 0].mean() > input_signal[..., 2].mean():
                    input_signal = input_signal[..., ::-1]
            
            image = Image.fromarray(input_signal)
            return image.convert('RGB')
        
        elif isinstance(input_signal, torch.Tensor):
            # Torch tensor
            if input_signal.dim() == 3:
                image_array = input_signal.permute(1, 2, 0).cpu().numpy()
            else:
                image_array = input_signal[0].permute(1, 2, 0).cpu().numpy()
            
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
            
            image = Image.fromarray(image_array)
            return image.convert('RGB')
        
        else:
            raise ValueError(f"Unsupported input type: {type(input_signal)}")
    
    def delete_last_interaction(self):
        # Delete the last interaction from current_interaction list.
        if len(self.current_interaction) > 0:
            self.current_interaction = self.current_interaction[:-1]
        else:
            raise ValueError("No interaction to delete.")
```

The Pipeline implementation is as follows:
```python
import torch
from typing import Optional, List, Union, Dict, Any
from PIL import Image
import numpy as np
from ...operators.flash_world_operator import FlashWorldOperator
from ...representations.point_clouds_generation.flash_world.flash_world_representation import FlashWorldRepresentation

class FlashWorldPipeline:
    # Pipeline for FlashWorld 3D scene generation.
    
    def __init__(
        self,
        representation_model: Optional[FlashWorldRepresentation] = None,
        operator: Optional[FlashWorldOperator] = None,
    ):
        # Initialize FlashWorld pipeline.
        # Args:
        #     representation_model: Pre-loaded FlashWorldRepresentation instance (optional)
        #     operator: FlashWorldOperator instance (optional)
        self.representation_model = representation_model
        self.operator = operator or FlashWorldOperator()
    
    @classmethod
    def from_pretrained(
        cls,
        representation_path: str,
        **kwargs
    ) -> 'FlashWorldPipeline':
        # Create pipeline instance from pretrained models.
        # Args:
        #     representation_path: HuggingFace repo ID for representation model
        #     **kwargs: Additional arguments passed to representation.from_pretrained()
        # Returns:
        #     FlashWorldPipeline instance
        representation_model = FlashWorldRepresentation.from_pretrained(
            pretrained_model_path=representation_path,
            **kwargs
        )
        
        return cls(representation_model=representation_model)
    
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
        # Process input and generate 3D scene representation.
        # Args:
        #     input_: Input image (path, PIL Image, numpy array, or tensor)
        #     interaction: Dictionary containing:
        #         - 'text_prompt': str, text description
        #         - 'cameras': torch.Tensor or List[Dict], camera parameters
        #     num_frames: Number of frames for generation
        #     image_height: Output image height
        #     image_width: Output image width
        #     image_index: Frame index for reference image
        #     return_video: If True, return video frames
        #     video_fps: FPS for video rendering
        # Returns:
        #     Dictionary containing:
        #         - 'scene_params': torch.Tensor, 3D Gaussian Splatting parameters
        #         - 'ref_w2c': torch.Tensor, reference world-to-camera transform
        #         - 'T_norm': torch.Tensor, normalization transform
        #         - 'video_frames': List[PIL.Image], rendered video frames (if return_video=True)
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
        # Convert list of camera dictionaries to tensor format.
        # Args:
        #     cameras: List of camera dicts with keys:
        #         - 'position': [x, y, z]
        #         - 'quaternion': [w, x, y, z]
        #         - 'fx', 'fy', 'cx', 'cy': camera intrinsics
        #     image_width: Image width
        #     image_height: Image height
        # Returns:
        #     Camera tensor of shape (N, 11)
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
        input_: Union[str, Image.Image, np.ndarray, torch.Tensor, None],
        text_prompt: str = "",
        cameras: Union[torch.Tensor, List[Dict[str, Any]]] = None,
        interactions: Optional[List[str]] = None,
        num_frames: int = 16,
        image_height: int = 480,
        image_width: int = 704,
        image_index: int = 0,
        return_video: bool = False,
        video_fps: int = 15,
        **kwargs
    ) -> Union[List[Image.Image], Dict[str, Any]]:
        # Main call interface for the pipeline.
        # Args:
        #     input_: Input image (path, PIL Image, numpy array, tensor, or None)
        #     text_prompt: Text description for scene generation
        #     cameras: Camera parameters (tensor or list of dicts). Ignored if interactions is provided.
        #     interactions: List of interaction strings (e.g., ["camera_rotate_left", "camera_forward"]).
        #                   If provided, cameras will be generated from these interactions.
        #     num_frames: Number of frames
        #     image_height: Output image height
        #     image_width: Output image width
        #     return_video: If True, return video frames as List[PIL.Image]
        #     video_fps: FPS for video rendering
        #     **kwargs: Additional arguments
        # Returns:
        #     If return_video=True: List[PIL.Image] of video frames
        #     Otherwise: Dict with scene_params, ref_w2c, T_norm
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
            cameras = interaction_result['cameras']
        elif cameras is None:
            # Create default cameras if not provided
            cameras = self._create_default_cameras(num_frames, image_width, image_height)
        
        interaction = {
            'text_prompt': text_prompt,
            'cameras': cameras,
        }
        
        result = self.process(
            input_=input_,
            interaction=interaction,
            num_frames=num_frames,
            image_height=image_height,
            image_width=image_width,
            image_index=image_index,
            return_video=return_video,
            video_fps=video_fps,
        )
        
        # Always return full result dict (don't filter video_frames)
        return result
    
    def _create_default_cameras(
        self,
        num_frames: int,
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        # Create default camera trajectory (circular path).
        # Args:
        #     num_frames: Number of frames
        #     image_width: Image width
        #     image_height: Image height
        # Returns:
        #     List of camera dictionaries
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
    
    def stream(self, *args, **kwds) -> Generator[torch.Tensor, List[str], None]:
        # Generator function supporting multi-round interactive inputs.
        # Should call __call__ internally.
        # Memory management must be handled here via the Memory module.
        yield self.__call__(*args, **kwds)
```

The Representation class implementation is as follows:
```python
import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from huggingface_hub import snapshot_download

from ...base_representation import BaseRepresentation

# Import FlashWorld models
from .flash_world.autoencoder_kl_wan import AutoencoderKLWan
from .flash_world.transformer_wan import WanTransformer3DModel
from .flash_world.reconstruction_model import WANDecoderPixelAligned3DGSReconstructionModel
from .flash_world.utils import (
    create_raymaps,
    normalize_cameras,
    sample_from_dense_cameras,
)

from transformers import T5TokenizerFast, UMT5EncoderModel
from diffusers import FlowMatchEulerDiscreteScheduler
import einops


@contextmanager
def onload_model(model, device, onload=False):
    # Context manager for moving model to GPU and back to CPU.
    if onload and device != "cpu":
        model.to(device) 
        try:
            yield model
        finally:
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    else:
        yield model


class FlashWorldRepresentation(BaseRepresentation):
    # Representation for FlashWorld 3D scene generation.
    
    def __init__(self, model: Optional[nn.Module] = None, device: Optional[str] = None):
        # Initialize FlashWorldRepresentation.
        # Args:
        #     model: Pre-loaded GenerationSystem model (optional)
        #     device: Device to run on ('cuda' or 'cpu')
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        
        if self.model is not None:
            self.model = self.model.to(self.device).eval()
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> 'FlashWorldRepresentation':
        # Create representation instance from pretrained model.
        # Args:
        #     pretrained_model_path: HuggingFace repo ID (e.g., "imlixinyang/FlashWorld")
        #     device: Device to run on
        #     **kwargs: Additional arguments
        # Returns:
        #     FlashWorldRepresentation instance
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Download from HuggingFace if needed
        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            print(f"Downloading weights from HuggingFace repo: {pretrained_model_path}")
            model_root = snapshot_download(pretrained_model_path)
            print(f"Model downloaded to: {model_root}")
        
        # Load model
        model = cls._load_generation_system(model_root, device, **kwargs)
        
        return cls(model=model, device=device)
    
    @staticmethod
    def _load_generation_system(model_root: str, device: str, **kwargs) -> nn.Module:
        # Load GenerationSystem model.
        # Find checkpoint file
        ckpt_path = kwargs.get('ckpt_path', None)
        if ckpt_path is None:
            # Try to find checkpoint in model_root
            ckpt_files = list(Path(model_root).glob("*.pt")) + list(Path(model_root).glob("*.pth")) + list(Path(model_root).glob("*.ckpt"))
            if ckpt_files:
                ckpt_path = str(ckpt_files[0])
            else:
                # Try common checkpoint names
                for name in ["model.ckpt", "checkpoint.pt", "model.pt", "flash_world.pt"]:
                    potential_path = os.path.join(model_root, name)
                    if os.path.exists(potential_path):
                        ckpt_path = potential_path
                        break
        
        offload_t5 = kwargs.get('offload_t5', False)
        offload_vae = kwargs.get('offload_vae', False)
        offload_transformer_during_vae = kwargs.get('offload_transformer_during_vae', False)
        
        # Initialize GenerationSystem
        generation_system = GenerationSystem(
            ckpt_path=ckpt_path,
            device=device,
            offload_t5=offload_t5,
            offload_vae=offload_vae,
            offload_transformer_during_vae=offload_transformer_during_vae,
        )
        
        return generation_system
    
    def api_init(self, api_key, endpoint):
        # Initialize API connection if needed.
        pass
    
    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Get 3D scene representation from input data.
        # Args:
        #     data: Dictionary containing:
        #         - 'text_prompt': str, text description
        #         - 'cameras': torch.Tensor, camera parameters (N, 11)
        #         - 'image': PIL.Image or None, reference image (optional)
        #         - 'image_index': int, frame index for reference image (default: 0)
        #         - 'image_height': int, output image height (default: 480)
        #         - 'image_width': int, output image width (default: 704)
        #         - 'num_frames': int, number of frames (default: 16)
        #         - 'video_fps': int, fps for video rendering (default: 15)
        #         - 'return_video': bool, whether to return video frames (default: False)
        # Returns:
        #     Dictionary containing:
        #         - 'scene_params': torch.Tensor, 3D Gaussian Splatting parameters
        #         - 'ref_w2c': torch.Tensor, reference world-to-camera transform
        #         - 'T_norm': torch.Tensor, normalization transform
        #         - 'video_frames': List[PIL.Image], rendered video frames (if requested)
        if self.model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() first.")
        
        text_prompt = data.get('text_prompt', "")
        cameras = data['cameras']  # Required
        image = data.get('image', None)
        image_index = data.get('image_index', 0)
        image_height = data.get('image_height', 480)
        image_width = data.get('image_width', 704)
        num_frames = data.get('num_frames', 16)
        video_fps = data.get('video_fps', 15)
        return_video = data.get('return_video', False)
        
        # Convert PIL Image to tensor if provided
        if image is not None:
            if isinstance(image, Image.Image):
                # Resize and center crop image to match target dimensions (same as FlashWorld/cli.py)
                image = image.convert('RGB')
                w, h = image.size
                
                # Calculate scale factor to maintain aspect ratio
                if image_height / h > image_width / w:
                    scale = image_height / h
                else:
                    scale = image_width / w
                
                # Calculate new dimensions for center crop
                new_h = int(image_height / scale)
                new_w = int(image_width / scale)
                
                # Center crop and resize to target dimensions
                image = image.crop((
                    (w - new_w) // 2, 
                    (h - new_h) // 2, 
                    new_w + (w - new_w) // 2, 
                    new_h + (h - new_h) // 2
                )).resize((image_width, image_height), Image.Resampling.BICUBIC)
                
                # Convert to tensor: (C, H, W) in range [-1, 1]
                image_array = np.array(image)
                image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1) / 255.0 * 2 - 1
            else:
                image_tensor = image
        else:
            image_tensor = None
        
        # Generate scene
        video_path = None
        if return_video:
            import tempfile
            video_path = tempfile.mktemp(suffix='.mp4')
        
        scene_params, ref_w2c, T_norm = self.model.generate(
            cameras=cameras,
            n_frame=num_frames,
            image=image_tensor,
            text=text_prompt,
            image_index=image_index,
            image_height=image_height,
            image_width=image_width,
            video_path=video_path,
            video_fps=video_fps,
        )
        
        result = {
            'scene_params': scene_params,
            'ref_w2c': ref_w2c,
            'T_norm': T_norm,
        }
        
        # Load video frames if generated
        if return_video and video_path and os.path.exists(video_path):
            import imageio
            video_frames = []
            reader = imageio.get_reader(video_path)
            for frame in reader:
                video_frames.append(Image.fromarray(frame))
            reader.close()
            os.remove(video_path)  # Clean up temp file
            result['video_frames'] = video_frames
        
        return result
```
"""
