import os
from typing import Optional, List, Union, Dict, Any, Generator
import numpy as np
from PIL import Image
import cv2
import torch

from ...operators.cut3r_operator import CUT3ROperator
from ...representations.point_clouds_generation.cut3r.cut3r_representation import (
    CUT3RRepresentation,
)


class CUT3RResult:
    """Container class for CUT3R results."""
    
    def __init__(
        self, 
        images: List[Image.Image],
        point_clouds: Optional[List[np.ndarray]] = None,
        depth_maps: Optional[List[np.ndarray]] = None,
        camera_poses: Optional[List[np.ndarray]] = None,
        data_type: str = "image"
    ):
        """
        Initialize CUT3R result container.
        
        Args:
            images: List of PIL Images (rendered point clouds or depth visualizations)
            point_clouds: List of point cloud arrays (optional)
            depth_maps: List of depth map arrays (optional)
            camera_poses: List of camera pose arrays (optional)
            data_type: Type of data ('image' or 'video')
        """
        self.images = images
        self.point_clouds = point_clouds
        self.depth_maps = depth_maps
        self.camera_poses = camera_poses
        self.data_type = data_type
    
    def save(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Save results to files.
        
        Args:
            output_dir: Output directory. If None, uses default.
            
        Returns:
            List of saved file paths
        """
        if output_dir is None:
            output_dir = "./cut3r_output"
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files: List[str] = []
        
        # Save images
        for i, img in enumerate(self.images):
            output_path = os.path.join(output_dir, f"frame_{i:06d}.png")
            img.save(output_path)
            saved_files.append(output_path)
        
        # Save point clouds if available
        if self.point_clouds is not None:
            pc_dir = os.path.join(output_dir, "point_clouds")
            os.makedirs(pc_dir, exist_ok=True)
            for i, pc in enumerate(self.point_clouds):
                output_path = os.path.join(pc_dir, f"pc_{i:06d}.npy")
                np.save(output_path, pc)
                saved_files.append(output_path)
        
        # Save depth maps if available
        if self.depth_maps is not None:
            depth_dir = os.path.join(output_dir, "depth_maps")
            os.makedirs(depth_dir, exist_ok=True)
            for i, depth in enumerate(self.depth_maps):
                # Ensure depth is 2D array (H, W) - follow CUT3R original code style
                if depth.ndim > 2:
                    # If batch dimension exists, take first item
                    depth = depth[0] if depth.ndim == 3 else depth.squeeze()
                elif depth.ndim < 2:
                    # If 1D, skip this depth map
                    print(f"Warning: Skipping depth map {i} with unexpected shape: {depth.shape}")
                    continue
                
                # Normalize depth values (follow CUT3R style: normalize to [0, 1] then scale to [0, 255])
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max > depth_min:
                    depth_norm = (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth_norm = np.zeros_like(depth)
                
                # Convert to uint8 and ensure it's 2D (H, W) - required by cv2.applyColorMap
                depth_uint8 = (depth_norm * 255).astype(np.uint8)
                if depth_uint8.ndim != 2:
                    depth_uint8 = depth_uint8.squeeze()
                    if depth_uint8.ndim != 2:
                        print(f"Warning: Skipping depth map {i} - cannot convert to 2D, shape: {depth_uint8.shape}")
                        continue
                
                # Apply colormap (requires 2D uint8 array) - follow CUT3R original code
                depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
                output_path = os.path.join(depth_dir, f"depth_{i:06d}.png")
                cv2.imwrite(output_path, depth_colored)
                saved_files.append(output_path)
        
        return saved_files
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]


class CUT3RPipeline:
    """Pipeline for CUT3R 3D scene reconstruction."""
    
    def __init__(
        self,
        representation_model: Optional[CUT3RRepresentation] = None,
        reasoning_model: Optional[Any] = None,
        synthesis_model: Optional[Any] = None,
        operator: Optional[CUT3ROperator] = None,
    ):
        """
        Initialize CUT3R pipeline.
        
        Args:
            representation_model: Pre-loaded CUT3RRepresentation instance (optional)
            reasoning_model: Reasoning model (not used for CUT3R, kept for compatibility)
            synthesis_model: Synthesis model (not used for CUT3R, kept for compatibility)
            operator: CUT3ROperator instance (optional)
        """
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.operator = operator or CUT3ROperator()
    
    @classmethod
    def from_pretrained(
        cls,
        representation_path: str,
        reasoning_path: Optional[str] = None,
        synthesis_path: Optional[str] = None,
        **kwargs
    ) -> 'CUT3RPipeline':
        """
        Create pipeline instance from pretrained models.
        
        Args:
            representation_path: HuggingFace repo ID for representation model
            reasoning_path: Not used for CUT3R (kept for compatibility)
            synthesis_path: Not used for CUT3R (kept for compatibility)
            **kwargs: Additional arguments passed to representation.from_pretrained()
            
        Returns:
            CUT3RPipeline instance
        """
        representation_model = CUT3RRepresentation.from_pretrained(
            pretrained_model_path=representation_path,
            **kwargs
        )
        
        # CUT3R doesn't use reasoning or synthesis models, but keep for compatibility
        reasoning_model = None
        synthesis_model = None
        
        return cls(
            representation_model=representation_model,
            reasoning_model=reasoning_model,
            synthesis_model=synthesis_model,
        )
    
    def process(
        self,
        input_: Union[str, Image.Image, np.ndarray, List[str], List[Image.Image], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> CUT3RResult:
        """
        Process input and generate 3D scene representation.
        
        Args:
            input_: Input image(s) - can be:
                - Image file path (str)
                - List of image file paths
                - PIL Image
                - List of PIL Images
                - Numpy array (H, W, 3)
                - List of numpy arrays
            interaction: Interaction string or dictionary
            **kwargs: Additional arguments:
                - output_type: "point_cloud", "depth_map", "camera_pose", or "all" (default: "all")
                - size: Input image size (default: auto-detected from model or 224)
                - vis_threshold: Confidence threshold for filtering point clouds (default: 1.0)
                - return_point_clouds: If True, include point clouds in result (default: True)
                - return_depth_maps: If True, include depth maps in result (default: True)
                - return_camera_poses: If True, include camera poses in result (default: True)
                
        Returns:
            CUT3RResult object containing processed results
        """
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")
        
        # Process input using operator's process_perception
        images_data = self.operator.process_perception(input_)
        if not isinstance(images_data, list):
            images_data = [images_data]
        
        # Process interaction
        if interaction is None:
            interaction_dict = {
                "data_type": "image",
                "output_type": "all"
            }
        elif isinstance(interaction, str):
            self.operator.get_interaction(interaction)
            interaction_dict = self.operator.process_interaction()
        else:
            interaction_dict = interaction
        
        # Prepare data for representation
        # Get size from kwargs or use representation model's default size
        size = kwargs.get('size', None)
        if size is None and self.representation_model is not None:
            size = getattr(self.representation_model, 'size', 224)
        elif size is None:
            size = 224
        
        data = {
            'images': images_data,
            'output_type': interaction_dict.get('output_type', kwargs.get('output_type', 'all')),
            'size': size,
            'vis_threshold': kwargs.get('vis_threshold', 1.0),
        }
        
        # Get representation
        results = self.representation_model.get_representation(data)
        
        # Convert results to PIL Images for visualization
        output_images = []
        
        # Prefer depth map visualization (more reliable for filtered point clouds)
        if 'depth_map' in results and results['depth_map']:
            # Use depth map visualization
            for depth in results['depth_map']:
                # Ensure depth is 2D array (H, W)
                if depth.ndim > 2:
                    # If batch dimension exists, take first item
                    depth = depth[0] if depth.ndim == 3 else depth.squeeze()
                elif depth.ndim < 2:
                    # If 1D, try to reshape (shouldn't happen normally)
                    raise ValueError(f"Unexpected depth shape: {depth.shape}")
                
                # Normalize depth values
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max > depth_min:
                    depth_norm = (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth_norm = np.zeros_like(depth)
                
                # Convert to uint8 and ensure it's 2D (H, W)
                depth_uint8 = (depth_norm * 255).astype(np.uint8)
                if depth_uint8.ndim != 2:
                    depth_uint8 = depth_uint8.squeeze()
                    if depth_uint8.ndim != 2:
                        raise ValueError(f"Depth array must be 2D after processing, got shape: {depth_uint8.shape}")
                
                # Apply colormap (requires 2D uint8 array)
                depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
                output_images.append(Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)))
        elif 'point_cloud' in results and results['point_cloud']:
            # Use point cloud visualization as fallback
            for pc, color in zip(results['point_cloud'], results.get('colors', [])):
                # Ensure point cloud is flattened (N, 3) format
                if pc.ndim == 3:
                    pc_2d = pc.reshape(-1, 3)
                else:
                    pc_2d = pc
                
                # Normalize and create image
                # For now, create a simple depth visualization
                depth = pc_2d[:, 2]
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_img = (depth_norm * 255).astype(np.uint8)
                
                # Try to reshape to square image if possible
                num_points = len(pc_2d)
                h = int(np.sqrt(num_points))
                w = num_points // h
                if h * w == num_points and h > 0 and w > 0:
                    depth_img = depth_img.reshape(h, w)
                    depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_VIRIDIS)
                    output_images.append(Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)))
                else:
                    # Fallback: use original input images if point cloud can't be visualized
                    pass
        
        # If no visualization was created, use input images as fallback
        if len(output_images) == 0:
            for img_data in images_data:
                if isinstance(img_data, np.ndarray):
                    img_uint8 = (img_data * 255).astype(np.uint8) if img_data.max() <= 1.0 else img_data.astype(np.uint8)
                    output_images.append(Image.fromarray(img_uint8))
        
        # Determine data type
        data_type = "image" if len(images_data) == 1 else "video"
        
        # Create result object
        result = CUT3RResult(
            images=output_images,
            point_clouds=results.get('point_cloud') if kwargs.get('return_point_clouds', True) else None,
            depth_maps=results.get('depth_map') if kwargs.get('return_depth_maps', True) else None,
            camera_poses=results.get('camera_pose') if kwargs.get('return_camera_poses', True) else None,
            data_type=data_type
        )
        
        return result
    
    def __call__(
        self,
        input_: Union[str, Image.Image, np.ndarray, List[str], List[Image.Image], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> CUT3RResult:
        """
        Main call interface for the pipeline.
        
        Args:
            input_: Input image(s)
            interaction: Interaction string or dictionary
            **kwargs: Additional arguments
            
        Returns:
            CUT3RResult object containing processed results as PIL Images or video frame list
        """
        return self.process(input_, interaction, **kwargs)
    
    def stream(
        self,
        input_: Union[str, Image.Image, np.ndarray, List[str], List[Image.Image], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[Union[torch.Tensor, List[str]], None, None]:
        """
        Stream processing interface for real-time interactive updates.
        
        Args:
            input_: Input image(s)
            interaction: Interaction string or dictionary
            **kwargs: Additional arguments
            
        Yields:
            Processed results as torch.Tensor or List[str] (for compatibility with diffusers-style streaming)
        """
        # For CUT3R, streaming is equivalent to regular processing
        # since inference is typically fast and not iterative
        result = self.process(input_, interaction, **kwargs)
        
        # Yield images as tensors for streaming compatibility
        for img in result.images:
            # Convert PIL Image to tensor
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            yield img_tensor


__all__ = ["CUT3RPipeline", "CUT3RResult"]

