"""
Wall-OSS Pipeline for VLA (Vision-Language-Action) synthesis.

This pipeline integrates the Wall-OSS operator and synthesis model
to provide a unified interface for action prediction inference.
"""

import torch
import os
from typing import Optional, Any, Union, Dict, List
from pathlib import Path
from PIL import Image

from ...operators.wall_oss_operator import WallOssOperator
from ...synthesis.vla_generation.wall_oss.wall_oss_synthesis import WallOssSynthesis


class WallOssPipeline:
    """
    Pipeline for Wall-OSS VLA (Vision-Language-Action) synthesis.
    
    Separates data preprocessing (operator) from model inference (synthesis).
    Designed for robotic action prediction tasks.
    """
    
    def __init__(
        self,
        operator: Optional[WallOssOperator] = None,
        synthesis_model: Optional[WallOssSynthesis] = None,
        device: str = 'cuda',
    ):
        """
        Initialize Wall-OSS Pipeline
        
        Args:
            operator: Wall-OSS operator instance
            synthesis_model: Wall-OSS synthesis model instance
            device: Device for inference
        """
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.device = device
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Union[str, Path],
        train_config_path: Optional[str] = None,
        train_config: Optional[Dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        system_prompt: Optional[str] = None,
        logger=None,
        **kwargs
    ) -> 'WallOssPipeline':
        """
        Load complete pipeline from pretrained model
        
        Args:
            pretrained_model_path: Path to pretrained Wall-OSS model
            train_config_path: Path to training config YAML file
            train_config: Training config dictionary (overrides train_config_path)
            device: Device for inference
            system_prompt: Custom system prompt
            logger: Logger instance
            **kwargs: Additional arguments
            
        Returns:
            WallOssPipeline: Initialized pipeline instance
        """
        if logger:
            logger.info(f"Loading Wall-OSS pipeline from {pretrained_model_path}")
        
        # Load synthesis model
        if logger:
            logger.info("Loading Wall-OSS synthesis model...")
        
        synthesis_model = WallOssSynthesis.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            train_config_path=train_config_path,
            train_config=train_config,
            device=device,
            **kwargs
        )
        
        # Initialize operator
        if logger:
            logger.info("Initializing Wall-OSS operator...")
        
        operator = WallOssOperator(
            processor=synthesis_model.processor,
            system_prompt=system_prompt,
        )
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create pipeline instance
        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            device=device,
        )
        
        if logger:
            logger.info("Wall-OSS pipeline loaded successfully")
        
        return pipeline

    def api_init(self, api_key, endpoint):
        """API-based inference not supported for Wall-OSS."""
        raise NotImplementedError("API init is not supported for Wall-OSS.")
    
    def process(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, Image.Image]] = None,
        messages: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process inputs through operator
        
        Args:
            text: Text prompt/question
            image: Image input
            messages: Pre-built messages
            **kwargs: Additional parameters
            
        Returns:
            Dict containing processed data
        """
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        
        # Process text interaction
        interaction_data = self.operator.process_interaction(
            text=text,
            messages=messages,
            **kwargs
        )
        
        # Process perception inputs
        perception_data = self.operator.process_perception(
            images=image,
            **kwargs
        )
        
        # Merge messages
        final_messages = interaction_data.get("messages", [])
        perception_messages = perception_data.get("messages", [])
        processed_images = perception_data.get("images", [])
        
        # Merge perception content into interaction messages
        for msg in final_messages:
            if msg.get("role") == "user":
                for p_msg in perception_messages:
                    if p_msg.get("role") == "user":
                        # Prepend image content before text
                        msg["content"] = p_msg["content"] + msg["content"]
        
        return {
            "messages": final_messages,
            "image": processed_images[0] if processed_images else None,
        }
    
    def __call__(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, Image.Image]] = None,
        messages: Optional[List[Dict]] = None,
        max_new_tokens: int = 1024,
        generation_kwargs: Optional[dict] = None,
        use_operator: bool = True,
        **kwargs
    ) -> str:
        """
        Generate action predictions
        
        Args:
            text: Text prompt/question
            image: Image input
            messages: Pre-built messages (if provided, other inputs are ignored)
            max_new_tokens: Maximum number of tokens to generate
            generation_kwargs: Additional generation parameters
            use_operator: Whether to use operator for preprocessing
            **kwargs: Additional parameters
            
        Returns:
            str: Generated action prediction response
        """
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        
        # Process inputs through operator if enabled
        if use_operator:
            processed_data = self.process(
                text=text,
                image=image,
                messages=messages,
                **kwargs
            )
            
            # Extract messages and image from processed data
            messages = processed_data.get("messages")
            image = processed_data.get("image")
        
        # Run inference
        result = self.synthesis_model.predict(
            image=image,
            text=text if not use_operator else None,
            messages=messages,
            max_new_tokens=max_new_tokens,
            generation_kwargs=generation_kwargs,
        )
        
        return result
    
    def save_pretrained(self, save_directory: str):
        """
        Save pipeline to directory
        
        Args:
            save_directory: Directory to save to
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save operator config
        if self.operator:
            operator_config = {
                'system_prompt': self.operator.system_prompt,
                'operation_types': self.operator.opration_types if hasattr(self.operator, 'opration_types') else []
            }
            torch.save(operator_config, os.path.join(save_directory, "operator_config.pt"))
        
        # Save pipeline config
        pipeline_config = {
            'device': self.device,
        }
        torch.save(pipeline_config, os.path.join(save_directory, "pipeline_config.pt"))
        
        print(f"Wall-OSS Pipeline saved to {save_directory}")
    
    def update_operator_config(self, **kwargs):
        """
        Update operator configuration
        
        Args:
            **kwargs: Configuration parameters
        """
        if self.operator:
            self.operator.update_config(**kwargs)
    
    def get_operator(self) -> Optional[WallOssOperator]:
        """Get operator instance"""
        return self.operator
    
    def get_synthesis_model(self) -> Optional[WallOssSynthesis]:
        """Get synthesis model instance"""
        return self.synthesis_model
