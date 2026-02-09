"""
Wall-OSS Operator for VLA (Vision-Language-Action) preprocessing.

This operator handles preprocessing for image and text inputs
for the Wall-OSS action prediction model.
"""

import numpy as np
from PIL import Image
import torch
from typing import Union, Optional, Dict, Any, List
from pathlib import Path

from .base_operator import BaseOperator


class WallOssOperator(BaseOperator):
    """
    Operator for Wall-OSS VLA preprocessing.
    
    Supports:
    - Text prompts (action prediction questions)
    - Image inputs (single observation images)
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        operation_types: List[str] = None,
    ):
        """
        Initialize Wall-OSS Operator
        
        Args:
            system_prompt: System prompt for the model
            operation_types: List of operation types
        """
        if operation_types is None:
            operation_types = [
                "text_processing",
                "image_processing",
                "action_prediction"
            ]
        
        super().__init__(operation_types)
        
        # Default system prompt for Wall-OSS VLA
        if system_prompt is None:
            self.system_prompt = (
                "You are a vision-language-action model that can understand visual scenes "
                "and predict appropriate actions for robotic manipulation tasks."
            )
        else:
            self.system_prompt = system_prompt
        
        # Initialize interaction template
        self.interaction_template = [
            "action_query",
            "visual_observation",
            "step_by_step_reasoning"
        ]
        self.interaction_template_init()
    
    def check_interaction(self, interaction):
        """Check if interaction type is valid"""
        if not isinstance(interaction, (str, dict, list)):
            raise TypeError(f"Invalid interaction type: {type(interaction)}")
        return True
    
    def get_interaction(self, interaction):
        """Get and store current interaction"""
        if self.check_interaction(interaction):
            self.current_interaction = interaction
    
    def load_image(self, image_input: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Load and preprocess image
        
        Args:
            image_input: Image path or PIL Image
            
        Returns:
            PIL Image in RGB mode
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input)
        else:
            pil_img = image_input
        
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        return pil_img
    
    def process_interaction(
        self,
        text: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        include_system_prompt: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process text interaction inputs
        
        Args:
            text: Text prompt/question
            messages: Pre-built messages (text will be appended if provided)
            include_system_prompt: Whether to include system prompt
            **kwargs: Additional parameters
            
        Returns:
            Dict containing:
                - messages: Processed messages with text
                - text: Original text prompt
        """
        # Store current interaction
        self.get_interaction(text or messages)
        
        result = {}
        
        # Build or use provided messages
        if messages is not None:
            # Use existing messages and append text if provided
            result["messages"] = messages.copy() if isinstance(messages, list) else messages
            
            # If text is provided, append it to the last user message or create new one
            if text:
                # Find last user message
                last_user_idx = None
                for i in range(len(result["messages"]) - 1, -1, -1):
                    if result["messages"][i].get("role") == "user":
                        last_user_idx = i
                        break
                
                if last_user_idx is not None:
                    # Append to existing user message
                    if isinstance(result["messages"][last_user_idx]["content"], list):
                        result["messages"][last_user_idx]["content"].append(
                            {"type": "text", "text": text}
                        )
                    else:
                        # Convert to list format if needed
                        result["messages"][last_user_idx]["content"] = [
                            {"type": "text", "text": result["messages"][last_user_idx]["content"]},
                            {"type": "text", "text": text}
                        ]
                else:
                    # No user message found, create new one
                    result["messages"].append({
                        "role": "user",
                        "content": [{"type": "text", "text": text}]
                    })
            
            result["text"] = text
        else:
            built_messages = []
            
            # Add system prompt if requested
            if include_system_prompt and self.system_prompt:
                built_messages.append({
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt}
                    ]
                })
            
            # Build user message content with text only
            content = []
            if text:
                content.append({"type": "text", "text": text})
            
            # Add user message
            if content:
                built_messages.append({
                    "role": "user",
                    "content": content
                })
            
            result["messages"] = built_messages
            result["text"] = text
        
        return result
    
    def process_perception(
        self,
        images: Optional[Union[str, Path, Image.Image, List]] = None,
        include_system_prompt: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process perception inputs (images for VLA)
        
        Args:
            images: Single image or list of images
            include_system_prompt: Whether to include system prompt
            **kwargs: Additional parameters
            
        Returns:
            Dict containing:
                - messages: Processed messages with perception data
                - images: Processed images
        """
        messages = []
        
        # Add system prompt if requested
        if include_system_prompt and self.system_prompt:
            messages.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt}
                ]
            })
        
        # Build user message content
        content = []
        processed_images = []
        
        # Add images
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            for img in images:
                processed_img = self.load_image(img)
                processed_images.append(processed_img)
                content.append({"type": "image"})
        
        # Add user message
        if content:
            messages.append({
                "role": "user",
                "content": content
            })
        
        result = {
            "messages": messages,
            "images": processed_images,
        }
        
        return result
    
    def update_config(self, **kwargs):
        """
        Update operator configuration
        
        Args:
            **kwargs: Configuration parameters to update
        """
        if "system_prompt" in kwargs:
            self.system_prompt = kwargs["system_prompt"]
