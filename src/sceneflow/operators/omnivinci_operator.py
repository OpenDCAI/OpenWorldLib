"""
OmniVinci Operator for multimodal data preprocessing.

This operator handles preprocessing for text, image, audio, and video inputs
for the OmniVinci model.
"""

import numpy as np
from PIL import Image
import torch
from typing import Union, Optional, Dict, Any, List
from pathlib import Path

from .base_operator import BaseOperator


class OmniVinciOperator(BaseOperator):
    """
    Operator for OmniVinci multimodal preprocessing.
    
    Supports:
    - Text prompts
    - Image inputs (single or multiple)
    - Audio inputs (single or multiple)
    - Video inputs (with optional audio track)
    """
    
    def __init__(
        self,
        processor=None,
        load_audio_in_video: bool = True,
        num_video_frames: int = 128,
        audio_length: str = "max_3600",
        system_prompt: Optional[str] = None,
        operation_types: List[str] = None,
    ):
        """
        Initialize OmniVinci Operator
        
        Args:
            processor: AutoProcessor instance
            load_audio_in_video: Whether to load audio track in video inputs
            num_video_frames: Number of frames to extract from video
            audio_length: Maximum audio length to process
            system_prompt: System prompt for the model
            operation_types: List of operation types
        """
        if operation_types is None:
            operation_types = [
                "text_processing",
                "image_processing",
                "audio_processing",
                "video_processing",
                "multimodal_processing"
            ]
        
        super().__init__(operation_types)
        
        self.processor = processor
        self.load_audio_in_video = load_audio_in_video
        self.num_video_frames = num_video_frames
        self.audio_length = audio_length
        
        # Default system prompt for OmniVinci
        if system_prompt is None:
            self.system_prompt = (
                "You are OmniVinci, an advanced multimodal AI assistant "
                "capable of understanding and processing text, images, audio, and video content."
            )
        else:
            self.system_prompt = system_prompt
        
        # Initialize interaction template
        self.interaction_template = [
            "text_prompt",
            "image_prompt",
            "audio_prompt",
            "video_prompt",
            "multimodal_prompt"
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
        """
        
        return str(image_input)
    
    def load_audio(self, audio_input: Union[str, Path, bytes]) -> Union[str, bytes]:
        """
        Load audio file
        
        Args:
            audio_input: Audio file path or bytes
            
        Returns:
            Audio path or bytes
        """
        if isinstance(audio_input, (str, Path)):
            return str(audio_input)
        return audio_input
    
    def load_video(self, video_input: Union[str, Path]) -> str:
        """
        Load video file
        
        Args:
            video_input: Video file path
            
        Returns:
            Video path as string
        """
        return str(video_input)
    
    def process_interaction(
        self,
        text: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        include_system_prompt: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process text interaction inputs
        
        Args:
            text: Text prompt
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
        audios: Optional[Union[str, Path, bytes, List]] = None,
        videos: Optional[Union[str, Path, List]] = None,
        include_system_prompt: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process perception inputs (images, audios, videos)
        
        Args:
            images: Single image or list of images
            audios: Single audio or list of audios
            videos: Single video or list of videos
            include_system_prompt: Whether to include system prompt
            **kwargs: Additional parameters
            
        Returns:
            Dict containing:
                - messages: Processed messages with perception data
                - load_audio_in_video: Whether to load audio in video
                - num_video_frames: Number of frames to extract from video
                - audio_length: Maximum audio length
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
        
        # Add videos first (as per OmniVinci's expected order)
        if videos is not None:
            if not isinstance(videos, list):
                videos = [videos]
            for video in videos:
                processed_video = self.load_video(video)
                content.append({"type": "video", "video": processed_video})
        
        # Add images
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            for img in images:
                processed_img = self.load_image(img)
                content.append({"type": "image", "image": processed_img})
        
        # Add audios
        if audios is not None:
            if not isinstance(audios, list):
                audios = [audios]
            for audio in audios:
                processed_audio = self.load_audio(audio)
                content.append({"type": "audio", "audio": processed_audio})
        
        # Add user message
        if content:
            messages.append({
                "role": "user",
                "content": content
            })
        
        result = {
            "messages": messages,
            "load_audio_in_video": self.load_audio_in_video,
            "num_video_frames": self.num_video_frames,
            "audio_length": self.audio_length,
        }
        
        return result
    
    def update_config(self, **kwargs):
        """
        Update operator configuration
        
        Args:
            **kwargs: Configuration parameters to update
        """
        if "load_audio_in_video" in kwargs:
            self.load_audio_in_video = kwargs["load_audio_in_video"]
        
        if "num_video_frames" in kwargs:
            self.num_video_frames = kwargs["num_video_frames"]
        
        if "audio_length" in kwargs:
            self.audio_length = kwargs["audio_length"]
        
        if "system_prompt" in kwargs:
            self.system_prompt = kwargs["system_prompt"]
