"""
OmniVinci Pipeline for multimodal reasoning.

This pipeline integrates the OmniVinci operator and reasoning model
to provide a unified interface for multimodal inference.
"""

import torch
import os
from typing import Optional, Any, Union, Dict, List
from pathlib import Path
from PIL import Image
import soundfile as sf
from ...operators.omnivinci_operator import OmniVinciOperator
from ...reasoning.general_reasoning.omnivinci.omnivinci_reasoning import OmniVinciReasoning
from ...memories.reasoning.omnivinci.omnivinci_memory import OmniVinciMemory


class OmniVinciPipeline:
    """
    Pipeline for OmniVinci multimodal reasoning.
    
    Separates data preprocessing (operator) from model inference (reasoning).
    """
    
    def __init__(
        self,
        operator: Optional[OmniVinciOperator] = None,
        reasoning_model: Optional[OmniVinciReasoning] = None,
        memory_module: Optional[OmniVinciMemory] = None,
        device: str = 'cuda',
        load_audio_in_video: bool = True,
        num_video_frames: int = 128,
        audio_length: str = "max_3600",
    ):
        """
        Initialize OmniVinci Pipeline
        
        Args:
            operator: OmniVinci operator instance
            reasoning_model: OmniVinci reasoning model instance
            memory_module: Memory module for conversation history
            device: Device for inference
            load_audio_in_video: Whether to load audio track in videos
            num_video_frames: Number of frames to extract from video
            audio_length: Maximum audio length to process
        """
        self.operator = operator
        self.reasoning_model = reasoning_model
        self.memory_module = memory_module if memory_module else OmniVinciMemory()
        self.device = device
        self.load_audio_in_video = load_audio_in_video
        self.num_video_frames = num_video_frames
        self.audio_length = audio_length
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Union[str, Path] = "./",
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: str = "torch.float16",
        device_map: Union[str, dict] = "auto",
        load_audio_in_video: bool = True,
        num_video_frames: int = 128,
        audio_length: str = "max_3600",
        system_prompt: Optional[str] = None,
        logger=None,
        **kwargs
    ) -> 'OmniVinciPipeline':
        """
        Load complete pipeline from pretrained model
        
        Args:
            pretrained_model_path: Path to pretrained model
            device: Device for inference
            torch_dtype: Data type for model weights
            device_map: Device mapping strategy
            load_audio_in_video: Whether to load audio in video inputs
            num_video_frames: Number of frames to extract from video
            audio_length: Maximum audio length to process
            system_prompt: Custom system prompt
            logger: Logger instance
            **kwargs: Additional arguments
            
        Returns:
            OmniVinciPipeline: Initialized pipeline instance
        """
        if logger:
            logger.info(f"Loading OmniVinci pipeline from {pretrained_model_path}")
        
        # Load reasoning model
        if logger:
            logger.info("Loading OmniVinci reasoning model...")
        
        reasoning_model = OmniVinciReasoning.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            device=device,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs
        )
        
        # Initialize operator
        if logger:
            logger.info("Initializing OmniVinci operator...")
        
        operator = OmniVinciOperator(
            processor=reasoning_model.processor,
            load_audio_in_video=load_audio_in_video,
            num_video_frames=num_video_frames,
            audio_length=audio_length,
            system_prompt=system_prompt,
        )
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize memory module
        memory_module = OmniVinciMemory()
        
        # Create pipeline instance
        pipeline = cls(
            operator=operator,
            reasoning_model=reasoning_model,
            memory_module=memory_module,
            device=device,
            load_audio_in_video=load_audio_in_video,
            num_video_frames=num_video_frames,
            audio_length=audio_length,
        )
        
        if logger:
            logger.info("OmniVinci pipeline loaded successfully")
        
        return pipeline

    def api_init(self, api_key, endpoint):
        # API-based inference is not implemented for OmniVinci yet.
        raise NotImplementedError("API init is not supported for OmniVinci.")

    
    def process(
        self,
        text: Optional[str] = None,
        images: Optional[Union[str, Path, Image.Image, List]] = None,
        audios: Optional[Union[str, Path, bytes, List]] = None,
        videos: Optional[Union[str, Path, List]] = None,
        messages: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process inputs through operator
        
        Args:
            text: Text prompt
            images: Image inputs
            audios: Audio inputs
            videos: Video inputs
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
            images=images,
            audios=audios,
            videos=videos,
            **kwargs
        )
        
        # Merge messages
        final_messages = interaction_data.get("messages", [])
        perception_messages = perception_data.get("messages", [])
        
        for msg in final_messages:
            if msg.get("role") == "user":
                for p_msg in perception_messages:
                    if p_msg.get("role") == "user":
                        msg["content"].extend(p_msg["content"])
        
        return {
            "messages": final_messages,
            "load_audio_in_video": perception_data.get("load_audio_in_video", self.load_audio_in_video),
            "num_video_frames": perception_data.get("num_video_frames", self.num_video_frames),
            "audio_length": perception_data.get("audio_length", self.audio_length),
        }
    
    def __call__(
        self,
        text: Optional[str] = None,
        images: Optional[Union[str, Path, Image.Image, List]] = None,
        audios: Optional[Union[str, Path, bytes, List]] = None,
        videos: Optional[Union[str, Path, List]] = None,
        messages: Optional[List[Dict]] = None,
        max_new_tokens: int = 1024,
        generation_kwargs: Optional[dict] = None,
        use_operator: bool = True,
        **kwargs
    ) -> str:
        """
        Generate predictions
        
        Args:
            text: Text prompt
            images: Image inputs
            audios: Audio inputs
            videos: Video inputs
            messages: Pre-built messages (if provided, other inputs are ignored)
            max_new_tokens: Maximum number of tokens to generate
            generation_kwargs: Additional generation parameters
            use_operator: Whether to use operator for preprocessing
            **kwargs: Additional parameters
            
        Returns:
            Generated text string
        """
        if self.reasoning_model is None:
            raise ValueError("Reasoning model is not initialized")
        
        # Process inputs through operator if enabled
        if use_operator:
            processed_data = self.process(
                text=text,
                images=images,
                audios=audios,
                videos=videos,
                messages=messages,
                **kwargs
            )
            # Extract messages and configuration from processed data
            messages = processed_data.get("messages")
            load_audio_in_video = processed_data.get("load_audio_in_video", self.load_audio_in_video)
            num_video_frames = processed_data.get("num_video_frames", self.num_video_frames)
            audio_length = processed_data.get("audio_length", self.audio_length)
        else:
            # Use raw inputs
            load_audio_in_video = kwargs.get("load_audio_in_video", self.load_audio_in_video)
            num_video_frames = kwargs.get("num_video_frames", self.num_video_frames)
            audio_length = kwargs.get("audio_length", self.audio_length)
        
        # Run inference
        result = self.reasoning_model.inference(
            messages=messages,
            max_new_tokens=max_new_tokens,
            generation_kwargs=generation_kwargs,
            load_audio_in_video=load_audio_in_video,
            num_video_frames=num_video_frames,
            audio_length=audio_length,
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
                'load_audio_in_video': self.operator.load_audio_in_video,
                'num_video_frames': self.operator.num_video_frames,
                'audio_length': self.operator.audio_length,
                'system_prompt': self.operator.system_prompt,
                'operation_types': self.operator.opration_types if hasattr(self.operator, 'opration_types') else []
            }
            torch.save(operator_config, os.path.join(save_directory, "operator_config.pt"))
        
        # Save pipeline config
        pipeline_config = {
            'device': self.device,
            'load_audio_in_video': self.load_audio_in_video,
            'num_video_frames': self.num_video_frames,
            'audio_length': self.audio_length,
        }
        torch.save(pipeline_config, os.path.join(save_directory, "pipeline_config.pt"))
        
        print(f"OmniVinci Pipeline saved to {save_directory}")
    
    def update_operator_config(self, **kwargs):
        """
        Update operator configuration
        
        Args:
            **kwargs: Configuration parameters
        """
        if self.operator:
            self.operator.update_config(**kwargs)
    
    def get_operator(self) -> Optional[OmniVinciOperator]:
        """Get operator instance"""
        return self.operator
    
    def get_reasoning_model(self) -> Optional[OmniVinciReasoning]:
        """Get reasoning model instance"""
        return self.reasoning_model
    
    def stream(
        self,
        text: Optional[str] = None,
        images: Optional[Union[str, Path, Image.Image, List]] = None,
        audios: Optional[Union[str, Path, bytes, List]] = None,
        videos: Optional[Union[str, Path, List]] = None,
        use_history: bool = True,
        max_new_tokens: int = 1024,
        generation_kwargs: Optional[dict] = None,
        reset_memory: bool = False,
        **kwargs
    ) -> str:
        """
        Stream-based generation with conversation memory
        
        Args:
            text: Text prompt for current turn
            images: Image inputs for current turn
            audios: Audio inputs for current turn
            videos: Video inputs for current turn
            use_history: Whether to include conversation history
            max_new_tokens: Maximum number of tokens to generate
            generation_kwargs: Additional generation parameters
            reset_memory: Whether to reset memory before processing
            **kwargs: Additional parameters
            
        Returns:
            Generated text string
        """
        if reset_memory:
            self.memory_module.manage(action="reset")
            print("--- Stream Started (Memory Reset) ---")
        
        # Build messages from history
        messages = None
        if use_history:
            messages = self.memory_module.select()
        
        # Process inputs through operator
        # Text will be merged into messages if messages exist
        processed_data = self.process(
            text=text,
            images=images,
            audios=audios,
            videos=videos,
            messages=messages,
            **kwargs
        )
        
        current_messages = processed_data.get("messages")
        load_audio_in_video = processed_data.get("load_audio_in_video", self.load_audio_in_video)
        num_video_frames = processed_data.get("num_video_frames", self.num_video_frames)
        audio_length = processed_data.get("audio_length", self.audio_length)
        
        # Run inference
        result = self.reasoning_model.inference(
            messages=current_messages,
            max_new_tokens=max_new_tokens,
            generation_kwargs=generation_kwargs,
            load_audio_in_video=load_audio_in_video,
            num_video_frames=num_video_frames,
            audio_length=audio_length,
        )
        
        response_text = result
        
        # Record to memory
        self.memory_module.record({
            'messages': current_messages,
            'response': response_text,
            'metadata': {
                'max_new_tokens': max_new_tokens,
                'load_audio_in_video': load_audio_in_video,
                'num_video_frames': num_video_frames,
                'audio_length': audio_length,
            }
        })
        
        return result
