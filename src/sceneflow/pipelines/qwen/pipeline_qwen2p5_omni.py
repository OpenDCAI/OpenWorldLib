"""
Qwen2.5-Omni Pipeline for multimodal reasoning.

This pipeline integrates the Qwen2.5-Omni operator and reasoning model
to provide a unified interface for multimodal inference.
"""

import torch
import os
from typing import Optional, Any, Union, Dict, List
from pathlib import Path
from PIL import Image
import soundfile as sf
from ...operators.qwen2p5_omni_operator import Qwen2p5OmniOperator
from ...reasoning.general_reasoning.qwen.qwen2p5_omni_reasoning import Qwen2p5OmniReasoning
from ...memories.reasoning.qwen.qwen_memory import QwenMemory


class Qwen2p5OmniPipeline:
    """
    Pipeline for Qwen2.5-Omni multimodal reasoning.
    
    Separates data preprocessing (operator) from model inference (reasoning).
    """
    
    def __init__(
        self,
        operator: Optional[Qwen2p5OmniOperator] = None,
        reasoning_model: Optional[Qwen2p5OmniReasoning] = None,
        memory_module: Optional[QwenMemory] = None,
        device: str = 'cuda',
        use_audio_in_video: bool = True,
    ):
        """
        Initialize Qwen2.5-Omni Pipeline
        
        Args:
            operator: Qwen2.5-Omni operator instance
            reasoning_model: Qwen2.5-Omni reasoning model instance
            memory_module: Memory module for conversation history
            device: Device for inference
            use_audio_in_video: Whether to use audio track in videos
        """
        self.operator = operator
        self.reasoning_model = reasoning_model
        self.memory_module = memory_module if memory_module else QwenMemory()
        self.device = device
        self.use_audio_in_video = use_audio_in_video
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Union[str, Path] = "Qwen/Qwen2.5-Omni-7B",
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = None,
        device_map: Union[str, dict] = "auto",
        use_audio_in_video: bool = False,
        system_prompt: Optional[str] = None,
        logger=None,
        **kwargs
    ) -> 'Qwen2p5OmniPipeline':
        """
        Load complete pipeline from pretrained model
        
        Args:
            pretrained_model_path: Path to pretrained model
            device: Device for inference
            torch_dtype: Data type for model weights
            attn_implementation: Attention implementation (e.g., "flash_attention_2")
            device_map: Device mapping strategy
            use_audio_in_video: Whether to use audio in video inputs
            system_prompt: Custom system prompt
            logger: Logger instance
            **kwargs: Additional arguments
            
        Returns:
            Qwen2p5_OmniPipeline: Initialized pipeline instance
        """
        if logger:
            logger.info(f"Loading Qwen2.5-Omni pipeline from {pretrained_model_path}")
        
        # Load reasoning model
        if logger:
            logger.info("Loading Qwen2.5-Omni reasoning model...")
        
        reasoning_model = Qwen2p5OmniReasoning.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            **kwargs
        )
        
        # Initialize operator
        if logger:
            logger.info("Initializing Qwen2.5-Omni operator...")
        
        operator = Qwen2p5OmniOperator(
            processor=reasoning_model.processor,
            use_audio_in_video=use_audio_in_video,
            system_prompt=system_prompt,
        )
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize memory module
        memory_module = QwenMemory()
        
        # Create pipeline instance
        pipeline = cls(
            operator=operator,
            reasoning_model=reasoning_model,
            memory_module=memory_module,
            device=device,
            use_audio_in_video=use_audio_in_video,
        )
        
        if logger:
            logger.info("Qwen2.5-Omni pipeline loaded successfully")
        
        return pipeline

    def api_init(self, api_key, endpoint):
        # API-based inference is not implemented for Qwen2.5-Omni yet.
        raise NotImplementedError("API init is not supported for Qwen2.5-Omni.")


    
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
            "use_audio_in_video": perception_data.get("use_audio_in_video", self.use_audio_in_video)
        }
    
    def __call__(
        self,
        text: Optional[str] = None,
        images: Optional[Union[str, Path, Image.Image, List]] = None,
        audios: Optional[Union[str, Path, bytes, List]] = None,
        videos: Optional[Union[str, Path, List]] = None,
        messages: Optional[List[Dict]] = None,
        max_new_tokens: int = 128,
        generation_kwargs: Optional[dict] = None,
        return_audio: bool = False,
        use_operator: bool = True,
        **kwargs
    ) -> Union[List[str], tuple]:
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
            return_audio: Whether to return generated audio
            use_operator: Whether to use operator for preprocessing
            **kwargs: Additional parameters
            
        Returns:
            If return_audio is False: List of generated text strings
            If return_audio is True: Tuple of (text strings, audio tensor)
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
            
            # Extract messages and use_audio_in_video from processed data
            messages = processed_data.get("messages")
            use_audio_in_video = processed_data.get("use_audio_in_video", self.use_audio_in_video)
        else:
            # Use raw inputs
            use_audio_in_video = kwargs.get("use_audio_in_video", self.use_audio_in_video)
        
        # Run inference
        if not return_audio:
            result = self.reasoning_model.inference(
                messages=messages,
                max_new_tokens=max_new_tokens,
                generation_kwargs=generation_kwargs,
                use_audio_in_video=use_audio_in_video,
                return_audio=return_audio,
            )
        else:
            result, audio = self.reasoning_model.inference(
                messages=messages,
                max_new_tokens=max_new_tokens,
                generation_kwargs=generation_kwargs,
                use_audio_in_video=use_audio_in_video,
                return_audio=return_audio,
            )
            return result, audio
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
                'use_audio_in_video': self.operator.use_audio_in_video,
                'system_prompt': self.operator.system_prompt,
                'operation_types': self.operator.opration_types if hasattr(self.operator, 'opration_types') else []
            }
            torch.save(operator_config, os.path.join(save_directory, "operator_config.pt"))
        
        # Save pipeline config
        pipeline_config = {
            'device': self.device,
            'use_audio_in_video': self.use_audio_in_video,
        }
        torch.save(pipeline_config, os.path.join(save_directory, "pipeline_config.pt"))
        
        print(f"Qwen2.5-Omni Pipeline saved to {save_directory}")
    
    def update_operator_config(self, **kwargs):
        """
        Update operator configuration
        
        Args:
            **kwargs: Configuration parameters
        """
        if self.operator:
            self.operator.update_config(**kwargs)
    
    def get_operator(self) -> Optional[Qwen2p5OmniOperator]:
        """Get operator instance"""
        return self.operator
    
    def get_reasoning_model(self) -> Optional[Qwen2p5OmniReasoning]:
        """Get reasoning model instance"""
        return self.reasoning_model
    
    def stream(
        self,
        text: Optional[str] = None,
        images: Optional[Union[str, Path, Image.Image, List]] = None,
        audios: Optional[Union[str, Path, bytes, List]] = None,
        videos: Optional[Union[str, Path, List]] = None,
        use_history: bool = True,
        max_new_tokens: int = 128,
        generation_kwargs: Optional[dict] = None,
        return_audio: bool = False,
        reset_memory: bool = False,
        **kwargs
    ) -> Union[List[str], tuple]:
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
            return_audio: Whether to return generated audio
            reset_memory: Whether to reset memory before processing
            **kwargs: Additional parameters
            
        Returns:
            If return_audio is False: List of generated text strings
            If return_audio is True: Tuple of (text strings, audio tensor)
        """
        if reset_memory:
            self.memory_module.manage(action="reset")
            print("--- Stream Started (Memory Reset) ---")
        
        # Build current turn messages
        messages = None
        if use_history:
            messages = self.memory_module.select()
        
        # Process inputs through operator
        processed_data = self.process(
            text=text,
            images=images,
            audios=audios,
            videos=videos,
            messages=messages,
            **kwargs
        )
        
        current_messages = processed_data.get("messages")
        use_audio_in_video = processed_data.get("use_audio_in_video", self.use_audio_in_video)
        breakpoint()
        # Run inference
        if not return_audio:
            result = self.reasoning_model.inference(
                messages=current_messages,
                max_new_tokens=max_new_tokens,
                generation_kwargs=generation_kwargs,
                use_audio_in_video=use_audio_in_video,
                return_audio=return_audio,
            )
            response_text = result[0] if isinstance(result, list) else result
        else:
            result, audio = self.reasoning_model.inference(
                messages=current_messages,
                max_new_tokens=max_new_tokens,
                generation_kwargs=generation_kwargs,
                use_audio_in_video=use_audio_in_video,
                return_audio=return_audio,
            )
            response_text = result[0] if isinstance(result, list) else result
        
        # Record to memory
        self.memory_module.record({
            'messages': current_messages,
            'response': response_text,
            'metadata': {
                'max_new_tokens': max_new_tokens,
                'return_audio': return_audio
            }
        })
        
        if return_audio:
            return result, audio
        return result

