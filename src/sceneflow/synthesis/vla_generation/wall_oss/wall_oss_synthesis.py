"""
Wall-OSS Synthesis Module for VLA (Vision-Language-Action) tasks.

This module wraps the Wall-OSS model for action prediction based on visual inputs.
"""

from typing import Optional, Union, Dict, Any
import torch
from PIL import Image
import yaml
import os

from ...base_synthesis import BaseSynthesis


try:
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
    WALL_X_AVAILABLE = True
except ImportError:
    WALL_X_AVAILABLE = False
    Qwen2_5_VLMoEForAction = None


class WallOssSynthesis(BaseSynthesis):
    """
    Synthesis wrapper for Wall-OSS VLA model.
    
    Based on Qwen2.5-VL with action prediction capabilities.
    Supports visual question answering for robotic action prediction.
    """
    
    def __init__(
        self,
        model: "Qwen2_5_VLMoEForAction",
        processor,
        train_config: Optional[Dict] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize Wall-OSS Synthesis
        
        Args:
            model: Qwen2_5_VLMoEForAction model instance
            processor: AutoProcessor instance
            train_config: Training configuration dictionary
            device: Device for inference
        """
        if not WALL_X_AVAILABLE:
            raise ImportError(
                "wall_x is not installed. Please install it to use WallOssSynthesis."
            )
        
        super().__init__()
        self.model = model
        self.processor = processor
        self.train_config = train_config or {}
        self.device = torch.device(device) if device is not None else self._get_default_device()
        
    @classmethod
    def get_default_config(cls, processor_path: str = "") -> Dict:
        """
        Get default training config for Wall-OSS model.
        
        This config is used for inference when no external config is provided.
        For training, you should provide a complete config via train_config parameter.
        
        Returns:
            Dict: Default configuration dictionary
        """
        return {
            # Model and paths configuration
            "model_type": "qwen2_5",
            "use_fast_tokenizer": False,
            "processor_path": processor_path,
            # Robot configuration - Define degrees of freedom for each component
            "dof_config": {
                "follow_left_ee_cartesian_pos": 3,
                "follow_left_ee_rotation": 3,
                "follow_left_gripper": 1,
                "follow_right_ee_cartesian_pos": 3,
                "follow_right_ee_rotation": 3,
                "follow_right_gripper": 1,
                "head_actions": 2,
                "height": 1,
                "car_pose": 3,
            },
            
            # Agent proprioception configuration
            "agent_pos_config": {
                "follow_left_ee_cartesian_pos": 3,
                "follow_left_ee_rotation": 3,
                "follow_left_gripper": 1,
                "follow_right_ee_cartesian_pos": 3,
                "follow_right_ee_rotation": 3,
                "follow_right_gripper": 1,
                "head_actions": 2,
                "height": 1,
                "car_pose": 3,
            },
            
            # Enable customized robot configuration
            "enable_customized_robot_config": True,
            "customized_robot_config": {
                "name": "physical-intelligence/libero",
                "customized_dof_config": {
                    "action_left_shoulder": 1,
                    "action_left_elbow": 1,
                    "action_left_forearm_roll": 1,
                    "action_left_wrist_angle": 1,
                    "action_left_wrist_rotate": 1,
                    "action_left_gripper": 1,
                    "action_right_waist": 1,
                    "action_right_shoulder": 1,
                    "action_right_elbow": 1,
                    "action_right_forearm_roll": 1,
                    "action_right_wrist_angle": 1,
                    "action_right_wrist_rotate": 1,
                    "action_right_gripper": 1,
                },
                "customized_agent_pos_config": {
                    "state_left_shoulder": 1,
                    "state_left_elbow": 1,
                    "state_left_forearm_roll": 1,
                    "state_left_wrist_angle": 1,
                    "state_left_wrist_rotate": 1,
                    "state_left_gripper": 1,
                    "state_right_waist": 1,
                    "state_right_shoulder": 1,
                    "state_right_elbow": 1,
                    "state_right_forearm_roll": 1,
                    "state_right_wrist_angle": 1,
                    "state_right_wrist_rotate": 1,
                    "state_right_gripper": 1,
                },
            },
            
            # Data configuration (for reference, not used during inference)
            "data": {
                "action_horizon": 32,
                "resolution": {
                    "face_view": 256,
                    "left_wrist_view": 256,
                    "right_wrist_view": 256,
                    "move1_view": 256,
                    "move2_view": 256,
                    "top_view": 256,
                    "wall_view": 256,
                    "multi_modal": 256,
                },
            },
        }
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        train_config_path: Optional[str] = None,
        train_config: Optional[Dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ) -> "WallOssSynthesis":
        """
        Load Wall-OSS model and processor from pretrained checkpoint.
        
        Args:
            pretrained_model_path: Path to pretrained model directory
            train_config_path: (Optional) Path to training config YAML file
            train_config: (Optional) Training config dictionary (overrides train_config_path)
                         If not provided, uses default config
            device: Device for inference
            **kwargs: Additional arguments
            
        Returns:
            WallOssSynthesis instance
        """
        if not WALL_X_AVAILABLE:
            raise ImportError(
                "wall_x is not installed. Please install it to use WallOssSynthesis."
            )
        
        # Load training config
        if train_config is None:
            if train_config_path is not None:
                # Load from provided path
                with open(train_config_path, "r") as f:
                    train_config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                # Try to load from model directory
                config_path = os.path.join(pretrained_model_path, "config.yml")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        train_config = yaml.load(f, Loader=yaml.FullLoader)
                else:
                    # Use default config
                    train_config = cls.get_default_config(pretrained_model_path)
        
        # Load processor
        from transformers import AutoProcessor
        processor_path = train_config.get("processor_path", pretrained_model_path)
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        
        # Load model
        model = Qwen2_5_VLMoEForAction.from_pretrained(
            pretrained_model_path, 
            train_config=train_config
        )
        
        # Move model to device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda":
            model = model.to(device, dtype=torch.bfloat16)
        else:
            model = model.to(device)
        
        model.eval()
        
        return cls(
            model=model,
            processor=processor,
            train_config=train_config,
            device=device
        )
    
    def api_init(self, api_key, endpoint):
        """API-based inference not supported for Wall-OSS."""
        raise NotImplementedError("API init is not supported for Wall-OSS.")
    
    def _get_default_device(self) -> torch.device:
        """Get default device for inference."""
        if hasattr(self.model, "device"):
            return self.model.device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    @torch.no_grad()
    def predict(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        messages: Optional[list] = None,
        max_new_tokens: int = 1024,
        generation_kwargs: Optional[dict] = None,
        **kwargs
    ) -> str:
        """
        Run Wall-OSS inference for action prediction.
        
        Args:
            image: PIL Image input
            text: Text prompt/question
            messages: Pre-built messages in chat format
            max_new_tokens: Maximum tokens to generate
            generation_kwargs: Additional generation parameters
            **kwargs: Additional parameters
            
        Returns:
            str: Generated action prediction response
        """
        # Build messages if not provided
        if messages is None:
            if image is None or text is None:
                raise ValueError("Either messages or both image and text must be provided")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text}
                    ],
                }
            ]
        
        # Apply chat template
        text_prompt = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Prepare inputs
        if image is not None:
            inputs = self.processor(
                text=[text_prompt], 
                images=[image], 
                return_tensors="pt"
            )
        else:
            # Extract image from messages if present
            inputs = self.processor(
                text=[text_prompt],
                return_tensors="pt"
            )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Prepare generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)
        
        # Generate
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode output
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
