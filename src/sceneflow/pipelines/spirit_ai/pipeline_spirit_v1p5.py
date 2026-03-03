import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Union, Generator, Any
from contextlib import nullcontext

from ..pipeline_utils import PipelineABC
from ...operators.spirit_v1p5_operator import SpiritV1p5Operator
from ...synthesis.vla_generation.spirit_ai.spirit_v1p5_synthesis import SpiritV1p5Synthesis
from ...memories.vla_synthesis.spirit_ai.spirit_v1p5_memory import SpiritV1p5Memory


class SpiritV1p5Pipeline(PipelineABC):
    """
    Spirit-VLA Pipeline for Vision-Language-Action inference.
    
    Handles the complete inference workflow including:
    - Image preprocessing
    - State normalization and transformation
    - Action prediction via diffusion transformer
    - Action postprocessing (delta to absolute conversion)
    """
    
    def __init__(
        self,
        synthesis: SpiritV1p5Synthesis,
        operator: SpiritV1p5Operator,
        memory: Optional[SpiritV1p5Memory] = None,
        device: str = "cuda",
        use_bf16: bool = True,
        used_chunk_size: int = 50,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.synthesis = synthesis.to(self.device)
        self.operator = operator.to(self.device)
        self.memory = memory or SpiritV1p5Memory(capacity=100)
        self.use_bf16 = use_bf16 and self.device.type == "cuda"
        self.used_chunk_size = used_chunk_size
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        norm_stats_path: Optional[str] = None,
        device: str = "cuda",
        use_bf16: bool = True,
        used_chunk_size: int = 50,
        **kwargs: Any,
    ):
        """
        Load pretrained Spirit-VLA model and initialize pipeline.
        
        Args:
            pretrained_model_path: Path to Spirit-VLA checkpoint
            norm_stats_path: Path to normalization statistics JSON file
            device: Device to load model on
            use_bf16: Whether to use bfloat16 precision
            used_chunk_size: Number of action steps to use from prediction
            **kwargs: Additional arguments passed to synthesis
        
        Returns:
            Initialized SpiritVLAPipeline instance
        """
        # Load normalization statistics
        raw_embodiment_stats = None
        if norm_stats_path and Path(norm_stats_path).exists():
            with open(norm_stats_path, "r") as f:
                raw_embodiment_stats = json.load(f)
        
        # Initialize synthesis (model wrapper)
        synthesis = SpiritV1p5Synthesis.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            device=device,
            **kwargs,
        )
        
        # Initialize operator (data processing)
        operator = SpiritV1p5Operator(
            raw_embodiment_stats=raw_embodiment_stats,
            target_image_size=(320, 240),
        )
        
        # Initialize memory
        memory = SpiritV1p5Memory(capacity=100)
        
        return cls(
            synthesis=synthesis,
            operator=operator,
            memory=memory,
            device=device,
            use_bf16=use_bf16,
            used_chunk_size=used_chunk_size,
        )
    
    def to(self, device: Union[str, torch.device]):
        """Move pipeline to specified device."""
        self.device = torch.device(device)
        self.synthesis.to(self.device)
        self.operator.to(self.device)
        return self
    
    def process(
        self,
        images: Dict[str, Union[str, Image.Image, np.ndarray, torch.Tensor]],
        raw_state: List[float],
        task: str,
        robot_type: str,
    ) -> Dict[str, Any]:
        """
        Process inputs through operator to prepare model-ready tensors.
        
        Args:
            images: Dictionary of camera images
            raw_state: Raw robot state vector
            task: Task description text
            robot_type: Robot type identifier
        
        Returns:
            Dictionary containing processed batch and metadata
        """
        # Process perception (images)
        image_tensors = self.operator.process_perception(images)
        
        # Process interaction (task, state, robot_type)
        interaction_data = self.operator.process_interaction(
            task=task,
            raw_state=raw_state,
            robot_type=robot_type,
        )
        
        # Construct batch for model
        batch = {
            key: tensor.unsqueeze(0).to(self.device)
            for key, tensor in image_tensors.items()
        }
        batch["observation.state"] = interaction_data["state_tensor"].unsqueeze(0).to(self.device)
        batch["task"] = [task]
        batch["robot_type"] = [robot_type]
        
        return {
            "batch": batch,
            "state_before_norm": interaction_data["state_before_norm"],
            "robot_type": robot_type,
            "raw_state": raw_state,
        }
    
    @torch.no_grad()
    def __call__(
        self,
        images: Dict[str, Union[str, Image.Image, np.ndarray, torch.Tensor]],
        raw_state: List[float],
        task: str,
        robot_type: str,
        return_all_steps: bool = True,
        noise: Optional[torch.Tensor] = None,
    ) -> Union[List[List[float]], List[float]]:
        """
        Execute inference to predict robot actions.
        
        Args:
            images: Dictionary mapping camera names to images
            raw_state: Raw robot state vector
            task: Task description text
            robot_type: Robot type ("ARX5", "UR5", "Franka", "aloha")
            return_all_steps: If True, return all action steps; else return first only
            noise: Optional noise tensor for diffusion sampling
        
        Returns:
            List of absolute target actions
        """
        # Process inputs
        processed = self.process(images, raw_state, task, robot_type)
        
        # Configure autocast context
        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.use_bf16 else nullcontext()
        )
        
        # Run inference
        with torch.inference_mode(), autocast_ctx:
            action_tensor = self.synthesis.predict(
                batch=processed["batch"],
                noise=noise,
            ).cpu()
        
        # Postprocess actions
        action_list = self.operator.process_output(
            action_np=action_tensor.squeeze(0).numpy(),
            state_np=processed["state_before_norm"].numpy(),
            robot_type=robot_type,
            used_chunk_size=self.used_chunk_size,
        )
        
        # Record to memory
        self.memory.record(
            data={
                "images": images,
                "state": raw_state,
                "task": task,
                "robot_type": robot_type,
                "actions": action_list,
            },
            metadata={"type": "inference_result"},
        )
        
        if return_all_steps:
            return action_list
        else:
            return action_list[0] if action_list else []
    
    def stream(
        self,
        images: Dict[str, Union[str, Image.Image, np.ndarray, torch.Tensor]],
        raw_state: List[float],
        task: str,
        robot_type: str,
        update_interval: int = 1,
    ) -> Generator[List[float], None, None]:
        """
        Stream action predictions for multi-round interactive inference.
        
        This generator yields individual actions one at a time, supporting
        real-time control loops. Memory management is handled internally.
        
        Args:
            images: Dictionary mapping camera names to images
            raw_state: Initial raw robot state vector
            task: Task description text
            robot_type: Robot type identifier
            update_interval: Steps between yielded actions
        
        Yields:
            Individual action vectors (absolute target positions)
        """
        # Get full action sequence
        action_list = self(
            images=images,
            raw_state=raw_state,
            task=task,
            robot_type=robot_type,
            return_all_steps=True,
        )
        
        # Stream actions with memory management
        for idx, action in enumerate(action_list):
            if idx % update_interval == 0:
                # Record streamed action to memory
                self.memory.record(
                    data={"action": action, "step": idx},
                    metadata={"type": "streamed_action"},
                )
                
                # Manage memory if needed
                self.memory.manage()
                
                yield action
    
    def reset_memory(self):
        """Clear all stored memory."""
        self.memory = SpiritV1p5Memory(capacity=self.memory.capacity)
    
    def get_memory_context(self, query: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant memory entries."""
        return self.memory.select(context_query=query)
