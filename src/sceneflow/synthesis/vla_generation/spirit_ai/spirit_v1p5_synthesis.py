import os
import json
import torch
from typing import Any, Dict, Optional, Union
from dataclasses import fields
from huggingface_hub import snapshot_download, hf_hub_download

from ...base_synthesis import BaseSynthesis        
from .spirit_v1p5.modeling_spirit_vla import SpiritVLAPolicy


class SpiritV1p5Synthesis(BaseSynthesis):
    """
    Synthesis wrapper for Spirit-VLA policy model.
    
    Handles:
    - Model loading and initialization
    - Inference with diffusion transformer
    - Action generation via flow matching
    """
    
    def __init__(
        self,
        policy: "SpiritVLAPolicy",
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Initialize Spirit-VLA Synthesis.
        
        Args:
            policy: Initialized SpiritVLAPolicy model
            device: Device for inference
        """
        super().__init__()
        self.device = device
        self.policy = policy.to(device)
        self.policy.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
        **kwargs,
    ):
        """
        Load pretrained Spirit-VLA model.
        
        Args:
            pretrained_model_path: Path to model checkpoint
            device: Device for inference
            strict: Whether to enforce strict weight loading
            **kwargs: Additional arguments
        
        Returns:
            Initialized SpiritV1p5Synthesis
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load policy
        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            # download from HuggingFace repo_id
            print(f"Downloading weights from HuggingFace repo: {pretrained_model_path}")
            model_root = snapshot_download(pretrained_model_path)
            print(f"Model downloaded to: {model_root}")
        policy = SpiritVLAPolicy.from_pretrained(model_root, strict=strict)
        
        return cls(policy=policy, device=device)
    
    def to(self, device: Union[str, torch.device]):
        """Move synthesis to specified device."""
        self.device = device
        self.policy.to(device)
        return self
    
    def api_init(self, api_key: str, endpoint: str):
        """
        Initialize API for online model calls.
        
        Note: Spirit-VLA runs locally, so this is a no-op.
        """
        pass
    
    @torch.no_grad()
    def predict(
        self,
        batch: Dict[str, Any],
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate action predictions from observations.
        
        Args:
            batch: Dictionary containing:
                - observation.images.cam_high: Main camera image
                - observation.images.cam_left_wrist: Left wrist camera
                - observation.images.cam_right_wrist: Right wrist camera
                - observation.state: Robot state tensor
                - task: List of task descriptions
                - robot_type: List of robot type identifiers
            noise: Optional noise tensor for diffusion sampling
        
        Returns:
            Action tensor of shape (B, T, action_dim)
        """
        return self.policy.select_action(batch, noise=noise)
    
    @property
    def config(self):
        """Get policy configuration."""
        return self.policy.config
    
    @property
    def chunk_size(self) -> int:
        """Get action chunk size."""
        return self.policy.config.chunk_size
    
    @property
    def n_action_steps(self) -> int:
        """Get number of action steps."""
        return self.policy.config.n_action_steps
    
    @property
    def max_action_dim(self) -> int:
        """Get maximum action dimension."""
        return self.policy.config.max_action_dim
    
    def compile(self, **kwargs):
        """
        Compile model for faster inference.
        
        Args:
            **kwargs: Arguments passed to torch.compile
        
        Returns:
            Self for chaining
        """
        self.policy._sample_actions_unified = torch.compile(
            self.policy._sample_actions_unified,
            **kwargs,
        )
        return self
    
    def quantize(self, quantization_type: str = "dynamic"):
        """
        Apply quantization for faster inference.
        
        Args:
            quantization_type: Type of quantization ("dynamic", "static")
        
        Returns:
            Self for chaining
        """
        if quantization_type == "dynamic":
            self.policy = torch.quantization.quantize_dynamic(
                self.policy,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
        return self
    
    @property
    def inner_policy(self):
        """Get the underlying policy model."""
        return self.policy
