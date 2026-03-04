import os
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin

from .controlnet_cosmos2_5 import Cosmos25ControlNet3DModel


class Cosmos25MultiControlNet3DModel(ModelMixin):
    """A model that manages and applies multiple ControlNets simultaneously.

    This class acts as a container for several `Cosmos25ControlNet3DModel` instances. It allows for
    applying different types of control (e.g., depth, edges, segmentation) in a single forward pass
    by iterating through the registered ControlNets, gathering their outputs, and aggregating them.

    Args:
        controlnets (`Dict[str, Cosmos25ControlNet3DModel]`):
            A dictionary where keys are unique names for the ControlNets and values are the
            `Cosmos25ControlNet3DModel` instances.
    """

    def __init__(self, controlnets: Dict[str, Cosmos25ControlNet3DModel]):
        super().__init__()
        self.nets = nn.ModuleDict(controlnets)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_cond: Dict[str, torch.Tensor],
        control_scale: Dict[str, float] = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        condition_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """The forward pass for the MultiControlNet. It iterates through each
        contained ControlNet.

        Args:
            hidden_states (`torch.Tensor`): Input latents from the base model.
            timestep (`torch.Tensor`): The current timestep.
            encoder_hidden_states (`torch.Tensor`): Text embeddings.
            control_cond (`Dict[str, torch.Tensor]`): A dictionary of control conditions, keyed by ControlNet name.
            control_scale (`Dict[str, float]`, defaults to `1.0`): A dictionary of scales for each ControlNet.
            attention_mask (`torch.Tensor`, *optional*): An attention mask.
            fps (`int`, *optional*): Frames per second for positional embeddings.
            condition_mask (`torch.Tensor`, *optional*): Mask for conditional frames (image-to-video).
            padding_mask (`torch.Tensor`, *optional*): Mask for padded areas.

        Returns:
            `Dict[str, torch.Tensor]`: A dictionary containing the aggregated control signals for each block ID.
        """
        assert len(control_cond) == len(control_scale), 'Mismatch between number of control conditions and scales.'

        outputs = dict()
        # Iterate over each registered ControlNet
        for key in control_cond:
            # Get the output from the individual ControlNet
            outputs_i = self.nets[key](
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                control_cond=control_cond[key],
                control_scale=control_scale[key],
                attention_mask=attention_mask,
                fps=fps,
                condition_mask=condition_mask,
                padding_mask=padding_mask,
            )
            # Aggregate the outputs from all ControlNets
            for name, val in outputs_i.items():
                if name not in outputs:
                    outputs[name] = val
                else:
                    outputs[name] += val
        return outputs

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """Saves each contained ControlNet to a subdirectory within the
        specified directory."""
        for name, controlnet in self.nets.items():
            controlnet.save_pretrained(os.path.join(save_directory, name), **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Optional[Union[str, os.PathLike]], **kwargs):
        """Loads multiple ControlNets from subdirectories of a given path."""
        names = os.listdir(pretrained_model_path)
        controlnets = dict()
        # Load each ControlNet from its respective subdirectory
        for name in names:
            controlnet = Cosmos25ControlNet3DModel.from_pretrained(os.path.join(pretrained_model_path, name), **kwargs)
            controlnets[name] = controlnet
        if len(controlnets) == 0:
            raise ValueError(f'No ControlNets found under {os.path.dirname(pretrained_model_path)}. ' f'Expected at least {pretrained_model_path}.')
        return cls(controlnets)
