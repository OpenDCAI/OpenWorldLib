from typing import Any

import torch

from ...operators.pi0_operator import PI0Operator
from ...synthesis.vla_generation.pi0.pi0_synthesis import PI0Synthesis


class PI0Pipeline:
    """Pipeline wrapper for PI0 policy inference using a dedicated operator.

    This pipeline supports both PI0 and PI0.5 modes based on the configuration.
    PI0.5 uses discrete state input and quantile-based normalization.
    """

    def __init__(
        self,
        synthesis: PI0Synthesis,
        operator: PI0Operator,
        original_action_dim: int = 14,
        device: str | torch.device | None = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.synthesis = synthesis.to(self.device)
        self.operator = operator.to(self.device)
        self.action_dim = self.synthesis.max_action_dim
        self.original_action_dim = original_action_dim
        self.resize_imgs_with_padding = (224, 224)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer_model_path: str,
        state_norm_stats: dict,
        action_norm_stats: dict,
        original_action_dim: int = 14,
        discrete_state_input: bool = False,
        device: str | torch.device | None = None,
        present_img_keys: list[str] | None = None,
        robot_type: str = 'aloha',
        use_delta_actions: bool = True,
        **policy_kwargs: Any,
    ) -> 'PI0Pipeline':
        """Create a PI0Pipeline from pretrained model.

        Args:
            model_path: Path to the pretrained PI0 policy weights.
            tokenizer_model_path: Path or hub id for the tokenizer used in prompt processing.
            state_norm_stats: Normalization stats for robot state; either {'mean','std'} or {'q01','q99'}.
            action_norm_stats: Normalization stats for actions; either {'mean','std'} or {'q01','q99'}.
            original_action_dim: Dimension of the environment's native action space before padding.
            discrete_state_input: If True, enables PI0.5 mode with discrete state input.
            device: Device to run inference on.
            present_img_keys: List of image keys to use.
            robot_type: One of 'aloha', 'libero', 'droid'.
            use_delta_actions: Whether the model uses delta actions (requires absolute conversion on output).

        Returns:
            PI0Pipeline instance.
        """
        synthesis = PI0Synthesis.from_pretrained(model_path, device=device, **policy_kwargs)
        operator = PI0Operator(
            state_norm_stats=state_norm_stats,
            action_norm_stats=action_norm_stats,
            tokenizer_model_path=tokenizer_model_path,
            robot_type=robot_type,
            resize_imgs_with_padding=(224, 224),
            discrete_state_input=discrete_state_input,
            present_img_keys=present_img_keys,
            original_action_dim=original_action_dim,
            use_delta_actions=use_delta_actions,
        )
        return cls(synthesis=synthesis, operator=operator, original_action_dim=original_action_dim, device=device)

    def to(self, device: str | torch.device):
        self.device = device
        self.synthesis.to(device)
        self.operator.to(device)
        return self

    def quantize(self) -> None:
        """Quantize via synthesis wrapper."""
        self.synthesis.quantize()

    def compile(self, **kwargs: Any) -> None:
        """Compile the `sample_actions` method using `torch.compile` for improved runtime speed."""
        self.synthesis.compile(**kwargs)

    def process(
        self,
        images: dict[str, torch.Tensor],
        task: str,
        state: torch.Tensor,
        add_batch_dim: bool = True,
    ):
        """Preprocess inputs (perception + interaction) to build model-ready tensors.
        
        Expects single-sample inputs (no batch dimension):
          - images: dict of (C, H, W) tensors
          - task: str
          - state: (state_dim,) tensor
        """
        ori_device = state.device if state is not None else self.device
        state = state.to(self.device)

        # Process perception (operates on single-sample: state 1D, images 3D)
        images, img_masks, state = self.operator.process_perception(images, state, pad_state=True)

        # Process interaction (operates on single-sample: state 1D)
        lang_tokens, lang_masks = self.operator.process_interaction(task=task, state=state)

        if add_batch_dim:
            images = [img.unsqueeze(0) for img in images]
            img_masks = [mask.unsqueeze(0) for mask in img_masks]
            lang_tokens = lang_tokens.unsqueeze(0)
            lang_masks = lang_masks.unsqueeze(0)
            state = state.unsqueeze(0)

        return {
            'images': images,
            'img_masks': img_masks,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'state': state,
            'ori_device': ori_device,
        }

    @torch.no_grad()
    def __call__(
        self,
        images: dict[str, torch.Tensor],
        task: str,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Run one forward pass from raw inputs to final action sequence.

        Args:
            images: Observation images of the robot. Each value is a tensor with shape (C,H,W).
            task: Natural language task description.
            state: The robot joint state tensor with shape (state_dim,).

        Returns:
            A tensor of predicted actions with shape (num_steps, original_action_dim) on the original input device.
        """
        processed = self.process(images, task, state, add_batch_dim=True)

        outputs = self.synthesis.predict(
            images=processed['images'],
            img_masks=processed['img_masks'],
            lang_tokens=processed['lang_tokens'],
            lang_masks=processed['lang_masks'],
            state=processed['state'],
        )

        # Remove batch dimension for single-sample post-processing
        # outputs: (1, n_steps, max_action_dim) -> (n_steps, max_action_dim)
        # state: (1, state_dim) -> (state_dim,)
        pred_action = self.operator.process_output(
            outputs[0],
            processed['state'][0],
            self.original_action_dim,
        )
        pred_action = pred_action.to(processed['ori_device'])

        return pred_action
