import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .base_operator import BaseOperator


class Normalize:
    """Normalize robot state vectors using mean/std or quantiles."""

    def __init__(self, stats: dict, *, use_quantiles: bool = False) -> None:
        self.EPSILON = 1e-6
        self.stats = stats
        self.use_quantiles = use_quantiles

        required_attrs = ['mean', 'std']
        if self.use_quantiles:
            required_attrs = ['q01', 'q99']

        for attr in required_attrs:
            if attr not in stats:
                raise AttributeError(f'stats object is missing the following attribute: {attr}')

        if self.use_quantiles:
            self.q01 = torch.tensor(stats['q01'], dtype=torch.float32)
            self.q99 = torch.tensor(stats['q99'], dtype=torch.float32)
        else:
            self.mean = torch.tensor(stats['mean'], dtype=torch.float32)
            self.std = torch.tensor(stats['std'], dtype=torch.float32)

    def to(self, device: torch.device | str) -> None:
        if self.use_quantiles:
            self.q01 = self.q01.to(device)
            self.q99 = self.q99.to(device)
        else:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_dim = x.shape[-1]
        if self.use_quantiles:
            return (x - self.q01[..., :x_dim]) / (self.q99[..., :x_dim] - self.q01[..., :x_dim] + self.EPSILON) * 2.0 - 1.0
        else:
            return (x - self.mean[..., :x_dim]) / (self.std[..., :x_dim] + self.EPSILON)


class Unnormalize:
    def __init__(self, stats: dict, *, use_quantiles: bool = False):
        self.EPSILON = 1e-6
        self.stats = stats
        self.use_quantiles = use_quantiles

        if self.use_quantiles:
            self.q01 = torch.tensor(stats['q01'], dtype=torch.float32)
            self.q99 = torch.tensor(stats['q99'], dtype=torch.float32)
        else:
            self.mean = torch.tensor(stats['mean'], dtype=torch.float32)
            self.std = torch.tensor(stats['std'], dtype=torch.float32)

    def to(self, device: torch.device | str) -> None:
        if self.use_quantiles:
            self.q01 = self.q01.to(device)
            self.q99 = self.q99.to(device)
        else:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_dim = x.shape[-1]
        if self.use_quantiles:
            return (x + 1.0) / 2.0 * (self.q99[..., :x_dim] - self.q01[..., :x_dim] + self.EPSILON) + self.q01[..., :x_dim]
        else:
            return x * (self.std[..., :x_dim] + self.EPSILON) + self.mean[..., :x_dim]


class AbsoluteActions:
    """Repacks delta actions into absolute action space."""

    def __init__(self):
        # If the robot has mobile base, masks of base action are False and it doesn't need to be specified explicitly.
        self.mask = torch.tensor([True, True, True, True, True, True, False, True, True, True, True, True, True, False])

    def to(self, device: torch.device | str) -> None:
        self.mask = self.mask.to(device)

    def __call__(self, data: dict) -> dict:
        if 'action' not in data or 'observation.state' not in data:
            return data
        state, action = data['observation.state'], data['action']
        dims = self.mask.shape[-1]
        action[..., :dims] += torch.where(self.mask, state[..., :dims], torch.zeros_like(state[..., :dims])).unsqueeze(-2)
        data['action'] = action
        return data


class AlohaInputs:
    """Inputs for the Aloha policy - converts Aloha state format to pi0 format."""

    def __init__(self, adapt_to_pi: bool = True) -> None:
        self.joint_flip_mask = torch.tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
        self.adapt_to_pi = adapt_to_pi

    def to(self, device: torch.device | str) -> None:
        self.joint_flip_mask = self.joint_flip_mask.to(device)

    def _gripper_from_angular_inv(self, value: torch.Tensor) -> torch.Tensor:
        # Directly inverts the gripper_from_angular function.
        value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
        return value - 0.5476

    def _gripper_to_angular(self, value: torch.Tensor) -> torch.Tensor:
        # Aloha transforms the gripper positions into a linear space. The following code
        # reverses this transformation to be consistent with pi0 which is pretrained in
        # angular space.
        value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

        def linear_to_radian(linear_position, arm_length, horn_radius):
            value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
            return torch.arcsin(torch.clip(value, -1.0, 1.0))

        value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
        return _normalize(value, min_val=0.5476, max_val=1.6296)

    def _decode_aloha(self, state: torch.Tensor) -> torch.Tensor:
        if self.adapt_to_pi:
            # Flip the joints.
            state[:14] = self.joint_flip_mask * state[:14]
            # Reverse the gripper transformation
            state[[6, 13]] = self._gripper_to_angular(state[[6, 13]])
        return state

    def __call__(self, data: dict) -> dict:
        """Decode Aloha-specific input formats into the pi0 training/runtime format."""
        state = self._decode_aloha(data['observation.state'])
        data['observation.state'] = state
        if 'action' in data:
            actions = data['action']
            actions[:, :14] = self.joint_flip_mask * actions[:, :14]
            actions[:, [6, 13]] = self._gripper_from_angular_inv(actions[:, [6, 13]])
            data['action'] = actions
        return data


class AlohaOutputs:
    """Outputs for the Aloha policy - converts pi0 output to Aloha format."""

    def __init__(self, original_action_dim: int, adapt_to_pi: bool = True):
        self.joint_flip_mask = torch.tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
        self.original_action_dim = original_action_dim
        self.adapt_to_pi = adapt_to_pi

    def to(self, device: torch.device | str) -> None:
        self.joint_flip_mask = self.joint_flip_mask.to(device)

    def _gripper_from_angular(self, value: torch.Tensor) -> torch.Tensor:
        value = value + 0.5476
        return _normalize(value, min_val=-0.6213, max_val=1.4910)

    def __call__(self, data: dict) -> dict:
        actions = data['action'][:, : self.original_action_dim]
        if self.adapt_to_pi:
            actions[:, :14] = self.joint_flip_mask * actions[:, :14]
            actions[:, [6, 13]] = self._gripper_from_angular(actions[:, [6, 13]])
        return {'action': actions}


class PadStatesAndActions:
    """Zero-pads states and actions to the model action dimension."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def _pad_to_dim(self, x: torch.Tensor, target_dim: int, axis: int = -1) -> torch.Tensor:
        current_dim = x.shape[axis]
        if current_dim < target_dim:
            shape = list(x.shape)
            shape[-1] = target_dim
            new_vector = torch.zeros(*shape, dtype=x.dtype, device=x.device)
            new_vector[..., :current_dim] = x
            x = new_vector
        return x

    def __call__(self, data: dict) -> dict:
        data['observation.state'] = self._pad_to_dim(data['observation.state'], self.action_dim, axis=-1)
        if 'action' in data:
            data['action'] = self._pad_to_dim(data['action'], self.action_dim, axis=-1)
        return data


def _normalize(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return x * (max_val - min_val) + min_val


def resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: float = -1.0) -> torch.Tensor:
    """Resize an image to fit inside the given (width, height) while preserving
    aspect ratio, then pad with the specified value so that the final image
    exactly matches the target size."""
    if img.ndim != 3:
        raise ValueError(f'(C,H,W) expected, but got {img.shape}')

    cur_height, cur_width = img.shape[1:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(img.unsqueeze(0), size=(resized_height, resized_width), mode='bilinear', align_corners=False).squeeze(0)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded_img = F.pad(resized_img, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
    return padded_img.squeeze(0)


class ImageTransform:
    def __init__(self, resize_imgs_with_padding: tuple[int, int], present_img_keys: list[str] | None = None, enable_image_aug: bool = False) -> None:
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.present_img_keys = present_img_keys
        if self.present_img_keys is None:
            self.present_img_keys = [
                'observation.images.cam_high',
                'observation.images.cam_left_wrist',
                'observation.images.cam_right_wrist',
            ]
        self.enable_image_aug = enable_image_aug
        self.width, self.height = resize_imgs_with_padding

    def __call__(self, data: dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Preprocesses input images: optionally scales and pads to a fixed size,
        then maps the pixel range from [0,1] to [-1,1]."""
        images = []
        img_masks = []

        for key in self.present_img_keys:
            if key not in data:
                raise ValueError(f'{key} not found in data. Please check the present_img_keys in the config or the dataset.')

            img = data[key]
            if self.resize_imgs_with_padding is not None:
                original_height, original_width = img.shape[1:]
                target_height, target_width = self.resize_imgs_with_padding
                if original_height != target_height or original_width != target_width:
                    img = resize_with_pad(img, *self.resize_imgs_with_padding, pad_value=0)

            # Normalize pixel values to [-1, 1]
            img = img * 2.0 - 1.0

            images.append(img)
            img_masks.append(torch.tensor(True, dtype=torch.bool, device=img.device))

        return images, img_masks


class PromptTokenizerTransform:
    def __init__(self, tokenizer_model_path: str, max_length: int, discrete_state_input: bool = False) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        self.tokenizer_max_length = max_length
        self.discrete_state_input = discrete_state_input

    def __call__(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the text input."""
        task = data['task'].strip().replace('_', ' ').replace('\n', ' ')
        device = data['observation.state'].device if 'observation.state' in data else torch.device('cpu')

        if self.discrete_state_input:
            assert 'observation.state' in data, 'discrete_state_input is True, but observation.state is not found.'
            discretized_state = torch.bucketize(data['observation.state'], torch.linspace(-1, 1, 256 + 1, device=device)[:-1]) - 1
            state_values = ' '.join([str(int(x)) for x in discretized_state.tolist()])
            task = f'Task: {task}, State: {state_values};\nAction: '
        else:
            # PaliGemma prompt has to end with a new line in Pi0
            task = f'{task}\n'

        tokenized_prompt = self.tokenizer(
            task,
            padding='max_length',
            padding_side='right',
            max_length=self.tokenizer_max_length,
            return_tensors='pt',
        )
        lang_tokens = tokenized_prompt['input_ids'][0].to(dtype=torch.int32, device=device)
        lang_masks = tokenized_prompt['attention_mask'][0].to(dtype=torch.bool, device=device)

        return lang_tokens, lang_masks


class PI0Operator(BaseOperator):
    """Operator for PI0 policy inference - handles preprocessing and postprocessing."""

    def __init__(
        self,
        state_norm_stats: dict,
        action_norm_stats: dict,
        tokenizer_model_path: str,
        resize_imgs_with_padding: tuple[int, int] = (224, 224),
        discrete_state_input: bool = False,
        present_img_keys: list[str] | None = None,
    ):
        # Base class initialization
        super().__init__(operation_types=[])
        
        # PI0 specific attributes
        self.device = 'cpu'
        self.state_norm_stats = state_norm_stats
        self.action_norm_stats = action_norm_stats
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.discrete_state_input = discrete_state_input
        self.pi05_enabled = discrete_state_input  # pi05 uses discrete state input

        # Input transforms
        self.aloha_inputs_transform = AlohaInputs()
        self.state_normalize_transform = Normalize(state_norm_stats, use_quantiles=self.pi05_enabled)
        self.pad_states_and_actions_transform = PadStatesAndActions(action_dim=32)
        self.image_transform = ImageTransform(
            resize_imgs_with_padding=resize_imgs_with_padding,
            present_img_keys=present_img_keys,
            enable_image_aug=False,
        )
        max_length = 200 if self.pi05_enabled else 48
        self.prompt_tokenizer_transform = PromptTokenizerTransform(
            tokenizer_model_path=tokenizer_model_path, max_length=max_length, discrete_state_input=discrete_state_input
        )

        # Output transforms
        self.state_unnormalize_transform = Unnormalize(action_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_unnormalize_transform = Unnormalize(action_norm_stats, use_quantiles=self.pi05_enabled)
        self.absolute_actions_transform = AbsoluteActions()
        self.aloha_outputs_transform = AlohaOutputs(original_action_dim=14)

    def to(self, device: str | torch.device):
        self.device = device
        self.aloha_inputs_transform.to(device)
        self.state_normalize_transform.to(device)
        self.state_unnormalize_transform.to(device)
        self.action_unnormalize_transform.to(device)
        self.absolute_actions_transform.to(device)
        self.aloha_outputs_transform.to(device)
        return self

    def process_perception(self, images: dict[str, torch.Tensor], state: torch.Tensor, pad_state: bool = True):
        """Process images and state for model input."""
        images = {k: v.to(self.device) for k, v in images.items()}
        state = state.to(self.device)

        # Apply Aloha input transform
        state = self.aloha_inputs_transform({'observation.state': state})['observation.state']

        # Normalize state
        state = self.state_normalize_transform(state)

        # Process images
        images, img_masks = self.image_transform(images)

        # Pad state if needed
        if pad_state:
            state = self.pad_states_and_actions_transform({'observation.state': state})['observation.state']

        return images, img_masks, state

    def process_interaction(self, task: str, state: torch.Tensor):
        """Process task description and state for tokenization."""
        lang_tokens, lang_masks = self.prompt_tokenizer_transform({'task': task, 'observation.state': state})
        return lang_tokens, lang_masks

    def process_output(self, pred_action: torch.Tensor, state: torch.Tensor, original_action_dim: int, **kwargs):
        """Process model output to final action."""
        # Update output transform with correct action dim
        self.aloha_outputs_transform = AlohaOutputs(original_action_dim=original_action_dim)

        # Unnormalize
        output_dict = {'action': pred_action, 'observation.state': state}
        output_dict['observation.state'] = self.state_unnormalize_transform(output_dict['observation.state'])
        output_dict['action'] = self.action_unnormalize_transform(output_dict['action'])

        # Convert to absolute actions
        output_dict = self.absolute_actions_transform(output_dict)

        # Apply Aloha output transform
        pred_action = self.aloha_outputs_transform(output_dict)['action']

        return pred_action

    # ====== Base class methods implementation ======
    


    def get_interaction(self, interaction: str | list[str]):
        """Append interaction(s) to the current list after validation."""
        if not isinstance(interaction, list):
            interaction = [interaction]
        for act in interaction:
            self.check_interaction(act)
            self.current_interaction.append(act)

    def check_interaction(self, interaction: str) -> bool:
        """Validate interaction/task; skip checks when no template is provided."""
        if not isinstance(interaction, str):
            raise ValueError('interaction must be a string')
        if self.interaction_template and interaction not in self.interaction_template:
            raise ValueError(f'{interaction} not in interaction_template: {self.interaction_template}')
        return True
    
    

    
    def get_interaction_history(self):
        """Get interaction history."""
        return self.interaction_history
    
    def delete_last_interaction(self):
        """Delete the last interaction from current_interaction."""
        if len(self.current_interaction) > 0:
            self.current_interaction = self.current_interaction[:-1]


