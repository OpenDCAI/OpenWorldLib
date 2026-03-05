import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .base_operator import BaseOperator


# ============================================================
# Normalize / Unnormalize
# ============================================================

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
            stats_dim = self.q01.shape[-1]
            d = min(x_dim, stats_dim)
            out = x.clone()
            out[..., :d] = (x[..., :d] - self.q01[..., :d]) / (self.q99[..., :d] - self.q01[..., :d] + self.EPSILON) * 2.0 - 1.0
            return out
        else:
            stats_dim = self.mean.shape[-1]
            d = min(x_dim, stats_dim)
            out = x.clone()
            out[..., :d] = (x[..., :d] - self.mean[..., :d]) / (self.std[..., :d] + self.EPSILON)
            return out


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
            stats_dim = self.q01.shape[-1]
            d = min(x_dim, stats_dim)
            out = x.clone()
            out[..., :d] = (x[..., :d] + 1.0) / 2.0 * (self.q99[..., :d] - self.q01[..., :d] + self.EPSILON) + self.q01[..., :d]
            return out
        else:
            stats_dim = self.mean.shape[-1]
            d = min(x_dim, stats_dim)
            out = x.clone()
            out[..., :d] = x[..., :d] * (self.std[..., :d] + self.EPSILON) + self.mean[..., :d]
            return out


# ============================================================
# Delta / Absolute action transforms (mask-configurable)
# ============================================================

class DeltaActions:
    """Converts absolute actions to delta actions relative to the current state.

    Args:
        mask: Boolean mask indicating which dims to convert. None means no-op.
              Length can be shorter than action dim; unmasked dims are unchanged.
    """

    def __init__(self, mask: tuple[bool, ...] | list[bool] | None = None):
        self.mask = torch.tensor(mask, dtype=torch.bool) if mask is not None else None

    def to(self, device: torch.device | str) -> None:
        if self.mask is not None:
            self.mask = self.mask.to(device)

    def __call__(self, data: dict) -> dict:
        if self.mask is None or 'action' not in data or 'observation.state' not in data:
            return data
        state, action = data['observation.state'], data['action']
        dims = self.mask.shape[-1]
        action[..., :dims] -= torch.where(self.mask, state[..., :dims], torch.zeros_like(state[..., :dims])).unsqueeze(-2)
        data['action'] = action
        return data


class AbsoluteActions:
    """Converts delta actions back to absolute actions by adding current state.

    Args:
        mask: Boolean mask indicating which dims to convert. None means no-op.
    """

    def __init__(self, mask: tuple[bool, ...] | list[bool] | None = None):
        if mask is None:
            # Default: Aloha dual-arm (backward compatible)
            mask = (True, True, True, True, True, True, False,
                    True, True, True, True, True, True, False)
        self.mask = torch.tensor(mask, dtype=torch.bool)

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


# ============================================================
# Robot-specific input transforms
# ============================================================

class AlohaInputs:
    """Inputs for the Aloha policy - converts Aloha state format to pi0 format.

    All methods expect single-sample tensors (no batch dimension):
      - state: (state_dim,)  1D
      - actions: (n_steps, action_dim)  2D
    """

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

    def _decode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Decode single-sample state (state_dim,) from Aloha to pi0 format."""
        if self.adapt_to_pi:
            state_dim = state.shape[-1]
            if state_dim >= 14:
                # Dual-arm: flip joints and convert gripper
                state[:14] = self.joint_flip_mask * state[:14]
                state[[6, 13]] = self._gripper_to_angular(state[[6, 13]])
            else:
                # Single-arm: only first 7 dims
                state[:7] = self.joint_flip_mask[:7] * state[:7]
                state[6] = self._gripper_to_angular(state[6:7])[0]
        return state

    def _encode_actions_inv(self, actions: torch.Tensor) -> torch.Tensor:
        """Inverse-encode actions (n_steps, action_dim) from Aloha to pi0 format."""
        if self.adapt_to_pi:
            action_dim = actions.shape[-1]
            if action_dim >= 14:
                actions[:, :14] = self.joint_flip_mask * actions[:, :14]
                actions[:, [6, 13]] = self._gripper_from_angular_inv(actions[:, [6, 13]])
            else:
                actions[:, :7] = self.joint_flip_mask[:7] * actions[:, :7]
                actions[:, 6] = self._gripper_from_angular_inv(actions[:, 6:7])[:, 0]
        return actions

    def __call__(self, data: dict) -> dict:
        """Decode Aloha-specific input formats into the pi0 training/runtime format.
        
        Expects single-sample data:
          - data['observation.state']: (state_dim,)
          - data['action']: (n_steps, action_dim) (optional, training only)
        """
        data['observation.state'] = self._decode_state(data['observation.state'])
        if 'action' in data:
            data['action'] = self._encode_actions_inv(data['action'])
        return data


class LiberoInputs:
    """Inputs for the Libero policy.

    Libero data does not need joint-flip or gripper-angular conversion.
    This transform only performs image key remapping to the model's canonical names.

    Expects single-sample data:
      - data['observation.state']: (state_dim,)   — passed through unchanged
      - data['action']: (n_steps, action_dim)      — passed through unchanged (optional)
    """

    def __init__(self) -> None:
        pass

    def to(self, device: torch.device | str) -> None:
        pass

    def __call__(self, data: dict) -> dict:
        # Libero state/action pass through without any robot-specific transform.
        return data


class DroidInputs:
    """Inputs for the DROID policy.

    DROID stores joint positions and gripper position separately.
    This transform concatenates them into a single state vector.

    Expects single-sample data:
      - data['observation.state']: (state_dim,)  — already concatenated joint+gripper
      - data['action']: (n_steps, action_dim)     — passed through unchanged (optional)
    """

    def __init__(self) -> None:
        pass

    def to(self, device: torch.device | str) -> None:
        pass

    def __call__(self, data: dict) -> dict:
        # If joint_position and gripper_position are provided separately, concatenate them.
        if 'observation.joint_position' in data and 'observation.gripper_position' in data:
            joint = data['observation.joint_position']
            gripper = data['observation.gripper_position']
            if gripper.ndim == 0:
                gripper = gripper.unsqueeze(0)
            data['observation.state'] = torch.cat([joint, gripper], dim=-1)
        return data


# ============================================================
# Robot-specific output transforms
# ============================================================

class AlohaOutputs:
    """Outputs for the Aloha policy - converts pi0 output to Aloha format.

    Expects single-sample tensors:
      - actions: (n_steps, action_dim)  2D
    """

    def __init__(self, original_action_dim: int, adapt_to_pi: bool = True):
        self.joint_flip_mask = torch.tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
        self.original_action_dim = original_action_dim
        self.adapt_to_pi = adapt_to_pi

    def to(self, device: torch.device | str) -> None:
        self.joint_flip_mask = self.joint_flip_mask.to(device)

    def _gripper_from_angular(self, value: torch.Tensor) -> torch.Tensor:
        value = value + 0.5476
        return _normalize(value, min_val=-0.6213, max_val=1.4910)

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions (n_steps, action_dim) from pi0 to Aloha format."""
        if self.adapt_to_pi:
            action_dim = actions.shape[-1]
            if action_dim >= 14:
                actions[:, :14] = self.joint_flip_mask * actions[:, :14]
                actions[:, [6, 13]] = self._gripper_from_angular(actions[:, [6, 13]])
            else:
                actions[:, :7] = self.joint_flip_mask[:7] * actions[:, :7]
                actions[:, 6] = self._gripper_from_angular(actions[:, 6:7])[:, 0]
        return actions

    def __call__(self, data: dict) -> dict:
        actions = data['action'][:, : self.original_action_dim]
        return {'action': self._encode_actions(actions)}


class LiberoOutputs:
    """Outputs for the Libero policy.

    Simply truncates actions to the original action dimension (typically 7).
    No robot-specific conversion needed.
    """

    def __init__(self, original_action_dim: int = 7):
        self.original_action_dim = original_action_dim

    def to(self, device: torch.device | str) -> None:
        pass

    def __call__(self, data: dict) -> dict:
        actions = data['action'][:, :self.original_action_dim]
        return {'action': actions}


class DroidOutputs:
    """Outputs for the DROID policy.

    Simply truncates actions to the original action dimension (typically 8).
    No robot-specific conversion needed.
    """

    def __init__(self, original_action_dim: int = 8):
        self.original_action_dim = original_action_dim

    def to(self, device: torch.device | str) -> None:
        pass

    def __call__(self, data: dict) -> dict:
        actions = data['action'][:, :self.original_action_dim]
        return {'action': actions}


# ============================================================
# Shared transforms (unchanged)
# ============================================================

class PadStatesAndActions:
    """Zero-pads states and actions to the model action dimension."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def _pad_to_dim(self, x: torch.Tensor, target_dim: int, axis: int = -1) -> torch.Tensor:
        current_dim = x.shape[axis]
        if (current_dim < target_dim):
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


# ============================================================
# Helper: boolean mask builder
# ============================================================

def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Build a boolean mask from signed dimension counts.

    Positive → True, negative → False.
    Example: make_bool_mask(6, -1, 6, -1) == (T,T,T,T,T,T, F, T,T,T,T,T,T, F)
    """
    result: list[bool] = []
    for d in dims:
        if d > 0:
            result.extend([True] * d)
        else:
            result.extend([False] * (-d))
    return tuple(result)


# ============================================================
# PI0Operator — supports aloha / libero / droid
# ============================================================

class PI0Operator(BaseOperator):
    """Operator for PI0 policy inference - handles preprocessing and postprocessing.

    Args:
        robot_type: One of 'aloha', 'libero', 'droid'. Controls which robot-specific
                    input/output transforms and delta-action masks are used.
        state_norm_stats: Normalization stats for robot state.
        action_norm_stats: Normalization stats for actions.
        tokenizer_model_path: Path or hub id for the tokenizer.
        resize_imgs_with_padding: Target (width, height) for images.
        discrete_state_input: If True, enables PI0.5 mode.
        present_img_keys: List of image dict keys expected from the environment.
        original_action_dim: Native action dimension before padding.
        use_delta_actions: Whether the model expects delta actions. If True, an
                           AbsoluteActions transform is appended to outputs.
    """

    # Supported robot types
    SUPPORTED_ROBOTS = ('aloha', 'libero', 'droid')

    def __init__(
        self,
        state_norm_stats: dict,
        action_norm_stats: dict,
        tokenizer_model_path: str,
        robot_type: str = 'aloha',
        resize_imgs_with_padding: tuple[int, int] = (224, 224),
        discrete_state_input: bool = False,
        present_img_keys: list[str] | None = None,
        original_action_dim: int = 14,
        use_delta_actions: bool = True,
    ):
        super().__init__(operation_types=[])

        if robot_type not in self.SUPPORTED_ROBOTS:
            raise ValueError(f"robot_type must be one of {self.SUPPORTED_ROBOTS}, got '{robot_type}'")

        # Core attributes
        self.device = 'cpu'
        self.robot_type = robot_type
        self.state_norm_stats = state_norm_stats
        self.action_norm_stats = action_norm_stats
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.discrete_state_input = discrete_state_input
        self.pi05_enabled = discrete_state_input
        self.original_action_dim = original_action_dim
        self.use_delta_actions = use_delta_actions

        # ---- Build robot-specific input/output transforms ----
        self.robot_inputs_transform, self.robot_outputs_transform, self.absolute_actions_transform = \
            self._build_robot_transforms(robot_type, original_action_dim, use_delta_actions)

        # ---- Shared transforms ----
        self.state_normalize_transform = Normalize(state_norm_stats, use_quantiles=self.pi05_enabled)
        self.pad_states_and_actions_transform = PadStatesAndActions(action_dim=32)
        self.image_transform = ImageTransform(
            resize_imgs_with_padding=resize_imgs_with_padding,
            present_img_keys=present_img_keys,
            enable_image_aug=False,
        )
        max_length = 200 if self.pi05_enabled else 48
        self.prompt_tokenizer_transform = PromptTokenizerTransform(
            tokenizer_model_path=tokenizer_model_path,
            max_length=max_length,
            discrete_state_input=discrete_state_input,
        )

        # Output transforms
        self.state_unnormalize_transform = Unnormalize(state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_unnormalize_transform = Unnormalize(action_norm_stats, use_quantiles=self.pi05_enabled)

    # ------------------------------------------------------------------
    # Robot transform factory
    # ------------------------------------------------------------------
    @staticmethod
    def _build_robot_transforms(
        robot_type: str,
        original_action_dim: int,
        use_delta_actions: bool,
    ) -> tuple:
        """Return (robot_inputs, robot_outputs, absolute_actions_or_None)."""

        if robot_type == 'aloha':
            robot_inputs = AlohaInputs(adapt_to_pi=True)
            robot_outputs = AlohaOutputs(original_action_dim=original_action_dim, adapt_to_pi=True)
            # Aloha: 6 joint delta + 1 gripper absolute, ×2 arms
            abs_mask = make_bool_mask(6, -1, 6, -1) if use_delta_actions else None
            absolute_actions = AbsoluteActions(mask=abs_mask) if abs_mask else None

        elif robot_type == 'libero':
            robot_inputs = LiberoInputs()
            robot_outputs = LiberoOutputs(original_action_dim=original_action_dim)
            # Libero: 6 joint delta + 1 gripper absolute
            abs_mask = make_bool_mask(6, -1) if use_delta_actions else None
            absolute_actions = AbsoluteActions(mask=abs_mask) if abs_mask else None

        elif robot_type == 'droid':
            robot_inputs = DroidInputs()
            robot_outputs = DroidOutputs(original_action_dim=original_action_dim)
            # DROID: 7 joint delta + 1 gripper absolute
            abs_mask = make_bool_mask(7, -1) if use_delta_actions else None
            absolute_actions = AbsoluteActions(mask=abs_mask) if abs_mask else None

        else:
            raise ValueError(f"Unknown robot_type: {robot_type}")

        return robot_inputs, robot_outputs, absolute_actions

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------
    def to(self, device: str | torch.device):
        self.device = device
        self.robot_inputs_transform.to(device)
        self.robot_outputs_transform.to(device)
        self.state_normalize_transform.to(device)
        self.state_unnormalize_transform.to(device)
        self.action_unnormalize_transform.to(device)
        if self.absolute_actions_transform is not None:
            self.absolute_actions_transform.to(device)
        return self

    # ------------------------------------------------------------------
    # Input processing
    # ------------------------------------------------------------------
    def process_perception(self, images: dict[str, torch.Tensor], state: torch.Tensor, pad_state: bool = True):
        """Process images and state for model input.

        Data flow:
          1. Robot-specific input transform (e.g. Aloha joint flip)
          2. State normalization
          3. Image preprocessing
          4. State padding to model dim
        """
        images = {k: v.to(self.device) for k, v in images.items()}
        state = state.to(self.device)

        # 1. Robot-specific input transform
        state = self.robot_inputs_transform({'observation.state': state})['observation.state']

        # 2. Normalize state
        state = self.state_normalize_transform(state)

        # 3. Process images
        images, img_masks = self.image_transform(images)

        # 4. Pad state
        if pad_state:
            state = self.pad_states_and_actions_transform({'observation.state': state})['observation.state']

        return images, img_masks, state

    def process_interaction(self, task: str, state: torch.Tensor):
        """Process task description and state for tokenization."""
        lang_tokens, lang_masks = self.prompt_tokenizer_transform({'task': task, 'observation.state': state})
        return lang_tokens, lang_masks

    # ------------------------------------------------------------------
    # Output processing
    # ------------------------------------------------------------------
    def process_output(self, pred_action: torch.Tensor, state: torch.Tensor, original_action_dim: int, **kwargs):
        """Process model output to final action.

        Data flow:
          1. Unnormalize state & action
          2. Delta → absolute action conversion (if applicable)
          3. Robot-specific output transform (e.g. Aloha joint unflip, dim truncation)
        """
        output_dict = {'action': pred_action, 'observation.state': state}

        # 1. Unnormalize
        output_dict['observation.state'] = self.state_unnormalize_transform(output_dict['observation.state'])
        output_dict['action'] = self.action_unnormalize_transform(output_dict['action'])

        # 2. Delta → absolute
        if self.absolute_actions_transform is not None:
            output_dict = self.absolute_actions_transform(output_dict)

        # 3. Robot-specific output transform
        pred_action = self.robot_outputs_transform(output_dict)['action']

        return pred_action

    # ------------------------------------------------------------------
    # Base class methods
    # ------------------------------------------------------------------

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


