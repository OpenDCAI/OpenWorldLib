import math
import random
from typing import Any

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoProcessor, AutoTokenizer

from .base_operator import BaseOperator


class Normalize:
    """Normalizes a tensor using mean/std or quantile-based scaling."""

    def __init__(self, stats: dict[int, dict[str, list[float]]], *, use_quantiles: bool = False, enable_clamp: bool = False):
        self.EPSILON = 1e-6
        self.use_quantiles = use_quantiles
        self.enable_clamp = enable_clamp

        required_attrs = ['mean', 'std'] if not self.use_quantiles else ['q01', 'q99']
        for attr in required_attrs:
            for key in stats:
                if attr not in stats[key]:
                    raise AttributeError(f'stats object is missing the following attribute: {attr}')

        if self.use_quantiles:
            self.q01 = {int(k): torch.tensor(stats[k]['q01'], dtype=torch.float32) for k in stats}
            self.q99 = {int(k): torch.tensor(stats[k]['q99'], dtype=torch.float32) for k in stats}
        else:
            self.mean = {int(k): torch.tensor(stats[k]['mean'], dtype=torch.float32) for k in stats}
            self.std = {int(k): torch.tensor(stats[k]['std'], dtype=torch.float32) for k in stats}

    def to(self, device: str | torch.device):
        if self.use_quantiles:
            for key in self.q01:
                self.q01[key] = self.q01[key].to(device)
            for key in self.q99:
                self.q99[key] = self.q99[key].to(device)
        else:
            for key in self.mean:
                self.mean[key] = self.mean[key].to(device)
            for key in self.std:
                self.std[key] = self.std[key].to(device)
        return self

    def __call__(self, x: torch.Tensor, embodiment_id: int = 0) -> torch.Tensor:
        x_dim = x.shape[-1]
        if self.use_quantiles:
            x = (x - self.q01[embodiment_id][..., :x_dim]) / (
                self.q99[embodiment_id][..., :x_dim] - self.q01[embodiment_id][..., :x_dim] + self.EPSILON
            ) * 2.0 - 1.0
        else:
            x = (x - self.mean[embodiment_id][..., :x_dim]) / (self.std[embodiment_id][..., :x_dim] + self.EPSILON)
        if self.enable_clamp:
            x = x.clamp(-1.0, 1.0)
        return x


class Unnormalize:
    """Unnormalizes a tensor using mean/std or quantile-based scaling."""

    def __init__(self, stats: dict[int, dict[str, list[float]]], *, use_quantiles: bool = False):
        self.EPSILON = 1e-6
        self.use_quantiles = use_quantiles
        required_attrs = ['mean', 'std'] if not self.use_quantiles else ['q01', 'q99']
        for attr in required_attrs:
            for key in stats:
                if attr not in stats[key]:
                    raise AttributeError(f'stats object is missing the following attribute: {attr}')

        if self.use_quantiles:
            self.q01 = {int(k): torch.tensor(stats[k]['q01'], dtype=torch.float32) for k in stats}
            self.q99 = {int(k): torch.tensor(stats[k]['q99'], dtype=torch.float32) for k in stats}
        else:
            self.mean = {int(k): torch.tensor(stats[k]['mean'], dtype=torch.float32) for k in stats}
            self.std = {int(k): torch.tensor(stats[k]['std'], dtype=torch.float32) for k in stats}

    def to(self, device: str | torch.device):
        if self.use_quantiles:
            for key in self.q01:
                self.q01[key] = self.q01[key].to(device)
            for key in self.q99:
                self.q99[key] = self.q99[key].to(device)
        else:
            for key in self.mean:
                self.mean[key] = self.mean[key].to(device)
            for key in self.std:
                self.std[key] = self.std[key].to(device)
        return self

    def __call__(self, x: torch.Tensor, embodiment_id: int = 0) -> torch.Tensor:
        x_dim = x.shape[-1]
        if self.use_quantiles:
            return (x + 1.0) / 2.0 * (self.q99[embodiment_id][..., :x_dim] - self.q01[embodiment_id][..., :x_dim] + self.EPSILON) + self.q01[
                embodiment_id
            ][..., :x_dim]
        else:
            return x * (self.std[embodiment_id][..., :x_dim] + self.EPSILON) + self.mean[embodiment_id][..., :x_dim]


class DeltaActions:
    """Repacks absolute actions into delta action space."""

    def __init__(self, mask: dict[int, list[bool]]):
        self.mask = {int(k): torch.tensor(mask[k]) for k in mask}

    def to(self, device: str | torch.device):
        for key in self.mask:
            self.mask[key] = self.mask[key].to(device)
        return self

    def __call__(self, data: dict) -> dict:
        if 'action' not in data or 'observation.state' not in data:
            return data
        embodiment_id = data['embodiment_id']
        dims = self.mask[embodiment_id].shape[-1]
        state, action = data['observation.state'], data['action']
        action[..., :dims] -= torch.where(self.mask[embodiment_id], state[..., :dims], torch.zeros_like(state[..., :dims])).unsqueeze(-2)
        data['action'] = action
        return data


class AbsoluteActions:
    """Repacks delta actions into absolute action space."""

    def __init__(self, mask: dict[int, list[bool]]):
        self.mask = {int(k): torch.tensor(mask[k]) for k in mask}

    def to(self, device: str | torch.device):
        for key in self.mask:
            self.mask[key] = self.mask[key].to(device)
        return self

    def __call__(self, data: dict) -> dict:
        if 'action' not in data or 'observation.state' not in data:
            return data
        embodiment_id = data['embodiment_id']
        state, action = data['observation.state'], data['action']
        dims = self.mask[embodiment_id].shape[-1]
        action[..., :dims] += torch.where(self.mask[embodiment_id], state[..., :dims], torch.zeros_like(state[..., :dims])).unsqueeze(-2)
        data['action'] = action
        return data


class PadStatesAndActions:
    """Zero-pads states and actions to the model action dimension."""

    def __init__(self, action_dim: int | None):
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
        if self.action_dim is None:
            raise ValueError('action_dim must be set before padding.')
        data['observation.state'] = self._pad_to_dim(data['observation.state'], self.action_dim, axis=-1)
        if 'action' in data:
            data['action'] = self._pad_to_dim(data['action'], self.action_dim, axis=-1)
        return data


def resize_image(img: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if img.ndim != 3:
        raise ValueError(f'(C,H,W) expected, but got {img.shape}')
    resized_img = F.interpolate(img.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
    return resized_img


def resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: float = -1.0) -> tuple[torch.Tensor, dict]:
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

    transform_params = {
        'original_size': (cur_width, cur_height),
        'ratio': ratio,
        'padding': (pad_left, pad_top),
    }
    return padded_img, transform_params


class RandomPoseTransform:
    """Applies a random crop, resize, and rotation to an image."""

    def __init__(self, crop_size: tuple[int, int], resize_size: tuple[int, int], rotation_degrees: tuple[float, float]):
        self.crop_size_h, self.crop_size_w = crop_size
        self.resize_size_h, self.resize_size_w = resize_size
        self.rotation_degrees = rotation_degrees

    def generate_params(self, h: int, w: int) -> dict[str, Any]:
        if h < self.crop_size_h or w < self.crop_size_w:
            raise ValueError(f'Required crop size {(self.crop_size_h, self.crop_size_w)} is larger than input image size {(h, w)}')
        i = torch.randint(0, h - self.crop_size_h + 1, size=(1,)).item()
        j = torch.randint(0, w - self.crop_size_w + 1, size=(1,)).item()
        crop_box = (j, i, self.crop_size_w, self.crop_size_h)
        angle = transforms.RandomRotation.get_params(self.rotation_degrees)
        return {'crop_box': crop_box, 'crop_size': (self.crop_size_w, self.crop_size_h), 'resize_size': (self.resize_size_w, self.resize_size_h), 'angle': angle}

    def apply_with_params(self, img: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        j, i, tw, th = params['crop_box']
        img = transforms.functional.crop(img, i, j, th, tw)
        img = transforms.functional.resize(img, (self.resize_size_h, self.resize_size_w))
        if params.get('angle') is not None:
            img = transforms.functional.rotate(img, params['angle'])
        return img

    def __call__(self, img: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        h, w = img.shape[-2:]
        params = self.generate_params(h, w)
        transformed_img = self.apply_with_params(img, params)
        return transformed_img, params


class ImageTransform:
    """Preprocesses a dictionary of images with optional augmentation."""

    def __init__(
        self,
        is_train: bool,
        resize_imgs_with_padding: tuple[int, int],
        present_img_keys: list[str] | None = None,
        enable_image_aug: bool = False,
        enable_depth_img: bool = False,
        depth_img_prefix_name: str | None = None,
        depth_img_mask_ratio: float = 0.5,
    ):
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.present_img_keys = present_img_keys or [
            'observation.images.cam_high',
            'observation.images.cam_left_wrist',
            'observation.images.cam_right_wrist',
        ]
        self.enable_image_aug = enable_image_aug
        self.width, self.height = resize_imgs_with_padding
        self.enable_depth_img = enable_depth_img
        self.depth_img_prefix_name = depth_img_prefix_name
        self.depth_img_mask_ratio = depth_img_mask_ratio if is_train else 0.0

        if self.enable_image_aug:
            self.color_jitter_transform = transforms.ColorJitter(
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
            )
            self.pose_transform = RandomPoseTransform(
                crop_size=(int(self.height * 0.95), int(self.width * 0.95)),
                resize_size=(self.height, self.width),
                rotation_degrees=(-5, 5),
            )

    def __call__(self, data: dict) -> tuple[list[torch.Tensor], list[torch.Tensor], dict]:
        images = []
        img_masks = []
        image_transform_params = {}

        for key in self.present_img_keys:
            if key not in data:
                raise ValueError(f'{key} not found in data. Please check the present_img_keys in the config or the dataset.')

            img = data[key]
            if self.enable_depth_img:
                assert self.depth_img_prefix_name is not None, 'depth_img_prefix_name is required'
                depth_img_key = key.replace('observation.images', self.depth_img_prefix_name)
                if depth_img_key in data and random.random() >= self.depth_img_mask_ratio:
                    depth_img = data[depth_img_key][0:1]
                else:
                    depth_img = torch.zeros_like(img[0:1])
                img = torch.cat([img, depth_img], dim=0)

            if self.resize_imgs_with_padding is not None:
                target_w, target_h = self.resize_imgs_with_padding
                original_h, original_w = img.shape[-2:]
                if original_h != target_h or original_w != target_w:
                    img, rwp_params = resize_with_pad(img, *self.resize_imgs_with_padding, pad_value=0)
                    if key == 'observation.images.cam_high':
                        image_transform_params['resize_with_pad'] = rwp_params

            if self.enable_image_aug:
                if key == 'observation.images.cam_high':
                    img, pose_params = self.pose_transform(img)
                    image_transform_params['pose_transform'] = pose_params
                img[:3, :, :] = self.color_jitter_transform(img[:3, :, :])

            img = img * 2.0 - 1.0
            images.append(img)
            img_masks.append(torch.tensor(True, dtype=torch.bool, device=img.device))

        return images, img_masks, image_transform_params


class TrajectoryTransform:
    """Transforms 2D trajectory data, including coordinate adjustments and normalization."""

    def __init__(self, step_interval: int | None = None, minmax_value: list[float] | None = None):
        self.step_interval = step_interval
        self.minmax_value = minmax_value
        if minmax_value is not None:
            assert minmax_value[2] > 0 and minmax_value[3] > 0, 'x_max and y_max must be greater than 0'
            self.min_value = torch.tensor([minmax_value[0], minmax_value[1], minmax_value[0], minmax_value[1]])
            self.max_value = torch.tensor([minmax_value[2], minmax_value[3], minmax_value[2], minmax_value[3]])
        else:
            self.min_value = None
            self.max_value = None
        self.traj_size = 4

    def to(self, device: str | torch.device):
        if self.min_value is not None:
            self.min_value = self.min_value.to(device)
        if self.max_value is not None:
            self.max_value = self.max_value.to(device)
        return self

    def __call__(self, data: dict, chunk_size: int, image_transform_params: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if 'perception.2d_traj' not in data or 'perception.2d_traj_is_pad' not in data:
            traj_chunk_size = (chunk_size // self.step_interval) if self.step_interval is not None else chunk_size
            return -torch.ones(traj_chunk_size, self.traj_size, dtype=torch.float32), torch.ones(traj_chunk_size, self.traj_size, dtype=torch.bool)

        if self.step_interval is not None:
            traj = data['perception.2d_traj'][:: self.step_interval]
            traj_is_pad = data['perception.2d_traj_is_pad'][:: self.step_interval]
        else:
            traj = data['perception.2d_traj']
            traj_is_pad = data['perception.2d_traj_is_pad']

        traj[torch.isnan(traj)] = -100

        if image_transform_params is not None:
            coords = traj.view(-1, 2, 2)

            if 'resize_with_pad' in image_transform_params:
                rwp = image_transform_params['resize_with_pad']
                ratio = rwp['ratio']
                pad_x, pad_y = rwp['padding']
                coords = coords / ratio
                coords[..., 0] += pad_x
                coords[..., 1] += pad_y

            if 'pose_transform' in image_transform_params:
                pose_p = image_transform_params['pose_transform']
                if pose_p.get('crop_box'):
                    crop_x, crop_y, _, _ = pose_p['crop_box']
                    coords[..., 0] -= crop_x
                    coords[..., 1] -= crop_y

                crop_w, crop_h = pose_p['crop_size']
                resize_w, resize_h = pose_p['resize_size']
                if crop_w > 0 and crop_h > 0:
                    scale_x = resize_w / crop_w
                    scale_y = resize_h / crop_h
                    coords[..., 0] *= scale_x
                    coords[..., 1] *= scale_y

                if pose_p.get('angle') is not None:
                    angle_rad = -math.radians(pose_p['angle'])
                    cos_a = math.cos(angle_rad)
                    sin_a = math.sin(angle_rad)
                    center_x, center_y = resize_w / 2, resize_h / 2
                    coords[..., 0] -= center_x
                    coords[..., 1] -= center_y
                    x_new = coords[..., 0] * cos_a - coords[..., 1] * sin_a
                    y_new = coords[..., 0] * sin_a + coords[..., 1] * cos_a
                    coords[..., 0] = x_new
                    coords[..., 1] = y_new
                    coords[..., 0] += center_x
                    coords[..., 1] += center_y

            traj = coords.view(-1, self.traj_size)

        traj_is_pad = traj_is_pad[:, None].expand(traj_is_pad.shape[0], self.traj_size)
        if self.minmax_value is not None:
            traj_is_pad = traj_is_pad | (traj < self.min_value[None, ...]) | (traj > self.max_value[None, ...])
            traj = traj.clamp(self.min_value[None, ...], self.max_value[None, ...])
            traj = traj / self.max_value[None, ...]
        return traj, traj_is_pad


class PromptTokenizerTransform:
    """Encodes task, state, and action information into token sequences for the policy model."""

    def __init__(
        self,
        is_train: bool,
        tokenizer_model_path: str,
        fast_tokenizer_path: str,
        max_length: int,
        discrete_state_input: bool = True,
        encode_action_input: bool = False,
        encoded_action_horizon: int | None = None,
        encode_sub_task_input: bool = False,
        text_token_length: int | None = 257152,
        autoregressive_inference_mode: bool = False,
        sample_ratios: dict | None = None,
    ):
        self.is_train = is_train
        self.device = 'cpu'
        self.paligemma_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        self.paligemma_tokenizer.add_bos_token = True
        self.processor = AutoProcessor.from_pretrained(tokenizer_model_path)
        self.fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)

        self.encode_action_input = encode_action_input
        self.discrete_state_input = discrete_state_input
        self.encode_sub_task_input = encode_sub_task_input

        self.encoded_action_horizon = encoded_action_horizon
        self.fast_skip_tokens = 128
        self.max_length = max_length
        self.text_token_length = text_token_length
        self.autoregressive_inference_mode = autoregressive_inference_mode

        self.sample_generator = SampleGenerator(sample_ratios) if is_train and sample_ratios is not None else None

    def to(self, device: str | torch.device):
        self.device = device
        return self

    def encode_action(self, action: torch.Tensor) -> dict:
        if self.encoded_action_horizon is not None:
            action_len = action.shape[1]
            horizon = int(self.encoded_action_horizon)
            assert action_len % horizon == 0, 'Action length must be divisible by encoded action horizon'
            step = action_len // horizon
            selected_indices = torch.arange(step - 1, action_len, step, device=action.device)
            action = action[:, selected_indices, :]

        batch_tokens = self.fast_tokenizer(action.to(torch.float32))
        fast_out = self.processor.tokenizer.pad({'input_ids': batch_tokens}, return_tensors='pt')
        act_ids = fast_out['input_ids'].squeeze(0)
        act_mask = fast_out['attention_mask'].squeeze(0)

        vocab_size = self.paligemma_tokenizer.vocab_size
        if self.text_token_length is not None:
            vocab_size = min(vocab_size, self.text_token_length)

        act_ids = vocab_size - 1 - self.fast_skip_tokens - act_ids
        act_ids[act_mask == 0] = self.paligemma_tokenizer.pad_token_id

        bos = self.paligemma_tokenizer('Action: ', add_special_tokens=False, return_tensors='pt')
        eos = self.paligemma_tokenizer('|<eos>', add_special_tokens=False, return_tensors='pt')

        final_act_ids = torch.cat(
            [
                bos['input_ids'].squeeze(0).to(act_ids.device),
                act_ids,
                eos['input_ids'].squeeze(0).to(act_ids.device),
            ],
            dim=0,
        )

        final_act_mask = torch.cat(
            [
                bos['attention_mask'].squeeze(0).to(act_mask.device),
                act_mask,
                eos['attention_mask'].squeeze(0).to(act_mask.device),
            ],
            dim=0,
        )

        return {'input_ids': final_act_ids, 'attention_mask': final_act_mask}

    def encode_sub_task(self, sub_task: str, add_eos: bool = True) -> dict:
        bos = self.paligemma_tokenizer('Subtask: ', add_special_tokens=False, return_tensors='pt')
        subtask_out = self.paligemma_tokenizer(
            [sub_task],
            add_special_tokens=False,
            return_tensors='pt',
            padding='longest',
            truncation=False,
        )
        final_subtask_ids = torch.cat(
            [
                bos['input_ids'].squeeze(0),
                subtask_out['input_ids'].squeeze(0),
            ],
            dim=0,
        )
        final_subtask_mask = torch.cat(
            [
                bos['attention_mask'].squeeze(0),
                subtask_out['attention_mask'].squeeze(0),
            ],
            dim=0,
        )

        if add_eos:
            eos = self.paligemma_tokenizer('<eos>', add_special_tokens=False, return_tensors='pt')
            final_subtask_ids = torch.cat([final_subtask_ids, eos['input_ids'].squeeze(0)], dim=0)
            final_subtask_mask = torch.cat([final_subtask_mask, eos['attention_mask'].squeeze(0)], dim=0)

        return {'input_ids': final_subtask_ids, 'attention_mask': final_subtask_mask}

    def create_input_tokens(
        self, task: str, state: torch.Tensor | None = None, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        prefix_texts = []
        cleaned = task.lower().strip().replace('_', ' ')

        main_task = cleaned.split(' subtask: ')[0]
        sub_task = None
        if ' subtask: ' in cleaned:
            sub_task = cleaned.split(' subtask: ')[1].split('\n')[0]

        encode_sub_task_input = self.encode_sub_task_input
        is_sub_task_train = self.is_train
        encode_action_input = self.encode_action_input
        if self.sample_generator is not None:
            encode_sub_task_input, is_sub_task_train, encode_action_input = self.sample_generator.get_sample()
        encode_sub_task_input = encode_sub_task_input and sub_task is not None

        predict_subtask = encode_sub_task_input and is_sub_task_train
        if predict_subtask or not self.discrete_state_input:
            prefix_texts.append(f'Task: {main_task}\n')
        elif self.discrete_state_input:
            assert state is not None, 'state is required when discrete_state_input is True'
            bins = torch.linspace(-1, 1, 256 + 1, device=self.device)[:-1]
            discretized = torch.bucketize(state, bins) - 1
            state_str = ' '.join(str(val.item()) for val in discretized)
            if encode_sub_task_input and not is_sub_task_train:
                prefix_texts.append(f'Task: {main_task}, Subtask: {sub_task}, State: {state_str};\n')
            else:
                prefix_texts.append(f'Task: {main_task}, State: {state_str};\n')
        else:
            raise ValueError('Invalid prefix text mode')

        prefix_out = self.paligemma_tokenizer(
            prefix_texts,
            add_special_tokens=True,
            return_tensors='pt',
            padding='longest',
            truncation=False,
        )
        prefix_ids = prefix_out['input_ids'][0]
        prefix_mask = prefix_out['attention_mask'][0]
        prefix_length = len(prefix_ids)
        fast_action_indicator = torch.zeros(prefix_length, dtype=torch.int32)

        assert prefix_length < self.max_length, f'Prefix length {prefix_length} is greater than max length {self.max_length}'

        final_ids = prefix_ids
        final_mask = prefix_mask
        if predict_subtask:
            encoded_sub_task = self.encode_sub_task(sub_task, add_eos=True)
            sub_task_ids = encoded_sub_task['input_ids']
            sub_task_mask = encoded_sub_task['attention_mask']
            final_ids = torch.cat([final_ids, sub_task_ids], dim=0)
            final_mask = torch.cat([final_mask, sub_task_mask], dim=0)
            fast_action_indicator = torch.cat([fast_action_indicator, torch.zeros_like(sub_task_mask)], dim=0)

        if encode_action_input and action is not None:
            encoded_action = self.encode_action(action[None])
            act_ids = encoded_action['input_ids']
            act_mask = encoded_action['attention_mask']
            final_ids = torch.cat([final_ids, act_ids], dim=0)
            final_mask = torch.cat([final_mask, act_mask], dim=0)
            fast_action_indicator = torch.cat([fast_action_indicator, torch.ones_like(act_mask)], dim=0)

        if final_ids.shape[0] > self.max_length and not self.autoregressive_inference_mode:
            final_ids = final_ids[: self.max_length]
            final_mask = final_mask[: self.max_length]
            fast_action_indicator = fast_action_indicator[: self.max_length]

        batch_inputs = {'input_ids': final_ids.tolist(), 'attention_mask': final_mask.tolist()}
        padding_side = 'left' if self.autoregressive_inference_mode else 'right'
        padded_output = self.paligemma_tokenizer.pad(
            batch_inputs, padding='max_length', padding_side=padding_side, max_length=self.max_length, return_tensors='pt'
        )
        final_ids = padded_output['input_ids']
        padded_mask = padded_output['attention_mask']

        att_mask = (padded_mask != 0).cumsum(dim=0) > prefix_length
        att_mask = att_mask & padded_mask

        loss_mask = (padded_mask != 0).cumsum(dim=0) > prefix_length
        loss_mask = loss_mask & padded_mask

        fast_action_indicator = F.pad(fast_action_indicator, (0, self.max_length - fast_action_indicator.shape[0]), mode='constant', value=0)

        return (
            final_ids.to(dtype=torch.int32, device=self.device),
            padded_mask.to(dtype=torch.bool, device=self.device),
            att_mask.to(dtype=torch.bool, device=self.device),
            loss_mask.to(dtype=torch.bool, device=self.device),
            fast_action_indicator.to(dtype=torch.bool, device=self.device),
            predict_subtask,
        )

    def __call__(self, data: dict) -> dict:
        if 'task' not in data:
            raise ValueError('No task found in data')

        task = data['task']
        state = data.get('observation.state', None)
        action = data.get('action', None)
        if action is not None and not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        return self.create_input_tokens(task, state, action)

    def extract_actions(self, tokens: list[list[int]], action_horizon: int, action_dim: int) -> torch.Tensor:
        assert len(tokens) == 1, 'Only support batch size 1'
        sequence = tokens[0].tolist() if hasattr(tokens[0], 'tolist') else list(tokens[0])

        bos_tokens = self.paligemma_tokenizer('Action: ', add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0).tolist()
        eos_tokens = self.paligemma_tokenizer('|<eos>', add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0).tolist()

        def find_subsequence(sequence_ids: list[int], pattern: list[int], start: int = 0) -> int:
            if not pattern:
                return -1
            max_start = len(sequence_ids) - len(pattern)
            for idx in range(start, max_start + 1):
                if sequence_ids[idx : idx + len(pattern)] == pattern:
                    return idx
            return -1

        bos_idx = find_subsequence(sequence, bos_tokens)
        if bos_idx == -1:
            return torch.zeros((1, 0, action_dim), dtype=torch.float32)

        action_start = bos_idx + len(bos_tokens)
        eos_idx = find_subsequence(sequence, eos_tokens, start=action_start)
        if eos_idx == -1:
            eos_idx = len(sequence)

        paligemma_action_ids = sequence[action_start:eos_idx]
        if not paligemma_action_ids:
            return torch.zeros((1, 0, action_dim), dtype=torch.float32)

        vocab_size = self.paligemma_tokenizer.vocab_size
        if self.text_token_length is not None:
            vocab_size = min(vocab_size, self.text_token_length)
        base_token_id = vocab_size - 1 - self.fast_skip_tokens

        fast_tokens: list[int] = []
        for paligemma_id in paligemma_action_ids:
            if paligemma_id > base_token_id:
                continue
            fast_id = base_token_id - paligemma_id
            if fast_id < 0:
                continue
            fast_tokens.append(int(fast_id))

        if not fast_tokens:
            return torch.zeros((1, 0, action_dim), dtype=torch.float32)

        decoded_actions = self.fast_tokenizer.decode(
            [fast_tokens],
            time_horizon=action_horizon,
            action_dim=action_dim,
        )
        actions = torch.tensor(decoded_actions, dtype=torch.float32)
        return actions


class SampleGenerator:
    """Generates random prompt format samples based on given ratios."""

    def __init__(self, sample_ratios: dict[str, float]):
        valid_sample_names = [
            'task_only',
            'task_with_subtask',
            'task_only_using_subtask_regression',
            'task_only_using_fast_regression',
            'task_with_subtask_using_fast_regression',
        ]
        assert all(name in valid_sample_names for name in sample_ratios.keys())
        assert all(0 <= sample_ratio <= 1 for sample_ratio in sample_ratios.values())
        if 'identity' not in sample_ratios:
            sample_ratios['identity'] = 1.0 - sum(sample_ratios.values())
        assert math.isclose(sum(sample_ratios.values()), 1.0, abs_tol=1e-6)
        self.sample_ratios = sample_ratios

    def get_sample(self) -> tuple[bool, bool, bool]:
        sample_type = random.random()
        prob_acc = 0.0
        sample_name = None
        for name, sample_ratio in self.sample_ratios.items():
            prob_acc += sample_ratio
            if sample_type < prob_acc:
                sample_name = name
                break

        encode_sub_task_input = sample_name in ['task_with_subtask', 'task_with_subtask_using_fast_regression', 'task_only_using_subtask_regression']
        is_sub_task_train = sample_name == 'task_only_using_subtask_regression'
        encode_action_input = sample_name in ['task_only_using_fast_regression', 'task_with_subtask_using_fast_regression']
        return encode_sub_task_input, is_sub_task_train, encode_action_input


class GigaBrain0Operator(BaseOperator):
    """Operator that handles all GigaBrain0 preprocessing and postprocessing."""

    def __init__(
        self,
        embodiment_id: int,
        state_norm_stats: dict,
        action_norm_stats: dict,
        delta_mask: list[bool],
        tokenizer_model_path: str,
        fast_tokenizer_path: str,
        *,
        resize_imgs_with_padding: tuple[int, int] = (224, 224),
        present_img_keys: list[str] | None = None,
        enable_image_aug: bool = False,
        enable_depth_img: bool = False,
        depth_img_prefix_name: str | None = None,
        discrete_state_input: bool = True,
        autoregressive_inference_mode: bool = False,
        text_max_length: int = 200,
    ) -> None:
        super().__init__()
        self.device = 'cpu'
        self.embodiment_id = embodiment_id
        # Interaction handling (no fixed template; accept free-text tasks).
        self.interaction_template = []
        self.interaction_template_init()

        # Transforms
        self.state_normalize = Normalize({embodiment_id: state_norm_stats}, use_quantiles=True)
        self.state_unnormalize = Unnormalize({embodiment_id: state_norm_stats}, use_quantiles=True)
        self.action_unnormalize = Unnormalize({embodiment_id: action_norm_stats}, use_quantiles=True)
        self.absolute_actions = AbsoluteActions({embodiment_id: delta_mask})
        self.pad_states_actions = PadStatesAndActions(action_dim=None)  # will be set later by pipeline

        self.image_transform = ImageTransform(
            is_train=False,
            resize_imgs_with_padding=resize_imgs_with_padding,
            present_img_keys=present_img_keys,
            enable_image_aug=enable_image_aug,
            enable_depth_img=enable_depth_img,
            depth_img_prefix_name=depth_img_prefix_name,
        )
        self.traj_transform = TrajectoryTransform()

        self.prompt_tokenizer = PromptTokenizerTransform(
            is_train=False,
            tokenizer_model_path=tokenizer_model_path,
            fast_tokenizer_path=fast_tokenizer_path,
            max_length=text_max_length,
            discrete_state_input=discrete_state_input,
            encode_action_input=False,
            encode_sub_task_input=True,
            autoregressive_inference_mode=autoregressive_inference_mode,
        )

    # Interaction --------------------------------------------------------------
    def check_interaction(self, interaction: str) -> bool:
        """Validate interaction/task; skip checks when no template is provided."""
        if not isinstance(interaction, str):
            raise ValueError('interaction must be a string')
        if self.interaction_template and interaction not in self.interaction_template:
            raise ValueError(f'{interaction} not in interaction_template: {self.interaction_template}')
        return True

    def get_interaction(self, interaction: str | list[str]):
        """Append interaction(s) to the current list after validation."""
        if not isinstance(interaction, list):
            interaction = [interaction]
        for act in interaction:
            self.check_interaction(act)
            self.current_interaction.append(act)

    def process_interaction(self, task: str | None = None, state: torch.Tensor | None = None, action: torch.Tensor | None = None):
        """Tokenize task/state/action; falls back to last recorded interaction when task is None."""
        if task is not None:
            self.get_interaction(task)
        if len(self.current_interaction) == 0:
            raise ValueError('No interaction/task provided to process_interaction')
        current_task = self.current_interaction[-1]

        if action is not None and not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        return self.prompt_tokenizer({'task': current_task, 'observation.state': state, 'action': action})

    def delete_last_interaction(self):
        super().delete_last_interaction()

    def set_action_dim(self, action_dim: int):
        """Delay setting action_dim until policy is known."""
        self.pad_states_actions.action_dim = action_dim

    def to(self, device: str | torch.device):
        self.device = device
        self.state_normalize.to(device)
        self.state_unnormalize.to(device)
        self.action_unnormalize.to(device)
        self.absolute_actions.to(device)
        self.prompt_tokenizer.to(device)
        self.traj_transform.to(device)
        return self

    # Preprocess ----------------------------------------------------------------
    def process_perception(self, images: dict[str, torch.Tensor], state: torch.Tensor | None, pad_state: bool = True):
        """Normalize state (if provided) and preprocess images."""
        images_proc, img_masks, image_params = self.image_transform(images)
        state_normed = None
        if state is not None and state.numel() > 0:
            state_normed = self.state_normalize(state, embodiment_id=self.embodiment_id)
        if pad_state and state_normed is not None:
            if self.pad_states_actions.action_dim is None:
                raise ValueError('action_dim must be set before padding state.')
            state_normed = self.pad_states_actions({'observation.state': state_normed})['observation.state']
        return images_proc, img_masks, image_params, state_normed

    def process_interaction(self, task: str, state: torch.Tensor | None = None, action: torch.Tensor | None = None):
        """Tokenize task/state/action to language tokens."""
        return self.prompt_tokenizer({'task': task, 'observation.state': state, 'action': action})

    # Postprocess ---------------------------------------------------------------
    def process_output(
        self,
        pred_action: torch.Tensor,
        state_proc: torch.Tensor,
        original_action_dim: int,
        image_transform_params: dict | None = None,
        traj_pred: torch.Tensor | None = None,
    ):
        """Unnormalize and convert delta->absolute, optionally undo image transforms on traj."""
        out = {'action': pred_action, 'embodiment_id': self.embodiment_id}
        if state_proc is not None:
            out['observation.state'] = state_proc
            out['observation.state'] = self.state_unnormalize(out['observation.state'], embodiment_id=self.embodiment_id)
        out['action'] = self.action_unnormalize(out['action'], embodiment_id=self.embodiment_id)
        out = self.absolute_actions(out)
        action_final = out['action'][:, :original_action_dim]

        if traj_pred is None:
            return action_final

        # Undo resize/pad if available
        if image_transform_params and 'resize_with_pad' in image_transform_params:
            ratio = image_transform_params['resize_with_pad']['ratio']
            pad_x, pad_y = image_transform_params['resize_with_pad']['padding']
            traj_pred[:, ::2] = (traj_pred[:, ::2] - pad_x) * ratio
            traj_pred[:, 1::2] = (traj_pred[:, 1::2] - pad_y) * ratio

        return action_final, traj_pred

    # Utilities -----------------------------------------------------------------
    def extract_actions(self, tokens: list[list[int]], action_horizon: int, action_dim: int) -> torch.Tensor:
        return self.prompt_tokenizer.extract_actions(tokens, action_horizon, action_dim)

    @property
    def tokenizer(self):
        return self.prompt_tokenizer.paligemma_tokenizer
