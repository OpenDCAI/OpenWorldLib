# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team.
# ==============================================================================

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    IDENTITY = "IDENTITY"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: Tuple


def create_stats_buffers(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, Tensor]] | None = None,
) -> dict[str, dict[str, nn.ParameterDict]]:
    stats_buffers = {}
    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue
        assert isinstance(norm_mode, NormalizationMode)
        if norm_mode is not NormalizationMode.MIN_MAX:
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")
        shape = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            shape = (c, 1, 1)
        min_v = torch.ones(shape, dtype=torch.float32) * torch.inf
        max_v = torch.ones(shape, dtype=torch.float32) * torch.inf
        buffer = nn.ParameterDict(
            {"min": nn.Parameter(min_v, requires_grad=False), "max": nn.Parameter(max_v, requires_grad=False)}
        )
        if stats is not None:
            if key not in stats:
                raise ValueError(f"Missing stats for feature `{key}` (expected `min`/`max`).")
            if "min" not in stats[key] or "max" not in stats[key]:
                raise ValueError(f"Stats for `{key}` must contain `min` and `max` for MIN_MAX normalization.")
            min_src, max_src = stats[key]["min"], stats[key]["max"]
            if isinstance(min_src, np.ndarray) and isinstance(max_src, np.ndarray):
                buffer["min"].data = torch.from_numpy(min_src).to(dtype=torch.float32)
                buffer["max"].data = torch.from_numpy(max_src).to(dtype=torch.float32)
            elif isinstance(min_src, torch.Tensor) and isinstance(max_src, torch.Tensor):
                buffer["min"].data = min_src.clone().to(dtype=torch.float32)
                buffer["max"].data = max_src.clone().to(dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected stats type for `{key}`: min={type(min_src)}, max={type(max_src)}")
        stats_buffers[key] = buffer
    return stats_buffers


def no_stats_error_str(name: str) -> str:
    return f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a pretrained model."


def build_norm_state(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, Tensor]] | None = None,
) -> tuple[dict[FeatureType, NormalizationMode], dict[str, nn.ParameterDict]]:
    norm_mode_map: dict[FeatureType, NormalizationMode] = {}
    for k, v in (norm_map or {}).items():
        ft = k if isinstance(k, FeatureType) else FeatureType(k)
        mode = v if isinstance(v, NormalizationMode) else NormalizationMode(v)
        if mode not in (NormalizationMode.IDENTITY, NormalizationMode.MIN_MAX):
            raise ValueError(f"Unsupported normalization mode: {mode}")
        norm_mode_map[ft] = mode
    stats_buffers = create_stats_buffers(features, norm_mode_map, stats)
    return norm_mode_map, stats_buffers


def pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def pad_and_cat(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = F.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)
    stacked_tensor = torch.cat(padded_tensors, dim=1)
    return stacked_tensor


def sample_beta(alpha: float, beta: float, bsize: int, device) -> torch.Tensor:
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([beta]))
    return m.sample((bsize,)).to(device).reshape((bsize,))


def sample_noise(shape, device) -> torch.Tensor:
    return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)


def sample_time(bsize: int, device) -> torch.Tensor:
    time_beta = sample_beta(1.5, 1.0, bsize, device)
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=torch.float32, device=device)


def preprocess_qwen_visual(
    sources,
    tokenizer,
    grid_thw_image: Optional[List] = None,
) -> Dict:
    grid_thw_image = grid_thw_image or []
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    visual_replicate_index_image = 0
    input_ids = []
    for source in sources:
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except Exception:
            pass
        input_id = []
        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except Exception:
                role = conv["from"]
                content = conv["value"]
            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        if visual_replicate_index_image < len(grid_thw_image):
                            replacement = (
                                "<|vision_start|>"
                                + "<|image_pad|>" * grid_thw_image[visual_replicate_index_image]
                                + "<|vision_end|>"
                            )
                            new_parts.append(replacement)
                            visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)
            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
        input_ids.append(input_id)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_ids = input_ids[:, : tokenizer.model_max_length]
    return dict(input_ids=input_ids)


def get_rope_index_3(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        return position_ids, mrope_position_deltas
