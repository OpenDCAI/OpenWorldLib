# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from __future__ import annotations

import os
import random
from typing import Any, Optional

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import InterpolationMode, Resize

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

from gamma_world._src.imaginaire.utils import log
from gamma_world._src.predict2_multiview.datasets.multiview import collate_fn
from gamma_world._src.gamma_world.datasets.multiview import AugmentationConfig


def _list_clip_ids(camera_root: str, suffix: str) -> set[str]:
    if not os.path.isdir(camera_root):
        return set()
    return {f[: -len(suffix)] for f in os.listdir(camera_root) if f.endswith(suffix)}


class LocalMultiviewVideoDataset(Dataset):


    def __init__(
        self,
        *,
        data_root: str,
        camera_keys: list[str],
        resolution_hw: tuple[int, int],
        num_video_frames: int,
        fps_downsample_factor: int,
        is_train: bool,
        camera_view_mapping: dict[str, int],
        camera_control_key_mapping: Optional[dict[str, str]] = None,
        add_view_prefix_to_caption: bool = False,
        camera_prefix_mapping: Optional[dict[str, str]] = None,
        single_caption_camera_name: Optional[str] = None,
        window_random_frame_offset_range: Optional[tuple[int, int]] = None,
        video_subdir: str = "video",
        hdmap_subdir: str = "hdmap",
        caption_subdir: str = "caption",
        val_holdout_frac: float = 0.05,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.camera_keys = list(camera_keys)
        self.resolution_hw = tuple(resolution_hw)
        self.num_video_frames = int(num_video_frames)
        self.fps_downsample_factor = int(fps_downsample_factor)
        self.is_train = is_train
        self.camera_view_mapping = dict(camera_view_mapping)
        self.camera_control_key_mapping = (
            dict(camera_control_key_mapping) if camera_control_key_mapping is not None else None
        )
        self.add_view_prefix_to_caption = add_view_prefix_to_caption
        self.camera_prefix_mapping = camera_prefix_mapping
        self.single_caption_camera_name = single_caption_camera_name
        self.window_random_frame_offset_range = window_random_frame_offset_range
        self.video_subdir = video_subdir
        self.hdmap_subdir = hdmap_subdir
        self.caption_subdir = caption_subdir

        if self.add_view_prefix_to_caption and self.camera_prefix_mapping is None:
            raise ValueError("camera_prefix_mapping is required when add_view_prefix_to_caption is True")
        if self.single_caption_camera_name and self.single_caption_camera_name not in self.camera_keys:
            raise ValueError(
                f"single_caption_camera_name {self.single_caption_camera_name} not in camera_keys {self.camera_keys}"
            )
        for cam in self.camera_keys:
            if cam not in self.camera_view_mapping:
                raise ValueError(f"camera {cam} missing from camera_view_mapping")

        self._resize = Resize(self.resolution_hw, interpolation=InterpolationMode.BILINEAR, antialias=True)

        clip_ids = self._discover_clip_ids()
        if not clip_ids:
            raise RuntimeError(f"No clips found under {data_root!r} for cameras {self.camera_keys}")

        n_total = len(clip_ids)
        n_val = max(1, int(round(n_total * val_holdout_frac))) if val_holdout_frac > 0 else 0
        train_ids = clip_ids[: n_total - n_val]
        val_ids = clip_ids[n_total - n_val :] if n_val > 0 else []
        self.clip_ids = train_ids if is_train else (val_ids or train_ids)

        log.info(
            f"LocalMultiviewVideoDataset: data_root={data_root}, cameras={self.camera_keys}, "
            f"is_train={is_train}, num_clips={len(self.clip_ids)} "
            f"(of {n_total} total; val_holdout={n_val})"
        )

    def _discover_clip_ids(self) -> list[str]:
        per_camera_ids: list[set[str]] = []
        for cam in self.camera_keys:
            video_ids = _list_clip_ids(os.path.join(self.data_root, self.video_subdir, cam), ".mp4")
            caption_ids = _list_clip_ids(os.path.join(self.data_root, self.caption_subdir, cam), ".txt")
            ids = video_ids & caption_ids
            if self.camera_control_key_mapping is not None:
                hdmap_ids = _list_clip_ids(os.path.join(self.data_root, self.hdmap_subdir, cam), ".mp4")
                ids &= hdmap_ids
            per_camera_ids.append(ids)
        common = set.intersection(*per_camera_ids) if per_camera_ids else set()
        return sorted(common)

    def __len__(self) -> int:
        return len(self.clip_ids)

    def _video_path(self, subdir: str, camera: str, clip_id: str) -> str:
        return os.path.join(self.data_root, subdir, camera, f"{clip_id}.mp4")

    def _caption_path(self, camera: str, clip_id: str) -> str:
        return os.path.join(self.data_root, self.caption_subdir, camera, f"{clip_id}.txt")

    def _decode_window(self, video_path: str, frame_indices: list[int]) -> tuple[torch.Tensor, float, tuple[int, int]]:
        from decord import VideoReader

        vr = VideoReader(video_path)
        fps = vr.get_avg_fps()
        frames = vr.get_batch(frame_indices).asnumpy()
        frames = rearrange(torch.from_numpy(frames), "t h w c -> t c h w")
        original_h, original_w = frames.shape[-2:]
        return self._resize(frames), float(fps), (int(original_h), int(original_w))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_id = self.clip_ids[idx]
        window_len = self.num_video_frames * self.fps_downsample_factor

        random_offset = 0
        if self.window_random_frame_offset_range is not None:
            random_offset = random.randint(*self.window_random_frame_offset_range)

        captions: list[str] = []
        multiview_frames: list[torch.Tensor] = []
        multiview_control: list[torch.Tensor] = []
        view_indices: list[int] = []
        view_indices_selection: list[int] = []
        camera_keys_selection: list[str] = []
        original_sizes: list[list[int]] = []

        chosen_start: Optional[int] = None
        chosen_fps: Optional[int] = None
        extracted_frame_ids: Optional[list[int]] = None

        for camera in self.camera_keys:
            video_path = self._video_path(self.video_subdir, camera, clip_id)
            from decord import VideoReader

            vr = VideoReader(video_path)
            total_frames = len(vr)
            avg_fps = vr.get_avg_fps()
            del vr

            if total_frames < window_len:
                raise RuntimeError(f"clip {clip_id} cam {camera}: only {total_frames} frames, need {window_len}")

            if chosen_start is None:
                max_start = total_frames - window_len - max(0, random_offset)
                if max_start < 0:
                    raise RuntimeError(
                        f"clip {clip_id} too short for window+offset: "
                        f"frames={total_frames} window={window_len} offset={random_offset}"
                    )
                chosen_start = random.randint(0, max_start) if self.is_train else 0
                chosen_fps = int(np.round(avg_fps))
            elif int(np.round(avg_fps)) != chosen_fps:
                raise RuntimeError(f"clip {clip_id} cam {camera}: fps {avg_fps} != ref fps {chosen_fps}")

            start = chosen_start + random_offset
            frame_indices = list(range(start, start + window_len, self.fps_downsample_factor))
            frames, _, original_hw = self._decode_window(video_path, frame_indices)
            assert len(frames) == self.num_video_frames

            if extracted_frame_ids is None:
                extracted_frame_ids = frame_indices
            elif frame_indices != extracted_frame_ids:
                raise RuntimeError("Frame indices diverged across cameras")

            multiview_frames.append(frames)
            original_sizes.append(list(original_hw))


            with open(self._caption_path(camera, clip_id), encoding="utf-8") as f:
                caption = f.read().strip()
            if self.single_caption_camera_name and camera != self.single_caption_camera_name:
                caption = ""
            if self.add_view_prefix_to_caption and caption:
                caption = f"{self.camera_prefix_mapping[camera]} {caption}"
            captions.append(caption)


            if self.camera_control_key_mapping is not None:
                hdmap_path = self._video_path(self.hdmap_subdir, camera, clip_id)
                control_frames, _, _ = self._decode_window(hdmap_path, frame_indices)
                if len(control_frames) != self.num_video_frames:
                    raise RuntimeError(f"clip {clip_id} cam {camera}: hdmap returned {len(control_frames)} frames")
                multiview_control.append(control_frames)

            view_indices.extend([self.camera_view_mapping[camera]] * self.num_video_frames)
            view_indices_selection.append(self.camera_view_mapping[camera])
            camera_keys_selection.append(camera)

        assert chosen_fps is not None and extracted_frame_ids is not None
        if chosen_fps % self.fps_downsample_factor != 0:
            raise RuntimeError(
                f"video fps {chosen_fps} not divisible by fps_downsample_factor {self.fps_downsample_factor}"
            )
        out_fps = chosen_fps / self.fps_downsample_factor

        front_pos = (
            torch.tensor(self.camera_keys.index(self.single_caption_camera_name), dtype=torch.int64)
            if self.single_caption_camera_name
            else None
        )
        if self.single_caption_camera_name and not self.add_view_prefix_to_caption:
            captions = [captions[int(front_pos.item())]]

        sample = {
            "__key__": clip_id,
            "__url__": self._video_path(self.video_subdir, self.camera_keys[0], clip_id),
            "video": rearrange(torch.cat(multiview_frames, dim=0), "t c h w -> c t h w"),
            "ai_caption": captions,
            "view_indices": torch.tensor(view_indices, dtype=torch.int64),
            "fps": torch.tensor(out_fps, dtype=torch.float64),
            "chunk_index": torch.tensor(0, dtype=torch.int64),
            "frame_indices": torch.tensor(extracted_frame_ids, dtype=torch.int64),
            "num_video_frames_per_view": torch.tensor(len(extracted_frame_ids), dtype=torch.int64),
            "view_indices_selection": torch.tensor(view_indices_selection, dtype=torch.int64),
            "camera_keys_selection": camera_keys_selection,
            "sample_n_views": torch.tensor(len(camera_keys_selection), dtype=torch.int64),
            "padding_mask": torch.zeros((1, *self.resolution_hw), dtype=torch.float32),
            "ref_cam_view_idx_sample_position": torch.tensor(-1, dtype=torch.int64),
            "front_cam_view_idx_sample_position": front_pos,
            "original_hw": torch.tensor(original_sizes, dtype=torch.int64),
        }
        if self.camera_control_key_mapping is not None:
            sample["control_input_hdmap_bbox"] = rearrange(torch.cat(multiview_control, dim=0), "t c h w -> c t h w")
        return sample


def _data_parallel_rank_world() -> tuple[int, int]:

    if USE_MEGATRON and parallel_state.is_initialized():
        return (
            parallel_state.get_data_parallel_rank(),
            parallel_state.get_data_parallel_world_size(),
        )
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return 0, 1


def get_local_multiview_video_loader(
    *,
    dataset_name: str = "local",
    is_train: bool,
    augmentation_config: AugmentationConfig = AugmentationConfig(),
    batch_size: int = 1,
    num_workers: int = 4,
    prefetch_factor: int | None = 1,
    data_root: str = "data",
    val_holdout_frac: float = 0.05,
    **kwargs: Any,
) -> DataLoader:

    del dataset_name

    kwargs.pop("object_store", None)
    kwargs.pop("max_shards", None)
    kwargs.pop("shuffle_buffer_size", None)

    dataset = LocalMultiviewVideoDataset(
        data_root=data_root,
        camera_keys=list(augmentation_config.camera_keys),
        resolution_hw=augmentation_config.resolution_hw,
        num_video_frames=augmentation_config.num_video_frames,
        fps_downsample_factor=augmentation_config.fps_downsample_factor,
        is_train=is_train,
        camera_view_mapping=augmentation_config.camera_view_mapping,
        camera_control_key_mapping=augmentation_config.camera_control_key_mapping,
        add_view_prefix_to_caption=augmentation_config.add_view_prefix_to_caption,
        camera_prefix_mapping=augmentation_config.camera_prefix_mapping,
        single_caption_camera_name=augmentation_config.single_caption_camera_name,
        window_random_frame_offset_range=augmentation_config.window_random_frame_offset_range,
        val_holdout_frac=val_holdout_frac,
    )

    rank, world = _data_parallel_rank_world()
    sampler = (
        DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=is_train, drop_last=is_train)
        if world > 1
        else None
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,
        pin_memory=False,
        drop_last=is_train,
        collate_fn=collate_fn,
    )
