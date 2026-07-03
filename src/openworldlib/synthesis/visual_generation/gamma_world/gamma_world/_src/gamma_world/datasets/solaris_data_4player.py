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

import glob
import io
import json
import os
from typing import Any

import attrs
import torch
import webdataset as wds
from einops import rearrange
from torchvision.transforms import InterpolationMode, Resize

from gamma_world._src.imaginaire import config
from gamma_world._src.imaginaire.datasets.decoders.json_loader import json_decoder
from gamma_world._src.imaginaire.datasets.decoders.video_decoder import video_naive_bytes
from gamma_world._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from gamma_world._src.imaginaire.datasets.webdataset.config.schema import DatasetConfig, DatasetInfo
from gamma_world._src.imaginaire.datasets.webdataset.distributors import ShardlistBasic
from gamma_world._src.imaginaire.utils import log
from gamma_world._src.imaginaire.utils.distributed import barrier, is_rank0
from gamma_world._src.predict2.datasets.cached_replay_dataloader import get_cached_replay_dataloader
from gamma_world._src.predict2_multiview.datasets.multiview import collate_fn
from gamma_world._src.gamma_world.datasets.sil_dataset import SILDataset

SOLARIS_4PLAYER_DATA_ROOT = "data/solar_webdata_4player"



SOLARIS_4PLAYER_EVAL_METADATA_CSV: str | None = None
SOLARIS_4PLAYER_TAR_KEY_PREFIX = "solaris_action__"

NUM_PLAYERS = 4
ACTION_SIDES: tuple[str, ...] = tuple(f"action_{i}" for i in range(NUM_PLAYERS))


def _load_eval_exclude_keys(csv_path: str) -> set[str]:

    import csv as _csv

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Solaris 4-player eval metadata CSV not found: {csv_path}. "
            f"Either fix the path in solaris_data_4player.py or disable "
            f"`augmentation_config.exclude_eval_keys`."
        )

    keys: set[str] = set()
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        if "video" not in (reader.fieldnames or []):
            raise RuntimeError(
                f"Solaris 4-player eval CSV {csv_path} is missing required "
                f"`video` column; got fieldnames={reader.fieldnames}"
            )
        for row in reader:
            base = os.path.splitext(os.path.basename(row["video"]))[0]
            keys.add(f"{SOLARIS_4PLAYER_TAR_KEY_PREFIX}{base}")

    if not keys:
        raise RuntimeError(f"Parsed 0 eval keys from {csv_path}; refusing to proceed silently.")
    return keys


def _verify_exclude_keys_against_tars(
    exclude_keys: set[str],
    data_root: str,
    min_matches_required: int = 3,
    max_tars_to_scan: int = 50,
) -> None:

    import random
    import tarfile
    import time

    if not is_rank0():
        barrier()
        return

    video_dir = os.path.join(data_root, "video")
    tar_files = sorted(glob.glob(os.path.join(video_dir, "*.tar")))
    if not tar_files:
        raise FileNotFoundError(f"No tar files found under {video_dir} for sanity check.")

    rng = random.Random(0)
    shuffled = tar_files.copy()
    rng.shuffle(shuffled)
    budget = min(max_tars_to_scan, len(shuffled))

    log.info(
        "[Solaris4P eval-filter sanity check] Sampling tars to confirm key format "
        "(rank-0 only, expect <30s)..."
    )
    t0 = time.time()

    matched_keys: list[str] = []
    total_scanned = 0
    tars_scanned = 0
    for tar_path in shuffled[:budget]:
        with tarfile.open(tar_path, "r") as tf:
            for m in tf.getmembers():
                key = os.path.splitext(m.name)[0]
                total_scanned += 1
                if key in exclude_keys:
                    matched_keys.append(key)
        tars_scanned += 1
        if len(matched_keys) >= min_matches_required:
            break

    elapsed = time.time() - t0

    if len(matched_keys) < min_matches_required:
        barrier()
        raise RuntimeError(
            f"Solaris 4-player eval-key sanity check FAILED: scanned "
            f"{total_scanned} keys across {tars_scanned} tars (budget={budget}) "
            f"and matched only {len(matched_keys)} of {len(exclude_keys)} eval "
            f"keys (required >= {min_matches_required}). Key format is likely "
            f"mismatched.\n"
            f"  Sample exclude keys (first 3): {list(exclude_keys)[:3]}"
        )

    log.info(
        f"[Solaris4P eval-filter sanity check] PASSED in {elapsed:.1f}s: "
        f"scanned {tars_scanned} tars / {total_scanned} keys, matched "
        f"{len(matched_keys)} eval keys (e.g. {matched_keys[:2]}) -- "
        f"filter is wired up correctly."
    )
    barrier()


@attrs.define(slots=False)
class Solaris4PlayerAugmentationConfig:


    resolution_hw: tuple[int, int] = (480, 832)
    num_video_frames: int = 93
    fps_downsample_factor: int = 1
    load_action: bool = False
    random_sampling: bool = True

    exclude_eval_keys: bool = False




    take_half_width: bool = False


class ExtractSolaris4PlayerFrames(Augmentor):


    def __init__(
        self,
        num_frames: int,
        resolution_hw: tuple[int, int],
        fps_downsample_factor: int = 1,
        load_action: bool = False,
        random_sampling: bool = True,
        exclude_keys: set[str] | None = None,
    ) -> None:
        super().__init__([], {})
        self.num_frames = num_frames
        self.resolution_hw = resolution_hw
        self.fps_downsample_factor = fps_downsample_factor
        self.load_action = load_action
        self.random_sampling = random_sampling
        self.exclude_keys = exclude_keys or set()

    def _read_action_json_from_tar(self, tar_url: Any, sample_key: str, side: str) -> dict | None:

        import tarfile

        url = tar_url
        action_tar_path = os.path.join(url.root, side, url.path)

        if not hasattr(self, "_action_tar_cache"):
            self._action_tar_cache: dict[str, Any] = {}

        if action_tar_path not in self._action_tar_cache:
            try:
                tf = tarfile.open(action_tar_path, "r")
                index = {}
                for m in tf.getmembers():
                    index[os.path.splitext(m.name)[0]] = m
                self._action_tar_cache[action_tar_path] = (tf, index)
            except Exception:
                self._action_tar_cache[action_tar_path] = None

        cached = self._action_tar_cache.get(action_tar_path)
        if cached is None:
            return None
        tf, index = cached

        member = index.get(sample_key)
        if member is None:
            return None
        try:
            f = tf.extractfile(member)
            return json.loads(f.read()) if f else None
        except Exception:
            return None

    def _parse_action_data(
        self, data: dict[str, Any], frame_indices: list[int], total_frames: int
    ) -> dict[str, torch.Tensor] | None:

        tar_url = data.get("__url__")
        sample_key = data.get("__key__", "")
        if tar_url is None:
            return None

        result: dict[str, torch.Tensor] = {}
        for side in ACTION_SIDES:
            action_json = self._read_action_json_from_tar(tar_url, sample_key, side)
            if action_json is None:
                return None

            keyboard_all = action_json.get("keyboard")
            camera_all = action_json.get("camera")
            if keyboard_all is None or camera_all is None:
                return None

            keyboard = [keyboard_all[i] for i in frame_indices]
            camera = [camera_all[i] for i in frame_indices]

            result[f"{side}_keyboard"] = torch.tensor(keyboard, dtype=torch.float32)
            result[f"{side}_camera"] = torch.tensor(camera, dtype=torch.float32)

        return result

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | None:

        if self.exclude_keys and data.get("__key__") in self.exclude_keys:
            return None
        try:
            from decord import VideoReader

            video_bytes = data["video"]
            vr = VideoReader(io.BytesIO(video_bytes))
            total_frames = len(vr)
            original_fps = vr.get_avg_fps()

            required_raw_frames = self.num_frames * self.fps_downsample_factor
            if total_frames < required_raw_frames:
                log.warning(
                    f"Solaris 4-player video has {total_frames} frames but need "
                    f"{required_raw_frames}, skipping sample {data.get('__key__', '?')}"
                )
                return None

            max_start = total_frames - required_raw_frames
            if self.random_sampling:
                start = torch.randint(0, max(max_start, 1), (1,)).item() if max_start > 0 else 0
            else:
                start = 0

            frame_indices = list(range(start, start + required_raw_frames, self.fps_downsample_factor))

            frames = vr.get_batch(frame_indices).asnumpy()
            frames = rearrange(torch.from_numpy(frames), "t h w c -> t c h w")

            full_w = frames.shape[-1]
            if full_w % NUM_PLAYERS != 0:
                raise RuntimeError(
                    f"4-player video width {full_w} not divisible by {NUM_PLAYERS} "
                    f"for sample {data.get('__key__', '?')}"
                )
            view_w = full_w // NUM_PLAYERS
            frames = torch.cat(
                [frames[..., i * view_w : (i + 1) * view_w] for i in range(NUM_PLAYERS)],
                dim=0,
            )
            sample_n_views = NUM_PLAYERS
            view_indices = torch.cat(
                [torch.full((self.num_frames,), i, dtype=torch.int64) for i in range(NUM_PLAYERS)]
            )

            original_h, original_w = frames.shape[-2:]

            frames = Resize(self.resolution_hw, interpolation=InterpolationMode.BILINEAR, antialias=True)(frames)

            caption_data = data.get("caption", {})
            if isinstance(caption_data, dict):
                caption = caption_data.get("prompt", "")
            elif isinstance(caption_data, str):
                caption = caption_data
            else:
                caption = ""

            fps = original_fps / self.fps_downsample_factor

            result = {
                "__key__": data["__key__"],
                "__url__": data["__url__"],
                "video": rearrange(frames, "t c h w -> c t h w"),
                "ai_caption": [caption],
                "view_indices": view_indices,
                "fps": torch.tensor(fps, dtype=torch.float64),
                "chunk_index": torch.tensor(0, dtype=torch.int64),
                "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
                "num_video_frames_per_view": torch.tensor(self.num_frames, dtype=torch.int64),
                "view_indices_selection": torch.arange(sample_n_views, dtype=torch.int64),
                "camera_keys_selection": ["solaris"],
                "sample_n_views": torch.tensor(sample_n_views, dtype=torch.int64),
                "padding_mask": torch.zeros((1, *self.resolution_hw), dtype=torch.float32),
                "ref_cam_view_idx_sample_position": torch.tensor(-1, dtype=torch.int64),
                "front_cam_view_idx_sample_position": torch.tensor(0, dtype=torch.int64),
                "original_hw": torch.tensor([[original_h, original_w]], dtype=torch.int64),
            }

            if self.load_action:
                action_data = self._parse_action_data(data, frame_indices, total_frames=total_frames)
                if action_data is not None:
                    result.update(action_data)

            return result
        except Exception as e:
            log.error(f"Error extracting solaris 4-player frames for {data.get('__key__', '?')}: {e}")
            return None


def get_solaris_4player_dataset_info(
    data_root: str = SOLARIS_4PLAYER_DATA_ROOT,
) -> list[DatasetInfo]:

    wdinfo_paths = glob.glob(os.path.join(data_root, "**", "wdinfo.json"), recursive=True)
    if not wdinfo_paths:
        raise FileNotFoundError(f"No wdinfo.json found under {data_root}")

    return [
        DatasetInfo(
            object_store_config=config.ObjectStoreConfig(enabled=False),
            wdinfo=wdinfo_paths,
            per_dataset_keys=["video", "caption"],
            source="solaris_action_4player",
        )
    ]


def get_solaris_4player_video_loader(
    *,
    augmentation_config: Solaris4PlayerAugmentationConfig = Solaris4PlayerAugmentationConfig(),
    data_root: str = SOLARIS_4PLAYER_DATA_ROOT,
    is_train: bool = True,
    batch_size: int = 1,
    num_workers: int = 4,
    prefetch_factor: int | None = 1,
    max_shards: int = 0,
    shuffle_buffer_size: int = 512,
    **kwargs: Any,
) -> Any:

    if augmentation_config.take_half_width:
        raise ValueError(
            "Solaris4PlayerAugmentationConfig.take_half_width=True is not "
            "supported: the 4-player loader always emits all 4 player views. "
            "If you inherited this from a 2-player experiment, override it "
            "back to False (or remove it) in your experiment config."
        )
    dataset_info = get_solaris_4player_dataset_info(data_root)

    exclude_keys: set[str] = set()
    if is_train and augmentation_config.exclude_eval_keys:
        if SOLARIS_4PLAYER_EVAL_METADATA_CSV is None:
            raise RuntimeError(
                "[Solaris4P eval-filter] `exclude_eval_keys=True` but "
                "SOLARIS_4PLAYER_EVAL_METADATA_CSV is not configured. Set the "
                "constant in solaris_data_4player.py or set "
                "`augmentation_config.exclude_eval_keys=False`."
            )
        exclude_keys = _load_eval_exclude_keys(SOLARIS_4PLAYER_EVAL_METADATA_CSV)
        log.info(
            f"[Solaris4P eval-filter] ENABLED: loaded {len(exclude_keys)} eval "
            f"sample keys from {SOLARIS_4PLAYER_EVAL_METADATA_CSV} -- these "
            f"will be skipped during training."
        )
        _verify_exclude_keys_against_tars(exclude_keys, data_root)
    elif is_train:
        log.warning(
            "[Solaris4P eval-filter] DISABLED: training data may overlap with "
            "any future inference eval set. Set "
            "`augmentation_config.exclude_eval_keys=True` once a 4-player eval "
            "CSV is available."
        )

    augmentations: dict[str, Augmentor] = {
        "extract_frames": ExtractSolaris4PlayerFrames(
            num_frames=augmentation_config.num_video_frames,
            resolution_hw=augmentation_config.resolution_hw,
            fps_downsample_factor=augmentation_config.fps_downsample_factor,
            load_action=augmentation_config.load_action,
            random_sampling=augmentation_config.random_sampling,
            exclude_keys=exclude_keys,
        ),
    }

    video_data_config = DatasetConfig(
        keys=[],
        buffer_size=shuffle_buffer_size,
        streaming_download=True,
        dataset_info=dataset_info,
        distributor=ShardlistBasic(
            shuffle=is_train,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=False,
            is_infinite_loader=is_train,
        ),
        decoders=[
            video_naive_bytes(),
            json_decoder,
        ],
        augmentation=augmentations,
        remove_extension_from_keys=True,
    )

    dataset = SILDataset(
        config=video_data_config,
        handler=wds.warn_and_continue,
        decoder_handler=wds.warn_and_continue,
        detshuffle=False,
    )

    if max_shards > 0:
        dataset.wdinfo.tar_files = dataset.wdinfo.tar_files[:max_shards]
        dataset.wdinfo.total_key_count = min(
            dataset.wdinfo.total_key_count,
            max_shards * dataset.wdinfo.chunk_size,
        )

    log.info(
        f"Solaris 4-player dataset ready: "
        f"{len(dataset.wdinfo.tar_files)} shards, {dataset.wdinfo.total_key_count} keys"
    )

    loader = get_cached_replay_dataloader(
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=None,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,
        pin_memory=False,
        collate_fn=collate_fn,
        cache_replay_name=f"solaris_4player_video_dataloader_{'train' if is_train else 'val'}",
    )

    return loader
