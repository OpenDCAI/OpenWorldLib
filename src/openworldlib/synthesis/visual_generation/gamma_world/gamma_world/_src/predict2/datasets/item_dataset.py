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

import dataclasses
import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gamma_world._src.imaginaire.utils import log
from gamma_world._src.imaginaire.utils.easy_io import easy_io


@dataclasses.dataclass
class ItemDatasetConfig:
    path: str
    length: int


class PromptOnlyItemDataset(torch.utils.data.Dataset):
    """
    A simple dataset class for handling sequences of pickle data read from a specified path.
    It supports reading from local paths or S3. It currently handles prompts and T5 embeddings.
    The class is mainly for debug and testing purposes.

    Args:
        path (str): The path to the dataset source. Can be a local file system path or an S3 bucket path.
        start_index (int, optional): The starting index of the dataset to consider (inclusive). Defaults to 0.
        end_index (int, optional): The ending index of the dataset to consider (exclusive). Defaults to 32.
        max_t5_length (int, optional): The maximum length for T5 embedding. Defaults to 512. The sequence will be padded to this length.

    Example:
        >>> dataset = PromptOnlyItemDataset(path='local/path/to/dataset', start_index=100, end_index=1100)
        >>> len(dataset)
        1000
        >>> item = dataset[0]  # Retrieve the first item in the dataset
    """

    def __init__(
        self,
        path: str = "s3://bucket/edify_video/v4/validation/item_dataset/sora_veo_v1",
        start_index: int = 0,  # inclusive
        end_index=32,  # exclusive
        max_t5_length: int = 512,
        height=704,
        width=1280,
        num_video_frames=136,
    ):
        self.start_index = start_index
        self.end_index = end_index
        self.path = path
        self.height = height
        self.width = width
        self.num_video_frames = num_video_frames
        log.warning(
            f"using path: {path} and default s3 credentials in easy_io. It is user's responsibility to set up the correct credentials."
        )
        max_length = easy_io.load(os.path.join(self.path, "meta_info.json"))["length"]
        assert max_length >= end_index, f"dataset {path} max_length: {max_length}, end_index: {end_index}"
        self.max_t5_length = max_t5_length

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, idx):
        while True:
            try:
                return self._getitem(idx)
            except Exception as e:
                log.error(f"Error in __getitem__ {e}")
                continue

    def _getitem(self, idx):
        """
        Retrieves a specific pickle file based on its index, performing preprocessing
        on text and image data and handling padding as needed.

        Args:
            idx (int): Index of the dataset item to retrieve.

        Returns:
            dict: A dictionary containing preprocessed dataset items, including embeddings, masks,
                and potentially transformed images.
        """
        item_fp = os.path.join(self.path, f"{self.start_index + idx:06d}.pkl")
        item = easy_io.load(item_fp)

        if item is None:
            raise ValueError(f"item is None: {item_fp}")
        # t5
        mask = torch.LongTensor(self.max_t5_length).zero_()
        length = len(item["t5_text_embeddings"])
        mask[0:length] = 1
        item["t5_text_mask"] = mask
        if length < self.max_t5_length:
            item["t5_text_embeddings"] = F.pad(
                item["t5_text_embeddings"], (0, 0, 0, self.max_t5_length - length), value=0
            ).float()
        item["t5_text_embeddings"] = item["t5_text_embeddings"][: self.max_t5_length]

        item["prompt"] = item["prompt"]
        item["__idx__"] = idx + self.start_index
        # hard coded conds
        item["fps"] = 30.0
        item["image_size"] = torch.Tensor([self.height, self.width, self.height, self.width]).float()
        item["padding_mask"] = torch.zeros((1, self.height, self.width)).float()
        item["num_frames"] = torch.zeros((1)) + self.num_video_frames

        return item


class PromptImageItemDataset(PromptOnlyItemDataset):
    def __init__(
        self,
        path: str = "s3://bucket/edify_video/v4/validation/item_dataset/sora_veo_v1",
        num_videos: int = 32,
        start_index: int = 0,  # inclusive
        end_index=32,  # exclusive
        max_t5_length: int = 512,
        height=704,
        width=1280,
        num_max_frames=136,
        augs=[],
        aug_labels=[],
        sigma_max_list=[],
        control_weight_list=[],
        pad_all_videos_to_same_length=True,
    ):
        self.start_index = start_index
        self.end_index = end_index
        self.path = path
        self.num_videos = num_videos
        self.height = height
        self.width = width
        self.num_max_frames = num_max_frames
        log.warning(
            f"using path: {path} and default s3 credentials in easy_io. It is user's responsibility to set up the correct credentials."
        )
        self.max_t5_length = max_t5_length

        valid_idx = [i for i, aug in enumerate(augs) if len(aug.comb) > 0]
        self.augs = [augs[i] for i in valid_idx]
        self.aug_labels = [aug_labels[i] for i in valid_idx]
        self.sigma_max_list = sigma_max_list
        self.control_weight_list = control_weight_list
        self.pad_all_videos_to_same_length = pad_all_videos_to_same_length

    def _getitem(self, idx):
        cur_idx = self.start_index + idx
        file_idx = cur_idx % self.num_videos
        aug_idx = (cur_idx // self.num_videos) % len(self.augs)
        sigma_idx = (cur_idx // (self.num_videos * len(self.augs))) % len(self.sigma_max_list)
        control_weight_idx = (cur_idx // (self.num_videos * len(self.augs) * len(self.sigma_max_list))) % len(
            self.control_weight_list
        )
        file_paths = easy_io.list_dir_or_file(self.path, recursive=True, list_dir=False, suffix=".mp4")
        file_paths = sorted([path[:-4] for path in file_paths])
        item_fp = os.path.join(self.path, file_paths[file_idx] + ".pkl")
        item = easy_io.load(item_fp)
        if item is None:
            raise ValueError(f"item is None: {item_fp}")
        # t5
        mask = torch.LongTensor(self.max_t5_length).zero_()
        if isinstance(item["t5_text_embeddings"], dict):
            t5_text_embeddings = list(item["t5_text_embeddings"].values())
            t5_text_embeddings = torch.cat([t5[:, :, None] for t5 in t5_text_embeddings], dim=2)
        else:
            t5_text_embeddings = item["t5_text_embeddings"]
        length = len(t5_text_embeddings)
        mask[0:length] = 1
        item["t5_text_mask"] = mask
        if length < self.max_t5_length:
            t5_text_embeddings = F.pad(t5_text_embeddings, (0, 0, 0, self.max_t5_length - length), value=0).float()
        item["t5_text_embeddings"] = t5_text_embeddings[: self.max_t5_length]

        item["prompt"] = item["prompt"]
        item["__idx__"] = file_idx
        item["filename"] = file_paths[file_idx]
        # hard coded conds
        item["image_size"] = torch.Tensor([self.height, self.width, self.height, self.width]).float()
        item["padding_mask"] = torch.zeros((1, self.height, self.width)).float()

        # Augmentation
        aug = self.augs[aug_idx]
        aug_label = self.aug_labels[aug_idx]

        # video mp4 file
        item_fp = os.path.join(self.path, file_paths[file_idx] + ".mp4")
        log.info(f"reading from {item_fp}")
        video_np, video_meta_data = easy_io.load(item_fp)  # (TxHxWx3)
        log.info(f"finished reading from {item_fp}")
        num_max_frames = self.num_max_frames
        if not self.pad_all_videos_to_same_length:
            num_max_frames = min(num_max_frames, video_np.shape[0])
        resized_video = np.zeros((num_max_frames, self.height, self.width, 3), dtype=np.uint8)  # THWC
        log.info(f"loading video: {resized_video.shape} from {item_fp}")
        for i in range(min(num_max_frames, video_np.shape[0])):
            resized_video[i] = cv2.resize(video_np[i], (self.width, self.height), interpolation=cv2.INTER_AREA)
        log.info(f"finished loading video: {resized_video.shape} from {item_fp}")
        item["video"] = resized_video.transpose((3, 0, 1, 2))

        # convert np array to tensor
        item["raw_video"] = torch.from_numpy(resized_video.transpose((3, 0, 1, 2)))  # CTHW
        item["num_frames"] = torch.zeros((1)) + video_np.shape[0]

        item = aug(item)
        item["aug_label"] = aug_label
        item["sigma_max"] = self.sigma_max_list[sigma_idx]
        item["hint_key"], item["control_weight"] = self.control_weight_list[control_weight_idx]
        item["fps"] = int(video_meta_data.get("fps"))
        return item


class PromptVideoItemDataset(PromptOnlyItemDataset):
    """
    Dataset for evaluation with prompt and video pairs.
    Expects pickle files with 'prompt' field and corresponding MP4 video files.

    Note:
        We intentionally do NOT load saved T5 embeddings from disk.
        Embeddings will be computed online in the model/callback (see text2world_model and ValLossComputation).

    Args:
        path (str): Path to dataset containing .pkl and .mp4 files
        start_index (int): Starting index (inclusive)
        end_index (int): Ending index (exclusive)
        max_t5_length (int): Maximum T5 sequence length
        height (int): Video height after resizing
        width (int): Video width after resizing
        num_video_frames (int): Number of video frames to load
    """

    def __init__(
        self,
        path: str = "s3://bucket/projects/edify_video/v4/validation/item_dataset/ptbench_video_val",
        start_index: int = 0,
        end_index: int = 32,
        max_t5_length: int = 512,
        height: int = 704,
        width: int = 1280,
        num_video_frames: int = 136,
    ):
        self.path = path
        self.height = height
        self.width = width
        self.num_video_frames = num_video_frames
        self.max_t5_length = max_t5_length

        log.warning(
            f"using path: {path} and default s3 credentials in easy_io. "
            f"It is user's responsibility to set up the correct credentials."
        )

        # Discover available MP4 files and create file list
        file_paths = easy_io.list_dir_or_file(self.path, recursive=True, list_dir=False, suffix=".mp4")
        self.file_paths = sorted([path[:-4] for path in file_paths])  # Remove .mp4 extension

        # Apply start_index and end_index to the discovered files
        self.file_paths = self.file_paths[start_index:end_index]

        log.info(
            f"Build PromptVideoItemDataset with path: {path}, "
            f"discovered {len(self.file_paths)} files (after slicing {start_index}:{end_index}), "
            f"video shape: ({num_video_frames}, {height}, {width})"
        )

    def __len__(self):
        return len(self.file_paths)

    def _getitem(self, idx):
        """
        Load a single item with prompt and video. T5 embeddings are NOT loaded.
        """
        # Use discovered file paths instead of sequential numbering
        file_idx = idx
        if file_idx >= len(self.file_paths):
            raise IndexError(f"Index {file_idx} out of range for {len(self.file_paths)} files")

        pkl_path = os.path.join(self.path, self.file_paths[file_idx] + ".pkl")
        video_path = os.path.join(self.path, self.file_paths[file_idx] + ".mp4")

        # Load pickle data (expects at least 'prompt')
        item = easy_io.load(pkl_path)
        if item is None:
            raise ValueError(f"item is None: {pkl_path}")
        # Do not load or pad t5_text_embeddings here; we switch to online computation

        # Load and process video
        log.info(f"Loading video from {video_path}")
        video_np, video_meta_data = easy_io.load(video_path)  # (T, H, W, 3)
        log.info(f"Loaded video with shape {video_np.shape} from {video_path}")

        # Resize video frames
        num_frames_to_load = min(self.num_video_frames, video_np.shape[0])
        resized_video = np.zeros((self.num_video_frames, self.height, self.width, 3), dtype=np.uint8)

        for i in range(min(num_frames_to_load, video_np.shape[0])):
            resized_video[i] = cv2.resize(video_np[i], (self.width, self.height), interpolation=cv2.INTER_AREA)

        # Convert to tensor format (C, T, H, W)
        video_cthw = resized_video.transpose((3, 0, 1, 2))  # CTHW
        item["video"] = torch.from_numpy(video_cthw)  # uint8 tensor, normalized later on GPU
        item["raw_video"] = item["video"].clone()

        # Set metadata
        # Keep prompt as-is. Online text embedding will use this field
        item["prompt"] = item["prompt"]
        item["__idx__"] = idx  # Use idx instead of item_idx
        item["__file__"] = self.file_paths[file_idx]  # Store actual file name
        item["fps"] = float(video_meta_data.get("fps", 30.0))
        item["image_size"] = torch.Tensor([self.height, self.width, self.height, self.width]).float()
        item["padding_mask"] = torch.zeros((1, self.height, self.width)).float()
        item["num_frames"] = torch.zeros((1)) + video_np.shape[0]

        log.info(
            f"Processed item {self.file_paths[file_idx]}: video shape {item['raw_video'].shape}, prompt length {len(item['prompt'])}"
        )

        return item


class PromptLVGItemDataset(PromptOnlyItemDataset):
    def __init__(
        self,
        path: str = "s3://bucket/projects/edify_video/v4/validation/item_dataset/lvg_video_extend_v0_val",
        start_index: int = 0,  # inclusive
        end_index=32,  # exclusive
        max_t5_length: int = 512,
        height=704,
        width=1280,
        video_length=121,  # length of each video clip
        num_overlap_frames=4,  # number of frames to encode
    ):
        self.start_index = start_index
        self.end_index = end_index
        self.path = path
        self.height = height
        self.width = width
        self.video_length = video_length
        log.warning(
            f"using path: {path} and default s3 credentials in easy_io. It is user's responsibility to set up the correct credentials."
        )
        log.info(
            f"Build item dataset with path: {path}, and start_index: {start_index}, end_index: {end_index}. video shape THW = ({video_length}, {height}, {width})"
        )
        self.max_t5_length = max_t5_length
        self.num_overlap_frames = num_overlap_frames

    def _getitem(self, idx):
        cur_idx = self.start_index + idx
        file_idx = cur_idx

        item_fp = os.path.join(self.path, f"{file_idx:06d}.pkl")
        item = easy_io.load(item_fp)
        input_image_or_video_ext = item["ext"]
        input_image_or_video_path = item_fp.replace(".pkl", f".{input_image_or_video_ext}")
        if item is None:
            raise ValueError(f"item is None: {item_fp}")
        # t5
        mask = torch.LongTensor(self.max_t5_length).zero_()
        length = len(item["t5_text_embeddings"])
        mask[0:length] = 1
        item["t5_text_mask"] = mask
        if length < self.max_t5_length:
            item["t5_text_embeddings"] = F.pad(
                item["t5_text_embeddings"], (0, 0, 0, self.max_t5_length - length), value=0
            ).float()
        item["t5_text_embeddings"] = item["t5_text_embeddings"][: self.max_t5_length]

        item["prompt"] = item["prompt"]
        item["__idx__"] = file_idx
        # hard coded conds
        item["fps"] = 30.0
        item["image_size"] = torch.Tensor([self.height, self.width, self.height, self.width]).float()
        item["padding_mask"] = torch.zeros((1, self.height, self.width)).float()
        item["num_frames"] = torch.zeros((1)) + self.video_length

        # video mp4 file
        if input_image_or_video_path.endswith(".mp4"):
            video_np, video_meta_data = easy_io.load(input_image_or_video_path)  # (TxHxWx3)
            assert len(video_np) > self.num_overlap_frames, (
                f"to support num_overlap_frames={self.num_overlap_frames}, need at least {self.num_overlap_frames} frames, but current video only have {len(video_np)} frames"
            )
            video_np = video_np[-self.num_overlap_frames :]  # Select the last num_overlap_frames frames
        else:
            video_np = np.array(easy_io.load(input_image_or_video_path))[None]  # (1xHxWx3)
            assert self.num_overlap_frames == 1, (
                f"image data is not supported when num_overlap_frames({self.num_overlap_frames}) > 1, need to set num_overlap_frames=1 or use video input"
            )

        resized_video = np.zeros((self.video_length, self.height, self.width, 3), dtype=np.uint8)
        for i in range(min(video_np.shape[0], resized_video.shape[0])):
            resized_video[i] = cv2.resize(video_np[i], (self.width, self.height), interpolation=cv2.INTER_AREA)
        item["video"] = resized_video.transpose((3, 0, 1, 2))
        return item


def calculate_indices(dataset_length: int, world_size: int, rank: int) -> Tuple[int, int, bool]:
    """
    Calculate the start and end indices for a given rank in a distributed setting.

    Args:
        dataset_length (int): The total length of the dataset.
        world_size (int): The number of distributed processes.
        rank (int): The rank of the current process.

    Returns:
        Tuple[int, int]: A tuple containing the start index (inclusive) and end index (exclusive).
    """
    # Calculate the number of samples per rank
    samples_per_rank = dataset_length // world_size
    remainder = dataset_length % world_size
    is_overflow = False

    # Calculate the start and end indices for this rank
    start_index = rank * samples_per_rank + min(rank, remainder)
    end_index = start_index + samples_per_rank + (1 if rank < remainder else 0)
    # take care of corner case where dataset_length is smaller than world size.
    if start_index >= dataset_length:  # when number of samples are not enough
        start_index = 0
        end_index = 1
        is_overflow = True

    return start_index, end_index, is_overflow
