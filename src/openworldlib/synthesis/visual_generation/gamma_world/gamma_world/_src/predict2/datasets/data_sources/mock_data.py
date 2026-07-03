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

"""
Mock data for fast testing and debugging. See `projects/cosmos/diffusion/v1/datasets/mock_data_test.py` for usage.
"""

from functools import partial

import torch

from gamma_world._src.imaginaire.datasets.mock_dataset import CombinedDictDataset, LambdaDataset
from gamma_world._src.predict2.datasets.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO


def get_image_dataset(
    resolution: str = "512",
    len_t5: int = 512,
    t5_dim: int = 1024,
    **kwargs,
):
    h, w = IMAGE_RES_SIZE_INFO[resolution]["9,16"]
    del kwargs
    return CombinedDictDataset(
        **{
            "images": LambdaDataset(partial(torch.randn, size=(3, h, w))),
            "t5_text_embeddings": LambdaDataset(partial(torch.randn, size=(len_t5, t5_dim))),
            "t5_text_mask": LambdaDataset(partial(torch.randint, low=0, high=2, size=(len_t5,), dtype=torch.int64)),
            "fps": LambdaDataset(lambda: 1.0),
            "image_size": LambdaDataset(partial(torch.tensor, [h, w, h, w], dtype=torch.float32)),
            "num_frames": LambdaDataset(lambda: 1),
            "padding_mask": LambdaDataset(partial(torch.zeros, size=(1, h, w))),
            "dataset_name": LambdaDataset(lambda: "image_data"),
            "raw_captions": LambdaDataset(lambda: "placeholder"),
            "ai_caption": LambdaDataset(lambda: "placeholder"),  # The text encoder augmentation method uses this field
            "__url__": LambdaDataset(lambda: "placeholder"),
            "__key__": LambdaDataset(lambda: "placeholder"),
        }
    )


def get_video_dataset(
    num_video_frames: int,
    resolution: str = "512",
    len_t5: int = 512,
    t5_dim: int = 1024,
    **kwargs,
):
    del kwargs
    h, w = VIDEO_RES_SIZE_INFO[resolution]["9,16"]

    def video_fn():
        return torch.randint(0, 255, size=(3, num_video_frames, h, w)).to(dtype=torch.uint8)

    return CombinedDictDataset(
        **{
            "video": LambdaDataset(video_fn),
            "t5_text_embeddings": LambdaDataset(partial(torch.randn, size=(len_t5, t5_dim))),
            "t5_text_mask": LambdaDataset(partial(torch.randint, low=0, high=2, size=(len_t5,), dtype=torch.int64)),
            "fps": LambdaDataset(lambda: 24.0),
            "image_size": LambdaDataset(partial(torch.tensor, [h, w, h, w], dtype=torch.float32)),
            "num_frames": LambdaDataset(lambda: num_video_frames),
            "padding_mask": LambdaDataset(partial(torch.zeros, size=(1, h, w))),
            "ai_caption": LambdaDataset(lambda: "placeholder"),
            "dataset_name": LambdaDataset(lambda: "video_data"),
            "chunk_index": LambdaDataset(lambda: 0),
            "frame_end": LambdaDataset(lambda: 0),
            "frame_start": LambdaDataset(lambda: 0),
            "n_orig_video_frames": LambdaDataset(lambda: 0),
            "__url__": LambdaDataset(lambda: "placeholder"),
            "__key__": LambdaDataset(lambda: "placeholder"),
        }
    )


def get_video_ctrlnet_dataset(
    h: int,
    w: int,
    num_video_frames: int,
    len_t5: int = 512,
    hint_key: str = "control_input_canny",
    **kwargs,
):
    del kwargs

    def video_fn():
        return torch.randint(0, 255, size=(3, num_video_frames, h, w)).to(dtype=torch.uint8)

    return CombinedDictDataset(
        **{
            "video": LambdaDataset(video_fn),
            hint_key: LambdaDataset(video_fn),
            "t5_text_embeddings": LambdaDataset(partial(torch.randn, size=(len_t5, 1024), dtype=torch.bfloat16)),
            "t5_text_mask": LambdaDataset(partial(torch.randint, low=0, high=2, size=(len_t5,), dtype=torch.int64)),
            "fps": LambdaDataset(lambda: 24.0),
            "image_size": LambdaDataset(partial(torch.tensor, [h, w, h, w], dtype=torch.float32)),
            "num_frames": LambdaDataset(lambda: num_video_frames),
            "padding_mask": LambdaDataset(partial(torch.zeros, size=(1, h, w), dtype=torch.bfloat16)),
        }
    )


def get_image_ctrlnet_dataset(
    h: int,
    w: int,
    len_t5: int = 512,
    hint_key: str = "control_input_canny",
    **kwargs,
):
    del kwargs
    return CombinedDictDataset(
        **{
            "images": LambdaDataset(partial(torch.randn, size=(3, h, w))),
            hint_key: LambdaDataset(partial(torch.randn, size=(3, h, w))),
            "t5_text_embeddings": LambdaDataset(partial(torch.randn, size=(len_t5, 1024))),
            "t5_text_mask": LambdaDataset(partial(torch.randint, low=0, high=2, size=(len_t5,), dtype=torch.int64)),
            "fps": LambdaDataset(lambda: 1.0),
            "image_size": LambdaDataset(partial(torch.tensor, [h, w, h, w], dtype=torch.float32)),
            "num_frames": LambdaDataset(lambda: 1),
            "padding_mask": LambdaDataset(partial(torch.zeros, size=(1, h, w))),
        }
    )
