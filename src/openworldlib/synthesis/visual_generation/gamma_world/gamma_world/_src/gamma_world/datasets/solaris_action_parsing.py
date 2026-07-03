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



import json
from typing import Optional

import torch

from gamma_world._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from gamma_world._src.imaginaire.utils import log


class SolarisActionParsing(Augmentor):


    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.num_frames = args.get("num_frames", 121)
        self.keyboard_dim = args.get("keyboard_dim", 23)
        self.camera_dim = args.get("camera_dim", 2)
        self.parse_plucker_fields = args.get("parse_plucker_fields", False)

    def _decode_json(self, raw_data, data_dict: dict, key: str) -> Optional[dict]:
        try:
            if isinstance(raw_data, dict):
                return raw_data
            elif isinstance(raw_data, bytes):
                return json.loads(raw_data.decode("utf-8"))
            elif isinstance(raw_data, str):
                return json.loads(raw_data)
            else:
                log.warning(
                    f"Unexpected solaris action type: {type(raw_data)}, "
                    f"url: {data_dict.get('__url__', 'unknown')}, key: {data_dict.get('__key__', 'unknown')}",
                    rank0_only=False,
                )
                return None
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning(
                f"Failed to parse solaris action JSON for {key}: {e}, "
                f"url: {data_dict.get('__url__', 'unknown')}, key: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

    def _pad_or_truncate(self, seq: list, target_len: int, dim: int) -> list:

        if len(seq) >= target_len:
            return seq[:target_len]
        pad = [0.0] * dim
        return seq + [pad] * (target_len - len(seq))

    def _parse_solaris_json(self, raw_data, data_dict: dict, key: str) -> Optional[dict]:
        action_json = self._decode_json(raw_data, data_dict, key)
        if action_json is None:
            return None

        keyboard = action_json.get("keyboard")
        camera = action_json.get("camera")
        if keyboard is None or camera is None:
            log.warning(
                f"Missing keyboard/camera in solaris action for {key}, "
                f"url: {data_dict.get('__url__', 'unknown')}, key: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        keyboard = self._pad_or_truncate(keyboard, self.num_frames, self.keyboard_dim)
        camera = self._pad_or_truncate(camera, self.num_frames, self.camera_dim)

        result = {
            "keyboard": torch.tensor(keyboard, dtype=torch.float32),
            "camera": torch.tensor(camera, dtype=torch.float32),
        }

        if self.parse_plucker_fields:
            camera_raw = action_json.get("camera_raw")
            positions = action_json.get("positions")
            if camera_raw is None or positions is None:
                log.warning(
                    f"Missing camera_raw/positions for plucker mode in {key}, "
                    f"url: {data_dict.get('__url__', 'unknown')}, key: {data_dict.get('__key__', 'unknown')}",
                    rank0_only=False,
                )
                return None
            camera_raw = self._pad_or_truncate(camera_raw, self.num_frames, 2)
            positions = self._pad_or_truncate(positions, self.num_frames, 3)
            result["camera_raw"] = torch.tensor(camera_raw, dtype=torch.float32)
            result["positions"] = torch.tensor(positions, dtype=torch.float32)

        return result

    def __call__(self, data_dict: dict) -> Optional[dict]:
        for key in self.input_keys:
            if key not in data_dict:
                log.warning(
                    f"Solaris action key '{key}' not found in data_dict, "
                    f"url: {data_dict.get('__url__', 'unknown')}, key: {data_dict.get('__key__', 'unknown')}",
                    rank0_only=False,
                )
                return None

            parsed = self._parse_solaris_json(data_dict[key], data_dict, key)
            if parsed is None:
                return None

            data_dict[f"{key}_keyboard"] = parsed["keyboard"]
            data_dict[f"{key}_camera"] = parsed["camera"]

            if self.parse_plucker_fields:
                data_dict[f"{key}_camera_raw"] = parsed["camera_raw"]
                data_dict[f"{key}_positions"] = parsed["positions"]

        return data_dict