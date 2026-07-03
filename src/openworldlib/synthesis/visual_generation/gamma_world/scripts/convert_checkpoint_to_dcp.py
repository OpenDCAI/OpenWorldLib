#!/usr/bin/env python
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
Convert a released ``.safetensors`` (or a ``.pt``/``.pth``) network checkpoint into a
DCP directory usable as a training-init checkpoint.

    python scripts/convert_checkpoint_to_dcp.py \
        --input  /path/to/causal/model.safetensors \
        --output /path/to/converted_causal_dcp

Writes ``<output>/model/`` (a model-only DCP, no optimizer/trainer state) and a
``<output>/conversion.json`` record. Keys are normalized to the ``net.`` prefix that
the trainer's checkpoint loader expects. For normal bidirectional/causal training pass
``checkpoint.load_path=<output>``; for DMD net init pass ``model.config.net_ckpt=<output>/model``.
"""

import argparse
import json
import os

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner, _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

_WRAPPER_PREFIXES = ("module.", "model.net.", "net.")


def load_raw_state_dict(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        print("weights_only=True failed; retrying with weights_only=False (the checkpoint is pickled).")
        obj = torch.load(path, map_location="cpu", weights_only=False)
    for wrapper_key in ("model", "state_dict", "generator", "generator_ema"):
        if isinstance(obj, dict) and wrapper_key in obj and isinstance(obj[wrapper_key], dict):
            return obj[wrapper_key]
    return obj


def normalize_key(key: str, prefix: str) -> str:
    for wrapper in _WRAPPER_PREFIXES:
        if key.startswith(wrapper):
            key = key[len(wrapper) :]
            break
    return prefix + key


def cast(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "preserve" or not tensor.is_floating_point():
        return tensor
    return tensor.to(torch.float32 if dtype == "float32" else torch.bfloat16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a .safetensors/.pt network checkpoint to DCP")
    parser.add_argument("--input", required=True, help=".safetensors / .pt / .pth network checkpoint")
    parser.add_argument("--output", required=True, help="output parent dir; the DCP is written to <output>/model")
    parser.add_argument("--prefix", default="net.", help="key prefix the trainer expects (default: net.)")
    parser.add_argument("--dtype", default="preserve", choices=["preserve", "float32", "bfloat16"])
    parser.add_argument("--no-strict", action="store_true", help="skip the read-back validation")
    parser.add_argument("--dry-run", action="store_true", help="inspect the key mapping without writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_raw_state_dict(args.input)
    state_dict = {normalize_key(key, args.prefix): cast(value, args.dtype) for key, value in raw.items()}

    print(f"{len(raw)} tensors -> {len(state_dict)} keys with prefix '{args.prefix}'")
    if args.dry_run:
        for key in list(state_dict)[:5]:
            print("  ", key, tuple(state_dict[key].shape), state_dict[key].dtype)
        return

    model_dir = os.path.join(args.output, "model")
    os.makedirs(model_dir, exist_ok=True)
    dcp.save(state_dict, storage_writer=FileSystemWriter(model_dir), planner=DefaultSavePlanner(), no_dist=True)

    with open(os.path.join(args.output, "conversion.json"), "w") as f:
        json.dump(
            {"input": args.input, "prefix": args.prefix, "dtype": args.dtype, "num_keys": len(state_dict)},
            f,
            indent=2,
        )

    if not args.no_strict:
        readback: dict = {}
        _load_state_dict(
            readback, storage_reader=FileSystemReader(model_dir), planner=_EmptyStateDictLoadPlanner(), no_dist=True
        )
        if len(readback) != len(state_dict):
            raise RuntimeError(f"DCP key count {len(readback)} != source {len(state_dict)}")
        bad = [key for key in readback if not key.startswith(args.prefix)]
        if bad:
            raise RuntimeError(f"DCP has keys without the '{args.prefix}' prefix: {bad[:5]}")
        print(f"validated: {len(readback)} keys, all prefixed '{args.prefix}'")

    print(f"wrote DCP to {model_dir}")


if __name__ == "__main__":
    main()
