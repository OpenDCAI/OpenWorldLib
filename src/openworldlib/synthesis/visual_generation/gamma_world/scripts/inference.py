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
import argparse
import json
import os
from typing import Optional

import mediapy as media
import torch

from gamma_world._src.gamma_world.inference.inference_bidirectional import BidirectionalInference
from gamma_world._src.gamma_world.inference.inference_i2v import I2VInference
from gamma_world._src.gamma_world.inference.model_specs import MODEL_SPECS
from gamma_world._src.imaginaire.utils import log
from gamma_world._src.imaginaire.visualize.video import save_img_or_video

def load_action(path: str, num_frames: int):
    data = json.load(open(path))
    keyboard = data["keyboard"]
    keyboard_tensor = torch.tensor(
        [keyboard[i] if i < len(keyboard) else [0.0] * 23 for i in range(num_frames)], dtype=torch.float32
    ).unsqueeze(0)
    camera = data.get("camera")
    camera_tensor = None
    if camera is not None:
        camera_tensor = torch.tensor(
            [camera[i] if i < len(camera) else [0.0, 0.0] for i in range(num_frames)], dtype=torch.float32
        ).unsqueeze(0)
    return keyboard_tensor, camera_tensor

def format_hydra_value(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_hydra_value(item) for item in value) + "]"
    if value == "":
        return "''"
    return str(value)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gamma-World inference")
    parser.add_argument("--mode", required=True, choices=list(MODEL_SPECS))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="net .safetensors (local path or hf:// URI); defaults to the selected mode checkpoint",
    )
    parser.add_argument("--vae", default=None, help="VAE path (local or hf://); defaults to the public HF tokenizer")
    parser.add_argument("--text-encoder", default=None, help="text-encoder path (local dir or hf://); defaults to the public HF model")
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="directory of example samples with first_frame image or first_frame_path.txt and actions",
    )
    parser.add_argument("--n-players", type=int, default=2, help="number of players in each eval-dir first_frame image")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, default=189)
    parser.add_argument("--num-conditional-frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--num-steps", type=int, default=None, help="denoising steps (ignored by causal_few_step)")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if args.n_players <= 0:
        parser.error("--n-players must be positive")
    return args

def load_example_sample(
    sample_dir: str,
    n_players: int,
    height: int,
    width: int,
    num_frames: int,
    prompt_override: Optional[str] = None,
):
    image_path = None
    for filename in ("first_frame.png", "first_frame.jpg", "first_frame.jpeg"):
        candidate = os.path.join(sample_dir, filename)
        if os.path.exists(candidate):
            image_path = candidate
            break
    pointer_path = os.path.join(sample_dir, "first_frame_path.txt")
    if image_path is None and os.path.exists(pointer_path):
        with open(pointer_path, "r", encoding="utf-8") as f:
            referenced_path = f.read().strip()
        if not referenced_path:
            raise ValueError(f"{pointer_path} is empty")
        image_path = referenced_path
        if not os.path.isabs(image_path):
            image_path = os.path.normpath(os.path.join(sample_dir, image_path))
    if image_path is None:
        raise FileNotFoundError(f"No first_frame image or first_frame_path.txt found under {sample_dir}")
    image = media.read_image(image_path)
    image_w = image.shape[1]
    if image_w % n_players != 0:
        raise ValueError(f"{image_path} width {image_w} is not divisible by n_players={n_players}")

    per_view_w = image_w // n_players
    images = [
        media.resize_image(image[:, player_idx * per_view_w : (player_idx + 1) * per_view_w], (height, width))
        for player_idx in range(n_players)
    ]

    if prompt_override is not None:
        prompt = prompt_override
    else:
        prompt_path = os.path.join(sample_dir, "prompt.txt")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"{prompt_path} not found; pass --prompt to override all eval samples")
        with open(prompt_path) as f:
            prompt = f.read().strip()

    action_paths = [os.path.join(sample_dir, f"action_{idx}.json") for idx in range(n_players)]
    if n_players == 2 and not all(os.path.exists(path) for path in action_paths):
        legacy_paths = [os.path.join(sample_dir, "action_left.json"), os.path.join(sample_dir, "action_right.json")]
        if all(os.path.exists(path) for path in legacy_paths):
            action_paths = legacy_paths

    actions = None
    if all(os.path.exists(path) for path in action_paths):
        actions = [load_action(path, num_frames) for path in action_paths]

    return images, prompt, actions

def generate_sample(engine, spec, images, prompt, actions, output_dir, args, num_steps) -> None:
    batch = engine.build_inference_batch(
        images, prompt, actions, num_frames=args.num_frames, num_conditional_frames=args.num_conditional_frames
    )
    engine.clear_cache()

    if spec.sampler == "bidirectional":
        video = engine.generate_single_shot(batch, guidance=args.guidance, seed=args.seed, num_steps=num_steps, shift=args.shift)
    else:
        video = engine.generate_from_batch(batch, guidance=args.guidance, seed=args.seed, num_steps=num_steps, shift=args.shift)

    if engine.rank0:
        video = ((video + 1.0) / 2.0).clamp(0, 1)
        os.makedirs(output_dir, exist_ok=True)
        save_img_or_video(video[0], os.path.join(output_dir, "generated"), fps=args.fps)
        log.info(f"Saved {tuple(video.shape)} to {output_dir}/generated")

def main() -> None:
    os.environ.setdefault("NVTE_FUSED_ATTN", "0")
    torch.set_grad_enabled(False)
    args = parse_args()
    spec = MODEL_SPECS[args.mode]
    checkpoint = args.checkpoint or spec.default_checkpoint
    num_steps = args.num_steps if args.num_steps is not None else spec.default_num_steps

    experiment_opts = [f"{key}={format_hydra_value(value)}" for key, value in spec.config_overrides.items()]
    engine_cls = BidirectionalInference if spec.sampler == "bidirectional" else I2VInference
    engine = engine_cls(
        experiment_name=spec.experiment,
        ckpt_path=checkpoint,
        config_file=spec.config_file,
        guidance=args.guidance,
        shift=args.shift,
        num_sampling_steps=num_steps or 35,
        seed=args.seed,
        context_parallel_size=args.context_parallel_size,
        experiment_opts=experiment_opts,
        vae_pth=args.vae,
        text_encoder_pth=args.text_encoder,
    )
    engine.fps = args.fps

    sample_dirs = []
    for root, _, files in os.walk(args.eval_dir):
        if any(name in files for name in ("first_frame.png", "first_frame.jpg", "first_frame.jpeg", "first_frame_path.txt")):
            sample_dirs.append(root)
    sample_dirs = sorted(sample_dirs)
    if args.max_eval_samples is not None:
        sample_dirs = sample_dirs[: args.max_eval_samples]
    if not sample_dirs:
        raise ValueError(f"No samples with first_frame image or first_frame_path.txt found under {args.eval_dir}")

    for sample_dir in sample_dirs:
        rel_name = os.path.relpath(sample_dir, args.eval_dir)
        images, prompt, actions = load_example_sample(
            sample_dir, args.n_players, args.height, args.width, args.num_frames, args.prompt
        )
        generate_sample(engine, spec, images, prompt, actions, os.path.join(args.output, rel_name), args, num_steps)

if __name__ == "__main__":
    main()
