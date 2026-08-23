"""
Single-GPU:
  CUDA_VISIBLE_DEVICES=1 python examples/run_rolling_forcing_from_json.py \
    --json_path worldeval_data/embodied_example_compressed/case1.json

# Memory note:
#   Rolling Forcing does not have LongLive-style prompt switch recache. This
#   script converts all JSON chunks into one long prompt and runs exactly one
#   native RollingForcing inference call. The rolling-window denoising, KV
#   cache, and attention-sink behavior therefore stay inside the migrated
#   Rolling Forcing runtime, matching the upstream inference path.
"""

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from json_chunk_utils import (
    add_organized_output_args,
    append_chunk_timestamp,
    build_chunk_prompt,
    build_global_prompt,
    choose_existing_path,
    count_video_frames,
    copy_organized_reference_files,
    load_records,
    prepare_organized_output_paths,
    resolve_chunk_num_frames,
    resolve_companion_video,
    should_use_chunk_prompts,
    write_chunk_timestamp_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Rolling Forcing from a chunked JSON. All JSON records are "
            "merged into one long-video prompt and generated in one native "
            "Rolling Forcing inference call so the rolling window and KV cache "
            "memory match upstream behavior."
        )
    )
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--wan_model_path", type=str, default=None)
    parser.add_argument("--generator_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default=None)
    add_organized_output_args(parser, "rolling_forcing_from_json")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=None, help="Optional per-record frame count override.")
    parser.add_argument("--default_chunk_frames", type=int, default=126)
    parser.add_argument("--min_chunk_frames", type=int, default=3)
    parser.add_argument(
        "--copy_reference",
        action="store_true",
        help="Copy the companion mp4 next to the organized output if it exists.",
    )
    return parser.parse_args()


def save_uint8_video(video_frames, output_path, fps=16):
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.detach().cpu()
        if video_frames.ndim == 5:
            video_frames = video_frames[0]
        video_frames = video_frames.numpy()

    with imageio.get_writer(output_path, fps=fps, quality=8) as writer:
        for frame in video_frames:
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)


def build_rolling_forcing_chunk_specs(records, args):
    global_prompt = build_global_prompt(records, args.prompt, max_chars=1800)
    use_chunk_prompts = should_use_chunk_prompts(records)

    chunk_specs = []
    for idx, record in enumerate(records):
        prompt = build_chunk_prompt(
            record,
            global_prompt=global_prompt,
            prompt_override=args.prompt,
            use_chunk_prompts=use_chunk_prompts,
        )
        if not prompt:
            print(f"Skipping record {idx}: no prompt text.")
            continue

        chunk_num_frames = args.num_frames if args.num_frames is not None else resolve_chunk_num_frames(
            record,
            fps=args.fps,
            default_chunk_frames=args.default_chunk_frames,
            min_chunk_frames=args.min_chunk_frames,
        )
        chunk_specs.append(
            {
                "idx": idx,
                "prompt": prompt,
                "num_frames": int(chunk_num_frames),
                "record": record,
            }
        )

    if not chunk_specs:
        raise ValueError("No valid prompt chunks found in JSON.")

    return chunk_specs, global_prompt, use_chunk_prompts


def resolve_rolling_forcing_request(chunk_specs, global_prompt, prompt_override=None, num_frame_per_block=3):
    total_num_frames = sum(int(chunk["num_frames"]) for chunk in chunk_specs)
    if total_num_frames % num_frame_per_block != 0:
        total_num_frames = ((total_num_frames + num_frame_per_block - 1) // num_frame_per_block) * num_frame_per_block

    if prompt_override:
        return prompt_override.strip(), total_num_frames

    chunk_prompts = []
    seen = set()
    for chunk in chunk_specs:
        prompt = " ".join(str(chunk["prompt"]).strip().split())
        if prompt and prompt not in seen:
            chunk_prompts.append(prompt)
            seen.add(prompt)

    if len(chunk_prompts) <= 1:
        return global_prompt, total_num_frames

    temporal_prompt = " ".join(
        f"Segment {idx + 1}: {prompt}" for idx, prompt in enumerate(chunk_prompts)
    )
    return temporal_prompt[:4000], total_num_frames


def no_reference_resolver(json_path):
    return json_path


def main():
    args = parse_args()

    from openworldlib.pipelines.rolling_forcing.pipeline_rolling_forcing import RollingForcingPipeline

    json_path = Path(args.json_path)
    records = load_records(json_path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"No valid records found in {json_path}")

    companion_resolver = resolve_companion_video if args.copy_reference else no_reference_resolver
    output_dir, output_path, json_copy_path, reference_video_path, reference_copy_path = prepare_organized_output_paths(
        args,
        json_path,
        companion_resolver=companion_resolver,
    )

    model_path = args.model_path or choose_existing_path(
        ["checkpoints/RollingForcing"],
        "checkpoints/RollingForcing",
    )
    wan_model_path = args.wan_model_path or choose_existing_path(
        ["checkpoints/Wan2.1-T2V-1.3B"],
        "checkpoints/Wan2.1-T2V-1.3B",
    )

    required_components = {"wan_model_path": wan_model_path}
    if args.generator_ckpt:
        required_components["generator_ckpt"] = args.generator_ckpt

    chunk_specs, global_prompt, use_chunk_prompts = build_rolling_forcing_chunk_specs(records, args)
    rolling_prompt, total_num_frames = resolve_rolling_forcing_request(
        chunk_specs,
        global_prompt=global_prompt,
        prompt_override=args.prompt,
        num_frame_per_block=3,
    )
    chunk_timestamps = []
    frame_cursor = 0
    for chunk in chunk_specs:
        append_chunk_timestamp(
            chunk_timestamps,
            record=chunk["record"],
            chunk_index=chunk["idx"],
            fps=args.fps,
            frame_start=frame_cursor,
            frame_count=chunk["num_frames"],
            requested_frames=chunk["num_frames"],
        )
        frame_cursor += chunk["num_frames"]

    print(f"Loaded JSON: {json_path}")
    print(f"Output directory: {output_dir}")
    print(f"Using model_path: {model_path}")
    print(f"Using wan_model_path: {wan_model_path}")
    print(f"Loaded records: {len(records)}")
    print(f"Prompt strategy: {'per-chunk prompts' if use_chunk_prompts else 'single global prompt'}")
    print(f"Global prompt snippet: {global_prompt[:200]!r}")
    print(f"RollingForcing total_num_frames={total_num_frames}")
    print(f"RollingForcing prompt snippet: {rolling_prompt[:240]!r}")
    print("Memory mode: one native RollingForcing inference call with rolling-window KV cache and attention sink.")

    pipeline = RollingForcingPipeline.from_pretrained(
        model_path=model_path,
        required_components=required_components,
        device=args.device,
    )

    for chunk in chunk_specs:
        print(
            f"[Chunk {chunk['idx']}] num_frames={chunk['num_frames']}, "
            f"prompt={chunk['prompt'][:140]!r}"
        )

    output_video = pipeline(
        prompt=rolling_prompt,
        num_frames=total_num_frames,
        seed=args.seed,
    )

    save_uint8_video(output_video, str(output_path), fps=args.fps)
    timestamp_path = write_chunk_timestamp_manifest(
        args,
        output_path=output_path,
        fps=args.fps,
        chunks=chunk_timestamps,
        total_frames=count_video_frames(output_video),
        notes=(
            ["total_frames includes RollingForcing block-size padding beyond the last chunk."]
            if total_num_frames > frame_cursor
            else None
        ),
    )
    if args.copy_reference:
        copy_organized_reference_files(json_path, json_copy_path, reference_video_path, reference_copy_path)
    else:
        json_copy_path.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Saved generated video: {output_path}")
    print(f"Saved chunk timestamps to: {timestamp_path}")
    print(f"Saved prompt JSON copy: {json_copy_path}")


if __name__ == "__main__":
    main()
