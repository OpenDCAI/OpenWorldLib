
"""
 Single-GPU:
   CUDA_VISIBLE_DEVICES=0 python examples/run_longlive_from_json.py \
     --json_path worldeval_data/gaming_example_compressed/case5.json
#
# With shorter per-chunk override for a quick smoke run:
#   CUDA_VISIBLE_DEVICES=0 python examples/run_longlive_from_json.py \
#     --json_path worldeval_data/gaming_example_compressed/case5.json \
#     --num_frames 24 \
#     --output_dir_name longlive_case5_smoke
#
# Memory note:
#   This script keeps all JSON chunks inside one LongLive interactive inference
#   call. LongLive's native InteractiveCausalInferencePipeline owns the KV cache
#   across chunks and calls _recache_after_switch() at switch_frame_indices.

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
            "Run LongLive from a chunked JSON. Each JSON record becomes one "
            "LongLive prompt segment, and all segments are generated in one "
            "native interactive inference pass so KV cache and recache behavior "
            "matches upstream LongLive."
        )
    )
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--wan_model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default=None)
    add_organized_output_args(parser, "longlive_from_json")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=None, help="Optional per-record frame count override.")
    parser.add_argument("--default_chunk_frames", type=int, default=24)
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


def build_longlive_chunk_specs(records, args):
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


def resolve_longlive_timeline(chunk_specs, num_frame_per_block=3):
    prompts = [chunk["prompt"] for chunk in chunk_specs]
    switch_frame_indices = []
    current_frame = 0

    for chunk in chunk_specs[:-1]:
        current_frame += int(chunk["num_frames"])
        switch_frame_indices.append(current_frame)

    total_num_frames = sum(int(chunk["num_frames"]) for chunk in chunk_specs)
    if total_num_frames % num_frame_per_block != 0:
        total_num_frames = ((total_num_frames + num_frame_per_block - 1) // num_frame_per_block) * num_frame_per_block

    switch_frame_indices = [
        min(max(1, int(index)), total_num_frames - 1)
        for index in switch_frame_indices
    ]

    return prompts, switch_frame_indices, total_num_frames


def build_longlive_chunk_timestamps(chunk_specs, switch_frame_indices, total_frames, fps):
    boundaries = [0, *[int(index) for index in switch_frame_indices], int(total_frames)]
    chunk_timestamps = []
    for chunk, frame_start, frame_end in zip(chunk_specs, boundaries[:-1], boundaries[1:]):
        nominal_frames = int(chunk["num_frames"])
        exported_frames = max(0, frame_end - frame_start)
        append_chunk_timestamp(
            chunk_timestamps,
            record=chunk["record"],
            chunk_index=chunk["idx"],
            fps=fps,
            frame_start=frame_start,
            frame_count=exported_frames,
            requested_frames=nominal_frames,
            extra={
                "prompt": chunk["prompt"],
                "switch_frame_start": frame_start,
                "switch_frame_end": frame_end,
                "nominal_frames": nominal_frames,
                "padding_frames": max(0, exported_frames - nominal_frames),
            },
        )
    return chunk_timestamps


def no_reference_resolver(json_path):
    return json_path


def main():
    args = parse_args()

    from openworldlib.pipelines.longlive.pipeline_longlive import LongLivePipeline

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
        ["checkpoints/LongLive", "Efficient-Large-Model/LongLive-1.3B"],
        "checkpoints/LongLive",
    )
    wan_model_path = args.wan_model_path or choose_existing_path(
        ["checkpoints/Wan2.1-T2V-1.3B", "Wan-AI/Wan2.1-T2V-1.3B"],
        "checkpoints/Wan2.1-T2V-1.3B",
    )

    chunk_specs, global_prompt, use_chunk_prompts = build_longlive_chunk_specs(records, args)
    prompts, switch_frame_indices, total_num_frames = resolve_longlive_timeline(
        chunk_specs,
        num_frame_per_block=3,
    )

    print(f"Loaded JSON: {json_path}")
    print(f"Output directory: {output_dir}")
    print(f"Using model_path: {model_path}")
    print(f"Using wan_model_path: {wan_model_path}")
    print(f"Loaded records: {len(records)}")
    print(f"Prompt strategy: {'per-chunk prompts' if use_chunk_prompts else 'single global prompt'}")
    print(f"Global prompt snippet: {global_prompt[:200]!r}")
    print(f"LongLive total_num_frames={total_num_frames}")
    print(f"LongLive switch_frame_indices={switch_frame_indices}")
    print("Memory mode: one native LongLive interactive inference call with KV cache and recache across chunks.")

    for chunk in chunk_specs:
        print(
            f"[Chunk {chunk['idx']}] num_frames={chunk['num_frames']}, "
            f"prompt={chunk['prompt'][:140]!r}"
        )

    pipeline = LongLivePipeline.from_pretrained(
        model_path=model_path,
        required_components={
            "wan_model_path": wan_model_path,
        },
        device=args.device,
    )
    if pipeline.memory_module is not None:
        pipeline.memory_module.manage(action="reset")

    output_video = pipeline.stream(
        prompts=prompts,
        switch_frame_indices=switch_frame_indices,
        num_frames=total_num_frames,
        seed=args.seed,
        reset=True,
    )

    save_uint8_video(output_video, str(output_path), fps=args.fps)
    actual_total_frames = count_video_frames(output_video)
    chunk_timestamps = build_longlive_chunk_timestamps(
        chunk_specs,
        switch_frame_indices=switch_frame_indices,
        total_frames=actual_total_frames,
        fps=args.fps,
    )
    timestamp_path = write_chunk_timestamp_manifest(
        args,
        output_path=output_path,
        fps=args.fps,
        chunks=chunk_timestamps,
        total_frames=actual_total_frames,
        notes=(
            ["Last chunk includes LongLive block-size padding frames when padding is needed."]
            if total_num_frames > sum(int(chunk["num_frames"]) for chunk in chunk_specs)
            else None
        ),
    )
    if args.copy_reference:
        copy_organized_reference_files(json_path, json_copy_path, reference_video_path, reference_copy_path)
    else:
        json_copy_path.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Saved to: {output_path}")
    print(f"Saved chunk timestamps to: {timestamp_path}")
    print(f"Copied JSON to: {json_copy_path}")
    if args.copy_reference:
        print(f"Copied reference video to: {reference_copy_path}")


if __name__ == "__main__":
    main()
