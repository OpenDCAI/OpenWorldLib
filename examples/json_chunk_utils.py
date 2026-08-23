import json
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image


INTERVAL_RE = re.compile(r"^[\[(]?\s*([0-9:.]+)\s*(?:,|-)\s*([0-9:.]+)\s*[\])]?\s*$")


def choose_existing_path(candidates, fallback):
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return fallback


def add_organized_output_args(parser, default_output_dir_name: str):
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default=default_output_dir_name,
        help="Folder name under --output_root for generated video and copied references.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory for organized outputs.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional generated video path. Defaults to <output_root>/<output_dir_name>/generated.mp4.",
    )
    parser.add_argument(
        "--timestamp_path",
        type=str,
        default=None,
        help=(
            "Optional chunk timestamp JSON path. Defaults to "
            "<generated video stem>_chunk_timestamps.json next to --output_path."
        ),
    )


def prepare_organized_output_paths(args, json_path: Path, companion_resolver=None):
    output_dir = Path(args.output_root) / args.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output_path) if args.output_path else output_dir / "generated.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolver = companion_resolver or resolve_companion_video
    reference_video_path = resolver(json_path)
    json_copy_path = output_dir / "prompt.json"
    reference_copy_path = output_dir / f"reference_{reference_video_path.name}"

    return output_dir, output_path, json_copy_path, reference_video_path, reference_copy_path


def copy_organized_reference_files(json_path, json_copy_path, reference_video_path, reference_copy_path):
    shutil.copy2(json_path, json_copy_path)
    shutil.copy2(reference_video_path, reference_copy_path)


def load_records(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text) -> str:
    return " ".join(str(text or "").strip().split())


def get_record_text(record, text_keys: Iterable[str] = ("prompt", "caption", "text", "instruction")) -> str:
    for key in text_keys:
        value = normalize_text(record.get(key, ""))
        if value:
            return value
    return ""


def build_global_prompt(records, prompt_override=None, max_chars=1800, text_keys: Iterable[str] = ("prompt", "caption", "text", "instruction")):
    if prompt_override:
        return normalize_text(prompt_override)
    text = " ".join(get_record_text(record, text_keys=text_keys) for record in records)
    text = normalize_text(text)
    return text[:max_chars] if text else "A first-person gameplay video."


def should_use_chunk_prompts(records, text_keys: Iterable[str] = ("prompt", "caption", "text", "instruction")) -> bool:
    texts = [get_record_text(record, text_keys=text_keys) for record in records]
    texts = [text for text in texts if text]
    return len(set(texts)) > 1


def build_chunk_prompt(record, global_prompt, prompt_override=None, use_chunk_prompts=False, text_keys: Iterable[str] = ("prompt", "caption", "text", "instruction")):
    record_text = get_record_text(record, text_keys=text_keys)
    if use_chunk_prompts and record_text:
        if prompt_override:
            base_prompt = normalize_text(prompt_override)
            if base_prompt and base_prompt != record_text:
                return f"{base_prompt} {record_text}".strip()
        return record_text
    return global_prompt


def resolve_companion_video(json_path: Path) -> Path:
    candidate = json_path.with_name(f"{json_path.stem}.mp4")
    if not candidate.exists():
        raise FileNotFoundError(f"Companion mp4 not found: {candidate}")
    return candidate


def load_first_frame(video_path: Path) -> Image.Image:
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read first frame from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    except ModuleNotFoundError:
        import imageio.v3 as iio

        frame = iio.imread(video_path, index=0)
        return Image.fromarray(frame)


def resolve_input_image(json_path: Path, image_path: Optional[str]) -> Image.Image:
    if image_path is not None:
        return Image.open(image_path).convert("RGB")
    return load_first_frame(resolve_companion_video(json_path))


def parse_timestamp_to_seconds(text: str) -> float:
    parts = text.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported timestamp format: {text!r}")


def parse_interval_seconds(interval_text: Optional[str]) -> Optional[float]:
    interval = parse_interval_range_seconds(interval_text)
    if interval is None:
        return None
    start_sec, end_sec = interval
    return end_sec - start_sec


def parse_interval_range_seconds(interval_text: Optional[str]) -> Optional[tuple[float, float]]:
    if not interval_text:
        return None
    match = INTERVAL_RE.match(str(interval_text).strip())
    if match is None:
        return None
    start_text, end_text = match.groups()
    try:
        start_sec = parse_timestamp_to_seconds(start_text)
        end_sec = parse_timestamp_to_seconds(end_text)
    except ValueError:
        return None
    if end_sec <= start_sec:
        return None
    return start_sec, end_sec


def default_chunk_timestamps_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_chunk_timestamps.json")


def resolve_chunk_timestamps_path(args, output_path: Path) -> Path:
    timestamp_path = getattr(args, "timestamp_path", None)
    if timestamp_path:
        return Path(timestamp_path)
    return default_chunk_timestamps_path(output_path)


def count_video_frames(video_frames) -> int:
    shape = getattr(video_frames, "shape", None)
    if shape is not None:
        if len(shape) == 5:
            return int(shape[1])
        if len(shape) == 4:
            return int(shape[0])
    return len(video_frames)


def build_chunk_timestamp_entry(
    record,
    chunk_index: int,
    fps: int,
    frame_start: int,
    frame_count: int,
    requested_frames: Optional[int] = None,
    dropped_overlap_frames: int = 0,
    extra: Optional[dict] = None,
) -> dict:
    frame_start = int(frame_start)
    frame_count = int(frame_count)
    frame_end = frame_start + frame_count
    fps = int(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive for chunk timestamps, got {fps}")

    source_interval = record.get("interval") if isinstance(record, dict) else None
    source_range = parse_interval_range_seconds(source_interval)
    source_start_sec = source_range[0] if source_range is not None else None
    source_end_sec = source_range[1] if source_range is not None else None

    entry = {
        "chunk_index": int(chunk_index),
        "source_interval": source_interval,
        "source_start_sec": source_start_sec,
        "source_end_sec": source_end_sec,
        "source_duration_sec": (
            source_end_sec - source_start_sec
            if source_start_sec is not None and source_end_sec is not None
            else None
        ),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_end_exclusive": frame_end,
        "num_exported_frames": frame_count,
        "generated_start_sec": frame_start / fps,
        "generated_end_sec": frame_end / fps,
        "generated_duration_sec": frame_count / fps,
        "requested_frames": int(requested_frames) if requested_frames is not None else None,
        "dropped_overlap_frames": int(dropped_overlap_frames),
    }
    if extra:
        entry.update(extra)
    return entry


def append_chunk_timestamp(
    chunks: list,
    record,
    chunk_index: int,
    fps: int,
    frame_start: int,
    frame_count: int,
    requested_frames: Optional[int] = None,
    dropped_overlap_frames: int = 0,
    extra: Optional[dict] = None,
) -> dict:
    entry = build_chunk_timestamp_entry(
        record=record,
        chunk_index=chunk_index,
        fps=fps,
        frame_start=frame_start,
        frame_count=frame_count,
        requested_frames=requested_frames,
        dropped_overlap_frames=dropped_overlap_frames,
        extra=extra,
    )
    chunks.append(entry)
    return entry


def write_chunk_timestamp_manifest(
    args,
    output_path: Path,
    fps: int,
    chunks: list,
    total_frames: Optional[int] = None,
    notes: Optional[list[str]] = None,
) -> Path:
    timestamp_path = resolve_chunk_timestamps_path(args, output_path)
    timestamp_path.parent.mkdir(parents=True, exist_ok=True)
    fps = int(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive for chunk timestamps, got {fps}")
    total_frames = int(total_frames if total_frames is not None else sum(chunk["num_exported_frames"] for chunk in chunks))

    manifest = {
        "version": 1,
        "video_path": str(output_path),
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / fps,
        "chunks": chunks,
    }
    if notes:
        manifest["notes"] = notes

    with open(timestamp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return timestamp_path


def align_to_4n_plus_1(frame_count: int, minimum: int, maximum: Optional[int] = None) -> int:
    minimum = max(5, minimum)
    min_valid = max(5, 4 * max(1, round((minimum - 1) / 4)) + 1)
    target = max(frame_count, min_valid)
    aligned = 4 * max(1, round((target - 1) / 4)) + 1

    if maximum is not None:
        max_valid = 4 * max(1, (maximum - 1) // 4) + 1
        if max_valid < min_valid:
            max_valid = min_valid
        aligned = min(aligned, max_valid)

    return max(min_valid, aligned)


def resolve_chunk_num_frames(
    record,
    fps: int,
    default_chunk_frames: int,
    min_chunk_frames: int = 1,
    max_chunk_frames: Optional[int] = None,
    align_4n_plus_1_frames: bool = False,
) -> int:
    duration_sec = parse_interval_seconds(record.get("interval"))
    if duration_sec is None:
        target_frames = default_chunk_frames
    else:
        target_frames = max(1, int(round(duration_sec * fps)))

    if align_4n_plus_1_frames:
        return align_to_4n_plus_1(
            target_frames,
            minimum=min_chunk_frames,
            maximum=max_chunk_frames,
        )

    if max_chunk_frames is not None:
        target_frames = min(target_frames, max_chunk_frames)
    return max(min_chunk_frames, target_frames)
