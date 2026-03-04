import os
import json
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# =========================
# 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
EPIC_ROOT = DATA_ROOT / "epic-kitchens"

CSV_PATH = EPIC_ROOT / "annotations.csv"
RAW_VIDEO_DIR = EPIC_ROOT / "videos_raw"
SAMPLES_DIR = EPIC_ROOT / "samples"
OUTPUT_JSON = EPIC_ROOT / "samples.json"

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

print(f"[EPIC] Project root: {PROJECT_ROOT}")
print(f"[EPIC] Data dir: {EPIC_ROOT}")

# =========================
# 读取 CSV
# =========================
df = pd.read_csv(CSV_PATH)
samples = []

# =========================
# 主循环
# =========================
for _, row in tqdm(df.iterrows(), total=len(df)):
    narration_id = row["narration_id"]
    video_id = row["video_id"]

    start_frame = int(row["start_frame"])
    stop_frame = int(row["stop_frame"]) + 10  # 防止结尾裁剪过紧
    narration = row["narration"]

    sample_dir = SAMPLES_DIR / narration_id
    sample_dir.mkdir(exist_ok=True)

    init_image_path = sample_dir / "init_frame.jpg"
    clip_video_path = sample_dir / f"{narration_id}.mp4"
    raw_video_path = RAW_VIDEO_DIR / f"{video_id}.mp4"

    cap = cv2.VideoCapture(str(raw_video_path))
    assert cap.isOpened(), f"Cannot open {raw_video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(clip_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    cur = start_frame
    first_frame_saved = False

    while cur <= stop_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if not first_frame_saved:
            cv2.imwrite(str(init_image_path), frame)
            first_frame_saved = True

        writer.write(frame)
        cur += 1

    cap.release()
    writer.release()

    # =========================
    # JSON 样本
    # =========================
    samples.append({
        "id": narration_id,
        "task_type": "i2v",
        "image_path": f"samples/{narration_id}/init_frame.jpg",
        "prompt": narration,
        "gt_video_path": f"samples/{narration_id}/{narration_id}.mp4",
        "metadata": {
            "participant_id": row["participant_id"],
            "video_id": video_id,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
            "verb": row["verb"],
            "verb_class": row["verb_class"],
            "noun": row["noun"],
            "noun_class": row["noun_class"],
            "all_nouns": row["all_nouns"],
            "all_noun_classes": row["all_noun_classes"]
        }
    })

# =========================
# 写 JSON
# =========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"[EPIC] Done ✔  Total samples: {len(samples)}")
print(f"[EPIC] JSON saved to: {OUTPUT_JSON}")
