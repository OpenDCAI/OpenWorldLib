import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# =========================
# 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
WOW_ROOT = DATA_ROOT / "wow"

BENCHMARK_ROOT = WOW_ROOT / "benchmark_samples"
OUTPUT_JSON = WOW_ROOT / "wowbench_converted.json"

IMAGE_NAME = "init_frame.jpg"
PROMPT_NAME = "prompt.txt"

WOW_ROOT.mkdir(parents=True, exist_ok=True)

print(f"[WoW] Project root: {PROJECT_ROOT}")
print(f"[WoW] Data dir: {WOW_ROOT}")

# =========================
# 下载数据
# =========================
try:
    snapshot_download(
        repo_id="WoW-world-model/WoW-1-Benchmark-Samples",
        repo_type="dataset",
        local_dir=str(WOW_ROOT),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4
    )
    print("[WoW] Download check completed.")
except Exception as e:
    print(f"[WoW] Download note: {e}")

# =========================
# 扫描样本
# =========================
cases = []
print("[WoW] Scanning benchmark_samples ...")

if not BENCHMARK_ROOT.exists():
    raise RuntimeError(f"benchmark_samples not found at {BENCHMARK_ROOT}")

for root, _, files in os.walk(BENCHMARK_ROOT):
    if not files:
        continue

    current_path = Path(root)
    case_id = current_path.name

    video_name = "video.mp4" if "video.mp4" in files else f"{case_id}.mp4"
    required = {IMAGE_NAME, PROMPT_NAME, video_name}

    if not required.issubset(set(files)):
        continue

    try:
        rel_class = current_path.relative_to(BENCHMARK_ROOT).parent
        class_str = rel_class.as_posix() if rel_class.as_posix() != "." else "General"

        image_rel = (current_path / IMAGE_NAME).relative_to(WOW_ROOT).as_posix()
        video_rel = (current_path / video_name).relative_to(WOW_ROOT).as_posix()
        prompt = (current_path / PROMPT_NAME).read_text(encoding="utf-8").strip()

        cases.append({
            "id": case_id,
            "task_type": "i2v",
            "class": class_str,
            "image_path": image_rel,
            "prompt": prompt,
            "gt_video_path": video_rel
        })

    except Exception as e:
        print(f"[WoW] Warning: skip {case_id}: {e}")

# =========================
# 写 JSON
# =========================
with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(cases, f, indent=2, ensure_ascii=False)

print(f"[WoW] Done ✔  Total samples: {len(cases)}")
print(f"[WoW] JSON saved to: {OUTPUT_JSON}")
