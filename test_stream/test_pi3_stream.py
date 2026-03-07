import sys
import os
sys.path.append("..")

from sceneflow.pipelines.pi3.pipeline_pi3 import Pi3Pipeline

MODE = "pi3x"  # or "pi3"
MODEL_PATH = {"pi3x": "yyfz233/Pi3X", "pi3": "yyfz233/Pi3"}[MODE]
IMAGE_INPUT = "../data/test_case1/ref_image.png"
VIDEO_INPUT = None
OUTPUT_DIR = "output_pi3_stream"

DATA_PATH = VIDEO_INPUT if VIDEO_INPUT is not None else IMAGE_INPUT

AVAILABLE_INTERACTIONS = [
    "forward", "backward", "left", "right",
    "forward_left", "forward_right", "backward_left", "backward_right",
    "camera_up", "camera_down", "camera_l", "camera_r",
    "camera_ul", "camera_ur", "camera_dl", "camera_dr",
    "camera_zoom_in", "camera_zoom_out",
]

pipeline = Pi3Pipeline.from_pretrained(model_path=MODEL_PATH, mode=MODE)

if VIDEO_INPUT is not None:
    result = pipeline(videos=VIDEO_INPUT, task_type="reconstruction", interval=10)
else:
    result = pipeline(images=IMAGE_INPUT, task_type="reconstruction", interval=10)
result.save(OUTPUT_DIR)

default_render = pipeline(task_type="render_view", view_index=0)
default_render.save(os.path.join(OUTPUT_DIR, "render_default.png"))

print(f"Reconstruction done. {result.camera_range['num_views']} views available.")
print(f"Mode: {MODE}")
print(f"Input: {DATA_PATH}")
print(f"Camera range: {result.camera_range}")
print("Available interactions:")
for i, name in enumerate(AVAILABLE_INTERACTIONS):
    print(f"  {i + 1}. {name}")
print("Tips: input multiple separated by comma (e.g., 'forward,left'). 'q' to quit.")

turn_idx = 0
while True:
    user_input = input(f"\n[Turn {turn_idx}] Enter interaction(s): ").strip().lower()
    if user_input in ["q", "quit", "exit"]:
        break

    signals = [s.strip() for s in user_input.split(",") if s.strip()]
    invalid = [s for s in signals if s not in AVAILABLE_INTERACTIONS]
    if invalid:
        print(f"Invalid: {invalid}. Choose from: {AVAILABLE_INTERACTIONS}")
        continue
    if not signals:
        continue

    rendered = pipeline.stream(interaction_signal=signals)
    rendered.save(f"{OUTPUT_DIR}/stream_turn_{turn_idx:03d}.png")
    print(f"Saved: stream_turn_{turn_idx:03d}.png")
    turn_idx += 1

print(f"Stream ended. Total turns: {turn_idx}")
