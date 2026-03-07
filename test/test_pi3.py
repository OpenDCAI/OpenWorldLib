import os
import sys
import cv2
import numpy as np

sys.path.append("..")

from sceneflow.pipelines.pi3.pipeline_pi3 import Pi3Pipeline

MODE = "pi3x"  # or "pi3"
MODEL_PATH = {"pi3x": "yyfz233/Pi3X", "pi3": "yyfz233/Pi3"}[MODE]
IMAGE_INPUT = "../data/test_case1/ref_image.png"
VIDEO_INPUT = None
OUTPUT_DIR = "output_pi3"

DATA_PATH = VIDEO_INPUT if VIDEO_INPUT is not None else IMAGE_INPUT

pipeline = Pi3Pipeline.from_pretrained(model_path=MODEL_PATH, mode=MODE)

if VIDEO_INPUT is not None:
    result = pipeline(videos=VIDEO_INPUT, task_type="reconstruction", interval=10)
else:
    result = pipeline(images=IMAGE_INPUT, task_type="reconstruction", interval=10)
result.save(OUTPUT_DIR)
print(f"Mode: {MODE}")
print(f"Input: {DATA_PATH}")
print(f"Views: {result.camera_range['num_views']}")
print(f"Camera range: {result.camera_range}")

rendered = pipeline(task_type="render_view", view_index=0)
rendered.save(os.path.join(OUTPUT_DIR, "render_default.png"))

rendered = pipeline(task_type="render_view", interactions=["forward", "left", "camera_r"])
rendered.save(os.path.join(OUTPUT_DIR, "render_interact.png"))

frames = pipeline(task_type="render_trajectory")
video_path = os.path.join(OUTPUT_DIR, "trajectory_video.mp4")
video = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    15,
    frames[0].size,
)
for frame in frames:
    video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video.release()
print(f"Trajectory video saved: {video_path}")
