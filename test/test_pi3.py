import sys
import os
import math
import numpy as np

sys.path.append("..")

from sceneflow.pipelines.pi3.pipeline_pi3 import Pi3Pipeline, render_point_cloud, _apply_camera_delta

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "yyfz233/Pi3X"
MODE = "pi3x"

IMAGE_INPUT = "../data/test_case1/ref_image.png"
VIDEO_INPUT = None
OUTPUT_DIR = "output_pi3"

# ============================================================
# Step 1: Load pipeline
# ============================================================
pipeline = Pi3Pipeline.from_pretrained(
    model_path=MODEL_PATH,
    mode=MODE,
)

# ============================================================
# Step 2: Inference (single-shot, all outputs at once)
# ============================================================
input_path = VIDEO_INPUT if VIDEO_INPUT is not None else IMAGE_INPUT
result = pipeline(images=input_path, interval=10)
result.save(OUTPUT_DIR)

print("Inference complete.")
print(f"  Views: {result.camera_range['num_views']}")
print(f"  Camera range: {result.camera_range}")

# ============================================================
# Step 3: Render from existing point cloud (no re-inference)
# ============================================================

# 3a: Render from an original camera viewpoint
rendered_default = pipeline.render_view(result=result, view_index=0)
rendered_default.save(os.path.join(OUTPUT_DIR, "render_default.png"))
print(f"Default view saved.")

# 3b: Render from a user-specified camera_view delta [dx,dy,dz,theta_x,theta_z]
rendered_custom = pipeline.render_view(
    result=result,
    camera_view=[0.3, 0.0, -0.2, 0.0, 0.1],
)
rendered_custom.save(os.path.join(OUTPUT_DIR, "render_custom.png"))
print(f"Custom view saved.")

# 3c: Interactive rendering via navigation signals
for signal in ["forward", "left", "camera_r"]:
    rendered = pipeline.stream(interaction_signal=signal, result=result)
    rendered.save(os.path.join(OUTPUT_DIR, f"render_interact_{signal}.png"))
    print(f"Interaction '{signal}' saved.")

# 3d: Render trajectory video (interpolate between original camera poses)
pts_all = result.numpy_data["points"][0]
masks = result.numpy_data["masks"][0].astype(bool)
colors_all = np.stack(result.input_images, axis=0)
pts = pts_all[masks].astype(np.float64)
cols = (colors_all[masks] * 255).clip(0, 255).astype(np.uint8)
h, w = pts_all.shape[1], pts_all.shape[2]

c2ws = [np.array(c["camera_to_world"], dtype=np.float64) for c in result.camera_params]
n_views = len(c2ws)
n_interp = 15
frames_np = []

for vi in range(n_views - 1):
    for j in range(n_interp):
        t = j / n_interp
        c2w = c2ws[vi] * (1 - t) + c2ws[vi + 1] * t
        img = render_point_cloud(pts, cols, c2w, h, w)
        frames_np.append(np.array(img))

img_last = render_point_cloud(pts, cols, c2ws[-1], h, w)
frames_np.append(np.array(img_last))

try:
    import cv2
    video_path = os.path.join(OUTPUT_DIR, "trajectory_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 15, (w, h))
    for f in frames_np:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Trajectory video saved: {video_path}")
except ImportError:
    print("cv2 not available, skipping video export.")

print("All done.")
