import os
import json

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from sceneflow.pipelines.lingbot_va.pipeline_lingbot_va import LingBotVAPipeline


MODEL_PATH = 'robbyant/lingbot-va-posttrain-robotwin'
IMAGE_DIR = 'data/test_vla/robotwin'
OUTPUT_PATH = 'outputs/lingbot_va_demo.png'
VIDEO_OUTPUT_PATH = 'outputs/lingbot_va_demo.mp4'
PROMPT = 'Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack.'
NUM_CHUNKS = 2
DECODE_VIDEO = True
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

OBS_CAM_KEYS = [
    'observation_images_cam_high',
    'observation_images_cam_left_wrist',
    'observation_images_cam_right_wrist',
]


def visualize_action(pred_action: np.ndarray, out_path: str, action_names: list[str] | None = None) -> None:
    """Visualize predicted action trajectories."""
    if pred_action.ndim == 1:
        pred_action = pred_action[None, :]
    num_dim, num_ts = pred_action.shape
    fig, axs = plt.subplots(num_dim, 1, figsize=(10, 2 * num_dim))
    if num_dim == 1:
        axs = [axs]
    time_axis = np.arange(num_ts)
    colors = plt.cm.viridis(np.linspace(0, 1, num_dim))
    action_names = action_names or [str(i) for i in range(num_dim)]

    for ax_idx in range(num_dim):
        ax = axs[ax_idx]
        ax.plot(time_axis, pred_action[ax_idx], label='Pred', color=colors[ax_idx], linewidth=1.5)
        ax.set_title(f'Channel {ax_idx}: {action_names[ax_idx]}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved action visualization to {out_path}')


if __name__ == '__main__':
    # Load norm stats from file
    norm_stats_path = 'data/test_vla/lingbot_va_norm_stats.json'
    with open(norm_stats_path, 'r') as f:
        norm_stat = json.load(f)

    # Load initial multi-view images
    img_dict = {}
    for k in OBS_CAM_KEYS:
        img_path = os.path.join(IMAGE_DIR, f'{k}.png')
        img_dict[k] = np.array(Image.open(img_path).convert('RGB'))

    # Build pipeline
    pipe = LingBotVAPipeline.from_pretrained(
        model_path=MODEL_PATH,
        device=DEVICE,
        norm_stat=norm_stat,
        obs_cam_keys=OBS_CAM_KEYS,
    )

    # Run inference
    output = pipe(
        images=img_dict,
        prompt=PROMPT,
        num_chunks=NUM_CHUNKS,
        decode_video=DECODE_VIDEO,
        guidance_scale=5.0,
        action_guidance_scale=1.0,
        num_inference_steps=25,
        action_num_inference_steps=50,
        video_exec_step=-1,
    )

    print(f'Predicted actions shape: {output.actions.shape}')
    visualize_action(output.actions, OUTPUT_PATH)

    # Save decoded video
    if output.video is not None:
        from diffusers.utils import export_to_video
        os.makedirs(os.path.dirname(VIDEO_OUTPUT_PATH), exist_ok=True)
        export_to_video(output.video, VIDEO_OUTPUT_PATH, fps=10)
        print(f'Saved video to {VIDEO_OUTPUT_PATH}')
