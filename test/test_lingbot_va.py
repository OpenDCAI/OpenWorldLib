import os
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from sceneflow.pipelines.lingbot_va.pipeline_lingbot_va import LingBotVAPipeline


# ─── RoboTwin config ─────────────────────────────────────────────────────────
def make_robotwin_config():
    used_action_channel_ids = list(range(0, 7)) + list(range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
    action_dim = 30
    _inv = [len(used_action_channel_ids)] * action_dim
    for _i, _j in enumerate(used_action_channel_ids):
        _inv[_j] = _i

    return SimpleNamespace(
        param_dtype=torch.bfloat16,
        patch_size=(1, 2, 2),
        env_type='robotwin_tshape',
        height=256,
        width=320,
        action_dim=action_dim,
        action_per_frame=16,
        frame_chunk_size=2,
        attn_window=72,
        obs_cam_keys=[
            'observation.images.cam_high',
            'observation.images.cam_left_wrist',
            'observation.images.cam_right_wrist',
        ],
        guidance_scale=5,
        action_guidance_scale=1,
        num_inference_steps=25,
        video_exec_step=-1,
        action_num_inference_steps=50,
        snr_shift=5.0,
        action_snr_shift=1.0,
        used_action_channel_ids=used_action_channel_ids,
        inverse_used_action_channel_ids=_inv,
        action_norm_method='quantiles',
        norm_stat={
            "q01": [
                -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
                -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
                -0.11975435614585876, -1, -1, -1, -1,
            ] + [0.0] * 16,
            "q99": [
                0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
                1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
                1, 1,
            ] + [0.0] * 14 + [1.0, 1.0],
        },
    )


MODEL_PATH = 'robbyant/lingbot-va-posttrain-robotwin'
IMAGE_DIR = 'data/test_vla/robotwin'
OUTPUT_PATH = 'outputs/lingbot_va_demo.png'
VIDEO_OUTPUT_PATH = 'outputs/lingbot_va_demo.mp4'
PROMPT = 'Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack.'
NUM_CHUNKS = 10
DECODE_VIDEO = True
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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
    config = make_robotwin_config()

    # Load initial multi-view images
    img_dict = {}
    for k in config.obs_cam_keys:
        img_path = os.path.join(IMAGE_DIR, f'{k}.png')
        img_dict[k] = np.array(Image.open(img_path).convert('RGB'))

    # Build pipeline
    pipe = LingBotVAPipeline.from_pretrained(
        model_path=MODEL_PATH,
        config=config,
        device=DEVICE,
    )

    # Run inference
    output = pipe(
        images=img_dict,
        prompt=PROMPT,
        num_chunks=NUM_CHUNKS,
        decode_video=DECODE_VIDEO,
    )

    print(f'Predicted actions shape: {output.actions.shape}')
    visualize_action(output.actions, OUTPUT_PATH)

    # Save decoded video
    if output.video is not None:
        from diffusers.utils import export_to_video
        os.makedirs(os.path.dirname(VIDEO_OUTPUT_PATH), exist_ok=True)
        export_to_video(output.video, VIDEO_OUTPUT_PATH, fps=10)
        print(f'Saved video to {VIDEO_OUTPUT_PATH}')
