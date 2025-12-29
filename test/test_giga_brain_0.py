import json
import os

import matplotlib
import numpy as np
import torch
import tyro

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from giga_datasets import load_dataset
from sceneflow.pipelines.giga_brain_0.pipeline_giga_brain_0 import GigaBrain0Pipeline


def inference_giga_brain_0(
    model_path: str,
    data_path: str,
    output_path: str,
    norm_stats_path: str,
    delta_mask: list[bool],
    embodiment_id: int,
    original_action_dim: int,
    action_chunk: int = 50,
    enable_2d_traj_output: bool = False,
    tokenizer_model_path: str = 'google/paligemma-3b-pt-224',
    fast_tokenizer_path: str = 'physical-intelligence/fast',
    depth_img_prefix_name: str | None = None,
    device: str = 'cuda:0',
):
    """示例：调用 GigaBrain0Pipeline 做推理并保存可视化结果。"""
    os.makedirs(output_path, exist_ok=True)

    with open(norm_stats_path, 'r') as f:
        norm_stats_data = json.load(f)['norm_stats']

    pipe = GigaBrain0Pipeline.from_pretrained(
        model_path=model_path,
        tokenizer_model_path=tokenizer_model_path,
        fast_tokenizer_path=fast_tokenizer_path,
        embodiment_id=embodiment_id,
        state_norm_stats=norm_stats_data['observation.state'],
        action_norm_stats=norm_stats_data['action'],
        delta_mask=delta_mask,
        original_action_dim=original_action_dim,
        depth_img_prefix_name=depth_img_prefix_name,
        device=device,
    )
    pipe.compile()

    dataset = load_dataset(
        [
            {
                '_class_name': 'LeRobotDataset',
                'data_path': data_path,
                'delta_info': {'action': action_chunk},
                'meta_name': 'meta',
            }
        ]
    )

    for idx in range(0, min(len(dataset), 1000), 100):
        data = dataset[idx]

        images = {
            'observation.images.cam_high': data['observation.images.cam_high'],
            'observation.images.cam_left_wrist': data['observation.images.cam_left_wrist'],
            'observation.images.cam_right_wrist': data['observation.images.cam_right_wrist'],
        }
        if pipe.operator.image_transform.enable_depth_img:
            images[f'{depth_img_prefix_name}.cam_high'] = data[f'{depth_img_prefix_name}.cam_high']
            images[f'{depth_img_prefix_name}.cam_left_wrist'] = data[f'{depth_img_prefix_name}.cam_left_wrist']
            images[f'{depth_img_prefix_name}.cam_right_wrist'] = data[f'{depth_img_prefix_name}.cam_right_wrist']

        task = data['task']
        state = data['observation.state']

        if enable_2d_traj_output:
            pred_action, traj_pred = pipe(images, task, state, enable_2d_traj_output=True)
        else:
            pred_action = pipe(images, task, state)

        # 可视化
        action_names = None
        if 'meta' in data and 'names' in data['meta'].info['features']['action']:
            action_names = data['meta'].info['features']['action']['names']

        visualize_action(
            data['action'].numpy(),
            pred_action.detach().cpu().numpy(),
            os.path.join(output_path, f'{idx}.png'),
            action_names,
        )
        if enable_2d_traj_output:
            visualize_traj(
                images['observation.images.cam_high'],
                traj_pred.detach().cpu().numpy(),
                os.path.join(output_path, f'{idx}_traj.png'),
            )


def visualize_action(gt_action: np.ndarray, pred_action: np.ndarray, out_path: str, action_names: list[str] | None = None) -> None:
    """动作轨迹对比可视化。"""
    pred_action = pred_action[:, :14]
    gt_action = gt_action[:, :14]
    num_ts, num_dim = gt_action.shape
    fig, axs = plt.subplots(num_dim, 1, figsize=(10, 2 * num_dim))
    time_axis = np.arange(num_ts) / 30.0
    colors = plt.cm.viridis(np.linspace(0, 1, num_dim))
    action_names = action_names or [str(i) for i in range(num_dim)]

    for ax_idx in range(num_dim):
        ax = axs[ax_idx]
        ax.plot(time_axis, gt_action[:, ax_idx], label='GT', color=colors[ax_idx], linewidth=2, linestyle='-')
        ax.plot(time_axis, pred_action[:, ax_idx], label='Pred', color=colors[ax_idx], linewidth=2, linestyle='--')
        ax.set_title(f'Joint {ax_idx}: {action_names[ax_idx]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_traj(images: np.ndarray, traj_pred: np.ndarray, out_path: str) -> None:
    """在图像上绘制 2D 轨迹。"""
    img = images
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    H, W = img.shape[:2]

    traj = traj_pred if isinstance(traj_pred, np.ndarray) else np.asarray(traj_pred)
    if traj.ndim == 1:
        traj = traj.reshape(1, 4)

    x1, y1, x2, y2 = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    mask1 = np.isfinite(x1) & np.isfinite(y1)
    mask2 = np.isfinite(x2) & np.isfinite(y2)

    fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax.imshow(img)
    ax.scatter(x1[mask1], y1[mask1], c='red', s=10)
    ax.scatter(x2[mask2], y2[mask2], c='red', s=10)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == '__main__':
    tyro.cli(inference_giga_brain_0)
