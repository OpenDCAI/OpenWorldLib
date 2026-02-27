"""Test file for PI0 and PI0.5 pipeline."""
import json
import os

import matplotlib
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from sceneflow.pipelines.pi0.pipeline_pi0 import PI0Pipeline


# PI0 Model Configuration
PI0_MODEL_PATH = 'lerobot/pi0_libero_finetuned'
PI05_MODEL_PATH = 'lerobot/pi05_libero_finetuned'  # Using same model for demo, in practice use different checkpoint
PI0_NORM_STATS_PATH = 'data/test_vla/pi0_norm_stats.json'
PI05_NORM_STATS_PATH = 'data/test_vla/pi0_5_norm_stats.json'
META_PATH = 'data/test_vla/meta.json'
# Using same image paths as test_giga_brain_0.py
MAIN_VIEW_PATH = 'data/test_vla/main_view.png'
WRIST_VIEW_PATH = 'data/test_vla/wrist_view.png'
TOKENIZER_MODEL_PATH = 'google/paligemma-3b-mix-224'

PRESENT_IMG_KEYS = [
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
]

# Output paths
PI0_OUTPUT_PATH = 'outputs/pi0_demo.png'
PI05_OUTPUT_PATH = 'outputs/pi05_demo.png'

# 基础配置
ORIGINAL_ACTION_DIM = 8
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def visualize_action(pred_action: np.ndarray, out_path: str, action_names: list[str] | None = None) -> None:
    if pred_action.ndim == 1:
        pred_action = pred_action[None, :]
    pred_action = pred_action[:, :ORIGINAL_ACTION_DIM]
    num_ts, num_dim = pred_action.shape
    fig, axs = plt.subplots(num_dim, 1, figsize=(10, 2 * num_dim))
    time_axis = np.arange(num_ts) / 30.0
    colors = plt.cm.viridis(np.linspace(0, 1, num_dim))
    action_names = action_names or [str(i) for i in range(num_dim)]

    for ax_idx in range(num_dim):
        ax = axs[ax_idx]
        ax.plot(time_axis, pred_action[:, ax_idx], label='Pred', color=colors[ax_idx], linewidth=2, linestyle='-')
        ax.set_title(f'Joint {ax_idx}: {action_names[ax_idx]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    # 禁用 torch.compile 错误抑制（如果需要使用 compile）
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    # 使用 test_giga_brain_0.py 的图像数据
    images = {
        'observation.images.cam_high': TF.to_tensor(Image.open(MAIN_VIEW_PATH).convert('RGB')),
        'observation.images.cam_left_wrist': TF.to_tensor(Image.open(WRIST_VIEW_PATH).convert('RGB')),
    }

    # 使用 meta.json 中的 task 和 state 数据
    with open(META_PATH, 'r') as f:
        meta_data = json.load(f)
    task = meta_data['task']
    # state 保持 1D (state_dim,)，Pipeline 内部会自动添加 batch 维度
    state = torch.tensor(meta_data['observation']['state'], dtype=torch.float32)

    # ===== PI0 测试 (continuous state input) =====
    # 加载 PI0 的 norm stats
    with open(PI0_NORM_STATS_PATH, 'r') as f:
        pi0_norm_stats_data = json.load(f)['norm_stats']

    # 兼容不同键名
    pi0_state_norm = pi0_norm_stats_data.get('observation.state', pi0_norm_stats_data.get('state'))
    pi0_action_norm = pi0_norm_stats_data.get('action', pi0_norm_stats_data.get('actions'))

    pipe = PI0Pipeline.from_pretrained(
        model_path=PI0_MODEL_PATH,
        tokenizer_model_path=TOKENIZER_MODEL_PATH,
        state_norm_stats=pi0_state_norm,
        action_norm_stats=pi0_action_norm,
        original_action_dim=ORIGINAL_ACTION_DIM,
        discrete_state_input=False,
        device=DEVICE,
        present_img_keys=PRESENT_IMG_KEYS,
    )
    pipe.compile()

    pred_action = pipe(images, task, state)
    print(pred_action)
    visualize_action(
        pred_action.detach().cpu().numpy(),
        PI0_OUTPUT_PATH,
        action_names=None,
    )

    # ===== PI0.5 测试 (discrete state input, quantile normalization) =====
    del pipe
    torch.cuda.empty_cache()

    # 加载 PI0.5 的 norm stats
    with open(PI05_NORM_STATS_PATH, 'r') as f:
        pi05_norm_stats_data = json.load(f)['norm_stats']

    # 兼容不同键名
    pi05_state_norm = pi05_norm_stats_data.get('observation.state', pi05_norm_stats_data.get('state'))
    pi05_action_norm = pi05_norm_stats_data.get('action', pi05_norm_stats_data.get('actions'))

    pipe05 = PI0Pipeline.from_pretrained(
        model_path=PI05_MODEL_PATH,
        tokenizer_model_path=TOKENIZER_MODEL_PATH,
        state_norm_stats=pi05_state_norm,
        action_norm_stats=pi05_action_norm,
        original_action_dim=ORIGINAL_ACTION_DIM,
        discrete_state_input=True,
        device=DEVICE,
        present_img_keys=PRESENT_IMG_KEYS,
    )
    pipe05.compile()

    pred_action_05 = pipe05(images, task, state)
    print(pred_action_05)
    visualize_action(
        pred_action_05.detach().cpu().numpy(),
        PI05_OUTPUT_PATH,
        action_names=None,
    )
