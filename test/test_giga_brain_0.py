import json
import os

import matplotlib
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from sceneflow.pipelines.giga_brain_0.pipeline_giga_brain_0 import GigaBrain0Pipeline


MODEL_PATH = 'open-gigaai/GigaBrain-0-3.5B-Base'
NORM_STATS_PATH = 'data/test_vla/norm_stats.json'
MAIN_VIEW_PATH = 'data/test_vla/main_view.png'
WRIST_VIEW_PATH = 'data/test_vla/wrist_view.png'
META_PATH = 'data/test_vla/meta.json'
OUTPUT_PATH = 'outputs/giga_brain_0_demo.png'

# 基础配置
ORIGINAL_ACTION_DIM = 14
DELTA_MASK = [True] * ORIGINAL_ACTION_DIM  # 如有准确掩码可替换
EMBODIMENT_ID = 0
TOKENIZER_MODEL_PATH = 'google/paligemma-3b-pt-224'
FAST_TOKENIZER_PATH = 'physical-intelligence/fast'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'



def visualize_action(pred_action: np.ndarray, out_path: str, action_names: list[str] | None = None) -> None:
    """仅可视化模型生成的动作轨迹。"""
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
    with open(NORM_STATS_PATH, 'r') as f:
        norm_stats_data = json.load(f)['norm_stats']

    # 兼容不同键名
    state_norm = norm_stats_data.get('observation.state', norm_stats_data.get('state'))
    action_norm = norm_stats_data.get('action', norm_stats_data.get('actions'))

    pipe = GigaBrain0Pipeline.from_pretrained(
        model_path=MODEL_PATH,
        tokenizer_model_path=TOKENIZER_MODEL_PATH,
        fast_tokenizer_path=FAST_TOKENIZER_PATH,
        embodiment_id=EMBODIMENT_ID,
        state_norm_stats=state_norm,
        action_norm_stats=action_norm,
        delta_mask=DELTA_MASK,
        original_action_dim=ORIGINAL_ACTION_DIM,
        depth_img_prefix_name=None,
        device=DEVICE,
        present_img_keys=['observation.images.cam_high', 'observation.images.cam_wrist'],
    )
    pipe.compile()

    images = {
        'observation.images.cam_high': TF.to_tensor(Image.open(MAIN_VIEW_PATH).convert('RGB')),
        'observation.images.cam_wrist': TF.to_tensor(Image.open(WRIST_VIEW_PATH).convert('RGB')),
    }

    with open(META_PATH, 'r') as f:
        meta_data = json.load(f)
    task = meta_data['task']
    state = torch.tensor(meta_data['observation']['state'], dtype=torch.float32)

    pred_action = pipe(images, task, state)
    print(pred_action)
    visualize_action(
        pred_action.detach().cpu().numpy(),
        OUTPUT_PATH,
        action_names=None,
    )
