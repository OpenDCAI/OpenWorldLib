"""Test file for PI0 and PI0.5 pipeline with Aloha / Libero / DROID data."""
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

# ============================================================
# Shared constants
# ============================================================
TOKENIZER_MODEL_PATH = 'google/paligemma-3b-mix-224'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ============================================================
# Per-robot data configurations
# ============================================================

# --- Aloha ---
ALOHA_DATA_DIR = 'data/test_vla/aloha'
ALOHA_PI0_MODEL_PATH = 'lerobot/pi0_base'
ALOHA_PI0_NORM_STATS_PATH = os.path.join(ALOHA_DATA_DIR, 'pi0_norm_stats.json')
ALOHA_STATE_PATH = os.path.join(ALOHA_DATA_DIR, 'state.json')
ALOHA_IMG_KEYS = [
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist',
]
ALOHA_IMG_FILES = {
    'observation.images.cam_high': os.path.join(ALOHA_DATA_DIR, 'observation_images_cam_high.png'),
    'observation.images.cam_left_wrist': os.path.join(ALOHA_DATA_DIR, 'observation_images_cam_left_wrist.png'),
    'observation.images.cam_right_wrist': os.path.join(ALOHA_DATA_DIR, 'observation_images_cam_right_wrist.png'),
}
ALOHA_ACTION_DIM = 14
ALOHA_DEFAULT_PROMPT = 'perform the task'

# --- Libero ---
LIBERO_DATA_DIR = 'data/test_vla/libero'
LIBERO_PI0_MODEL_PATH = 'lerobot/pi0_libero_finetuned'
LIBERO_PI05_MODEL_PATH = 'lerobot/pi05_libero_finetuned'
LIBERO_PI0_NORM_STATS_PATH = os.path.join(LIBERO_DATA_DIR, 'pi0_norm_stats.json')
LIBERO_PI05_NORM_STATS_PATH = os.path.join(LIBERO_DATA_DIR, 'pi0_5_norm_stats.json')
LIBERO_META_PATH = os.path.join(LIBERO_DATA_DIR, 'meta.json')
LIBERO_IMG_KEYS = [
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
]
LIBERO_IMG_FILES = {
    'observation.images.cam_high': os.path.join(LIBERO_DATA_DIR, 'main_view.png'),
    'observation.images.cam_left_wrist': os.path.join(LIBERO_DATA_DIR, 'wrist_view.png'),
}
LIBERO_ACTION_DIM = 7

# --- DROID ---
DROID_DATA_DIR = 'data/test_vla/droid'
DROID_PI0_MODEL_PATH = 'lerobot/pi0_base'
DROID_PI05_MODEL_PATH = 'lerobot/pi0_base'  # In practice use pi05_droid checkpoint
DROID_PI0_NORM_STATS_PATH = os.path.join(DROID_DATA_DIR, 'pi0_norm_states_droid_joint.json')
DROID_PI05_NORM_STATS_PATH = os.path.join(DROID_DATA_DIR, 'pi05_norm_states_droid_joint.json')
DROID_STEP_DATA_PATH = os.path.join(DROID_DATA_DIR, 'step_data.json')
DROID_IMG_KEYS = [
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist',
]
DROID_IMG_FILES = {
    'observation.images.cam_high': os.path.join(DROID_DATA_DIR, 'exterior_image_1_left.png'),
    'observation.images.cam_left_wrist': os.path.join(DROID_DATA_DIR, 'wrist_image_left.png'),
    'observation.images.cam_right_wrist': os.path.join(DROID_DATA_DIR, 'exterior_image_2_left.png'),
}
DROID_ACTION_DIM = 8


# ============================================================
# Utilities
# ============================================================

def load_norm_stats(path: str) -> tuple[dict, dict]:
    """Load norm stats and return (state_norm, action_norm)."""
    with open(path, 'r') as f:
        data = json.load(f)['norm_stats']
    state_norm = data.get('observation.state', data.get('state'))
    action_norm = data.get('action', data.get('actions'))
    return state_norm, action_norm


def load_images(img_files: dict[str, str]) -> dict[str, torch.Tensor]:
    """Load images from file paths into (C, H, W) float32 tensors in [0, 1]."""
    return {key: TF.to_tensor(Image.open(path).convert('RGB')) for key, path in img_files.items()}


def visualize_action(pred_action: np.ndarray, out_path: str, action_dim: int, action_names: list[str] | None = None) -> None:
    """Visualize predicted action trajectories."""
    if pred_action.ndim == 1:
        pred_action = pred_action[None, :]
    pred_action = pred_action[:, :action_dim]
    num_ts, num_dim = pred_action.shape
    fig, axs = plt.subplots(num_dim, 1, figsize=(10, 2 * num_dim))
    if num_dim == 1:
        axs = [axs]
    time_axis = np.arange(num_ts) / 30.0
    colors = plt.cm.viridis(np.linspace(0, 1, num_dim))
    action_names = action_names or [str(i) for i in range(num_dim)]

    for ax_idx in range(num_dim):
        ax = axs[ax_idx]
        ax.plot(time_axis, pred_action[:, ax_idx], label='Pred', color=colors[ax_idx], linewidth=2)
        ax.set_title(f'Joint {ax_idx}: {action_names[ax_idx]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  -> Saved visualization to {out_path}')


def run_test(
    name: str,
    model_path: str,
    norm_stats_path: str,
    images: dict[str, torch.Tensor],
    task: str,
    state: torch.Tensor,
    robot_type: str,
    action_dim: int,
    out_path: str,
    discrete_state_input: bool = False,
    use_delta_actions: bool = True,
    present_img_keys: list[str] | None = None,
) -> None:
    """Run a single PI0 / PI0.5 inference test."""
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')
    print(f'  Model:       {model_path}')
    print(f'  Robot:       {robot_type}')
    print(f'  Action dim:  {action_dim}')
    print(f'  Delta:       {use_delta_actions}')
    print(f'  PI0.5 mode:  {discrete_state_input}')
    print(f'  State shape: {state.shape}')

    state_norm, action_norm = load_norm_stats(norm_stats_path)

    pipe = PI0Pipeline.from_pretrained(
        model_path=model_path,
        tokenizer_model_path=TOKENIZER_MODEL_PATH,
        state_norm_stats=state_norm,
        action_norm_stats=action_norm,
        original_action_dim=action_dim,
        discrete_state_input=discrete_state_input,
        device=DEVICE,
        present_img_keys=present_img_keys,
        robot_type=robot_type,
        use_delta_actions=use_delta_actions,
    )
    pipe.compile()

    pred_action = pipe(images, task, state)
    print(f'  Pred shape:  {pred_action.shape}')
    print(f'  Pred sample: {pred_action[0, :action_dim].tolist()}')

    visualize_action(pred_action.detach().cpu().numpy(), out_path, action_dim)

    # Cleanup
    del pipe
    torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    # ==============================================================
    # PI0 Tests — Aloha / Libero / DROID
    # ==============================================================

    # ----- PI0 + Aloha -----
    aloha_data = json.load(open(ALOHA_STATE_PATH, 'r'))
    aloha_state = torch.tensor(aloha_data['observation.state'], dtype=torch.float32)
    aloha_images = load_images(ALOHA_IMG_FILES)

    run_test(
        name='PI0 + Aloha (dual-arm, 14-dim)',
        model_path=ALOHA_PI0_MODEL_PATH,
        norm_stats_path=ALOHA_PI0_NORM_STATS_PATH,
        images=aloha_images,
        task=ALOHA_DEFAULT_PROMPT,
        state=aloha_state,
        robot_type='aloha',
        action_dim=ALOHA_ACTION_DIM,
        out_path='outputs/pi0_aloha_demo.png',
        discrete_state_input=False,
        use_delta_actions=True,
        present_img_keys=ALOHA_IMG_KEYS,
    )

    # ----- PI0 + Libero -----
    libero_meta = json.load(open(LIBERO_META_PATH, 'r'))
    libero_state = torch.tensor(libero_meta['observation']['state'], dtype=torch.float32)
    libero_task = libero_meta['task']
    libero_images = load_images(LIBERO_IMG_FILES)

    run_test(
        name='PI0 + Libero (single-arm, 7-dim)',
        model_path=LIBERO_PI0_MODEL_PATH,
        norm_stats_path=LIBERO_PI0_NORM_STATS_PATH,
        images=libero_images,
        task=libero_task,
        state=libero_state,
        robot_type='libero',
        action_dim=LIBERO_ACTION_DIM,
        out_path='outputs/pi0_libero_demo.png',
        discrete_state_input=False,
        use_delta_actions=True,
        present_img_keys=LIBERO_IMG_KEYS,
    )

    # ----- PI0 + DROID -----
    droid_data = json.load(open(DROID_STEP_DATA_PATH, 'r'))
    # DROID state = joint_position (7) + gripper_position (1) = 8 dims
    droid_joint = droid_data['observation_numeric']['joint_position']
    droid_gripper = droid_data['observation_numeric']['gripper_position']
    droid_state = torch.tensor(droid_joint + droid_gripper, dtype=torch.float32)
    droid_task = droid_data['language_instruction']
    droid_images = load_images(DROID_IMG_FILES)

    run_test(
        name='PI0 + DROID (joint position, 8-dim)',
        model_path=DROID_PI0_MODEL_PATH,
        norm_stats_path=DROID_PI0_NORM_STATS_PATH,
        images=droid_images,
        task=droid_task,
        state=droid_state,
        robot_type='droid',
        action_dim=DROID_ACTION_DIM,
        out_path='outputs/pi0_droid_demo.png',
        discrete_state_input=False,
        use_delta_actions=True,
        present_img_keys=DROID_IMG_KEYS,
    )

    # ==============================================================
    # PI0.5 Tests — Libero / DROID
    # ==============================================================

    # ----- PI0.5 + Libero -----
    run_test(
        name='PI0.5 + Libero (single-arm, 7-dim, quantile norm)',
        model_path=LIBERO_PI05_MODEL_PATH,
        norm_stats_path=LIBERO_PI05_NORM_STATS_PATH,
        images=libero_images,
        task=libero_task,
        state=libero_state,
        robot_type='libero',
        action_dim=LIBERO_ACTION_DIM,
        out_path='outputs/pi05_libero_demo.png',
        discrete_state_input=True,
        use_delta_actions=False,
        present_img_keys=LIBERO_IMG_KEYS,
    )

    # ----- PI0.5 + DROID -----
    run_test(
        name='PI0.5 + DROID (joint position, 8-dim, quantile norm)',
        model_path=DROID_PI05_MODEL_PATH,
        norm_stats_path=DROID_PI05_NORM_STATS_PATH,
        images=droid_images,
        task=droid_task,
        state=droid_state,
        robot_type='droid',
        action_dim=DROID_ACTION_DIM,
        out_path='outputs/pi05_droid_demo.png',
        discrete_state_input=True,
        use_delta_actions=True,
        present_img_keys=DROID_IMG_KEYS,
    )

    print(f'\n{"="*60}')
    print('  All tests completed!')
    print(f'{"="*60}')
