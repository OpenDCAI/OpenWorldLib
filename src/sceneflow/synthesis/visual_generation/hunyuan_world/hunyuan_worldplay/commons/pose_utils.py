import json
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from ..generate_custom_trajectory import generate_camera_trajectory_local


_MAPPING = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}


def one_hot_to_one_dimension(one_hot: torch.Tensor) -> torch.Tensor:
    return torch.tensor([_MAPPING[tuple(row.tolist())] for row in one_hot])


def parse_pose_string(pose_string: str) -> list[dict]:
    forward_speed = 0.08
    yaw_speed = np.deg2rad(3)
    pitch_speed = np.deg2rad(3)

    motions: list[dict] = []
    commands = [cmd.strip() for cmd in pose_string.split(",")]

    for cmd in commands:
        if not cmd:
            continue

        parts = cmd.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid pose command: {cmd}. Expected format: 'action-duration'")

        action = parts[0].strip()
        try:
            duration = float(parts[1].strip())
        except ValueError as e:
            raise ValueError(f"Invalid duration in command: {cmd}") from e

        if duration < 0:
            raise ValueError(f"Invalid duration in command: {cmd}")
        num_frames = int(duration)

        if action == "w":
            for _ in range(num_frames):
                motions.append({"forward": forward_speed})
        elif action == "s":
            for _ in range(num_frames):
                motions.append({"forward": -forward_speed})
        elif action == "a":
            for _ in range(num_frames):
                motions.append({"right": -forward_speed})
        elif action == "d":
            for _ in range(num_frames):
                motions.append({"right": forward_speed})
        elif action == "up":
            for _ in range(num_frames):
                motions.append({"pitch": pitch_speed})
        elif action == "down":
            for _ in range(num_frames):
                motions.append({"pitch": -pitch_speed})
        elif action == "left":
            for _ in range(num_frames):
                motions.append({"yaw": -yaw_speed})
        elif action == "right":
            for _ in range(num_frames):
                motions.append({"yaw": yaw_speed})
        else:
            raise ValueError(
                f"Unknown action: {action}. Supported actions: w, s, a, d, up, down, left, right"
            )

    return motions


def pose_string_to_json(pose_string: str) -> dict:
    motions = parse_pose_string(pose_string)
    poses = generate_camera_trajectory_local(motions)

    intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ]

    pose_json: dict = {}
    for i, p in enumerate(poses):
        pose_json[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}

    return pose_json


def _load_pose_json_from_path(path: str) -> dict:
    if not os.path.exists(path):
        raise ValueError(f"Pose json not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def pose_to_input(pose_data, latent_num: int, tps: bool = False):
    if isinstance(pose_data, str):
        if pose_data.endswith(".json"):
            pose_json = _load_pose_json_from_path(pose_data)
        else:
            pose_json = pose_string_to_json(pose_data)
    elif isinstance(pose_data, dict):
        pose_json = pose_data
    else:
        raise ValueError(f"Invalid pose_data type: {type(pose_data)}. Expected str or dict.")

    pose_keys = list(pose_json.keys())
    latent_num_from_pose = len(pose_keys)
    if latent_num_from_pose != latent_num:
        raise ValueError(
            f"pose corresponds to {latent_num_from_pose * 4 - 3} frames, num_frames "
            f"must be set to {latent_num_from_pose * 4 - 3} to ensure alignment."
        )

    intrinsic_list = []
    w2c_list = []
    for i in range(latent_num):
        t_key = pose_keys[i]
        c2w = np.array(pose_json[t_key]["extrinsic"])
        w2c = np.linalg.inv(c2w)
        w2c_list.append(w2c)
        intrinsic = np.array(pose_json[t_key]["K"])
        intrinsic[0, 0] /= intrinsic[0, 2] * 2
        intrinsic[1, 1] /= intrinsic[1, 2] * 2
        intrinsic[0, 2] = 0.5
        intrinsic[1, 2] = 0.5
        intrinsic_list.append(intrinsic)

    w2c_list = np.array(w2c_list)
    intrinsic_list = torch.tensor(np.array(intrinsic_list))

    c2ws = np.linalg.inv(w2c_list)
    C_inv = np.linalg.inv(c2ws[:-1])
    relative_c2w = np.zeros_like(c2ws)
    relative_c2w[0, ...] = c2ws[0, ...]
    relative_c2w[1:, ...] = C_inv @ c2ws[1:, ...]
    trans_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
    rotate_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)

    move_norm_valid = 0.0001
    for i in range(1, relative_c2w.shape[0]):
        move_dirs = relative_c2w[i, :3, 3]
        move_norms = np.linalg.norm(move_dirs)
        if move_norms > move_norm_valid:
            move_norm_dirs = move_dirs / move_norms
            angles_rad = np.arccos(move_norm_dirs.clip(-1.0, 1.0))
            trans_angles_deg = angles_rad * (180.0 / torch.pi)
        else:
            trans_angles_deg = torch.zeros(3)

        R_rel = relative_c2w[i, :3, :3]
        r = R.from_matrix(R_rel)
        rot_angles_deg = r.as_euler("xyz", degrees=True)

        if move_norms > move_norm_valid:
            if (not tps) or (tps and abs(rot_angles_deg[1]) < 5e-2 and abs(rot_angles_deg[0]) < 5e-2):
                if trans_angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1
                elif trans_angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1

                if trans_angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1
                elif trans_angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1

        if rot_angles_deg[1] > 5e-2:
            rotate_one_hot[i, 0] = 1
        elif rot_angles_deg[1] < -5e-2:
            rotate_one_hot[i, 1] = 1

        if rot_angles_deg[0] > 5e-2:
            rotate_one_hot[i, 2] = 1
        elif rot_angles_deg[0] < -5e-2:
            rotate_one_hot[i, 3] = 1

    trans_one_hot = torch.tensor(trans_one_hot)
    rotate_one_hot = torch.tensor(rotate_one_hot)

    trans_one_label = one_hot_to_one_dimension(trans_one_hot)
    rotate_one_label = one_hot_to_one_dimension(rotate_one_hot)
    action_one_label = trans_one_label * 9 + rotate_one_label

    return torch.as_tensor(w2c_list), torch.as_tensor(intrinsic_list), action_one_label


def pose_to_latent_num(pose_data) -> int:
    if isinstance(pose_data, str):
        if pose_data.endswith(".json"):
            pose_json = _load_pose_json_from_path(pose_data)
        else:
            pose_json = pose_string_to_json(pose_data)
    elif isinstance(pose_data, dict):
        pose_json = pose_data
    else:
        raise ValueError(f"Invalid pose_data type: {type(pose_data)}. Expected str or dict.")
    return len(list(pose_json.keys()))
