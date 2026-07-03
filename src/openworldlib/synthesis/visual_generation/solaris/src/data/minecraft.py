
import numpy as np

from .utils import resize_letterbox

CAMERA_SCALER = 360.0 / 2400.0
HOTBAR_KEYS_NUM = 9
ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",  # 11
    "back",  # 12
    "left",  # 13
    "right",  # 14
    "jump",  # 15
    "sneak",  # 16
    "sprint",  # 17
    "swapHands",  # 18
    "attack",  # 19
    "use",  # 20
    "pickItem",  # 21
    "drop",  # 22
    "cameraX",
    "cameraY",
]


KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    # IMPORTANT: must be float dtype; otherwise assigning non-integer degrees
    # (e.g. 5.625) will be truncated when written into this array.
    "camera": np.array([0.0, 0.0], dtype=np.float32),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}


def read_act_slice_vpt(
    actions_json,
    start,
    stop,
):

    attack_is_stuck = False
    last_hotbar = 0

    episode_actions = []
    for i in range(0, stop):
        step_data = actions_json[i]
        if i == 0:
            # Check if attack will be stuck down
            if step_data["mouse"]["newButtons"] == [0]:
                attack_is_stuck = True
        elif attack_is_stuck:
            # Check if we press attack down, then it might not be stuck
            if 0 in step_data["mouse"]["newButtons"]:
                attack_is_stuck = False
        # If still stuck, remove the action
        if attack_is_stuck:
            step_data["mouse"]["buttons"] = [
                button for button in step_data["mouse"]["buttons"] if button != 0
            ]

        action, _ = json_action_to_env_action(step_data)

        # Update hotbar selection
        current_hotbar = step_data["hotbar"]
        if current_hotbar != last_hotbar:
            action["hotbar.{}".format(current_hotbar + 1)] = 1
        last_hotbar = current_hotbar

        if i < start:
            continue

        episode_actions.append(action)
    episode_actions = one_hot_actions(episode_actions)
    return episode_actions


def compress_mouse_linear(degrees):
    """
    Performs linear compression of mouse movement by simple division.
    In Matrix-Game-2, 1 degree corresponds to 1/15 camera conditioning value,
    so the default scaling is 1/15.
    According to the author, the maximum conditioning value is 0.4, so we shoud
    clip the input to 6 degrees?

    Args:
        dx: Mouse movement delta (in pixels)
        scaling: Scaling factor for linear compression (default: 0.01)

    Returns:
        Linearly scaled mouse movement value in [-20.0 * scaling, 20.0 * scaling] *inclusive*
    """
    scaling = 1.0 / 15
    max_val = 20
    degrees = np.clip(degrees, -max_val, max_val)
    return degrees * scaling


def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
    Returns (minerl_action, is_null_action)
    """
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0.0, 0.0], dtype=np.float32)

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    # dy and dx are in pixels (up to 2400)
    # camera actions are in degrees (-180 to 180)
    camera_action[0] = mouse["dx"] * CAMERA_SCALER
    camera_action[1] = mouse["dy"] * CAMERA_SCALER
    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action


def read_obs_slice_decord(
    video,
    start,
    stop,
    resize,
):
    try:
        # Shift frames to the right by one so that frame at t is influenced by action at t.
        frames = video.get_batch(range(start + 1, stop + 1)).asnumpy()
    except Exception as e:
        raise ValueError(
            f"Could not read frames from video: {e}. "
            "Ensure the video is properly loaded and the range is valid."
        )

    episode_frames = np.array(
        [
            resize_letterbox(frame, *resize) if resize is not None else frame
            for frame in frames
        ]
    )
    return episode_frames


def convert_act_slice_mineflayer(
    actions,
):
    """
    Convert MineFlayer action objects to one-hot action array.

    Args:
        actions: List of action dictionaries with MineFlayer format

    Returns:
        2D numpy array with shape (len(actions), len(ACTION_KEYS))
    """
    actions_one_hot = np.zeros((len(actions), len(ACTION_KEYS)), dtype=np.float32)

    for i, action_obj in enumerate(actions):
        action_data = action_obj["action"]

        # Map boolean movement actions to ACTION_KEYS indices
        if action_data["forward"]:
            actions_one_hot[i, ACTION_KEYS.index("forward")] = 1
        if action_data["back"]:
            actions_one_hot[i, ACTION_KEYS.index("back")] = 1
        if action_data["left"]:
            actions_one_hot[i, ACTION_KEYS.index("left")] = 1
        if action_data["right"]:
            actions_one_hot[i, ACTION_KEYS.index("right")] = 1
        if action_data["jump"]:
            actions_one_hot[i, ACTION_KEYS.index("jump")] = 1
        if action_data["sprint"]:
            actions_one_hot[i, ACTION_KEYS.index("sprint")] = 1
        if action_data["sneak"]:
            actions_one_hot[i, ACTION_KEYS.index("sneak")] = 1
        # In VPT's terms on hot slots "attack", "use", and "pickItem" are LMB, RMB, and MMB respectively.
        # So we can map those actions one to one.
        # The remaining mineflayer actions we need to map using the underlying mouse/keyboard buttons.
        # In minecraft/mineflayer:
        # "mount" is RMB, "dismount" is left shift,
        # "place_block" is RMB, "place_entity" is RMB,
        # "mine" is LMB
        if action_data["attack"]:
            actions_one_hot[i, ACTION_KEYS.index("attack")] = 1  # LMB
        if action_data["use"]:
            actions_one_hot[i, ACTION_KEYS.index("use")] = 1  # RMB

        if action_data["mount"]:
            actions_one_hot[i, ACTION_KEYS.index("use")] = 1  # RMB

        if action_data["dismount"]:
            actions_one_hot[i, ACTION_KEYS.index("sneak")] = 1  # left shift

        if action_data["place_block"]:
            actions_one_hot[i, ACTION_KEYS.index("use")] = 1  # RMB
        if action_data["place_entity"]:
            actions_one_hot[i, ACTION_KEYS.index("use")] = 1  # RMB
        if action_data["mine"]:
            actions_one_hot[i, ACTION_KEYS.index("attack")] = 1  # LMB
        for hotbar_idx in range(HOTBAR_KEYS_NUM):
            if action_data.get("hotbar.{}".format(hotbar_idx + 1)):
                actions_one_hot[
                    i, ACTION_KEYS.index("hotbar.{}".format(hotbar_idx + 1))
                ] = 1

        # Convert camera values from radians to degrees (no normalization)
        # MineFlayer camera: [yaw, pitch] in radians
        # ACTION_KEYS: cameraX (yaw), cameraY (pitch)
        yaw_rad = action_data["camera"][0]
        pitch_rad = action_data["camera"][1]

        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)

        actions_one_hot[i, ACTION_KEYS.index("cameraX")] = yaw_deg
        actions_one_hot[i, ACTION_KEYS.index("cameraY")] = pitch_deg

    return actions_one_hot


def one_hot_actions(actions):
    actions_one_hot = np.zeros((len(actions), len(ACTION_KEYS)), dtype=np.float32)
    for i, current_actions in enumerate(actions):
        for j, action_key in enumerate(ACTION_KEYS):
            if action_key.startswith("camera"):
                if action_key == "cameraX":
                    value = current_actions["camera"][0]
                elif action_key == "cameraY":
                    value = current_actions["camera"][1]
                else:
                    raise ValueError(f"Unknown camera action key: {action_key}")
            else:
                value = current_actions[action_key]
                assert 0 <= value <= 1, f"Action value must be in [0, 1] got {value}"
            actions_one_hot[i, j] = value

    return actions_one_hot
