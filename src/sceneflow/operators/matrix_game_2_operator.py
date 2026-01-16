from .base_operator import BaseOperator

import torch
from torchvision.transforms import v2
import random

def combine_data(data, num_frames=57, keyboard_dim=6, mouse=True):
    assert num_frames % 4 == 1
    keyboard_condition = torch.zeros((num_frames, keyboard_dim))
    if mouse == True:
        mouse_condition = torch.zeros((num_frames, 2))
    
    current_frame = 0
    selections = [12]

    while current_frame < num_frames:
        rd_frame = selections[random.randint(0, len(selections) - 1)]
        rd = random.randint(0, len(data) - 1)
        k = data[rd]['keyboard_condition']
        if mouse == True:
            m = data[rd]['mouse_condition']
        
        if current_frame == 0:
            keyboard_condition[:1] = k[:1]
            if mouse == True:
                mouse_condition[:1] = m[:1]
            current_frame = 1
        else:
            rd_frame = min(rd_frame, num_frames - current_frame)
            repeat_time = rd_frame // 4
            keyboard_condition[current_frame:current_frame+rd_frame] = k.repeat(repeat_time, 1)
            if mouse == True:
                mouse_condition[current_frame:current_frame+rd_frame] = m.repeat(repeat_time, 1)
            current_frame += rd_frame
    if mouse == True:
        return {
                "keyboard_condition": keyboard_condition,
                "mouse_condition": mouse_condition
            }
    return {"keyboard_condition": keyboard_condition}

def Bench_actions_universal(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        # "back",
        "left",
        "right",
    ]
    actions_double_action = [
        "forward_left",
        "forward_right",
        # "back_left",
        # "back_right",
    ]

    actions_single_camera = [   
        "camera_l",
        "camera_r",
        # "camera_ur",
        # "camera_ul",
        # "camera_dl",
        # "camera_dr" 
        # "camera_up",
        # "camera_down",
    ]
    actions_to_test = actions_double_action * 5 + actions_single_camera * 5 + actions_single_action * 5
    for action in (actions_single_action + actions_double_action):
        for camera in (actions_single_camera):
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    # print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1, "left": 2, "right": 3
    }

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_up":  [CAM_VALUE, 0],
        "camera_down": [-CAM_VALUE, 0],
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
        "camera_ur":  [CAM_VALUE, CAM_VALUE],
        "camera_ul":  [CAM_VALUE, -CAM_VALUE],
        "camera_dr":  [-CAM_VALUE, CAM_VALUE],
        "camera_dl":  [-CAM_VALUE, -CAM_VALUE],
    }

    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0, 0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=4, mouse=True)


def Bench_actions_gta_drive(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        "back",
    ]

    actions_single_camera = [   
        "camera_l",
        "camera_r",
    ]
    actions_to_test = actions_single_camera * 2 + actions_single_action * 2
    for action in (actions_single_action):
        for camera in (actions_single_camera):
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    # print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1
    }

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
    }
    
    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=2, mouse=True)

def Bench_actions_templerun(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "jump",
        "slide",
        "leftside",
        "rightside",
        "turnleft",
        "turnright",
        "nomove"
    ]

    actions_to_test = actions_single_action

    base_action = actions_single_action

    KEYBOARD_IDX = { 
        "nomove": 0, "jump": 1, "slide": 2, "turnleft": 3,
        "turnright": 4, "leftside": 5, "rightside": 6
    }

    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=7, mouse=False)


def encode_actions(action_list, mode):
    """
    将一个动作 list 编码成 keyboard_condition / mouse_condition (一次性的)
    """
    if mode == "universal":
        KEYBOARD_IDX = {"forward":0, "back":1, "left":2, "right":3}
        CAM_MAP = {"camera_l":[0,-0.1], "camera_r":[0,0.1]}
        keyboard_dim = 4
        mouse = True
        COMBINATION_MAP = {
            "forward_left": ["forward", "left"],
            "forward_right": ["forward", "right"],
            "back_left": ["back", "left"], 
            "back_right": ["back", "right"]
        }
    elif mode == "gta_drive":
        KEYBOARD_IDX = {"forward":0, "back":1}
        CAM_MAP = {"camera_l":[0,-0.1], "camera_r":[0,0.1]}
        keyboard_dim = 2
        mouse = True
    else: # templerun
        KEYBOARD_IDX = {
            "nomove":0,"jump":1,"slide":2,
            "turnleft":3,"turnright":4,
            "leftside":5,"rightside":6,
        }
        CAM_MAP = {}
        keyboard_dim = 7
        mouse = False

    keyboard = torch.zeros(keyboard_dim)
    if mouse:
        mouse_value = torch.zeros(2)

    for act in action_list:
        if act in COMBINATION_MAP:
            for sub_act in COMBINATION_MAP[act]:
                if sub_act in KEYBOARD_IDX:
                    keyboard[KEYBOARD_IDX[sub_act]] = 1
        if act in KEYBOARD_IDX:
            keyboard[KEYBOARD_IDX[act]] = 1
        if mouse and act in CAM_MAP:
            mouse_value[:] = torch.tensor(CAM_MAP[act])

    if mouse:
        return keyboard, mouse_value
    return keyboard, None


def resizecrop(image, th, tw):
    w, h = image.size
    if h / w > th / tw:
        new_w = int(w)
        new_h = int(new_w * th / tw)
    else:
        new_h = int(h)
        new_w = int(new_h * tw / th)
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    image = image.crop((left, top, right, bottom))
    return image


class MatrixGame2Operator(BaseOperator):

    def __init__(self, operation_types=[], mode="universal", interaction_template=[]):
        super().__init__(operation_types=operation_types)
        self.mode = mode
        if mode == 'universal':
            interaction_template = ["forward", "left", "right", "forward_left", "forward_right",
                                    "camera_l", "camera_r"]
        elif mode == 'gta_drive':
            interaction_template = ["forward", "back", "camera_l", "camera_r"]
        elif mode == 'templerun':
            interaction_template = ["jump","slide","leftside","rightside",
                                    "turnleft","turnright","nomove"]
        self.interaction_template = interaction_template
        self.interaction_template_init()

        self.current_interaction = []  # 保存用户按顺序输入的动作组

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def check_interaction(self, interaction):
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template")
        return True

    def get_interaction(self, interaction_list):
        # 用户传进来是 list
        for act in interaction_list:
            self.check_interaction(act)
        self.current_interaction.append(interaction_list)

    def _build_sequence(self, num_frames, frames_per_action=4):
        if len(self.current_interaction) == 0:
            raise RuntimeError("No interaction registered")

        cur_interaction = self.current_interaction[-1]

        total_actions = len(cur_interaction)
        available_frames = num_frames
        frames_per_action = max(frames_per_action, available_frames // total_actions)
        
        if frames_per_action < 1:
            frames_per_action = 1

        padded_actions = []
        for action in cur_interaction:
            padded_actions.extend([action] * frames_per_action)

        while len(padded_actions) < num_frames:
            padded_actions.append(padded_actions[-1])

        padded_actions = padded_actions[:num_frames]

        keyboard_list = []
        mouse_list = []
        mouse_enabled = (self.mode != "templerun")
        
        for action in padded_actions:
            kb, ms = encode_actions([action], self.mode)
            keyboard_list.append(kb)
            if mouse_enabled:
                mouse_list.append(ms)
        
        keyboard_tensor = torch.stack(keyboard_list)
        if mouse_enabled:
            mouse_tensor = torch.stack(mouse_list)
            return {
                "keyboard_condition": keyboard_tensor,
                "mouse_condition": mouse_tensor
            }
        
        return {"keyboard_condition": keyboard_tensor}

    # multi_turn 不用额外修改，外部调整num_frames以及输入interaction即可
    def process_action_universal(self, num_frames):
        return self._build_sequence(num_frames)

    def process_action_gta_drive(self, num_frames):
        return self._build_sequence(num_frames)

    def process_action_templerun(self, num_frames):
        return self._build_sequence(num_frames)
    
    def process_interaction(self, num_frames):
        if self.mode == "universal":
            return self.process_action_universal(num_frames)
        elif self.mode == "gta_drive":
            return self.process_action_gta_drive(num_frames)
        elif self.mode == "templerun":
            return self.process_action_templerun(num_frames)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def process_perception(self,
                           input_image,
                           num_output_frames,
                           resize_H=352,
                           resize_W=640,
                           device: str = "cuda",
                           weight_dtype = torch.bfloat16,):
        image = resizecrop(input_image, resize_H, resize_W)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=weight_dtype, device=device)

        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs={"tiled": True, "tile_size": [resize_H//8, resize_W//8], "tile_stride": [resize_H//16+1, resize_W//16-2]}

        return {
            "image": image,
            "img_cond": img_cond,
            "tiler_kwargs": tiler_kwargs
        }
