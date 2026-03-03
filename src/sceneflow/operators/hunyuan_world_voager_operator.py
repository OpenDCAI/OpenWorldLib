import numpy as np
from PIL import Image
import torch
import argparse
import os
import json
import imageio
import pyexr
import cv2

from .base_operator import BaseOperator


def camera_list(
    num_frames=49,
    type="forward",
    Width=512,
    Height=512,
    fx=256,
    fy=256
):
    # assert type in ["forward", "backward", "left", "right", "camera_l", "camera_r"]

    start_pos = np.array([0, 0, 0])
    end_pos = np.array([0, 0, 0])
    if type == "forward":
        end_pos = np.array([0, 0, 1])
    elif type == "backward":
        end_pos = np.array([0, 0, -1])
    elif type == "left":
        end_pos = np.array([-1, 0, 0])
    elif type == "right":
        end_pos = np.array([1, 0, 0])

    cx = Width // 2
    cy = Height // 2

    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    intrinsics = np.stack([intrinsic] * num_frames)

    # Interpolate camera positions along a straight line
    camera_centers = np.linspace(start_pos, end_pos, num_frames)
    target_start = np.array([0, 0, 100])  # Target point
    if type == "camera_l":
        target_end = np.array([-100, 0, 0])
    elif type == "camera_r":
        target_end = np.array([100, 0, 0])
    else:
        target_end = np.array([0, 0, 100])
    target_points = np.linspace(target_start, target_end, num_frames * 2)[:num_frames]

    extrinsics = []
    for t, target_point in zip(camera_centers, target_points):
        if type == "left" or type == "right":
            target_point = t + target_point
        
        z = (target_point - t)
        z = z / np.linalg.norm(z)
        x = np.array([1, 0, 0])
        y = np.cross(z, x)
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)

        R = np.stack([x, y, z], axis=0)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ t
        extrinsics.append(w2c)
    extrinsics = np.stack(extrinsics)

    return intrinsics, extrinsics


class HunyuanWorldVoyagerOperator(BaseOperator):
    def __init__(self, 
                 operation_types=["action_instruction"],
                 interaction_template = ["forward", "backward", "left", "right", "camera_l", "camera_r"]
        ):
        super(HunyuanWorldVoyagerOperator, self).__init__()
        self.interaction_template = interaction_template
        self.interaction_template_init()

        self.opration_types = operation_types

    def check_interaction(self, interaction):
        if interaction not in self.interaction_template:
            raise ValueError(f"Interaction {interaction} not in interaction_template")
        return True
    
    def get_interaction(self, interaction):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def process_interaction(self,
                            num_frames,
                            Width=512,
                            Height=512,
                            fx=256,
                            fy=256
                            ):
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return camera_list(
                    num_frames=num_frames,
                    type=now_interaction,
                    Width=Width,
                    Height=Height,
                    fx=fx,
                    fy=fy)

    def process_perception(self, input_image, device):
        if isinstance(input_image, np.ndarray):
            image_tensor = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        elif isinstance(input_image, Image.Image):
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            input_image = np.array(input_image)
            image_tensor = torch.tensor(input_image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        else:
            image_tensor = input_image.to(device)
        return input_image, image_tensor
