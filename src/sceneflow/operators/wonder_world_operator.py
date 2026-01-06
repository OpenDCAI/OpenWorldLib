from .base_operator import BaseOperator
import torch
import copy
import numpy as np


class WonderWorldOperator(BaseOperator):
    def __init__(self,
                 operation_types=["action_instruction"],
                 interaction_template=["forward", "left", "right", "backward", 
                                     "camera_l", "camera_r", "camera_up", "camera_down"],
                 camera_speed=0.001,
                 rotation_range_theta=0.37,
                 camera_speed_multiplier_rotation=1.5,
                 interp_frames=7,
                 xyz_scale=1000,
                 device="cuda"
        ):
        super().__init__(operation_types)
        self.interaction_template = interaction_template
        self.current_interaction = []
        
        # Camera parameters
        self.camera_speed = camera_speed
        self.rotation_range_theta = rotation_range_theta
        self.camera_speed_multiplier_rotation = camera_speed_multiplier_rotation
        self.interp_frames = interp_frames
        self.xyz_scale = xyz_scale
        self.device = device
        
        # Interaction to rotation path mapping
        self.interaction_to_code = {
            "forward": 0,
            "left": 1,
            "right": -1,
            "backward": 3,
            "camera_l": 2,
            "camera_r": -2,
            "camera_up": 4,
            "camera_down": -4
        }
    
    def check_interaction(self, interaction):
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template")
        return True

    def get_interaction(self, interaction):
        """接收交互指令"""
        if not isinstance(interaction, list):
            interaction = [interaction]
        for act in interaction:
            self.check_interaction(act)
        self.current_interaction.extend(interaction)
    
    def clear_interaction(self):
        """清空当前交互缓存"""
        self.current_interaction = []
    
    def process_interaction(self):
        """
        根据当前的interaction生成对应的相机轨迹视图矩阵
        返回: 4x4视图矩阵列表（flatten成16个元素的list）
        """
        if not self.current_interaction:
            return []
        
        # 1. 将交互转换为rotation_path编码
        rotation_path = []
        for interaction in self.current_interaction:
            code = self.interaction_to_code.get(interaction, 0)
            rotation_path.append(code)
        
        # 2. 初始视图矩阵: 单位矩阵对应 [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        # 这是经过 xy_negate 后的结果
        initial_view_matrix = torch.tensor([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], device=self.device, dtype=torch.float)
        
        # 3. 从初始矩阵反推出原始的R和T
        # 因为 view_matrix_final = view_matrix @ xy_negate
        # 所以 view_matrix = view_matrix_final @ xy_negate.inverse()
        xy_negate_matrix = torch.tensor([
            [-1, 0, 0, 0], 
            [0, -1, 0, 0], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ], device=self.device, dtype=torch.float)
        
        # xy_negate 的逆矩阵就是它自己（对角矩阵且对角元素为±1）
        initial_matrix_before_negate = initial_view_matrix @ xy_negate_matrix
        
        current_R = initial_matrix_before_negate[:3, :3].clone()
        current_T = initial_matrix_before_negate[3, :3].unsqueeze(0) / self.xyz_scale
        
        # 4. 生成相机轨迹
        view_matrices = []
        reference_R = current_R.clone()
        reference_T = current_T.clone()
        
        move_left_count = 0
        move_right_count = 0
        move_up_count = 0
        move_down_count = 0
        
        for rotation in rotation_path:
            # Forward (code: 0)
            if rotation == 0:
                forward_speed_multiplier = -1.0
                right_multiplier = 0
                up_multiplier = 0
                camera_speed = self.camera_speed
                
                if any([move_left_count, move_right_count, move_up_count, move_down_count]):
                    current_R = reference_R.clone()
                    current_T = reference_T.clone()
                    move_left_count = move_right_count = move_up_count = move_down_count = 0
            
            # Left/Right rotation (code: ±1)
            elif abs(rotation) == 1:
                if any([move_left_count, move_right_count, move_up_count, move_down_count]):
                    current_R = reference_R.clone()
                    current_T = reference_T.clone()
                    move_left_count = move_right_count = move_up_count = move_down_count = 0
                
                theta_frame = torch.tensor(self.rotation_range_theta / (self.interp_frames + 1)) * rotation
                sin = torch.sum(torch.stack([torch.sin(i * theta_frame) for i in range(1, self.interp_frames + 2)]))
                cos = torch.sum(torch.stack([torch.cos(i * theta_frame) for i in range(1, self.interp_frames + 2)]))
                
                forward_speed_multiplier = -1.0 / (self.interp_frames + 1) * cos.item()
                right_multiplier = -1.0 / (self.interp_frames + 1) * sin.item()
                up_multiplier = 0
                camera_speed = self.camera_speed * self.camera_speed_multiplier_rotation
                
                theta = torch.tensor(self.rotation_range_theta * rotation)
                rotation_matrix = torch.tensor([
                    [torch.cos(theta), 0, torch.sin(theta)],
                    [0, 1, 0],
                    [-torch.sin(theta), 0, torch.cos(theta)]
                ], device=self.device, dtype=torch.float)
                current_R = rotation_matrix @ current_R
            
            # Camera left/right (code: ±2)
            elif abs(rotation) == 2:
                if rotation > 0:
                    move_left_count += 1
                else:
                    move_right_count += 1
                
                if (rotation > 0 and move_right_count != 0) or (rotation < 0 and move_left_count != 0):
                    current_R = reference_R.clone()
                    current_T = reference_T.clone()
                    move_right_count = move_left_count = 0
                
                forward_speed_multiplier = right_multiplier = up_multiplier = 0
                camera_speed = 0
                
                theta = torch.tensor(self.rotation_range_theta * rotation / 2)
                rotation_matrix = torch.tensor([
                    [torch.cos(theta), 0, torch.sin(theta)],
                    [0, 1, 0],
                    [-torch.sin(theta), 0, torch.cos(theta)]
                ], device=self.device, dtype=torch.float)
                current_R = rotation_matrix @ current_R
            
            # Backward (code: 3)
            elif rotation == 3:
                continue
            
            # Camera up/down (code: ±4)
            elif abs(rotation) == 4:
                if rotation > 0:
                    move_up_count += 1
                else:
                    move_down_count += 1
                
                if (rotation > 0 and move_down_count != 0) or (rotation < 0 and move_up_count != 0):
                    current_R = reference_R.clone()
                    current_T = reference_T.clone()
                    move_up_count = move_down_count = 0
                
                forward_speed_multiplier = right_multiplier = up_multiplier = 0
                camera_speed = 0
                
                theta = torch.tensor(self.rotation_range_theta * rotation / 4)
                rotation_matrix = torch.tensor([
                    [1, 0, 0],
                    [0, torch.cos(theta), -torch.sin(theta)],
                    [0, torch.sin(theta), torch.cos(theta)]
                ], device=self.device, dtype=torch.float)
                current_R = rotation_matrix @ current_R
            
            else:
                forward_speed_multiplier = right_multiplier = up_multiplier = 0
                camera_speed = 0
            
            # Apply translation
            move_dir = torch.tensor([
                [-right_multiplier, -up_multiplier, -forward_speed_multiplier]
            ], device=self.device, dtype=torch.float)
            current_T = current_T + camera_speed * move_dir
            
            # 构建4x4视图矩阵
            view_matrix = torch.eye(4, device=self.device, dtype=torch.float)
            view_matrix[:3, :3] = current_R
            view_matrix[3, :3] = current_T.squeeze() * self.xyz_scale
            
            # 应用xy取反
            view_matrix_final = view_matrix @ xy_negate_matrix
            
            # 转换为列表格式（16个元素）
            view_matrices.append(view_matrix_final.flatten().tolist())
            
            # Update reference
            reference_R = current_R.clone()
            reference_T = current_T.clone()
        
        return view_matrices

    def process_perception(self, image):
        """处理感知信息（预留接口）"""
        pass