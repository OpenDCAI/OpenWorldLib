import torch
import numpy as np
import copy
import torch.nn.functional as F
from pytorch3d.renderer import PerspectiveCameras

class WonderJourneyOperator:
    def __init__(self, device='cuda', **kwargs):
        self.device = device
        
        self.interaction_template = {
            "movement": ["straight", "turn_right", "turn_left", "stop"],
            "text": str, 
            "config": ["load_trajectory"]
        }
        
        self.current_interaction = []
        
        self.current_camera = None
        self.predefined_cameras = [] # 存储预设轨迹
        self.use_predefined = False  # 是否开启用户上传轨迹模式
        
        self.config = {
            "init_focal_length": 512,
            "forward_speed_multiplier": 0.05, 
            "camera_speed_multiplier_rotation": 0.8, 
            "rotation_range_theta": 0.05, # 
            "frames_per_scene": 60, # 一个动作持续多少帧
            "acceleration_frames": 10, # 加速/减速缓冲帧数 (原代码是 total/2)
            "random_walk_scale_vertical": 0.2,# 随机扰动幅度
            "right_multiplier": 0.0,
        }
        self.config.update(kwargs)
        
        # 用于实现加减速和连贯性的部分
        self.state = {
            "frame_counter": 0,# 当前片段执行到了第几帧 (rc)
            "total_frames": 0,
            "is_rotating": False,
            "rotation_dir": 0,
            "speed_up": True,# 是否启用起步加速
            "speed_down": True,# 是否启用结束减速
            "global_frame_idx": 0# 全局帧数 (如果开启 predefined 模式的话)
        }

    def check_interaction(self, interaction):
        i_type = interaction.get("type")
        content = interaction.get("content")
        
        if i_type == "movement":
            if content not in self.interaction_template["movement"]:
                raise ValueError(f"Movement '{content}' not in template")
        elif i_type == "config":
            if content not in self.interaction_template["config"]:
                raise ValueError(f"Config command '{content}' not in template")
        
        return True

    def get_interaction(self, interaction):

        self.check_interaction(interaction)
        self.current_interaction.append(interaction)
        
        i_type = interaction.get("type")
        content = interaction.get("content")

        # 预设轨迹加载
        if i_type == "config" and content == "load_trajectory":
            paths = interaction.get("paths", {})
            self.load_predefined_trajectory(paths.get("intrinsics"), paths.get("extrinsics"))
            self.use_predefined = True
            print("Switched to Predefined Trajectory Mode.")
            return

        # 处理运动指令 (重置状态机以触发加减速逻辑)
        if i_type == "movement":
            self.use_predefined = False 
            frames = interaction.get("frames", self.config["frames_per_scene"])
            
            # 重置计数器
            self.state["frame_counter"] = 0 
            self.state["total_frames"] = frames
            self.state["speed_up"] = True   
            self.state["speed_down"] = True
            
            if content == "straight":
                self.state["is_rotating"] = False
                self.state["rotation_dir"] = 0
            elif content == "turn_right":
                self.state["is_rotating"] = True
                self.state["rotation_dir"] = 1
            elif content == "turn_left":
                self.state["is_rotating"] = True
                self.state["rotation_dir"] = -1
            elif content == "stop":
                self.state["is_rotating"] = False
                self.state["rotation_dir"] = 0

    def init_camera(self):
        # FrameSyn.get_init_camera
        focal = self.config["init_focal_length"]
        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = focal
        K[0, 1, 1] = focal
        K[0, 0, 2] = 256
        K[0, 1, 2] = 256
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        self.current_camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device)
        return self.current_camera

    def load_predefined_trajectory(self, intrinsics_path, extrinsics_path):
        # FrameSyn.__init__ 中的 predefined 部分
        if not intrinsics_path or not extrinsics_path:
            raise ValueError("Intrinsics or Extrinsics path missing.")
            
        intrinsics = np.load(intrinsics_path).astype(np.float32)
        extrinsics = np.load(extrinsics_path).astype(np.float32)

        intrinsics = torch.from_numpy(intrinsics).to(self.device)
        extrinsics = torch.from_numpy(extrinsics).to(self.device)

        # Pad to 4x4
        Ks = F.pad(intrinsics, (0, 1, 0, 1), value=0)
        Ks[:, 2, 3] = Ks[:, 3, 2] = 1

        Rs, ts = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
        # PerspectiveCameras operate on row-vector matrices
        Rs = Rs.movedim(1, 2)

        self.predefined_cameras = [
            PerspectiveCameras(K=K.unsqueeze(0), R=R.T.unsqueeze(0), T=t.unsqueeze(0), device=self.device)
            for K, R, t in zip(Ks, Rs, ts)
        ]
        self.state["global_frame_idx"] = 0
        print(f"Loaded {len(self.predefined_cameras)} frames from trajectory.")

    def process_interaction(self):
        # 计算下一帧相机状态

        if self.current_camera is None:
            self.init_camera()

        # 如果：预设轨迹模式
        if self.use_predefined:
            idx = self.state["global_frame_idx"]
            if idx < len(self.predefined_cameras):
                self.current_camera = self.predefined_cameras[idx]
                self.state["global_frame_idx"] += 1
                return self.current_camera
            else:
                return self.current_camera

        # 如果: 不用用户自己的轨迹
        if not self.current_interaction:
            return self.current_camera

        next_camera = copy.deepcopy(self.current_camera)
        
        rc = self.state["frame_counter"]
        total_frames = self.state["total_frames"]
        is_rotating = self.state["is_rotating"]
        rotation_dir = self.state["rotation_dir"]
        
        v = self.config['forward_speed_multiplier']
        theta_range = self.config["rotation_range_theta"]
        
        # Move logic
        
        # 旋转
        if is_rotating:
            # FrameSyn.get_next_camera_rotation(if is rotating)
            theta = torch.tensor(theta_range * rotation_dir)
            rotation_matrix = torch.tensor(
                [[torch.cos(theta), 0, torch.sin(theta)], 
                 [0, 1, 0], 
                 [-torch.sin(theta), 0, torch.cos(theta)]],
                device=self.device,
            )
            # 更新朝向 R + 计算朝向
            next_camera.R[0] = rotation_matrix @ next_camera.R[0]
            theta_current = theta * (total_frames + 2 - rc)
            
            next_camera.move_dir = torch.tensor([
                -v * torch.sin(theta_current).item(), 
                0.0, 
                v * torch.cos(theta_current).item()
            ], device=self.device)
            
        # 直行，包括随机扰动
        else:
            # FrameSyn.get_next_camera_rotation (else branch)
            k = self.config['camera_speed_multiplier_rotation'] # e.g. 0.8
            acceleration_frames = self.config.get("acceleration_frames", total_frames // 2)
            
            # Speed Up
            if self.state["speed_up"] and rc <= acceleration_frames:
                factor = (k + (1-k) * (rc / acceleration_frames))
                next_camera.move_dir = torch.tensor([0.0, 0.0, v * factor], device=self.device)
                
            # Speed Down
            elif self.state["speed_down"] and rc > total_frames - acceleration_frames:
                factor = (k + (1-k) * ((total_frames - rc + 1) / acceleration_frames))
                next_camera.move_dir = torch.tensor([0.0, 0.0, v * factor], device=self.device)
            
            # 匀速
            else:
                next_camera.move_dir = torch.tensor([0.0, 0.0, v], device=self.device)

            theta_wobble = torch.tensor(2 * torch.pi * rc / (total_frames + 1))
            wobble = -self.config["random_walk_scale_vertical"] * 0.01 * torch.sin(theta_wobble).item()
            next_camera.move_dir[1] = wobble # 修改 Y 轴位移

        current_content = self.current_interaction[-1].get("content")
        if current_content == "stop":
             next_camera.move_dir = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        next_camera.T += next_camera.move_dir

        # 更新状态
        self.state["frame_counter"] += 1
        self.current_camera = next_camera
        
        return next_camera

    def delete_last_interaction(self):
        if self.current_interaction:
            self.current_interaction.pop()
            # 重置当前片段的计数器
            self.state["frame_counter"] = 0