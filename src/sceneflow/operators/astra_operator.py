import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from einops import rearrange
import os
import json
import imageio
from pathlib import Path
from torchvision.transforms import v2
from typing import Optional


class AstraOperator(object): 
    def __init__(self, device="cuda"):
        self.device = device
        self.interaction_template = ["camera_l", "camera_r", "forward", "backward", "forward_left", "forward_right", "s_curve", "left_right"]
        self.current_interaction = []
        
        # 预处理转换 (来自 InlineVideoEncoder)
        self.frame_process = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def check_interaction(self, interaction):
        """
        检验输入是否符合astra输入格式
        """
        if interaction not in self.interaction_template:
            raise ValueError(f"Invalid interaction: {interaction}. Allowed: {self.interaction_template}")
        return True

    def get_interaction(self, interaction):
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, modality_type="sekai", start_frame=0, initial_condition_frames=1, total_frames_to_generate=8, use_real_poses=False, scene_info_path=None, encoded_data=None):
        """
        根据当前的 interaction (direction) 生成对应的 Camera Embedding。
        """
        direction = self.current_interaction[-1] if self.current_interaction else "forward"
        model_dtype = torch.bfloat16 
        
        # Load scene info for nuscenes if needed
        scene_info = None
        if modality_type == "nuscenes" and scene_info_path and os.path.exists(scene_info_path):
            with open(scene_info_path, 'r') as f:
                scene_info = json.load(f)

        if modality_type == "sekai":
            camera_embedding_full = self.generate_sekai_camera_embeddings_sliding(
                encoded_data.get('cam_emb', None) if encoded_data else None,
                start_frame,
                initial_condition_frames,
                total_frames_to_generate,
                0,
                use_real_poses=use_real_poses,
                direction=direction
            )
        elif modality_type == "nuscenes":
            camera_embedding_full = self.generate_nuscenes_camera_embeddings_sliding(
                scene_info,
                start_frame,
                initial_condition_frames,
                total_frames_to_generate
            )
        elif modality_type == "openx":
            camera_embedding_full = self.generate_openx_camera_embeddings_sliding(
                encoded_data,
                start_frame,
                initial_condition_frames,
                total_frames_to_generate,
                use_real_poses=use_real_poses
            )
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")
            
        return camera_embedding_full.to(self.device, dtype=model_dtype)

    def process_perception(self, condition_video=None, condition_image=None):
        """
        加载图像或视频文件，预处理为 Tensor Frames，供 Synthesis 编码使用。
        逻辑来自 InlineVideoEncoder.load_video_frames 和 image_to_frame_stack
        """
        frames = None
        
        if condition_video:
            video_path = Path(condition_video).expanduser().resolve()
            if not video_path.exists():
                raise FileNotFoundError(f"File not Found: {video_path}")
                
            reader = imageio.get_reader(str(video_path))
            frame_list = []
            for frame_data in reader:
                frame = Image.fromarray(frame_data)
                # Crop and Resize
                frame = self._crop_and_resize(frame)
                frame_list.append(self.frame_process(frame))
            reader.close()
            
            if frame_list:
                frames = torch.stack(frame_list, dim=0)
                frames = rearrange(frames, "T C H W -> C T H W")
                
        elif condition_image:
            image_path = Path(condition_image).expanduser().resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"File not Found: {image_path}")
            
            image = Image.open(str(image_path)).convert("RGB")
            # Crop and Resize
            image = self._crop_and_resize(image)
            frame = self.frame_process(image)
            # Repeat 10 times (Default in infer_demo.py)
            frames = torch.stack([frame for _ in range(10)], dim=0)
            frames = rearrange(frames, "T C H W -> C T H W")
        
        return frames

    @staticmethod
    def _crop_and_resize(image: Image.Image) -> Image.Image:
        """来自 InlineVideoEncoder 的静态方法"""
        target_w, target_h = 832, 480
        return v2.functional.resize(
            image,
            (round(target_h), round(target_w)),
            interpolation=v2.InterpolationMode.BILINEAR,
        )

    def delete_last_interaction(self):
        self.current_interaction = self.current_interaction[:-1]
    
    def compute_relative_pose(self, pose_a, pose_b, use_torch=False):
        """Compute relative pose matrix of camera B with respect to camera A"""
        assert pose_a.shape == (4, 4), f"Camera A extrinsic matrix should be (4,4), got {pose_a.shape}"
        assert pose_b.shape == (4, 4), f"Camera B extrinsic matrix should be (4,4), got {pose_b.shape}"
        if use_torch:
            if not isinstance(pose_a, torch.Tensor): pose_a = torch.from_numpy(pose_a).float()
            if not isinstance(pose_b, torch.Tensor): pose_b = torch.from_numpy(pose_b).float()
            pose_a_inv = torch.inverse(pose_a)
            relative_pose = torch.matmul(pose_b, pose_a_inv)
        else:
            if not isinstance(pose_a, np.ndarray): pose_a = np.array(pose_a, dtype=np.float32)
            if not isinstance(pose_b, np.ndarray): pose_b = np.array(pose_b, dtype=np.float32)
            pose_a_inv = np.linalg.inv(pose_a)
            relative_pose = np.matmul(pose_b, pose_a_inv)
        return relative_pose

    def generate_sekai_camera_embeddings_sliding(self, cam_data, start_frame, initial_condition_frames, new_frames, total_generated, use_real_poses=True, direction="left"):
        """Generate camera embeddings for Sekai dataset - sliding window version"""
        time_compression_ratio = 4
        framepack_needed_frames = 1 + 16 + 2 + 1 + new_frames
        
        if use_real_poses and cam_data is not None and 'extrinsic' in cam_data:
            print("🔧 Using real Sekai camera data")
            cam_extrinsic = cam_data['extrinsic']
            max_needed_frames = max(start_frame + initial_condition_frames + new_frames, framepack_needed_frames, 30)
            relative_poses = []
            for i in range(max_needed_frames):
                frame_idx = i * time_compression_ratio
                next_frame_idx = frame_idx + time_compression_ratio
                if next_frame_idx < len(cam_extrinsic):
                    cam_prev = cam_extrinsic[frame_idx]
                    cam_next = cam_extrinsic[next_frame_idx]
                    relative_pose = self.compute_relative_pose(cam_prev, cam_next)
                    relative_poses.append(torch.as_tensor(relative_pose[:3, :]))
                else:
                    relative_poses.append(torch.zeros(3, 4))
            pose_embedding = torch.stack(relative_poses, dim=0)
            pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
            mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
            condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
            mask[start_frame:condition_end] = 1.0
            camera_embedding = torch.cat([pose_embedding, mask], dim=1)
            return camera_embedding.to(torch.bfloat16)
        else:
            max_needed_frames = max(start_frame + initial_condition_frames + new_frames, framepack_needed_frames, 30)
            print(f"🔧 Generating Sekai synthetic camera frames: {max_needed_frames}, Direction: {direction}")
            CONDITION_FRAMES = initial_condition_frames
            STAGE_1 = new_frames//2
            STAGE_2 = new_frames - STAGE_1
            relative_poses = []
            
            # --- START of infer_demo.py logic copy ---
            for i in range(max_needed_frames):
                pose = np.eye(4, dtype=np.float32)
                if direction=="forward":
                    if i >= CONDITION_FRAMES and i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                        pose[2, 3] = -0.03
                elif direction == "backward":
                    if i >= CONDITION_FRAMES and i < CONDITION_FRAMES + STAGE_1 + STAGE_2:
                        pose[2, 3] = +0.03
                elif direction=="camera_l":
                    if i >= CONDITION_FRAMES and i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                        yaw_per_frame = 0.03
                        pose[0, 0] = np.cos(yaw_per_frame); pose[0, 2] = np.sin(yaw_per_frame)
                        pose[2, 0] = -np.sin(yaw_per_frame); pose[2, 2] = np.cos(yaw_per_frame)
                        pose[2, 3] = -0.00
                elif direction=="camera_r":
                    if i >= CONDITION_FRAMES and i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                        yaw_per_frame = -0.03
                        pose[0, 0] = np.cos(yaw_per_frame); pose[0, 2] = np.sin(yaw_per_frame)
                        pose[2, 0] = -np.sin(yaw_per_frame); pose[2, 2] = np.cos(yaw_per_frame)
                        pose[2, 3] = -0.00
                elif direction=="forward_left":
                     if i >= CONDITION_FRAMES and i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                        yaw_per_frame = 0.03
                        forward_speed = 0.03
                        pose[0, 0] = np.cos(yaw_per_frame); pose[0, 2] = np.sin(yaw_per_frame)
                        pose[2, 0] = -np.sin(yaw_per_frame); pose[2, 2] = np.cos(yaw_per_frame)
                        pose[2, 3] = -forward_speed
                elif direction=="forward_right":
                     if i >= CONDITION_FRAMES and i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                        yaw_per_frame = -0.03
                        forward_speed = 0.03
                        pose[0, 0] = np.cos(yaw_per_frame); pose[0, 2] = np.sin(yaw_per_frame)
                        pose[2, 0] = -np.sin(yaw_per_frame); pose[2, 2] = np.cos(yaw_per_frame)
                        pose[2, 3] = -forward_speed
                elif direction=="s_curve":
                     yaw_per_frame = 0.03 if i < CONDITION_FRAMES+STAGE_1 else -0.03
                     if i >= CONDITION_FRAMES and i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                         cos_yaw = np.cos(yaw_per_frame); sin_yaw = np.sin(yaw_per_frame)
                         pose[0, 0] = cos_yaw; pose[0, 2] = sin_yaw
                         pose[2, 0] = -sin_yaw; pose[2, 2] = cos_yaw
                         pose[2, 3] = -0.03
                         if i >= CONDITION_FRAMES+STAGE_1 and i < CONDITION_FRAMES+STAGE_1+STAGE_2//3:
                             pose[0, 3] = -0.01 
                elif direction=="left_right":
                     yaw_per_frame = 0.03 if i < CONDITION_FRAMES+STAGE_1 else -0.03
                     if i >= CONDITION_FRAMES and i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                         cos_yaw = np.cos(yaw_per_frame); sin_yaw = np.sin(yaw_per_frame)
                         pose[0, 0] = cos_yaw; pose[0, 2] = sin_yaw
                         pose[2, 0] = -sin_yaw; pose[2, 2] = cos_yaw
                         pose[2, 3] = -0.00
                
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
            # --- END of infer_demo.py logic copy ---

            pose_embedding = torch.stack(relative_poses, dim=0)
            pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
            mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
            condition_end = min(start_frame + initial_condition_frames + 1, max_needed_frames)
            mask[start_frame:condition_end] = 1.0
            camera_embedding = torch.cat([pose_embedding, mask], dim=1)
            return camera_embedding.to(torch.bfloat16)

    def generate_openx_camera_embeddings_sliding(self, encoded_data, start_frame, initial_condition_frames, new_frames, use_real_poses):
        """Generate camera embeddings for OpenX dataset"""
        # (完整逻辑请保留 infer_demo.py 中的内容，此处仅为结构示例，实际操作中需粘贴全部代码)
        time_compression_ratio = 4
        framepack_needed_frames = 1 + 16 + 2 + 1 + new_frames
        
        if use_real_poses and encoded_data is not None and 'cam_emb' in encoded_data and 'extrinsic' in encoded_data['cam_emb']:
            print("🔧 Using OpenX real camera data")
            cam_extrinsic = encoded_data['cam_emb']['extrinsic']
            max_needed_frames = max(start_frame + initial_condition_frames + new_frames, framepack_needed_frames, 30)
            relative_poses = []
            for i in range(max_needed_frames):
                frame_idx = i * time_compression_ratio
                next_frame_idx = frame_idx + time_compression_ratio
                if next_frame_idx < len(cam_extrinsic):
                    cam_prev = cam_extrinsic[frame_idx]
                    cam_next = cam_extrinsic[next_frame_idx]
                    relative_pose = self.compute_relative_pose(cam_prev, cam_next)
                    relative_poses.append(torch.as_tensor(relative_pose[:3, :]))
                else:
                    relative_poses.append(torch.zeros(3, 4))
            pose_embedding = torch.stack(relative_poses, dim=0)
            pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
            mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
            mask[start_frame:min(start_frame + initial_condition_frames, max_needed_frames)] = 1.0
            camera_embedding = torch.cat([pose_embedding, mask], dim=1)
            return camera_embedding.to(torch.bfloat16)
        else:
            print("🔧 Using OpenX synthetic camera data")
            max_needed_frames = max(start_frame + initial_condition_frames + new_frames, framepack_needed_frames, 30)
            relative_poses = []
            for i in range(max_needed_frames):
                roll_per_frame = 0.02; pitch_per_frame = 0.01; yaw_per_frame = 0.015; forward_speed = 0.003
                pose = np.eye(4, dtype=np.float32)
                cos_roll = np.cos(roll_per_frame); sin_roll = np.sin(roll_per_frame)
                cos_pitch = np.cos(pitch_per_frame); sin_pitch = np.sin(pitch_per_frame)
                cos_yaw = np.cos(yaw_per_frame); sin_yaw = np.sin(yaw_per_frame)
                pose[0, 0] = cos_yaw * cos_pitch
                pose[0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
                pose[0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
                pose[1, 0] = sin_yaw * cos_pitch
                pose[1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
                pose[1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
                pose[2, 0] = -sin_pitch
                pose[2, 1] = cos_pitch * sin_roll
                pose[2, 2] = cos_pitch * cos_roll
                pose[0, 3] = forward_speed * 0.5
                pose[1, 3] = forward_speed * 0.3
                pose[2, 3] = -forward_speed
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
            pose_embedding = torch.stack(relative_poses, dim=0)
            pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
            mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
            mask[start_frame:min(start_frame + initial_condition_frames, max_needed_frames)] = 1.0
            camera_embedding = torch.cat([pose_embedding, mask], dim=1)
            return camera_embedding.to(torch.bfloat16)

    def generate_nuscenes_camera_embeddings_sliding(self, scene_info, start_frame, initial_condition_frames, new_frames):
        """Generate camera embeddings for NuScenes dataset"""
        time_compression_ratio = 4
        framepack_needed_frames = 1 + 16 + 2 + 1 + new_frames
        
        if scene_info is not None and 'keyframe_poses' in scene_info:
            print("🔧 Using NuScenes real pose data")
            keyframe_poses = scene_info['keyframe_poses']
            if len(keyframe_poses) == 0:
                max_needed_frames = max(framepack_needed_frames, 30)
                pose_sequence = torch.zeros(max_needed_frames, 7, dtype=torch.float32)
                mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
                mask[start_frame:min(start_frame + initial_condition_frames, max_needed_frames)] = 1.0
                camera_embedding = torch.cat([pose_sequence, mask], dim=1)
                return camera_embedding.to(torch.bfloat16)
            
            reference_pose = keyframe_poses[0]
            max_needed_frames = max(framepack_needed_frames, 30)
            pose_vecs = []
            for i in range(max_needed_frames):
                if i < len(keyframe_poses):
                    current_pose = keyframe_poses[i]
                    translation = torch.tensor(np.array(current_pose['translation']) - np.array(reference_pose['translation']), dtype=torch.float32)
                    rotation = torch.tensor(current_pose['rotation'], dtype=torch.float32)
                    pose_vec = torch.cat([translation, rotation], dim=0)
                else:
                    pose_vec = torch.cat([torch.zeros(3, dtype=torch.float32), torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)], dim=0)
                pose_vecs.append(pose_vec)
            pose_sequence = torch.stack(pose_vecs, dim=0)
            mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
            mask[start_frame:min(start_frame + initial_condition_frames, max_needed_frames)] = 1.0
            camera_embedding = torch.cat([pose_sequence, mask], dim=1)
            return camera_embedding.to(torch.bfloat16)
        else:
            print("🔧 Using NuScenes synthetic pose data")
            max_needed_frames = max(framepack_needed_frames, 30)
            pose_vecs = []
            for i in range(max_needed_frames):
                angle = i * 0.04; radius = 15.0
                x = radius * np.sin(angle); y = 0.0; z = radius * (1 - np.cos(angle))
                translation = torch.tensor([x, y, z], dtype=torch.float32)
                yaw = angle + np.pi/2
                rotation = torch.tensor([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)], dtype=torch.float32)
                pose_vec = torch.cat([translation, rotation], dim=0)
                pose_vecs.append(pose_vec)
            pose_sequence = torch.stack(pose_vecs, dim=0)
            mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
            mask[start_frame:min(start_frame + initial_condition_frames, max_needed_frames)] = 1.0
            camera_embedding = torch.cat([pose_sequence, mask], dim=1)
            return camera_embedding.to(torch.bfloat16)
