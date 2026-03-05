from .base_operator import BaseOperator
import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as TF
from einops import rearrange
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def interpolate_camera_poses(
    src_indices: np.ndarray, 
    src_rot_mat: np.ndarray, 
    src_trans_vec: np.ndarray, 
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    # Interpolate translation
    interp_func_trans = interp1d(
        src_indices, 
        src_trans_vec, 
        axis=0, 
        kind='linear', 
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    # Interpolate rotation
    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    # Ensure no sudden change in qw
    quats = src_quat_vec.as_quat().copy()  # [N, 4]
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4))
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses).float()


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    Rot = T[:, :3, :3] # [B,3,3]
    trans = T[:, :3, 3:] # [B,3,1]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)[None, :, :].repeat(T.shape[0], 1, 1)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor, 
    framewise: bool = False, 
    normalize_trans: bool = True, 
) -> torch.Tensor:
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    # Ensure identity matrix for 1st frame
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        # Compute pose between i and i+1
        relative_poses_framewise = torch.bmm(SE3_inverse(relative_poses[:-1]), relative_poses[1:])
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans:
        translations = relative_poses[:, :3, 3] # [f, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        # Only normalize when moving
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


@torch.no_grad()
def create_meshgrid(n_frames: int, height: int, width: int, bias: float = 0.5, device='cuda', dtype=torch.float32) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias # [h*w, 2]
    grid_xy = grid_xy[None, ...].repeat(n_frames, 1, 1) # [f, h*w, 2]
    return grid_xy


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
):
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype) # [f, h*w, 2]
    fx, fy, cx, cy = Ks.chunk(4, dim=-1) # [f, 1]

    i = grid_xy[..., 0] # [f, h*w]
    j = grid_xy[..., 1] # [f, h*w]
    zs = torch.ones_like(i) # [f, h*w]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1) # [f, h*w, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True) # [f, h*w, 3]

    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2) # [f, h*w, 3]
    rays_o = c2ws_mat[:, :3, 3] # [f, 3]
    rays_o = rays_o[:, None, :].expand_as(rays_d) # [f, h*w, 3]
    
    plucker_embeddings = torch.cat([rays_o, rays_d], dim=-1) # [f, h*w, 6]
    plucker_embeddings = plucker_embeddings.view([n_frames, height, width, 6]) # [f*h*w, 6]
    return plucker_embeddings


def get_Ks_transformed(
    Ks: torch.Tensor,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
):
    fx, fy, cx, cy = Ks.chunk(4, dim=-1) # [f, 1]

    scale_x = width_resize / width_org
    scale_y = height_resize / height_org

    fx_resize = fx * scale_x
    fy_resize = fy * scale_y
    cx_resize = cx * scale_x
    cy_resize = cy * scale_y

    crop_offset_x = (width_resize - width_final) / 2
    crop_offset_y = (height_resize - height_final) / 2

    cx_final = cx_resize - crop_offset_x
    cy_final = cy_resize - crop_offset_y
    
    Ks_transformed = torch.zeros_like(Ks)
    Ks_transformed[:, 0:1] = fx_resize
    Ks_transformed[:, 1:2] = fy_resize
    Ks_transformed[:, 2:3] = cx_final
    Ks_transformed[:, 3:4] = cy_final

    return Ks_transformed


class TrajectoryGenerator:
    """
    Tool class for generating camera trajectories from text commands.
    OpenCV coordinate system:
        X: Right
        Y: Down
        Z: Forward
    """
    def __init__(self):
        self.trans_speed = 0.1 
        self.rot_speed = np.radians(0.2)
        self.zoom_speed = 0.01

        self.COMMAND_MAPPING = {
            # Translation
            "forward":         np.array([ 0,  0,  1,  0,  0,  0,  0]) * self.trans_speed,
            "backward":        np.array([ 0,  0, -1,  0,  0,  0,  0]) * self.trans_speed,
            "left":            np.array([-1,  0,  0,  0,  0,  0,  0]) * self.trans_speed,
            "right":           np.array([ 1,  0,  0,  0,  0,  0,  0]) * self.trans_speed,
            
            # Diagonal Translation
            "forward_left":    np.array([-1,  0,  1,  0,  0,  0,  0]) * self.trans_speed,
            "forward_right":   np.array([ 1,  0,  1,  0,  0,  0,  0]) * self.trans_speed,
            "backward_left":   np.array([-1,  0, -1,  0,  0,  0,  0]) * self.trans_speed,
            "backward_right":  np.array([ 1,  0, -1,  0,  0,  0,  0]) * self.trans_speed,

            # Camera Pan/Tilt
            "camera_up":        np.array([ 0,  0,  0, -1,  0,  0,  0]) * self.rot_speed,
            "camera_down":      np.array([ 0,  0,  0,  1,  0,  0,  0]) * self.rot_speed,
            "camera_l":         np.array([ 0,  0,  0,  0, -1,  0,  0]) * self.rot_speed,
            "camera_r":         np.array([ 0,  0,  0,  0,  1,  0,  0]) * self.rot_speed,
            
            # Diagonal Rotation
            "camera_ul":        np.array([ 0,  0,  0, -1, -1,  0,  0]) * self.rot_speed,
            "camera_ur":        np.array([ 0,  0,  0, -1,  1,  0,  0]) * self.rot_speed,
            "camera_dl":        np.array([ 0,  0,  0,  1, -1,  0,  0]) * self.rot_speed,
            "camera_dr":        np.array([ 0,  0,  0,  1,  1,  0,  0]) * self.rot_speed,
            
            # Zooming
            "camera_zoom_in":   np.array([ 0,  0,  0,  0,  0,  0,  1]) * self.zoom_speed,
            "camera_zoom_out":  np.array([ 0,  0,  0,  0,  0,  0, -1]) * self.zoom_speed,
        }

    def generate(self, action_list: list, num_frames: int, height: int, width: int):
        """Generate c2ws and Ks matrices"""
        delta_vec = np.zeros(7) # [tx, ty, tz, rx, ry, rz, zoom]
        valid_cmds = 0
        for action in action_list:
            if action in self.COMMAND_MAPPING:
                delta_vec += self.COMMAND_MAPPING[action]
                valid_cmds += 1
            else:
                print(f"[Warning] Unknown action command: {action}")
        
        if valid_cmds == 0:
            pass 

        c2ws = []
        Ks = [] 
        current_pose = np.eye(4)
        
        d_trans = delta_vec[:3]
        d_rot_euler = delta_vec[3:6]
        d_zoom = delta_vec[6]

        base_fx = max(width, height)
        base_fy = base_fx
        cx = width / 2.0
        cy = height / 2.0

        for i in range(num_frames):
            c2ws.append(current_pose.copy())
            
            current_zoom_factor = 1.0 + (d_zoom * i)
            current_zoom_factor = max(0.1, current_zoom_factor) 

            K = np.array([base_fx * current_zoom_factor, base_fy * current_zoom_factor, cx, cy], dtype=np.float32)
            Ks.append(K)
            
            delta_mat = np.eye(4)
            if np.linalg.norm(d_rot_euler) > 1e-8:
                r = Rotation.from_euler('xyz', d_rot_euler, degrees=False)
                delta_mat[:3, :3] = r.as_matrix()
            delta_mat[:3, 3] = d_trans
            current_pose = current_pose @ delta_mat

        c2ws = np.array(c2ws, dtype=np.float32)
        Ks = np.array(Ks, dtype=np.float32) # [num_frames, 4]

        return c2ws, Ks


class LingBotOperator(BaseOperator):
    def __init__(self, operation_types=["textual_instruction", "action_instruction"]):
        super().__init__(operation_types=operation_types)
        # Supported interaction templates: prompt, action_path (file), action_list (commands)
        self.interaction_template = ["prompt", "action_path", "action_list"]
        self.interaction_template_init()
        self.current_interaction = {}
        
        # Initialize trajectory generator
        self.traj_generator = TrajectoryGenerator()

    def check_interaction(self, interaction):
        if not isinstance(interaction, dict):
             raise ValueError("Interaction must be a dictionary")
        return True

    def get_interaction(self, interaction):
        self.check_interaction(interaction)
        self.current_interaction = interaction

    def process_perception(self, input_image, resize_H=480, resize_W=832, device="cuda"):
        if not isinstance(input_image, Image.Image):
            raise ValueError("Input image must be a PIL Image")
        img = input_image.resize((resize_W, resize_H), Image.BICUBIC)
        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)
        return {
            "image_tensor": img_tensor, 
            "pil_image": img
        }

    def process_interaction(self, resize_H=480, resize_W=832, num_frames=81, device="cuda"): 
        prompt = self.current_interaction.get("prompt", "")
        action_path = self.current_interaction.get("action_path", None)
        action_list = self.current_interaction.get("action_list", None)
        
        camera_data = None
        
        c2ws = None
        Ks = None
        
        # Branch 1: Generate from command list
        if action_list is not None and isinstance(action_list, list):
            print(f"Generating trajectory from commands: {action_list}")
            c2ws, Ks_np = self.traj_generator.generate(action_list, num_frames, resize_H, resize_W)
            Ks = torch.from_numpy(Ks_np).float()
        
        # Branch 2: Load from file path (legacy mode)
        elif action_path is not None:
            # Load original data
            c2ws = np.load(os.path.join(action_path, "poses.npy"))
            Ks_orig = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
            
            # Align frame count
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            real_frame_num = min(num_frames, len_c2ws)
            c2ws = c2ws[:real_frame_num]
            
            # Transform intrinsics (adapt to resize)
            Ks = get_Ks_transformed(Ks_orig,
                                    height_org=480,
                                    width_org=832,
                                    height_resize=resize_H,
                                    width_resize=resize_W,
                                    height_final=resize_H,
                                    width_final=resize_W)
            Ks = Ks[0].repeat(len(c2ws), 1) # [N, 4]
            
        
        # Common processing logic (Generate Embedding)
        if c2ws is not None and Ks is not None:
            len_c2ws = len(c2ws)
            
            # 1. Interpolate/smooth trajectory
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
            )
            
            # 2. Compute relative poses
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=False)
            
            # 3. Adjust Ks length to match downsampled c2ws
            Ks_downsampled = Ks[:len(c2ws_infer)]
            
            c2ws_infer = c2ws_infer.to(device)
            Ks_downsampled = Ks_downsampled.to(device)
            
            # 4. Compute Plucker Embedding
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks_downsampled, resize_H, resize_W)
            
            camera_data = {
                "c2ws_plucker_emb": c2ws_plucker_emb
            }

        return {
            "prompt": prompt,
            "camera_data": camera_data
        }