from PIL import Image
from PIL import ImageFilter
import cv2
import numpy as np
import scipy
import scipy.signal
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import torch
import io
import logging
from pathlib import Path

from collections import deque
from torchvision.transforms import ToTensor
import os
import yaml
import shutil
from .general_utils import save_video
from datetime import datetime
from pytorch3d.renderer import PerspectiveCameras
from datetime import datetime
from diffusers.configuration_utils import FrozenDict
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from ..scene.cameras import Camera


def convert_pt3d_cam_to_3dgs_cam(pt3d_cam: PerspectiveCameras, xyz_scale=1):
    transform_matrix_pt3d = pt3d_cam.get_world_to_view_transform().get_matrix()[0]
    transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
    transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale
    transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()
    opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=torch.device('cuda')))
    transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d
    transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
    c2w = np.array(transform_matrix)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    focal_length = pt3d_cam.K[0, 0, 0].item()
    half_img_size_x = pt3d_cam.K[0, 0, 2].item()
    fovx = 2*np.arctan(half_img_size_x / focal_length)
    half_img_size_y = pt3d_cam.K[0, 1, 2].item()
    fovy = 2*np.arctan(half_img_size_y / focal_length)
    tdgs_cam = Camera(R=R, T=T, FoVx=fovx, FoVy=fovy)
    return tdgs_cam


def rotate_pytorch3d_camera(camera:PerspectiveCameras, angle_rad:float, axis='x'):
    """
    Rotate a PyTorch3D camera object around the specified axis by the given angle.
    It should keep its own location in the world frame.
    This means that the following equation should hold:
    x_world @ P_w2c^new = x_world @ P_w2c^old @ P^(-1),
    where P^(-1) denotes the inverse of the desired transform matrix.
    
    Parameters:
        camera (PyTorch3D Camera): The camera object to rotate.
        angle_rad (float): The angle in radians by which to rotate the camera.
        axis (str): The axis around which to rotate the camera. Can be 'x', 'y', or 'z'.
    
    Returns:
        PyTorch3D Camera: The rotated camera object.
    """
    if axis == 'x':
        R = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angle_rad), -torch.sin(angle_rad)],
            [0, torch.sin(angle_rad), torch.cos(angle_rad)]
        ]).float()
    elif axis == 'y':
        R = torch.tensor([
            [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
            [0, 1, 0],
            [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
        ]).float()
    elif axis == 'z':
        R = torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
            [torch.sin(angle_rad), torch.cos(angle_rad), 0],
            [0, 0, 1]
        ]).float()
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    # Construct the pytorch3d-style P matrix from R and T. P=[[R', 0], [T, 1]]
    P = torch.eye(4)
    P[:3, :3] = R.transpose(0, 1)
    Pinv = torch.inverse(P).to(camera.device)

    P_old = camera.get_world_to_view_transform().get_matrix()
    P_new = P_old @ Pinv
    T_new = P_new[:, 3, :3]
    R_new = P_new[:, :3, :3]

    new_camera = camera.clone()
    new_camera.T = T_new
    new_camera.R = R_new
    
    return new_camera


def translate_pytorch3d_camera(camera:PerspectiveCameras, translation:torch.Tensor):
    """
    Translate a PyTorch3D camera object by the given translation vector.
    It should keep its own orientation in the world frame.
    This means that the following equation should hold:
    x_world @ P_w2c^new = x_world @ P_w2c^old @ P^(-1),
    where P^(-1) denotes the inverse of the desired transform matrix.
    
    Parameters:
        camera (PyTorch3D Camera): The camera object to translate.
        translation (torch.Tensor): The translation vector to apply to the camera.
    
    Returns:
        PyTorch3D Camera: The translated camera object.
    """
    # Construct the pytorch3d-style P matrix from R and T. P=[[R', 0], [T, 1]]
    P = torch.eye(4)
    P[3, :3] = translation
    Pinv = torch.inverse(P).to(camera.device)

    P_old = camera.get_world_to_view_transform().get_matrix()
    P_new = P_old @ Pinv
    T_new = P_new[:, 3, :3]
    R_new = P_new[:, :3, :3]

    new_camera = camera.clone()
    new_camera.T = T_new
    new_camera.R = R_new
    
    return new_camera


def find_biggest_connected_inpaint_region(mask):
    H, W = mask.shape
    visited = torch.zeros((H, W), dtype=torch.bool)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # up, right, down, left
    
    def bfs(i, j):
        queue = deque([(i, j)])
        region = []
        
        while queue:
            x, y = queue.popleft()
            if 0 <= x < H and 0 <= y < W and not visited[x, y] and mask[x, y] == 1:
                visited[x, y] = True
                region.append((x, y))
                for dx, dy in directions:
                    queue.append((x + dx, y + dy))
                    
        return region
    
    max_region = []
    
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and not visited[i, j]:
                current_region = bfs(i, j)
                if len(current_region) > len(max_region):
                    max_region = current_region
    
    mask_connected = torch.zeros((H, W)).to(mask.device)
    for x, y in max_region:
        mask_connected[x, y] = 1
    return mask_connected


def edge_pad(img, mask, mode=1):
    if mode == 0:
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res0 = 1 - nmask
        res1 = nmask
        p0 = np.stack(res0.nonzero(), axis=0).transpose()
        p1 = np.stack(res1.nonzero(), axis=0).transpose()
        min_dists, min_dist_idx = cKDTree(p1).query(p0, 1)
        loc = p1[min_dist_idx]
        for (a, b), (c, d) in zip(p0, loc):
            img[a, b] = img[c, d]
    elif mode == 1:
        record = {}
        kernel = [[1] * 3 for _ in range(3)]
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res = scipy.signal.convolve2d(
            nmask, kernel, mode="same", boundary="fill", fillvalue=1
        )
        res[nmask < 1] = 0
        res[res == 9] = 0
        res[res > 0] = 1
        ylst, xlst = res.nonzero()
        queue = [(y, x) for y, x in zip(ylst, xlst)]
        # bfs here
        cnt = res.astype(np.float32)
        acc = img.astype(np.float32)
        step = 1
        h = acc.shape[0]
        w = acc.shape[1]
        offset = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while queue:
            target = []
            for y, x in queue:
                val = acc[y][x]
                for yo, xo in offset:
                    yn = y + yo
                    xn = x + xo
                    if 0 <= yn < h and 0 <= xn < w and nmask[yn][xn] < 1:
                        if record.get((yn, xn), step) == step:
                            acc[yn][xn] = acc[yn][xn] * cnt[yn][xn] + val
                            cnt[yn][xn] += 1
                            acc[yn][xn] /= cnt[yn][xn]
                            if (yn, xn) not in record:
                                record[(yn, xn)] = step
                                target.append((yn, xn))
            step += 1
            queue = target
        img = acc.astype(np.uint8)
    else:
        nmask = mask.copy()
        ylst, xlst = nmask.nonzero()
        yt, xt = ylst.min(), xlst.min()
        yb, xb = ylst.max(), xlst.max()
        content = img[yt : yb + 1, xt : xb + 1]
        img = np.pad(
            content,
            ((yt, mask.shape[0] - yb - 1), (xt, mask.shape[1] - xb - 1), (0, 0)),
            mode="edge",
        )
    return img, mask


def gaussian_noise(img, mask):
    noise = np.random.randn(mask.shape[0], mask.shape[1], 3)
    noise = (noise + 1) / 2 * 255
    noise = noise.astype(np.uint8)
    nmask = mask.copy()
    nmask[mask > 0] = 1
    img = nmask[:, :, np.newaxis] * img + (1 - nmask[:, :, np.newaxis]) * noise
    return img, mask


def cv2_telea(img, mask, radius=5):
    ret = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    return ret, mask


def cv2_ns(img, mask, radius=5):
    ret = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
    return ret, mask


def mean_fill(img, mask):
    avg = img.mean(axis=0).mean(axis=0)
    img[mask < 1] = avg
    return img, mask

def estimate_scale_and_shift(x, y, init_method='identity', optimize_scale=True):
    assert len(x.shape) == 1 and len(y.shape) == 1, "Inputs should be 1D tensors"
    assert x.shape[0] == y.shape[0], "Input tensors should have the same length"

    n = x.shape[0]

    if init_method == 'identity':
        shift_init = 0.
        scale_init = 1.
    elif init_method == 'median':
        shift_init = (torch.median(y) - torch.median(x)).item()
        scale_init = (torch.sum(torch.abs(y - torch.median(y))) / n / (torch.sum(torch.abs(x - torch.median(x))) / n)).item()
    else:
        raise ValueError("init_method should be either 'identity' or 'median'")
    shift = torch.tensor(shift_init).cuda().requires_grad_()
    scale = torch.tensor(scale_init).cuda().requires_grad_()

    # Set optimizer and scheduler
    optimizer = torch.optim.Adam([shift, scale], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    # Optimization loop
    for step in range(1000):  # Set the range to the number of steps you find appropriate
        optimizer.zero_grad()
        if optimize_scale:
            loss = torch.abs((x.detach() + shift) * scale - y.detach()).mean()
        else:
            loss = torch.abs(x.detach() + shift - y.detach()).mean()
        loss.backward()
        if step == 0:
            print(f"Iteration {step + 1}: L1 Loss = {loss.item():.4f}")
        optimizer.step()
        scheduler.step(loss)

        # Early stopping condition if needed
        if step > 20 and scheduler._last_lr[0] < 1e-6:  # You might want to adjust these conditions
            print(f"Iteration {step + 1}: L1 Loss = {loss.item():.4f}")
            break

    if optimize_scale:
        return scale.item(), shift.item()
    else:
        return 1., shift.item()


def save_depth_map(depth_map, file_name, vmin=None, vmax=None, save_clean=False):
    depth_map = np.squeeze(depth_map)
    if depth_map.ndim != 2:
        raise ValueError("Depth map after squeezing must be 2D.")

    dpi = 100  # Adjust this value if necessary
    figsize = (depth_map.shape[1] / dpi, depth_map.shape[0] / dpi)  # Width, Height in inches

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cax = ax.imshow(depth_map, cmap='viridis', vmin=vmin, vmax=vmax)

    if not save_clean:
        # Standard save with labels and color bar
        cbar = fig.colorbar(cax)
        ax.set_title("Depth Map")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
    else:
        # Clean save without labels, color bar, or axis
        plt.axis('off')
        ax.set_aspect('equal', adjustable='box')

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert('RGB')  # Convert to RGB
    img = img.resize((depth_map.shape[1], depth_map.shape[0]), Image.Resampling.LANCZOS)  # Resize to original dimensions
    img.save(file_name, format='png')
    buf.close()
    plt.close()
    return img


"""
Apache-2.0 license
https://github.com/hafriedlander/stable-diffusion-grpcserver/blob/main/sdgrpcserver/services/generate.py
https://github.com/parlance-zz/g-diffuser-bot/tree/g-diffuser-bot-beta2
_handleImageAdjustment
"""

functbl = {
    "gaussian": gaussian_noise,
    "edge_pad": edge_pad,
    "cv2_ns": cv2_ns,
    "cv2_telea": cv2_telea,
}

def soft_stitching(source_img, target_img, mask, blur_size=11, sigma=2.5):
    # Apply Gaussian blur to the mask to create a soft transition area
    # The size of the kernel and the standard deviation can be adjusted
    # for more or less blending

    # blur_size  # Size of the Gaussian kernel, must be odd
    # sigma       # Standard deviation of the Gaussian kernel
    
    # Ensure the mask is float for blurring
    soft_mask = mask.float()

    # Adding padding to reduce edge effects during blurring
    padding = blur_size // 2
    soft_mask = F.pad(soft_mask, (padding, padding, padding, padding), mode='reflect')
    
    # Apply the Gaussian blur
    blurred_mask = gaussian_blur(soft_mask, kernel_size=(blur_size, blur_size), sigma=(sigma, sigma))
    
    # Remove the padding
    blurred_mask = blurred_mask[:, :, padding:-padding, padding:-padding]
    
    # Ensure the mask is within 0 and 1 after blurring
    blurred_mask = torch.clamp(blurred_mask, 0, 1)
    
    # Blend the images based on the blurred mask
    stitched_img = source_img * blurred_mask + target_img * (1 - blurred_mask)
    
    return stitched_img

def prepare_scheduler(scheduler):
    if hasattr(scheduler, "is_scale_input_called"):
        scheduler.is_scale_input_called = True  # to surpress the warning
    return scheduler


def load_example_yaml(example_name, yaml_path):
    with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
    yaml_data = None
    for d in data:
        if d['name'] == example_name:
            yaml_data = d
            break
    return yaml_data


def merge_frames(all_rundir, fps=10, save_dir=None, is_forward=False, save_depth=False, save_gif=True):
    """
    Merge frames from multiple run directories into a single directory with continuous naming.
    
    Parameters:
        all_rundir (list of pathlib.Path): Directories containing the run data.
        save_dir (pathlib.Path): Directory where all frames should be saved.
    """

    # Ensure save_dir/frames exists
    save_frames_dir = save_dir / 'frames'
    save_frames_dir.mkdir(parents=True, exist_ok=True)

    if save_depth:
        save_depth_dir = save_dir / 'depth'
        save_depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize a counter for the new filenames
    global_counter = 0
    
    # Iterate through all provided run directories
    if is_forward:
        all_rundir = all_rundir[::-1]
    for rundir in all_rundir:
        # Ensure the rundir and the frames subdir exist
        if not rundir.exists():
            print(f"Warning: {rundir} does not exist. Skipping...")
            continue
        
        frames_dir = rundir / 'images' / 'frames'
        if not frames_dir.exists():
            print(f"Warning: {frames_dir} does not exist. Skipping...")
            continue

        if save_depth:
            depth_dir = rundir / 'images' / 'depth'
            if not depth_dir.exists():
                print(f"Warning: {depth_dir} does not exist. Skipping...")
                continue
        
        # Get all .png files in the frames directory, assuming no nested dirs
        frame_files = sorted(frames_dir.glob('*.png'), key=lambda x: int(x.stem))
        if save_depth:
            depth_files = sorted(depth_dir.glob('*.png'), key=lambda x: int(x.stem))
        
        # Copy and rename each file
        for i, frame_file in enumerate(frame_files):
            # Form the new path and copy the file
            new_frame_path = save_frames_dir / f"{global_counter}.png"
            shutil.copy(str(frame_file), str(new_frame_path))

            if save_depth:
                # Form the new path and copy the file
                new_depth_path = save_depth_dir / f"{global_counter}.png"
                shutil.copy(str(depth_files[i]), str(new_depth_path))
            
            # Increment the global counter
            global_counter += 1
    
    last_keyframe_name = 'kf1.png' if is_forward else 'kf2.png'
    last_keyframe = all_rundir[-1] / 'images' / last_keyframe_name
    new_frame_path = save_frames_dir / f"{global_counter}.png"
    shutil.copy(str(last_keyframe), str(new_frame_path))

    if save_depth:
        last_depth_name = 'kf1_depth.png' if is_forward else 'kf2_depth.png'
        last_depth = all_rundir[-1] / 'images' / last_depth_name
        new_depth_path = save_depth_dir / f"{global_counter}.png"
        shutil.copy(str(last_depth), str(new_depth_path))

    frames = []
    for frame_file in sorted(save_frames_dir.glob('*.png'), key=lambda x: int(x.stem)):
        frame_image = Image.open(frame_file)
        frame = ToTensor()(frame_image).unsqueeze(0)
        frames.append(frame)

    if save_depth:
        depth = []
        for depth_file in sorted(save_depth_dir.glob('*.png'), key=lambda x: int(x.stem)):
            depth_image = Image.open(depth_file)
            depth_frame = ToTensor()(depth_image).unsqueeze(0)
            depth.append(depth_frame)

    video = (255 * torch.cat(frames, dim=0)).to(torch.uint8).detach().cpu()
    video_reverse = (255 * torch.cat(frames[::-1], dim=0)).to(torch.uint8).detach().cpu()

    save_video(video, save_dir / "output.mp4", fps=fps, save_gif=save_gif)
    save_video(video_reverse, save_dir / "output_reverse.mp4", fps=fps, save_gif=save_gif)

    if save_depth:
        depth_video = (255 * torch.cat(depth, dim=0)).to(torch.uint8).detach().cpu()
        depth_video_reverse = (255 * torch.cat(depth[::-1], dim=0)).to(torch.uint8).detach().cpu()

        save_video(depth_video, save_dir / "output_depth.mp4", fps=fps, save_gif=save_gif)
        save_video(depth_video_reverse, save_dir / "output_depth_reverse.mp4", fps=fps, save_gif=save_gif)


def merge_keyframes(all_keyframes, save_dir, save_folder='keyframes', fps=1):
    """
    Save a list of PIL images sequentially into a directory.

    Parameters:
        all_keyframes (list): A list of PIL Image objects.
        save_dir (Path): A pathlib Path object indicating where to save the images.
    """
    # Ensure that the save_dir exists
    save_path = save_dir / save_folder
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save each keyframe with a sequential filename
    for i, frame in enumerate(all_keyframes):
        frame.save(save_path / f'{i}.png')

    all_keyframes = [ToTensor()(frame).unsqueeze(0) for frame in all_keyframes]
    all_keyframes = torch.cat(all_keyframes, dim=0)
    video = (255 * all_keyframes).to(torch.uint8).detach().cpu()
    video_reverse = (255 * all_keyframes.flip(0)).to(torch.uint8).detach().cpu()

    save_video(video, save_dir / "keyframes.mp4", fps=fps)
    save_video(video_reverse, save_dir / "keyframes_reverse.mp4", fps=fps)

class SimpleLogger:
    def __init__(self, log_path):
        # Ensure log_path is a Path object, whether provided as str or Path
        if not isinstance(log_path, Path):
            log_path = Path(log_path)
        
        # Ensure the file ends with '.log'
        if not log_path.name.endswith('.txt'):
            raise ValueError("Log file must end with '.txt' extension")

        # Create the directory if it does not exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(str(log_path))
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def print(self, message, attach_time=False):
        if attach_time:
            current_time = datetime.now().strftime("[%H:%M:%S]")
            self.logger.info(current_time)
        self.logger.info(message)
