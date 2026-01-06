import kornia
import numpy as np
import torch
from matplotlib import cm
from torchvision.io import write_video
import imageio


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


class LatentStorer:
    def __init__(self):
        self.latent = None

    def __call__(self, i, t, latent):
        self.latent = latent


def sobel_filter(disp, mode="sobel", beta=10.0):
    sobel_grad = kornia.filters.spatial_gradient(disp, mode=mode, normalized=False)
    sobel_mag = torch.sqrt(sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2)
    alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

    return alpha


def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
    depth,
    near_plane=None,
    far_plane=None,
    cmap="viridis",
):
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)

    return colored_image


def save_video(video, path, fps=10, save_gif=True):
    video = video.permute(0, 2, 3, 1)
    video_codec = "libx264"
    video_options = {
        "crf": "30",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",
    }
    write_video(str(path), video, fps=fps, video_codec=video_codec, options=video_options)
    if not save_gif:
        return
    video_np = video.cpu().numpy()
    gif_path = str(path).replace('.mp4', '.gif')
    imageio.mimsave(gif_path, video_np, duration=1000/fps, loop=0)


### additional functions from wonderworld 3DGS codebase ###
def quaternion2rotmat(q):
    # check if q is normalized
    assert torch.allclose(torch.norm(q, dim=-1), torch.ones(q.size(0), device=q.device)), "quaternion is not normalized"

    R = torch.zeros((q.size(0), 3, 3), device=q.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def rotmat2quaternion(R):
    assert R.size(1) == 3 and R.size(2) == 3, "R must be of shape [B, 3, 3]"

    B = R.size(0)
    q = torch.zeros((B, 4), device=R.device, dtype=R.dtype)

    r11, r12, r13 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    r21, r22, r23 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    r31, r32, r33 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    # Calculate trace
    trace = r11 + r22 + r33

    # Case where the trace is positive
    s = torch.sqrt(trace + 1.0) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (r32 - r23) / s
    q[:, 2] = (r13 - r31) / s
    q[:, 3] = (r21 - r12) / s

    # Cases where the trace is negative
    t1 = (r11 > r22) & (r11 > r33)  # case for x dominant
    t2 = (r22 > r11) & (r22 > r33)  # case for y dominant
    t3 = (r33 > r11) & (r33 > r22)  # case for z dominant

    # Recalculate s for different cases
    s1 = torch.sqrt(1.0 + r11 - r22 - r33) * 2
    s2 = torch.sqrt(1.0 + r22 - r11 - r33) * 2
    s3 = torch.sqrt(1.0 + r33 - r11 - r22) * 2

    q[t1, 0] = (r32[t1] - r23[t1]) / s1[t1]
    q[t1, 1] = 0.25 * s1[t1]
    q[t1, 2] = (r12[t1] + r21[t1]) / s1[t1]
    q[t1, 3] = (r13[t1] + r31[t1]) / s1[t1]

    q[t2, 0] = (r13[t2] - r31[t2]) / s2[t2]
    q[t2, 1] = (r12[t2] + r21[t2]) / s2[t2]
    q[t2, 2] = 0.25 * s2[t2]
    q[t2, 3] = (r23[t2] + r32[t2]) / s2[t2]

    q[t3, 0] = (r21[t3] - r12[t3]) / s3[t3]
    q[t3, 1] = (r13[t3] + r31[t3]) / s3[t3]
    q[t3, 2] = (r23[t3] + r32[t3]) / s3[t3]
    q[t3, 3] = 0.25 * s3[t3]

    return q


def normal2rotation(n):
    n = torch.nn.functional.normalize(n, dim=1)  # Normalize the input normal vector
    proxy_x = torch.tensor([1, 0, 0], dtype=torch.float32, device=n.device).expand_as(n)
    proxy_y = torch.tensor([0, 1, 0], dtype=torch.float32, device=n.device).expand_as(n)
    
    # Determine whether n is more parallel to proxy_x or proxy_y
    dot_x = torch.abs(torch.sum(n * proxy_x, dim=1))
    dot_y = torch.abs(torch.sum(n * proxy_y, dim=1))
    
    # Allocate storage for x_dir and y_dir
    x_dir = torch.zeros_like(n)
    y_dir = torch.zeros_like(n)
    
    # Case 0: more parallel with proxy_x
    mask_case_0 = dot_x > dot_y
    x_dir[mask_case_0] = torch.cross(proxy_y[mask_case_0], n[mask_case_0])
    y_dir[mask_case_0] = torch.cross(n[mask_case_0], x_dir[mask_case_0])

    # Case 1: more parallel with proxy_y
    mask_case_1 = ~mask_case_0
    y_dir[mask_case_1] = torch.cross(n[mask_case_1], proxy_x[mask_case_1])
    x_dir[mask_case_1] = torch.cross(y_dir[mask_case_1], n[mask_case_1])
    
    # Normalize the direction vectors to ensure they are unit vectors
    x_dir = torch.nn.functional.normalize(x_dir, dim=1)
    y_dir = torch.nn.functional.normalize(y_dir, dim=1)
    
    # Stack the direction vectors to form the rotation matrix
    R = torch.stack([x_dir, y_dir, n], dim=-1)

    # Convert the rotation matrix to a quaternion using the corrected function
    q = rotmat2quaternion(R)
    return q


def rotation2normal(q):
    R = quaternion2rotmat(q)
    normal = R[:, :, 2]
    return normal
