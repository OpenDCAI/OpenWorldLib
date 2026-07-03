import io
import math
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager

import cv2
import imageio
import loguru
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from matplotlib.colors import Normalize


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_normal_angles(normals1, normals2, eps=1e-8):
    """
    Compute the per-pixel angle between two normal maps, in degrees.

    Args:
        normals1: First normal map, shaped (H, W, 3) or (B, H, W, 3)
        normals2: Second normal map, with the same shape as normals1
        eps: Small value to avoid division by zero

    Returns:
        angles: Angle matrix shaped (H, W) or (B, H, W), in degrees
    """
    # Ensure the input shapes match
    assert normals1.shape == normals2.shape

    # Compute the dot product: (x1x2 + y1y2 + z1z2)
    dot_product = torch.sum(normals1 * normals2, dim=-1)

    # Compute the norm of each vector
    norm1 = torch.norm(normals1, dim=-1)
    norm2 = torch.norm(normals2, dim=-1)

    # Compute the cosine value, avoiding division by zero
    cos_theta = dot_product / (norm1 * norm2 + eps)

    # Clamp to [-1, 1] to avoid overflow from numerical error
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Compute radians and convert to degrees
    angles_rad = torch.acos(cos_theta)
    angles_deg = torch.rad2deg(angles_rad)

    return angles_deg


def point_padding(points):
    pad = torch.ones_like(points)[..., 0:1]
    return torch.cat([points, pad], dim=-1)


def np_point_padding(points):
    pad = np.ones_like(points)[..., 0:1]
    return np.concatenate([points, pad], axis=-1)


def load_video(video_path):
    # Prevent OpenCV from starting internal threads and causing heavy CPU contention.
    # In multiprocessing environments, OpenCV internal threads are usually set to 0 or 1.
    cv2.setNumThreads(0)

    frames = []
    cap = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            # A log or exception could be added here.
            return []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame)
            frames.append(pil_frame)

    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        # Decide whether to raise the exception or return partial frames as needed.
        # raise e

    finally:
        # Always release the file handle, even if an error occurs.
        cap.release()

    return frames


def get_last_video_frame(video_path):
    """decord supports random access, so read the last frame directly by index."""
    vr = VideoReader(video_path, ctx=cpu(0))
    last_frame = vr[-1].asnumpy()  # Directly index the last frame; decord seeks internally
    return last_frame


def save_video(
        frames: np.ndarray,
        output_path: str,
        fps: int = 24,
        codec: str = "libx264"  # General MP4 codec with the best compatibility
):
    """
    Save a video with imageio at the specified FPS, supporting multiple input frame formats.

    Args:
    - frames: Input frame array, which must be shaped (f, h, w, c), where:
      - f: frame count, h: height, w: width, c: channel count (1 or 3)
      - dtype: float32 (values 0-1) or uint8 (values 0-255)
    - output_path: Output video path, such as ./output.mp4
    - fps: Video frame rate, in frames per second
    - codec: Video codec; libx264 is the default MP4 codec and has the best compatibility
    """
    # 1. Validate input shape
    if len(frames.shape) != 4:
        raise ValueError(f"输入帧形状必须为 (f, h, w, c)，当前形状：{frames.shape}")
    f, h, w, c = frames.shape
    if c not in [1, 3]:
        raise ValueError(f"通道数必须为1（灰度）或3（彩色），当前通道数：{c}")

    # 2. Convert uniformly to uint8 (0-255)
    processed_frames = frames.copy()
    if processed_frames.dtype == np.float32:
        # float32 (0-1) -> uint8 (0-255); clip outliers first
        processed_frames = np.clip(processed_frames, 0.0, 1.0)  # Ensure values stay in 0-1
        processed_frames = (processed_frames * 255).astype(np.uint8)
    elif processed_frames.dtype == np.uint8:
        # Clip uint8 outliers directly to 0-255
        processed_frames = np.clip(processed_frames, 0, 255)
    else:
        raise TypeError(f"仅支持 float32/uint8 类型，当前类型：{processed_frames.dtype}")

    # 3. Expand single-channel grayscale to 3 channels for codec compatibility
    if c == 1:
        processed_frames = np.repeat(processed_frames, 3, axis=-1)  # (f,h,w,1) → (f,h,w,3)

    # 4. Write the video at the specified FPS
    with imageio.get_writer(output_path, fps=fps, codec=codec) as writer:
        for idx, frame in enumerate(processed_frames):
            writer.append_data(frame)


def load_16bit_png_depth(depth_png: str) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def save_16bit_png_depth(depth: np.ndarray, depth_png: str):
    # Ensure the numpy array's dtype is float32, then cast to float16, and finally reinterpret as uint16
    depth_uint16 = np.array(depth, dtype=np.float32).astype(np.float16).view(np.uint16)

    # Create a PIL Image from the 16-bit depth values and save it
    depth_pil = Image.fromarray(depth_uint16)

    if not depth_png.endswith(".png"):
        print("ERROR DEPTH FILE:", depth_png)
        raise NotImplementedError

    try:
        depth_pil.save(depth_png)
    except:
        print("ERROR DEPTH FILE:", depth_png)
        raise NotImplementedError


def adjust_image_size(h, w):
    """
    Adjust h and w so that:
    1. h_new is divisible by 16
    2. w_new is divisible by 16
    3. (h_new//16) * (w_new//16) is divisible by 8

    Return h_new and w_new as close as possible to the original size, rounded up.
    """
    # Round h_new up to a multiple of 16
    h_new = math.ceil(h / 16) * 16

    # Count the number of factor-2 terms in a = h_new // 16
    a = h_new // 16
    p = 0
    temp = a
    while temp > 0 and temp % 2 == 0:
        p += 1
        temp //= 2

    # b = w_new//16 must be divisible by 2^(3-p) so that a*b is divisible by 8
    required_factor = 1 << max(0, 3 - p)  # 2^max(0, 3-p)

    # Round w_new up to the smallest value that satisfies the condition
    b = math.ceil(w / 16)
    b = math.ceil(b / required_factor) * required_factor
    w_new = b * 16

    return h_new, w_new


def rank0_log(message, level="INFO"):
    if int(os.environ.get('RANK', '0')) == 0:
        loguru.logger.opt(depth=1).log(level, message)


class Timer:
    """Concise multi-part timer."""

    def __init__(self):
        self.records = defaultdict(list)
        self._start_times = {}

    def start(self, name: str):
        """Start timing."""
        self._start_times[name] = time.perf_counter()

    def end(self, name: str):
        """End timing."""
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.records[name].append(elapsed)
            del self._start_times[name]
            return elapsed
        return 0

    @contextmanager
    def track(self, name: str):
        """Time a block with a context manager."""
        self.start(name)
        try:
            yield
        finally:
            self.end(name)

    def summary(self):
        """Print a statistical summary, splitting IO and non-IO parts."""

        # Aggregate by category
        io_total_time = 0.0
        non_io_total_time = 0.0
        io_records = {}
        non_io_records = {}

        for name, times in self.records.items():
            total_time = sum(times)
            if "[IO]" in name:
                io_total_time += total_time
                io_records[name] = times
            else:
                non_io_total_time += total_time
                non_io_records[name] = times

        overall_total = io_total_time + non_io_total_time

        # Print the table header
        print("\n" + "=" * 80)
        print(f"{'Part':<35} {'Calls':>8} {'Total':>10} {'Mean':>10} {'Min':>10} {'Max':>10}")
        print("=" * 80)

        # Print non-IO parts
        if non_io_records:
            print("-" * 80)
            print("[Compute Parts]")
            print("-" * 80)
            for name, times in non_io_records.items():
                print(
                    f"{name:<35} {len(times):>8} {sum(times):>10.4f} "
                    f"{sum(times) / len(times):>10.4f} {min(times):>10.4f} {max(times):>10.4f}"
                )

        # Print IO parts
        if io_records:
            print("-" * 80)
            print("[IO Parts]")
            print("-" * 80)
            for name, times in io_records.items():
                print(
                    f"{name:<35} {len(times):>8} {sum(times):>10.4f} "
                    f"{sum(times) / len(times):>10.4f} {min(times):>10.4f} {max(times):>10.4f}"
                )

        # Print the summary
        print("=" * 80)
        print(f"{'[Summary] Compute Total':<35} {non_io_total_time:>10.4f}s ({non_io_total_time / overall_total * 100 if overall_total > 0 else 0:>6.2f}%)")
        print(f"{'[Summary] IO Total':<35} {io_total_time:>10.4f}s ({io_total_time / overall_total * 100 if overall_total > 0 else 0:>6.2f}%)")
        print(f"{'[Summary] Overall Total':<35} {overall_total:>10.4f}s")
        print("=" * 80)



def split_n_into_d_parts(N: int, D: int) -> list[int]:
    """
    Split integer N evenly into D parts, minimizing differences between parts (only 0 or 1).
    :param N: Positive integer or 0 to split (total count)
    :param D: Number of parts to split into (positive integer)
    :return: Integer list of length D whose sum is N
    :raises ValueError: Raised when D is not a positive integer
    """
    # Validate that the number of parts is a positive integer
    if not isinstance(D, int) or D <= 0:
        raise ValueError
    # Handle the case where N is not an integer
    if not isinstance(N, int):
        raise ValueError

    # Core calculation: base value plus remainder
    q, r = divmod(N, D)  # Equivalent to q=N//D and r=N%D; divmod is more concise
    # The first D-r parts are q and the last r parts are q+1, preserving the sum and minimizing differences
    result = [q] * (D - r) + [q + 1] * r
    return result

def color_print(msg, level="info"):
    """
    Color print helper with a minimal call site.
    :param msg: String to print, such as the final content from an f-string
    :param level: Print level: info (green), failed (red), warn (yellow); defaults to info
    """
    # Color-code dictionary; add key-value pairs to extend levels
    COLOR_MAP = {
        "info": "\033[32m",  # Green: normal
        "error": "\033[31m",  # Red: failure/urgent
        "warning": "\033[33m"     # Yellow: warning (optional extension)
    }
    COLOR_RESET = "\033[0m"   # Required color reset
    # Concatenate the color code, content, and reset code before printing
    print(f"{COLOR_MAP.get(level, COLOR_MAP['info'])}{msg}{COLOR_RESET}")


def sample_align_nframe(N, n):
    if n > (N - 1):
        raise ValueError

    # 1. Core logic: N-1 positions are available (1 to N-1); N-1 must be included, and intervals decrease over time
    # We need to choose n points. That gives n-1 internal intervals plus one starting offset.
    # Total free units = (N-1) - n
    extra_space = (N - 1) - n

    # 2. Allocate free units to the starting point and early intervals
    # To make the initial intervals larger, allocate the extra space (extra_space)
    # first to: a. the starting position of the first point; b. the gaps between earlier indices

    # Minimum extra increment assigned to each slot (start slot + interval slots)
    num_allocatable_slots = n  # 1 starting position + n-1 intervals
    base_extra = extra_space // num_allocatable_slots
    remainder = extra_space % num_allocatable_slots

    # 3. Build the allocation sequence, placing larger increments first
    allocations = [base_extra + 1] * remainder + [base_extra] * (num_allocatable_slots - remainder)

    # 4. Generate indices
    indices = []
    # First point: start from 1 and add the first allocation
    current = 1 + allocations[0]
    indices.append(current)

    # Subsequent points: step size is 1 (base step) plus the allocation
    for i in range(1, len(allocations)):
        current += (1 + allocations[i])
        indices.append(current)

    return np.array(indices)


def colorize_depth(
        depth,
        colormap='plasma',
        min_depth=None,
        max_depth=None,
        inverse=True,
        save_path=None,
        return_pil=False,
        show_colorbar=False,
        colorbar_label='Depth',
        colorbar_width=0.03,  # Colorbar width ratio
        colorbar_pad=0.02,  # Gap between the colorbar and image
        colorbar_ticks=5,  # Number of colorbar ticks
        figsize_scale=1.0,  # Image scale ratio
        dpi=150,  # DPI when saving
        title=None,  # Optional title
):
    """
    Color visualization for depth maps.

    Args:
        depth: [H,W], [1,H,W], or [B,1,H,W], as a numpy array or torch tensor
        colormap: 'plasma', 'turbo', 'inferno', 'magma', 'viridis', 'jet'
        min_depth: Minimum depth value for normalization; computed automatically if None
        max_depth: Maximum depth value for normalization; computed automatically if None
        inverse: True means near areas are red and far areas are purple
        save_path: Save path
        return_pil: True returns a PIL Image; False returns a numpy array
        show_colorbar: Whether to show the colorbar
        colorbar_label: Colorbar label
        colorbar_width: Colorbar width ratio
        colorbar_pad: Gap between the colorbar and image
        colorbar_ticks: Number of colorbar ticks
        figsize_scale: Image scale ratio
        dpi: DPI when saving
        title: Optional title

    Returns:
        Colored depth map as a numpy array or PIL Image
    """
    # Convert to numpy [H, W]
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    depth = np.squeeze(depth)

    # Ensure it is 2D
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth, got shape {depth.shape}")

    # Get the valid depth range
    valid = (depth > 0) & np.isfinite(depth)

    if min_depth is None:
        # min_depth = np.percentile(depth[valid], 2) if valid.any() else 0
        min_depth = depth[valid].min().item()
    if max_depth is None:
        # max_depth = np.percentile(depth[valid], 98) if valid.any() else 1
        max_depth = depth[valid].max().item()

    # Normalize
    depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)

    # Reverse the color direction
    if inverse:
        depth_norm = 1.0 - depth_norm

    # Set invalid regions
    depth_norm[~valid] = 0

    # Apply the colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth_norm)
    colored = (colored[:, :, :3] * 255).astype(np.uint8)

    # ========== Without ColorBar (original logic) ==========
    if not show_colorbar:
        if save_path:
            img = Image.fromarray(colored)
            img.save(save_path)

        if return_pil:
            return Image.fromarray(colored)
        return colored

    # ========== With ColorBar (new) ==========
    h, w = depth.shape
    figsize = (w / 100 * figsize_scale, h / 100 * figsize_scale)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Display the depth map using original normalized values so the colorbar maps correctly
    depth_display = depth.copy()
    depth_display[~valid] = np.nan  # Set invalid regions to NaN so they are not displayed

    # Create a Normalize object
    if inverse:
        # When reversed, the colorbar mapping should also be reversed
        norm = Normalize(vmin=max_depth, vmax=min_depth)
    else:
        norm = Normalize(vmin=min_depth, vmax=max_depth)

    im = ax.imshow(depth_display, cmap=colormap, norm=norm)
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=12)

    # Add the colorbar
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=colorbar_width,
        pad=colorbar_pad,
        label=colorbar_label
    )

    # Set colorbar ticks
    tick_values = np.linspace(min_depth, max_depth, colorbar_ticks)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{v:.2f}' for v in tick_values])

    plt.tight_layout()

    # Convert the matplotlib image to numpy/PIL
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)

    # Convert to a numpy array or PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    result_img = Image.open(buf)

    plt.close(fig)

    if return_pil:
        return result_img
    return np.array(result_img)[:, :, :3]