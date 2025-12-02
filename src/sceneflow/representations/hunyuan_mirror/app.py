import gc
import os
import shutil
import time
from datetime import datetime
import io
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

from mirror_src.utils.inference_utils import load_and_preprocess_images
from mirror_src.utils.geometry import (
    depth_edge,
    normals_edge
)
from mirror_src.utils.visual_util import (
    convert_predictions_to_glb_scene,
    segment_sky,
    download_file_from_url
)
from mirror_src.utils.save_utils import save_camera_params, save_gs_ply, process_ply_to_splat, convert_gs_to_ply
from mirror_src.utils.render_utils import render_interpolated_video
import onnxruntime


# Initialize model - this will be done on GPU when needed
model = None

# Global variable to store current terminal output
current_terminal_output = ""

# Helper class to capture terminal output
class TeeOutput:
    """Capture output while still printing to console"""
    def __init__(self, max_chars=10000):
        self.terminal = sys.stdout
        self.log = io.StringIO()
        self.max_chars = max_chars  # 限制最大字符数
    
    def write(self, message):
        global current_terminal_output
        self.terminal.write(message)
        self.log.write(message)
        
        # 获取当前内容并限制长度
        content = self.log.getvalue()
        if len(content) > self.max_chars:
            # 只保留最后 max_chars 个字符
            content = "...(earlier output truncated)...\n" + content[-self.max_chars:]
            self.log = io.StringIO()
            self.log.write(content)
        
        current_terminal_output = self.log.getvalue()
    
    def flush(self):
        self.terminal.flush()
    
    def getvalue(self):
        return self.log.getvalue()
    
    def clear(self):
        global current_terminal_output
        self.log = io.StringIO()
        current_terminal_output = ""

# -------------------------------------------------------------------------
# Model inference
# -------------------------------------------------------------------------
@spaces.GPU(duration=120)
def run_model(
    target_dir,
    confidence_percentile: float = 10,
    edge_normal_threshold: float = 5.0,
    edge_depth_threshold: float = 0.03,
    apply_confidence_mask: bool = True,
    apply_edge_mask: bool = True,
):
    """
    Run the WorldMirror model on images in the 'target_dir/images' folder and return predictions.
    """
    global model
    import torch  # Ensure torch is available in function scope
    
    from mirror_src.models.models.worldmirror import WorldMirror
    from mirror_src.models.utils.geometry import depth_to_world_coords_points

    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Initialize model if not already done
    if model is None:
        model = WorldMirror.from_pretrained("/opt/tiger/WMflow/ckpts", local_files_only=True).to(device)
    else:
        model.to(device)
    
    model.eval()

    # Load images using WorldMirror's load_images function
    print("Loading images...")
    image_folder_path = os.path.join(target_dir, "images")
    image_file_paths = [os.path.join(image_folder_path, path) for path in os.listdir(image_folder_path)]
    img = load_and_preprocess_images(image_file_paths).to(device)

    print(f"Loaded {img.shape[1]} images")
    if img.shape[1] == 0:
        raise ValueError("No images found. Check your upload.")

    # Run model inference
    print("Running inference...")
    inputs = {}
    inputs['img'] = img
    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_amp:
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32
    with torch.amp.autocast('cuda', enabled=bool(use_amp), dtype=amp_dtype):
        predictions = model(inputs)

    # img
    imgs = inputs["img"].permute(0, 1, 3, 4, 2)
    imgs = imgs[0].detach().cpu().numpy() # S H W 3

    # depth output
    depth_preds = predictions["depth"]
    depth_conf = predictions["depth_conf"]
    depth_preds = depth_preds[0].detach().cpu().numpy() # S H W 1
    depth_conf = depth_conf[0].detach().cpu().numpy() # S H W

    # normal output
    normal_preds = predictions["normals"] # S H W 3
    normal_preds = normal_preds[0].detach().cpu().numpy() # S H W 3

    # camera parameters
    camera_poses = predictions["camera_poses"][0].detach().cpu().numpy() # [S,4,4]
    camera_intrs = predictions["camera_intrs"][0].detach().cpu().numpy() # [S,3,3]
    
    # points output
    pts3d_preds = depth_to_world_coords_points(predictions["depth"][0, ..., 0], predictions["camera_poses"][0], predictions["camera_intrs"][0])[0]
    pts3d_preds = pts3d_preds.detach().cpu().numpy()  # S H W 3
    pts3d_conf = depth_conf              # S H W

    # sky mask segmentation
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url(
            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
        )
    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    sky_mask_list = []
    for i, img_path in enumerate([os.path.join(image_folder_path, path) for path in os.listdir(image_folder_path)]):
        sky_mask = segment_sky(img_path, skyseg_session)
        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != imgs.shape[1] or sky_mask.shape[1] != imgs.shape[2]:
            sky_mask = cv2.resize(sky_mask, (imgs.shape[2], imgs.shape[1]))
        sky_mask_list.append(sky_mask)
    sky_mask = np.stack(sky_mask_list, axis=0) # [S, H, W]
    sky_mask = sky_mask>0

    # mask computation
    final_mask_list = []    
    for i in range(inputs["img"].shape[1]):
        final_mask = None
        if apply_confidence_mask:
            # compute confidence mask based on the pointmap confidence
            confidences = pts3d_conf[i, :, :] # [H, W]
            percentile_threshold = np.quantile(confidences, confidence_percentile / 100.0)
            conf_mask = confidences >= percentile_threshold
            if final_mask is None:
                final_mask = conf_mask
            else:
                final_mask = final_mask & conf_mask
        if apply_edge_mask:
            # compute edge mask based on the normalmap
            normal_pred = normal_preds[i] # [H, W, 3]
            normal_edges = normals_edge(
                normal_pred, tol=edge_normal_threshold, mask=final_mask
            )
            # compute depth mask based on the depthmap
            depth_pred = depth_preds[i, :, :, 0] # [H, W]
            depth_edges = depth_edge(
                depth_pred, rtol=edge_depth_threshold, mask=final_mask
            )
            edge_mask = ~(depth_edges & normal_edges)
            if final_mask is None:
                final_mask = edge_mask
            else:
                final_mask = final_mask & edge_mask
        final_mask_list.append(final_mask)

    if final_mask_list[0] is not None:
        final_mask = np.stack(final_mask_list, axis=0) # [S, H, W]
    else:
        final_mask = np.ones(pts3d_conf.shape[:3], dtype=bool) # [S, H, W]

    # gaussian splatting output
    if "splats" in predictions:
        splats_dict = {}
        splats_dict['means'] = predictions["splats"]["means"]
        splats_dict['scales'] = predictions["splats"]["scales"]
        splats_dict['quats'] = predictions["splats"]["quats"]
        splats_dict['opacities'] = predictions["splats"]["opacities"]
        if "sh" in predictions["splats"]:
            splats_dict['sh'] = predictions["splats"]["sh"]
        if "colors" in predictions["splats"]:
            splats_dict['colors'] = predictions["splats"]["colors"]

    # output lists
    outputs = {}
    outputs['images'] = imgs
    outputs['world_points'] = pts3d_preds
    outputs['depth'] = depth_preds
    outputs['normal'] = normal_preds
    outputs['final_mask'] = final_mask
    outputs['sky_mask'] = sky_mask
    outputs['camera_poses'] = camera_poses
    outputs['camera_intrs'] = camera_intrs
    if "splats" in predictions:
        outputs['splats'] = splats_dict
    
    # Process data for visualization tabs (depth, normal)
    processed_data = prepare_visualization_data(
        outputs, inputs
    )

    # Clean up
    torch.cuda.empty_cache()

    return outputs, processed_data


# -------------------------------------------------------------------------
# Update and navigation function
# -------------------------------------------------------------------------
def update_view_info(current_view, total_views, view_type="Depth"):
        """Update view information display"""
        return f"""
        <div style='text-align: center; padding: 10px; background: #f8f8f8; color: #999; border-radius: 8px; margin-bottom: 10px;'>
            <strong>{view_type} View Navigation</strong> | 
            Current: View {current_view} / {total_views} views
        </div>
        """
        
def update_view_selectors(processed_data):
    """Update view selector sliders and info displays based on available views"""
    if processed_data is None or len(processed_data) == 0:
        num_views = 1
    else:
        num_views = len(processed_data)

    # 确保 num_views 至少为 1
    num_views = max(1, num_views)

    # 更新滑块的最大值和视图信息，使用 gr.update() 而不是创建新组件
    depth_slider_update = gr.update(minimum=1, maximum=num_views, value=1, step=1)
    normal_slider_update = gr.update(minimum=1, maximum=num_views, value=1, step=1)
    
    # 更新视图信息显示
    depth_info_update = update_view_info(1, num_views, "Depth")
    normal_info_update = update_view_info(1, num_views, "Normal")

    return (
        depth_slider_update,  # depth_view_slider
        normal_slider_update,  # normal_view_slider
        depth_info_update,    # depth_view_info
        normal_info_update,   # normal_view_info
    )

def get_view_data_by_index(processed_data, view_index):
    """Get view data by index, handling bounds"""
    if processed_data is None or len(processed_data) == 0:
        return None

    view_keys = list(processed_data.keys())
    if view_index < 0 or view_index >= len(view_keys):
        view_index = 0

    return processed_data[view_keys[view_index]]

def update_depth_view(processed_data, view_index):
    """Update depth view for a specific view index"""
    view_data = get_view_data_by_index(processed_data, view_index)
    if view_data is None or view_data["depth"] is None:
        return None

    return render_depth_visualization(view_data["depth"], mask=view_data.get("mask"))

def update_normal_view(processed_data, view_index):
    """Update normal view for a specific view index"""
    view_data = get_view_data_by_index(processed_data, view_index)
    if view_data is None or view_data["normal"] is None:
        return None

    return render_normal_visualization(view_data["normal"], mask=view_data.get("mask"))

def initialize_depth_normal_views(processed_data):
    """Initialize the depth and normal view displays with the first view data"""
    if processed_data is None or len(processed_data) == 0:
        return None, None

    # Use update functions to ensure confidence filtering is applied from the start
    depth_vis = update_depth_view(processed_data, 0)
    normal_vis = update_normal_view(processed_data, 0)

    return depth_vis, normal_vis


# -------------------------------------------------------------------------
# File upload and update preview gallery
# -------------------------------------------------------------------------
def process_uploaded_files(files, time_interval=1.0):
    """
    Process uploaded files by extracting video frames or copying images.
    
    Args:
        files: List of uploaded file objects (videos or images)
        time_interval: Interval in seconds for video frame extraction
        
    Returns:
        tuple: (target_dir, image_paths) where target_dir is the output directory
               and image_paths is a list of processed image file paths
    """
    gc.collect()
    torch.cuda.empty_cache()

    # Create unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    images_dir = os.path.join(target_dir, "images")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(images_dir)

    image_paths = []

    if files is None:
        return target_dir, image_paths

    video_exts = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".3gp"]

    for file_data in files:
        # Get file path
        if isinstance(file_data, dict) and "name" in file_data:
            src_path = file_data["name"]
        else:
            src_path = str(file_data)

        ext = os.path.splitext(src_path)[1].lower()
        base_name = os.path.splitext(os.path.basename(src_path))[0]

        # Process video: extract frames
        if ext in video_exts:
            cap = cv2.VideoCapture(src_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = int(fps * time_interval)

            frame_count = 0
            saved_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % interval == 0:
                    dst_path = os.path.join(images_dir, f"{base_name}_{saved_count:06}.png")
                    cv2.imwrite(dst_path, frame)
                    image_paths.append(dst_path)
                    saved_count += 1
            cap.release()
            print(f"Extracted {saved_count} frames from: {os.path.basename(src_path)}")

        # Process HEIC/HEIF: convert to JPEG
        elif ext in [".heic", ".heif"]:
            try:
                with Image.open(src_path) as img:
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    dst_path = os.path.join(images_dir, f"{base_name}.jpg")
                    img.save(dst_path, "JPEG", quality=95)
                    image_paths.append(dst_path)
                    print(f"Converted HEIC: {os.path.basename(src_path)} -> {os.path.basename(dst_path)}")
            except Exception as e:
                print(f"HEIC conversion failed for {src_path}: {e}")
                dst_path = os.path.join(images_dir, os.path.basename(src_path))
                shutil.copy(src_path, dst_path)
                image_paths.append(dst_path)

        # Process regular images: copy directly
        else:
            dst_path = os.path.join(images_dir, os.path.basename(src_path))
            shutil.copy(src_path, dst_path)
            image_paths.append(dst_path)

    image_paths = sorted(image_paths)

    print(f"Processed files to {images_dir}")
    return target_dir, image_paths

# Handle file upload and update preview gallery
def update_gallery_on_upload(input_video, input_images, time_interval=1.0):
    """
    Process uploaded files immediately when user uploads or changes files,
    and display them in the gallery. Returns (target_dir, image_paths).
    If nothing is uploaded, returns None and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = process_uploaded_files(input_video, input_images, time_interval)
    return (
        None,
        target_dir,
        image_paths,
        "Upload complete. Click 'Reconstruct' to begin 3D processing.",
    )
        
# -------------------------------------------------------------------------
# Init function
# -------------------------------------------------------------------------
def prepare_visualization_data(
    model_outputs, input_views
):
    """Transform model predictions into structured format for display components"""
    visualization_dict = {}

    # Iterate through each input view
    nviews = input_views["img"].shape[1]
    for idx in range(nviews):
        # Extract RGB image data
        rgb_image = input_views["img"][0, idx].detach().cpu().numpy()

        # Retrieve 3D coordinate predictions
        world_coordinates = model_outputs["world_points"][idx]

        # Build view-specific data structure
        current_view_info = {
            "image": rgb_image,
            "points3d": world_coordinates,
            "depth": None,
            "normal": None,
            "mask": None,
        }

        # Apply final segmentation mask from model
        segmentation_mask = model_outputs["final_mask"][idx].copy()

        current_view_info["mask"] = segmentation_mask
        current_view_info["depth"] = model_outputs["depth"][idx].squeeze()

        surface_normals = model_outputs["normal"][idx]
        current_view_info["normal"] = surface_normals

        visualization_dict[idx] = current_view_info

    return visualization_dict

@spaces.GPU(duration=120)
def gradio_demo(
    target_dir,
    frame_selector="All",
    show_camera=False,
    filter_sky_bg=False,
    show_mesh=False,
    filter_ambiguous=False,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    # Capture terminal output
    tee = TeeOutput()
    old_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        if not os.path.isdir(target_dir) or target_dir == "None":
            terminal_log = tee.getvalue()
            sys.stdout = old_stdout
            return None, "No valid target directory found. Please upload first.", None, None, None, None, None, None, None, None, None, None, None, None, terminal_log

        start_time = time.time()
        gc.collect()
        torch.cuda.empty_cache()

        # Prepare frame_selector dropdown
        target_dir_images = os.path.join(target_dir, "images")
        all_files = (
            sorted(os.listdir(target_dir_images))
            if os.path.isdir(target_dir_images)
            else []
        )
        all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
        frame_selector_choices = ["All"] + all_files

        print("Running WorldMirror model...")
        with torch.no_grad():
            predictions, processed_data = run_model(target_dir)

        # Save predictions
        prediction_save_path = os.path.join(target_dir, "predictions.npz")
        np.savez(prediction_save_path, **predictions)

        # Save camera parameters as JSON
        camera_params_file = save_camera_params(
            predictions['camera_poses'], 
            predictions['camera_intrs'], 
            target_dir
        )

        # Handle None frame_selector
        if frame_selector is None:
            frame_selector = "All"

        # Build a GLB file name
        glbfile = os.path.join(
            target_dir,
            f"glbscene_{frame_selector.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_camera}_mesh{show_mesh}.glb",
        )

        # Convert predictions to GLB
        glbscene = convert_predictions_to_glb_scene(
            predictions,
            filter_by_frames=frame_selector,
            show_camera=show_camera,
            mask_sky_bg=filter_sky_bg,
            as_mesh=show_mesh,  # Use the show_mesh parameter
            mask_ambiguous=filter_ambiguous
        )
        glbscene.export(file_obj=glbfile)
        
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds")
        log_msg = (
            f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."
        )
        # Convert predictions to 3dgs ply
        gs_file = None
        splat_mode = 'ply'
        if "splats" in predictions:
            # Get Gaussian parameters (already filtered by GaussianSplatRenderer)
            means = predictions["splats"]["means"][0].reshape(-1, 3)
            scales = predictions["splats"]["scales"][0].reshape(-1, 3)
            quats = predictions["splats"]["quats"][0].reshape(-1, 4)
            colors = (predictions["splats"]["sh"][0] if "sh" in predictions["splats"] else predictions["splats"]["colors"][0]).reshape(-1, 3)
            opacities = predictions["splats"]["opacities"][0].reshape(-1)
            
            # Convert to torch tensors if needed
            if not isinstance(means, torch.Tensor):
                means = torch.from_numpy(means)
            if not isinstance(scales, torch.Tensor):
                scales = torch.from_numpy(scales)
            if not isinstance(quats, torch.Tensor):
                quats = torch.from_numpy(quats)
            if not isinstance(colors, torch.Tensor):
                colors = torch.from_numpy(colors)
            if not isinstance(opacities, torch.Tensor):
                opacities = torch.from_numpy(opacities)
            
            if splat_mode == 'ply':
                gs_file = os.path.join(target_dir, "gaussians.ply")
                save_gs_ply(
                    gs_file,
                    means,
                    scales,
                    quats,
                    colors,
                    opacities
                )
                print(f"Saved Gaussian Splatting PLY to: {gs_file}")
                print(f"File exists: {os.path.exists(gs_file)}")
                if os.path.exists(gs_file):
                    print(f"File size: {os.path.getsize(gs_file)} bytes")
            elif splat_mode == 'splat':
                # Save Gaussian splat
                plydata = convert_gs_to_ply(
                        means,
                        scales,
                        quats,
                        colors,
                        opacities
                    )
                gs_file = os.path.join(target_dir, "gaussians.splat")
                gs_file = process_ply_to_splat(plydata, gs_file)

        # Initialize depth and normal view displays with processed data
        depth_vis, normal_vis = initialize_depth_normal_views(
            processed_data
        )

        # Update view selectors and info displays based on available views
        depth_slider, normal_slider, depth_info, normal_info = update_view_selectors(
            processed_data
        )

        # Automatically generate render video
        # Generate render video if possible
        rgb_video_path = None
        depth_video_path = None
        
        if "splats" in predictions:
            # try:
            from pathlib import Path
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Get camera parameters and image dimensions
            camera_poses = torch.tensor(predictions['camera_poses']).unsqueeze(0).to(device)
            camera_intrs = torch.tensor(predictions['camera_intrs']).unsqueeze(0).to(device)
            H, W = predictions['images'].shape[1], predictions['images'].shape[2]
            
            # Render video
            out_path = Path(target_dir) / "rendered_video"
            render_interpolated_video(
                model.gs_renderer, 
                predictions["splats"], 
                camera_poses, 
                camera_intrs, 
                (H, W), 
                out_path, 
                interp_per_pair=15, 
                loop_reverse=True,
                save_mode="split"
            )
            
            # Check output files
            rgb_video_path = str(out_path) + "_rgb.mp4"
            depth_video_path = str(out_path) + "_depth.mp4"
            
            if not os.path.exists(rgb_video_path) and not os.path.exists(depth_video_path):
                rgb_video_path = None
                depth_video_path = None
                
        # Cleanup
        del predictions
        gc.collect()
        torch.cuda.empty_cache()

        # Get terminal output and restore stdout
        terminal_log = tee.getvalue()
        sys.stdout = old_stdout

        return (
            glbfile,
            log_msg,
            gr.Dropdown(choices=frame_selector_choices, value=frame_selector, interactive=True),
            processed_data,
            depth_vis,
            normal_vis,
            depth_slider,
            normal_slider,
            depth_info,
            normal_info,
            camera_params_file,
            gs_file,
            rgb_video_path,
            depth_video_path,
            terminal_log,
        )
    
    except Exception as e:
        # In case of error, still restore stdout
        terminal_log = tee.getvalue()
        sys.stdout = old_stdout
        print(f"Error occurred: {e}")
        raise


# -------------------------------------------------------------------------
# Helper functions for visualization
# -------------------------------------------------------------------------
def render_depth_visualization(depth_map, mask=None):
    """Generate a color-coded depth visualization image with masking capabilities"""
    if depth_map is None:
        return None

    # Create working copy and identify positive depth values
    depth_copy = depth_map.copy()
    positive_depth_mask = depth_copy > 0

    # Combine with user-provided mask for filtering
    if mask is not None:
        positive_depth_mask = positive_depth_mask & mask

    # Perform percentile-based normalization on valid regions
    if positive_depth_mask.sum() > 0:
        valid_depth_values = depth_copy[positive_depth_mask]
        lower_bound = np.percentile(valid_depth_values, 5)
        upper_bound = np.percentile(valid_depth_values, 95)

        depth_copy[positive_depth_mask] = (depth_copy[positive_depth_mask] - lower_bound) / (upper_bound - lower_bound)

    # Convert to RGB using matplotlib colormap
    import matplotlib.pyplot as plt

    color_mapper = plt.cm.turbo_r
    rgb_result = color_mapper(depth_copy)
    rgb_result = (rgb_result[:, :, :3] * 255).astype(np.uint8)

    # Mark invalid regions with white color
    rgb_result[~positive_depth_mask] = [255, 255, 255]

    return rgb_result

def render_normal_visualization(normal_map, mask=None):
    """Convert surface normal vectors to RGB color representation for display"""
    if normal_map is None:
        return None

    # Make a working copy to avoid modifying original data
    normal_display = normal_map.copy()

    # Handle masking by zeroing out invalid regions
    if mask is not None:
        masked_regions = ~mask
        normal_display[masked_regions] = [0, 0, 0]  # Zero out masked pixels

    # Transform from [-1, 1] to [0, 1] range for RGB display
    normal_display = (normal_display + 1.0) / 2.0
    normal_display = (normal_display * 255).astype(np.uint8)

    return normal_display


def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def get_terminal_output():
    """
    Get current terminal output for real-time display
    """
    global current_terminal_output
    return current_terminal_output

# -------------------------------------------------------------------------
# FunctionExample scene metadata extraction
# -------------------------------------------------------------------------
def extract_example_scenes_metadata(base_directory):
    """
    Extract comprehensive metadata for all scene directories containing valid images.
    
    Args:
        base_directory: Root path where example scene directories are located
        
    Returns:
        Collection of dictionaries with scene details (title, location, preview, etc.)
    """
    from glob import glob
    
    # Return empty list if base directory is missing
    if not os.path.exists(base_directory):
        return []
    
    # Define supported image format extensions
    VALID_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    
    scenes_data = []
    
    # Process each subdirectory in the base directory
    for directory_name in sorted(os.listdir(base_directory)):
        current_directory = os.path.join(base_directory, directory_name)
        
        # Filter out non-directory items
        if not os.path.isdir(current_directory):
            continue
        
        # Gather all valid image files within the current directory
        discovered_images = []
        for file_format in VALID_IMAGE_FORMATS:
            # Include both lowercase and uppercase format variations
            discovered_images.extend(glob(os.path.join(current_directory, f'*.{file_format}')))
            discovered_images.extend(glob(os.path.join(current_directory, f'*.{file_format.upper()}')))
        
        # Skip directories without any valid images
        if not discovered_images:
            continue
        
        # Ensure consistent image ordering
        discovered_images.sort()
        
        # Construct scene metadata record
        scene_record = {
            'name': directory_name,
            'path': current_directory,
            'thumbnail': discovered_images[0],
            'num_images': len(discovered_images),
            'image_files': discovered_images,
        }
        
        scenes_data.append(scene_record)
    
    return scenes_data

def load_example_scenes(scene_name, scenes):
    """
    Initialize and prepare an example scene for 3D reconstruction processing.
    
    Args:
        scene_name: Identifier of the target scene to load
        scenes: List containing all available scene configurations
        
    Returns:
        Tuple containing processed scene data and status information
    """
    # Locate the target scene configuration by matching names
    target_scene_config = None
    for scene_config in scenes:
        if scene_config["name"] == scene_name:
            target_scene_config = scene_config
            break

    # Handle case where requested scene doesn't exist
    if target_scene_config is None:
        return None, None, None, "Scene not found"

    # Prepare image file paths for processing pipeline
    # Extract all image file paths from the selected scene
    image_file_paths = []
    for img_file_path in target_scene_config["image_files"]:
        image_file_paths.append(img_file_path)

    # Process the scene images through the standard upload pipeline
    processed_target_dir, processed_image_list = process_uploaded_files(image_file_paths, 1.0)

    # Return structured response with scene data and user feedback
    status_message = f"Successfully loaded scene '{scene_name}' containing {target_scene_config['num_images']} images. Click 'Reconstruct' to begin 3D processing."
    
    return (
        None,  # Reset reconstruction visualization
        None,  # Reset gaussian splatting output
        processed_target_dir,  # Provide working directory path
        processed_image_list,  # Update image gallery display
        status_message,
    )


# -------------------------------------------------------------------------
# UI and event handling
# -------------------------------------------------------------------------
theme = gr.themes.Base()

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #a9b8f8 0%, #7081e8 60%, #4254c5 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    .normal-weight-btn button,
    .normal-weight-btn button span,
    .normal-weight-btn button *,
    .normal-weight-btn * {
        font-weight: 400 !important;
    }
    .terminal-output {
        max-height: 400px !important;
        overflow-y: auto !important;
    }
    .terminal-output textarea {
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        color: #333 !important;
        background-color: #f8f9fa !important;
        max-height: 400px !important;
    }
    .example-gallery {
        width: 100% !important;
    }
    .example-gallery img {
        width: 100% !important;
        height: 280px !important;
        object-fit: contain !important;
        aspect-ratio: 16 / 9 !important;
    }
    .example-gallery .grid-wrap {
        width: 100% !important;
    }
    
    /* 滑块导航样式 */
    .depth-tab-improved .gradio-slider input[type="range"] {
        height: 8px !important;
        border-radius: 4px !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }

    .depth-tab-improved .gradio-slider input[type="range"]::-webkit-slider-thumb {
        height: 20px !important;
        width: 20px !important;
        border-radius: 50% !important;
        background: #fff !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
    }

    .depth-tab-improved button {
        transition: all 0.3s ease !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    .depth-tab-improved button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    .normal-tab-improved .gradio-slider input[type="range"] {
        height: 8px !important;
        border-radius: 4px !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }

    .normal-tab-improved .gradio-slider input[type="range"]::-webkit-slider-thumb {
        height: 20px !important;
        width: 20px !important;
        border-radius: 50% !important;
        background: #fff !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
    }

    .normal-tab-improved button {
        transition: all 0.3s ease !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    .normal-tab-improved button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }

    #depth-view-info, #normal-view-info {
        animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    """
) as demo:
    # State variables for the tabbed interface
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")
    processed_data_state = gr.State(value=None)
    current_view_index = gr.State(value=0)  # Track current view index for navigation

    # Header and description
    gr.HTML(
    """
    <div style="text-align: center;">
    <h1>
        <span style="background: linear-gradient(90deg, #3b82f6, #1e40af); -webkit-background-clip: text; background-clip: text; color: transparent; font-weight: bold;">WorldMirror:</span> 
        <span style="color: #555555;">Universal 3D World Reconstruction with Any Prior Prompting</span>
    </h1>
    <p>
    <a href="https://arxiv.org/abs/2510.10726">📄 ArXiv Paper</a> |
    <a href="https://3d-models.hunyuan.tencent.com/world/">🌐 Project Page</a> |
    <a href="https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror">💻 GitHub Repository</a> | 
    <a href="https://huggingface.co/tencent/HunyuanWorld-Mirror">🤗 Hugging Face Model</a>
    </p>
    </div>
    <div style="font-size: 16px; line-height: 1.5;">
        <p>WorldMirror supports any combination of inputs (images, intrinsics, poses, and depth) and multiple outputs including point clouds, camera parameters, depth maps, normal maps, and 3D Gaussian Splatting (3DGS). </p>
    <h3>How to Use:</h3>
    <ol>
        <li><strong>Upload Your Data:</strong> Click the "Upload Video or Images" button to add your files. Videos are automatically extracted into frames at one-second intervals.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction.</li>
            <li><strong>Visualize:</strong> Explore multiple reconstruction results across different tabs:
                <ul>
                    <li><strong>3D View:</strong> Interactive point cloud/mesh visualization with camera poses (downloadable as GLB)</li>
                    <li><strong>3D Gaussian Splatting:</strong> Interactive 3D Gaussian Splatting visualization with RGB and depth videos (downloadable as PLY)</li>
                    <li><strong>Depth Maps:</strong> Per-view depth estimation results (downloadable as PNG)</li>
                    <li><strong>Normal Maps:</strong> Per-view surface orientation visualization (downloadable as PNG)</li>
                    <li><strong>Camera Parameters:</strong> Estimated camera poses and intrinsics (downloadable as JSON)</li>
                </ul>
            </li>
    </ol>
    <p><strong style="color: #3b82f6;">Please note: Loading data and displaying 3D effects may take a moment. For faster performance, we recommend downloading the code from our GitHub and running it locally.</strong></p>
    </div>
    """)

    output_path_state = gr.Textbox(label="Output Path", visible=False, value="None")

    # Main UI components
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            file_upload = gr.File(
                file_count="multiple",
                label="Upload Video or Images",
                interactive=True,
                file_types=["image", "video"],
                height="200px",
            )
            time_interval = gr.Slider(
                minimum=0.1,
                maximum=10.0,
                value=1.0,
                step=0.1,
                label="Video Sample interval",
                interactive=True,
                visible=True,
                scale=4,
            )
            resample_btn = gr.Button(
                "Resample",
                visible=True,
                scale=1,
                elem_classes=["normal-weight-btn"],
            )
            image_gallery = gr.Gallery(
                label="Image Preview",
                columns=4,
                height="200px",
                show_download_button=True,
                object_fit="contain",
                preview=True
            )
            
            terminal_output = gr.Textbox(
                label="Terminal Output",
                lines=6,
                max_lines=6,
                interactive=False,
                show_copy_button=True,
                container=True,
                elem_classes=["terminal-output"],
                autoscroll=True
            )

        with gr.Column(scale=3):
            log_output = gr.Markdown(
                "Upload video or images first, then click Reconstruct to start processing",
                elem_classes=["custom-log"],
            )

            with gr.Tabs() as tabs:
                with gr.Tab("3D Gaussian Splatting", id=1) as gs_tab:
                    with gr.Row():
                        with gr.Column(scale=3):
                            gs_output = gr.Model3D(
                                label="Gaussian Splatting",
                                height=500,
                            )
                        with gr.Column(scale=1):
                            gs_rgb_video = gr.Video(
                                label="Rendered RGB Video",
                                height=250,
                                autoplay=False,
                                loop=False,
                                interactive=False,
                            )
                            gs_depth_video = gr.Video(
                                label="Rendered Depth Video",
                                height=250,
                                autoplay=False,
                                loop=False,
                                interactive=False,
                            )
                with gr.Tab("Point Cloud/Mesh", id=0):
                    reconstruction_output = gr.Model3D(
                        label="3D Pointmap/Mesh",
                        height=500,
                        zoom_speed=0.4,
                        pan_speed=0.4,
                    )
                with gr.Tab("Depth", elem_classes=["depth-tab-improved"]):
                    depth_view_info = gr.HTML(
                        value="<div style='text-align: center; padding: 10px; background: #f8f8f8; color: #999; border-radius: 8px; margin-bottom: 10px;'>"
                              "<strong>Depth View Navigation</strong> | Current: View 1 / 1 views</div>",
                        elem_id="depth-view-info"
                    )
                    depth_view_slider = gr.Slider(
                        minimum=1, 
                        maximum=1, 
                        step=1, 
                        value=1,
                        label="View Selection Slider",
                        interactive=True,
                        elem_id="depth-view-slider"
                    )
                    depth_map = gr.Image(
                        type="numpy",
                        label="Depth Map",
                        format="png",
                        interactive=False,
                        height=340
                    )
                with gr.Tab("Normal", elem_classes=["normal-tab-improved"]):
                    normal_view_info = gr.HTML(
                        value="<div style='text-align: center; padding: 10px; background: #f8f8f8; color: #999; border-radius: 8px; margin-bottom: 10px;'>"
                              "<strong>Normal View Navigation</strong> | Current: View 1 / 1 views</div>",
                        elem_id="normal-view-info"
                    )
                    normal_view_slider = gr.Slider(
                        minimum=1, 
                        maximum=1, 
                        step=1, 
                        value=1,
                        label="View Selection Slider",
                        interactive=True,
                        elem_id="normal-view-slider"
                    )
                    normal_map = gr.Image(
                        type="numpy",
                        label="Normal Map",
                        format="png",
                        interactive=False,
                        height=340
                    )
                with gr.Tab("Camera Parameters", elem_classes=["camera-tab"]):
                    with gr.Row():
                        gr.HTML("")
                        camera_params = gr.DownloadButton(
                            label="Download Camera Parameters",
                            scale=1,
                            variant="primary",
                        )
                        gr.HTML("")
                    
            with gr.Row():
                reconstruct_btn = gr.Button(
                    "Reconstruct", 
                    scale=1, 
                    variant="primary"
                )
                clear_btn = gr.ClearButton(
                    [
                        file_upload,
                        reconstruction_output,
                        log_output,
                        output_path_state,
                        image_gallery,
                        depth_map,
                        normal_map,
                        depth_view_slider,
                        normal_view_slider,
                        depth_view_info,
                        normal_view_info,
                        camera_params,
                        gs_output,
                        gs_rgb_video,
                        gs_depth_video,
                    ],
                    scale=1,
                )
                
            with gr.Row():
                frame_selector = gr.Dropdown(
                        choices=["All"], value="All", label="Show Points of a Specific Frame"
                    )
                
            gr.Markdown("### Reconstruction Options: (not applied to 3DGS)")
            with gr.Row():
                show_camera = gr.Checkbox(label="Show Camera", value=True)
                show_mesh = gr.Checkbox(label="Show Mesh", value=True)
                filter_ambiguous = gr.Checkbox(label="Filter low confidence & depth/normal edges", value=True)
                filter_sky_bg = gr.Checkbox(label="Filter Sky Background", value=False)

        with gr.Column(scale=1):            
            gr.Markdown("### Click to load example scenes")
            realworld_scenes = extract_example_scenes_metadata("examples/realistic") if os.path.exists("examples/realistic") else extract_example_scenes_metadata("examples")
            generated_scenes = extract_example_scenes_metadata("examples/stylistic") if os.path.exists("examples/stylistic") else []
            
            # If no subdirectories exist, fall back to single gallery
            if not os.path.exists("examples/realistic") and not os.path.exists("examples/stylistic"):
                # Fallback: use all scenes from examples directory
                all_scenes = extract_example_scenes_metadata("examples")
                if all_scenes:
                    gallery_items = [
                        (scene["thumbnail"], f"{scene['name']}\n📷 {scene['num_images']} images")
                        for scene in all_scenes
                    ]
                    
                    example_gallery = gr.Gallery(
                        value=gallery_items,
                        label="Example Scenes",
                        columns=1,
                        rows=None,
                        height=800,
                        object_fit="contain",
                        show_label=False,
                        interactive=True,
                        preview=False,
                        allow_preview=False,
                        elem_classes=["example-gallery"]
                    )
                    
                    def handle_example_selection(evt: gr.SelectData):
                        if evt:
                            result = load_example_scenes(all_scenes[evt.index]["name"], all_scenes)
                            return result
                        return (None, None, None, None, "No scene selected")
                    
                    example_gallery.select(
                        fn=handle_example_selection,
                        outputs=[
                            reconstruction_output,
                            gs_output,
                            output_path_state,
                            image_gallery,
                            log_output,
                        ],
                    )
            else:
                # Tabbed interface for categorized examples
                with gr.Tabs():
                    with gr.Tab("🌍 Realistic Cases"):
                        if realworld_scenes:
                            realworld_items = [
                                (scene["thumbnail"], f"{scene['name']}\n📷 {scene['num_images']} images")
                                for scene in realworld_scenes
                            ]
                            
                            realworld_gallery = gr.Gallery(
                                value=realworld_items,
                                label="Real-world Examples",
                                columns=1,
                                rows=None,
                                height=750,
                                object_fit="contain",
                                show_label=False,
                                interactive=True,
                                preview=False,
                                allow_preview=False,
                                elem_classes=["example-gallery"]
                            )
                            
                            def handle_realworld_selection(evt: gr.SelectData):
                                if evt:
                                    result = load_example_scenes(realworld_scenes[evt.index]["name"], realworld_scenes)
                                    return result
                                return (None, None, None, None, "No scene selected")
                            
                            realworld_gallery.select(
                                fn=handle_realworld_selection,
                                outputs=[
                                    reconstruction_output,
                                    gs_output,
                                    output_path_state,
                                    image_gallery,
                                    log_output,
                                ],
                            )
                        else:
                            gr.Markdown("No real-world examples available")
                    
                    with gr.Tab("🎨 Stylistic Cases"):
                        if generated_scenes:
                            generated_items = [
                                (scene["thumbnail"], f"{scene['name']}\n📷 {scene['num_images']} images")
                                for scene in generated_scenes
                            ]
                            
                            generated_gallery = gr.Gallery(
                                value=generated_items,
                                label="Generated Examples",
                                columns=1,
                                rows=None,
                                height=750,
                                object_fit="contain",
                                show_label=False,
                                interactive=True,
                                preview=False,
                                allow_preview=False,
                                elem_classes=["example-gallery"]
                            )
                            
                            def handle_generated_selection(evt: gr.SelectData):
                                if evt:
                                    result = load_example_scenes(generated_scenes[evt.index]["name"], generated_scenes)
                                    return result
                                return (None, None, None, None, "No scene selected")
                            
                            generated_gallery.select(
                                fn=handle_generated_selection,
                                outputs=[
                                    reconstruction_output,
                                    gs_output,
                                    output_path_state,
                                    image_gallery,
                                    log_output,
                                ],
                            )
                        else:
                            gr.Markdown("No generated examples available")
    
    # -------------------------------------------------------------------------
    # Click logic
    # -------------------------------------------------------------------------
    reconstruct_btn.click(fn=clear_fields, inputs=[], outputs=[]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            output_path_state,
            frame_selector,
            show_camera,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous
        ],
        outputs=[
            reconstruction_output,
            log_output,
            frame_selector,
            processed_data_state,
            depth_map,
            normal_map,
            depth_view_slider,
            normal_view_slider,
            depth_view_info,
            normal_view_info,
            camera_params,
            gs_output,
            gs_rgb_video,
            gs_depth_video,
            terminal_output,
        ],
    ).then(
        fn=lambda: "False",
        inputs=[],
        outputs=[is_example],  # set is_example to "False"
    )

    # -------------------------------------------------------------------------
    # Live update logic
    # -------------------------------------------------------------------------
    def refresh_3d_scene(
        workspace_path,
        frame_selector,
        show_camera,
        is_example,
        filter_sky_bg=False,
        show_mesh=False,
        filter_ambiguous=False
    ):
        """
        Refresh 3D scene visualization
        
        Load prediction data from workspace, generate or reuse GLB scene files based on current parameters,
        and return file paths needed for the 3D viewer.
        
        Args:
            workspace_path: Workspace directory path for reconstruction results
            frame_selector: Frame selector value for filtering points from specific frames
            show_camera: Whether to display camera positions
            is_example: Whether this is an example scene
            filter_sky_bg: Whether to filter sky background
            show_mesh: Whether to display as mesh mode
            filter_ambiguous: Whether to filter low-confidence ambiguous areas
            
        Returns:
            tuple: (GLB scene file path, Gaussian point cloud file path, status message)
        """

        # If example scene is clicked, skip processing directly
        if is_example == "True":
            return (
                gr.update(),
                gr.update(),
                "No reconstruction results available. Please click the Reconstruct button first.",
            )

        # Validate workspace directory path
        if not workspace_path or workspace_path == "None" or not os.path.isdir(workspace_path):
            return (
                gr.update(),
                gr.update(),
                "No reconstruction results available. Please click the Reconstruct button first.",
            )

        # Check if prediction data file exists
        prediction_file_path = os.path.join(workspace_path, "predictions.npz")
        if not os.path.exists(prediction_file_path):
            return (
                gr.update(),
                gr.update(),
                f"Prediction file does not exist: {prediction_file_path}. Please run reconstruction first.",
            )

        # Load prediction data
        prediction_data = np.load(prediction_file_path, allow_pickle=True)
        predictions = {key: prediction_data[key] for key in prediction_data.keys() if key != 'splats'}

        # Generate GLB scene file path (named based on parameter combination)
        safe_frame_name = frame_selector.replace('.', '_').replace(':', '').replace(' ', '_')
        scene_filename = f"scene_{safe_frame_name}_cam{show_camera}_mesh{show_mesh}_edges{filter_ambiguous}_sky{filter_sky_bg}.glb"
        scene_glb_path = os.path.join(workspace_path, scene_filename)

        # If GLB file doesn't exist, generate new scene file
        if not os.path.exists(scene_glb_path):
            scene_model = convert_predictions_to_glb_scene(
                predictions,
                filter_by_frames=frame_selector,
                show_camera=show_camera,
                mask_sky_bg=filter_sky_bg,
                as_mesh=show_mesh,
                mask_ambiguous=filter_ambiguous
            )
            scene_model.export(file_obj=scene_glb_path)

        # Find Gaussian point cloud file
        gaussian_file_path = os.path.join(workspace_path, "gaussians.ply")
        if not os.path.exists(gaussian_file_path):
            gaussian_file_path = None

        return (
            scene_glb_path,
            gaussian_file_path,
            "3D scene updated.",
        )
    
    def refresh_view_displays_on_filter_update(
        workspace_dir,
        sky_background_filter,
        current_processed_data,
        depth_slider_position,
        normal_slider_position,
    ):
        """
        Refresh depth and normal view displays when filter settings change
        
        When the background filter checkbox state changes, regenerate processed data and update all view displays.
        This ensures that filter effects are reflected in real-time in the depth map and normal map visualizations.
        
        Args:
            workspace_dir: Workspace directory path containing prediction data and images
            sky_background_filter: Sky background filter enable status
            current_processed_data: Currently processed visualization data
            depth_slider_position: Current position of the depth view slider
            normal_slider_position: Current position of the normal view slider
            
        Returns:
            tuple: (updated processed data, depth visualization result, normal visualization result)
        """
        
        # Validate workspace directory validity
        if not workspace_dir or workspace_dir == "None" or not os.path.isdir(workspace_dir):
            return current_processed_data, None, None

        # Build and check prediction data file path
        prediction_data_path = os.path.join(workspace_dir, "predictions.npz")
        if not os.path.exists(prediction_data_path):
            return current_processed_data, None, None

        try:
            # Load raw prediction data
            raw_prediction_data = np.load(prediction_data_path, allow_pickle=True)
            predictions_dict = {key: raw_prediction_data[key] for key in raw_prediction_data.keys()}

            # Load image data using WorldMirror's load_images function
            images_directory = os.path.join(workspace_dir, "images")
            image_file_paths = [os.path.join(images_directory, path) for path in os.listdir(images_directory)]
            img = load_and_preprocess_images(image_file_paths)
            img = img.detach().cpu().numpy()

            # Regenerate processed data with new filter settings
            refreshed_data = {}
            for view_idx in range(img.shape[1]):
                view_data = {
                    "image": img[0, view_idx],
                    "points3d": predictions_dict["world_points"][view_idx],
                    "depth": None,
                    "normal": None,
                    "mask": None,
                }
                mask = predictions_dict["final_mask"][view_idx].copy()
                if sky_background_filter:
                    sky_mask = predictions_dict["sky_mask"][view_idx]
                    mask = mask & sky_mask
                view_data["mask"] = mask
                view_data["depth"] = predictions_dict["depth"][view_idx].squeeze()
                view_data["normal"] = predictions_dict["normal"][view_idx]
                refreshed_data[view_idx] = view_data

            # Get current view indices from slider positions (convert to 0-based indices)
            current_depth_index = int(depth_slider_position) - 1 if depth_slider_position else 0
            current_normal_index = int(normal_slider_position) - 1 if normal_slider_position else 0

            # Update depth and normal views with new filter data
            updated_depth_visualization = update_depth_view(refreshed_data, current_depth_index)
            updated_normal_visualization = update_normal_view(refreshed_data, current_normal_index)

            return refreshed_data, updated_depth_visualization, updated_normal_visualization

        except Exception as error:
            print(f"Error occurred while refreshing view displays: {error}")
            return current_processed_data, None, None

    frame_selector.change(
        refresh_3d_scene,
        [
            output_path_state,
            frame_selector,
            show_camera,
            is_example,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous
        ],
        [reconstruction_output, gs_output, log_output],
    )
    show_camera.change(
        refresh_3d_scene,
        [
            output_path_state,
            frame_selector,
            show_camera,
            is_example,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous
        ],
        [reconstruction_output, gs_output, log_output],
    )
    show_mesh.change(
        refresh_3d_scene,
        [
            output_path_state,
            frame_selector,
            show_camera,
            is_example,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous
        ],
        [reconstruction_output, gs_output, log_output],
    )
    
    filter_sky_bg.change(
        refresh_3d_scene,
        [
            output_path_state,
            frame_selector,
            show_camera,
            is_example,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous
        ],
        [reconstruction_output, gs_output, log_output],
    ).then(
        fn=refresh_view_displays_on_filter_update,
        inputs=[
            output_path_state,
            filter_sky_bg,
            processed_data_state,
            depth_view_slider,
            normal_view_slider,
        ],
        outputs=[
            processed_data_state,
            depth_map,
            normal_map,
        ],
    )
    filter_ambiguous.change(
        refresh_3d_scene,
        [
            output_path_state,
            frame_selector,
            show_camera,
            is_example,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous
        ],
        [reconstruction_output, gs_output, log_output],
    ).then(
        fn=refresh_view_displays_on_filter_update,
        inputs=[
            output_path_state,
            filter_sky_bg,
            processed_data_state,
            depth_view_slider,
            normal_view_slider,
        ],
        outputs=[
            processed_data_state,
            depth_map,
            normal_map,
        ],
    )

    # -------------------------------------------------------------------------
    # Auto update gallery when user uploads or changes files
    # -------------------------------------------------------------------------
    def update_gallery_on_file_upload(files, interval):
        if not files:
            return None, None, None, ""
        
        # Capture terminal output
        tee = TeeOutput()
        old_stdout = sys.stdout
        sys.stdout = tee
        
        try:
            target_dir, image_paths = process_uploaded_files(files, interval)
            terminal_log = tee.getvalue()
            sys.stdout = old_stdout
            
            return (
                target_dir,
                image_paths,
                "Upload complete. Click 'Reconstruct' to begin 3D processing.",
                terminal_log,
            )
        except Exception as e:
            terminal_log = tee.getvalue()
            sys.stdout = old_stdout
            print(f"Error occurred: {e}")
            raise

    def resample_video_with_new_interval(files, new_interval, current_target_dir):
        """Resample video with new slider value"""
        if not files:
            return (
                current_target_dir,
                None,
                "No files to resample.",
                "",
            )

        # Check if we have videos to resample
        video_extensions = [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
            ".webm",
            ".m4v",
            ".3gp",
        ]
        has_video = any(
            os.path.splitext(
                str(file_data["name"] if isinstance(file_data, dict) else file_data)
            )[1].lower()
            in video_extensions
            for file_data in files
        )

        if not has_video:
            return (
                current_target_dir,
                None,
                "No videos found to resample.",
                "",
            )

        # Capture terminal output
        tee = TeeOutput()
        old_stdout = sys.stdout
        sys.stdout = tee
        
        try:
            # Clean up old target directory if it exists
            if (
                current_target_dir
                and current_target_dir != "None"
                and os.path.exists(current_target_dir)
            ):
                shutil.rmtree(current_target_dir)

            # Process files with new interval
            target_dir, image_paths = process_uploaded_files(files, new_interval)
            
            terminal_log = tee.getvalue()
            sys.stdout = old_stdout

            return (
                target_dir,
                image_paths,
                f"Video resampled with {new_interval}s interval. Click 'Reconstruct' to begin 3D processing.",
                terminal_log,
            )
        except Exception as e:
            terminal_log = tee.getvalue()
            sys.stdout = old_stdout
            print(f"Error occurred: {e}")
            raise

    file_upload.change(
        fn=update_gallery_on_file_upload,
        inputs=[file_upload, time_interval],
        outputs=[output_path_state, image_gallery, log_output, terminal_output],
    )

    resample_btn.click(
        fn=resample_video_with_new_interval,
        inputs=[file_upload, time_interval, output_path_state],
        outputs=[output_path_state, image_gallery, log_output, terminal_output],
    )

    # -------------------------------------------------------------------------
    # Navigation for Depth, Normal tabs
    # -------------------------------------------------------------------------
    def navigate_with_slider(processed_data, target_view):
        """Navigate to specified view using slider"""
        if processed_data is None or len(processed_data) == 0:
            return None, update_view_info(1, 1)
        
        # Check if target_view is None or invalid value, and safely convert to int
        try:
            if target_view is None:
                target_view = 1
            else:
                target_view = int(float(target_view))  # Convert to float first then int, handle decimal input
        except (ValueError, TypeError):
            target_view = 1
        
        total_views = len(processed_data)
        # Ensure view index is within valid range
        view_index = max(1, min(target_view, total_views)) - 1
        
        # Update depth map
        depth_vis = update_depth_view(processed_data, view_index)
        
        # Update view information
        info_html = update_view_info(view_index + 1, total_views)
        
        return depth_vis, info_html

    def navigate_with_slider_normal(processed_data, target_view):
        """Navigate to specified normal view using slider"""
        if processed_data is None or len(processed_data) == 0:
            return None, update_view_info(1, 1, "Normal")
        
        # Check if target_view is None or invalid value, and safely convert to int
        try:
            if target_view is None:
                target_view = 1
            else:
                target_view = int(float(target_view))  # Convert to float first then int, handle decimal input
        except (ValueError, TypeError):
            target_view = 1
        
        total_views = len(processed_data)
        # Ensure view index is within valid range
        view_index = max(1, min(target_view, total_views)) - 1
        
        # Update normal map
        normal_vis = update_normal_view(processed_data, view_index)
        
        # Update view information
        info_html = update_view_info(view_index + 1, total_views, "Normal")
        
        return normal_vis, info_html

    def handle_depth_slider_change(processed_data, target_view):
        return navigate_with_slider(processed_data, target_view)
    
    def handle_normal_slider_change(processed_data, target_view):
        return navigate_with_slider_normal(processed_data, target_view)
    
    depth_view_slider.change(
        fn=handle_depth_slider_change,
        inputs=[processed_data_state, depth_view_slider],
        outputs=[depth_map, depth_view_info]
    )
    
    normal_view_slider.change(
        fn=handle_normal_slider_change,
        inputs=[processed_data_state, normal_view_slider],
        outputs=[normal_map, normal_view_info]
    )
    
    # -------------------------------------------------------------------------
    # Real-time terminal output update
    # -------------------------------------------------------------------------
    # Use a timer to periodically update terminal output
    timer = gr.Timer(value=0.5)  # Update every 0.5 seconds
    timer.tick(
        fn=get_terminal_output,
        inputs=[],
        outputs=[terminal_output]
    )
    
    gr.HTML("""
    <hr style="margin-top: 40px; margin-bottom: 20px;">
    <div style="text-align: center; font-size: 14px; color: #666; margin-bottom: 20px;">
        <h3>Acknowledgements</h3>
        <p>🔗 <a href="https://github.com/microsoft/MoGe">MoGe2 on HuggingFace</a> | 🔗 <a href="https://github.com/facebookresearch/vggt">VGGT on HuggingFace</a></p>
    </div>
    """)

    demo.queue().launch(
        show_error=True,
        share=False,
        server_name="127.0.0.1",
        server_port=8080,
        ssr_mode=False,
    )
