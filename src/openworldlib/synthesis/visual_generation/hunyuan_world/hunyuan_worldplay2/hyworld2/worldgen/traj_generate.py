import argparse
import copy
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import trimesh
import utils3d
from PIL import Image
from moge.model.v2 import MoGeModel
from openai import OpenAI
from scipy.spatial import cKDTree
from tqdm import tqdm
from transformers import Sam3Processor, Sam3Model

from src.camera_utils import add_scene_cam, get_c2w, CAM_COLORS, interpolate_poses, compute_lookat_xy_angle
from src.general_utils import save_16bit_png_depth, set_seed, adjust_image_size, Timer, rank0_log
from src.navi_utils import (
    find_robust_center,
    project_center_to_3d,
    get_max_size_center,
    save_visualization,
    pil_image_to_base64,
    get_navigation_instruction,
    deduplicate_ordered,
    get_bearing_and_direction,
    get_topk_seg_data,
    get_mask_edge_points_3d,
    create_and_save_combined_pcd,
    process_single_scene,
    process_trajectories,
    select_reconstruct_via_fps,
    filter_and_select_diverse_trajectories,
    compute_trajectory_similarity_matrix,
    visualize_comparison
)
from src.panorama_utils import (
    split_panorama_image,
    split_panorama_depth,
    rotate_around_z_axis,
    pred_pano_depth,
    get_view_point_from_panorama_point,
    convert_rgbd2pcd_panorama,
    convert_rgbd2mesh_panorama,
    smooth_sky_depth_boundary,
    erp_distance_ray_to_normal
)
from src.pointcloud import point_rendering
from src.seg_utils import get_zim_mask, build_gd_model, build_zim_model
from src.vlm_utils import get_qwen_caption_format

os.environ["TOKENIZERS_PARALLELISM"] = "false"
timer = Timer()

# Runtime environment.
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

LLM_ADDR = "localhost"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

LLM_PORT = 8000

SAM_BATCH_SIZE = 4
MAX_MASK_COUNT = 5
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
ZIM_REPO_ID = "naver-iv/zim-anything-vitl"
ZIM_SUBFOLDER = "zim_vit_l_2092"
GD_REPO_ID = "IDEA-Research/grounding-dino-tiny"
SAM3_REPO_ID = "facebook/sam3"
MOGE_ID = "Ruicheng/moge-2-vitl-normal"


def resolve_hf_checkpoint(repo_id, allow_patterns=None, subfolder=None, required_files=None):
    """Download a Hugging Face model to the local cache and return its usable path."""
    from huggingface_hub import snapshot_download

    repo_root = snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        cache_dir=HF_CACHE_DIR,
    )
    checkpoint_dir = os.path.join(repo_root, subfolder) if subfolder else repo_root
    required_files = required_files or []
    missing_files = [filename for filename in required_files if not os.path.exists(os.path.join(checkpoint_dir, filename))]
    if missing_files:
        raise FileNotFoundError(f"Checkpoint '{repo_id}' is missing files in {checkpoint_dir}: {missing_files}")
    return checkpoint_dir


def resolve_zim_checkpoint():
    return resolve_hf_checkpoint(
        ZIM_REPO_ID,
        allow_patterns=[f"{ZIM_SUBFOLDER}/*"],
        subfolder=ZIM_SUBFOLDER,
        required_files=["encoder.onnx", "decoder.onnx"],
    )


def resolve_gd_checkpoint():
    return resolve_hf_checkpoint(
        GD_REPO_ID,
        allow_patterns=["*.json", "*.txt", "model.safetensors"],
        required_files=["config.json", "preprocessor_config.json", "model.safetensors"],
    )


def save_image(splitted_image, path):
    splitted_image.save(path)


def save_depth(depth_np, path):
    save_16bit_png_depth(depth_np, path)


def save_view_initial_data(args_tuple):
    """Save initial data for one view in the IO thread pool."""
    (view_i, scene_path, projected_pcd, point_mask, projected_uv, splitted_image) = args_tuple

    projected_pcd.export(f"{scene_path}/render_results/view{view_i}/points.ply")
    point_mask.save(f"{scene_path}/render_results/view{view_i}/point_mask.png")
    np.save(f"{scene_path}/render_results/view{view_i}/projected_xy.npy", projected_uv)
    splitted_image.save(f"{scene_path}/render_results/view{view_i}/start_frame.png")

    return view_i


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", default=None, type=str, help="target path")
    parser.add_argument("--fov_x", default=120, type=float, help="panorama split fov x")
    parser.add_argument("--fov_y", default=90, type=float, help="panorama split fov y")
    parser.add_argument("--seed", default=1024, type=int, help="random seed for reproducibility")
    parser.add_argument("--split_view_num", default=3, type=int, help="final split view num")
    parser.add_argument("--splitted_resolution", default=480, type=int, help="splitted resolution")
    parser.add_argument("--nframe", default=21, type=int, help="number of frames for trajectory generation")
    parser.add_argument("--distance_threshold", default=0.1, type=float, help="distance threshold for obstacle avoidance")
    parser.add_argument("--obs_iteration_limit", default=3, type=int, help="obstacle avoidance iteration limit")
    parser.add_argument("--rotation_deg", default=120, type=float, help="rotation degree (left/right)")
    parser.add_argument("--rotation_up", default=45, type=float, help="rotation degree up")
    parser.add_argument("--up_right", default=60, type=float, help="rotation degree up-and-right")
    parser.add_argument("--obs_decay", default=2 / 3, type=float, help="obstacle decay factor")
    parser.add_argument("--contract", default=8.0, type=float, help="depth contract factor, the overall depth range: [0, median_depth * contract * 2]")
    parser.add_argument("--skip_exist", action="store_true", help="skip existing videos")

    # navigation params
    parser.add_argument("--apply_nav_traj", action="store_true", help="apply navigation trajectory")
    parser.add_argument("--wonder_topk", type=int, default=3)
    parser.add_argument("--recon_topk", type=int, default=5)
    parser.add_argument("--move_dist", type=float, default=8.0)
    parser.add_argument("--radius_threshold", type=float, default=4.0)
    parser.add_argument("--min_angle_threshold", type=float, default=40.0)
    parser.add_argument("--traj_sim_threshold", type=float, default=0.7)
    parser.add_argument("--traj_sim_threshold_recon", type=float, default=0.7)
    parser.add_argument("--apply_up_route", action="store_true", help="Whether to render up views")
    parser.add_argument("--apply_recon_iteration", action="store_true", help="Whether to apply reconstruction iteration")
    parser.add_argument("--eloop_dist", type=float, default=0.25)
    parser.add_argument("--force_vlm", action="store_true", help="force VLM output")

    # NavMesh Params
    parser.add_argument("--cellSize", type=float, default=0.1)
    parser.add_argument("--cellHeight", type=float, default=0.1)
    parser.add_argument("--agentHeight", type=float, default=0.2)
    parser.add_argument("--agentRadius", type=float, default=0.1)
    parser.add_argument("--agentMaxClimb", type=float, default=0.1)
    parser.add_argument("--maxSlope", type=float, default=30.0)
    parser.add_argument("--roof_height_threshold", type=float, default=0.1)

    # multi-process params
    parser.add_argument("--node_rank", type=int, default=0, help="local rank for multi-node")
    parser.add_argument("--node_size", type=int, default=1, help="world size for multi-node")

    # VLM server params
    parser.add_argument("--llm_addr", type=str, default=LLM_ADDR, help="vLLM server address")
    parser.add_argument("--llm_port", type=int, default=LLM_PORT, help="vLLM server port")
    parser.add_argument("--llm_name", type=str, default=MODEL_NAME, help="VLM model name served by vLLM")

    args = parser.parse_args()

    # Override globals with argparse values
    LLM_ADDR = args.llm_addr
    LLM_PORT = args.llm_port
    MODEL_NAME = args.llm_name

    device = torch.device("cuda")
    set_seed(args.seed)

    print("Models Initializing...")
    zim_predictor = build_zim_model("vit_l", resolve_zim_checkpoint(), device=device)
    gd_processor, gd_model = build_gd_model(resolve_gd_checkpoint(), device=device)

    depth_model = MoGeModel.from_pretrained(MOGE_ID).to(device).eval()

    # VLM & SAM3
    client = OpenAI(api_key="EMPTY", base_url=f"http://{LLM_ADDR}:{LLM_PORT}/v1")
    sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("Models Initializing over.")

    # Near-view rotations used by regular trajectory generation.
    camera_candidates_near = [
        {"type": "normal", "backward-forward": 0, "left-right": 0, "rotation": [-args.rotation_deg, 0], "name": "right-rotation"},
        {"type": "normal", "backward-forward": 0, "left-right": 0, "rotation": [args.rotation_deg, 0], "name": "left-rotation"},
    ]
    if args.up_right > 0:
        camera_candidates_near.append({"type": "aerial", "backward-forward": 0, "left-right": 0, "rotation": [-args.up_right, -args.rotation_up], "name": "up-right-aerial"})
    else:
        camera_candidates_near.append({"type": "normal", "backward-forward": 0, "left-right": 0, "rotation": [0, -args.rotation_up], "name": "up-rotation"})

    if os.path.exists(f"{args.target_path}/panorama.png"):
        scene_list = [args.target_path]  # single path VLM inference
    else:
        scene_list = glob(f"{args.target_path}/*")

    scene_list.sort()
    scene_list = scene_list[args.node_rank::args.node_size]

    for scene_path in tqdm(scene_list):

        # ======================================== Stage1: Regular Trajectory Generation ========================================
        if not args.skip_exist and os.path.exists(f"{scene_path}/render_results"):
            print(f"Delete existing {scene_path}/render_results")
            render_results_list = glob(f"{scene_path}/render_results/*")
            render_results_list = [r for r in render_results_list if r.split('/')[-1] not in ("full_depth_prediction.pt", "global_mesh.ply",
                                                                                              "global_normal.npy", "global_pcd.ply", "sky_mask.png", "sky_pcd.ply")]
            for path in render_results_list:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

        print(f"Stage1: Start regular trajectory generation for scene: {scene_path.split('/')[-1]}")

        image_path = f"{scene_path}/panorama_sr.png" if os.path.exists(f"{scene_path}/panorama_sr.png") else f"{scene_path}/panorama.png"

        full_img = Image.open(image_path)
        if full_img.size[1] > 1920:
            full_img = full_img.resize((3840, 1920), resample=Image.Resampling.BICUBIC)
        width_origin, height_origin = full_img.size

        # get meta info
        if os.path.exists(f"{scene_path}/meta_info.json"):
            with open(f"{scene_path}/meta_info.json", "r") as f:
                meta_info = json.load(f)
        else:
            meta_info = {}
            base64_image = pil_image_to_base64(full_img)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": get_qwen_caption_format("env_cls")},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ]}
            ]
            print(f"Qwen3-VL labeling meta information for {scene_path}...")
            with timer.track("Qwen3-VL labeling meta information"):
                response = client.chat.completions.create(model=MODEL_NAME, messages=messages, max_tokens=1024, temperature=0.0, seed=1024)
                clean_text = response.choices[0].message.content.strip().replace('[', '').replace(']', '').replace('"', '').replace("'", "").replace("```json", "").replace("```", "")
            meta_info["scene_type"] = clean_text
            with open(f"{scene_path}/meta_info.json", "w") as write:
                json.dump(meta_info, write, indent=2)

        os.makedirs(f"{scene_path}/render_results", exist_ok=True)

        if os.path.exists(f"{scene_path}/render_results/sky_mask.png"):
            sky_mask = np.array(Image.open(f"{scene_path}/render_results/sky_mask.png")) / 255
            sky_mask = ~torch.from_numpy(sky_mask).bool()
        elif os.path.exists(f"{scene_path}/sky_mask.png"):
            sky_mask = Image.open(f"{scene_path}/sky_mask.png").convert("L")  # UE sky mask is inverted.
            sky_mask = Image.fromarray(255 - np.array(sky_mask))
            sky_mask = sky_mask.resize((width_origin, height_origin), resample=Image.Resampling.NEAREST)
            sky_mask.save(f"{scene_path}/render_results/sky_mask.png")
            sky_mask = np.array(sky_mask) / 255
            sky_mask = ~torch.from_numpy(sky_mask).bool()
        else:
            with timer.track("Get sky mask"):
                if meta_info["scene_type"] == "outdoor":
                    sky_mask = torch.tensor(~get_zim_mask(full_img, "sky.", 0.3, 0.3, zim_predictor, gd_processor, gd_model, DEVICE=device))
                    # FIXME: Treat sky-dominant scenes as all non-sky for now.
                    if sky_mask.float().mean() > 0.9:
                        rank0_log(f"Sky mask is too high for {scene_path} ({sky_mask.float().mean()}), set to all non-sky")
                        sky_mask[:] = False
                else:
                    sky_mask = torch.zeros((full_img.size[1], full_img.size[0])).bool()
            # save sky mask
            transforms.ToPILImage()(((~sky_mask).float() * 255).to(torch.uint8)).save(f"{scene_path}/render_results/sky_mask.png")

        # predict depth
        if os.path.exists(f"{scene_path}/render_results/full_depth_prediction.pt"):
            full_depth = torch.load(f"{scene_path}/render_results/full_depth_prediction.pt", weights_only=False)
        else:
            with timer.track("Predict panorama depth"):
                full_depth = pred_pano_depth(depth_model, full_img)
                edge_mask = torch.from_numpy(utils3d.numpy.depth_edge(full_depth["distance"].cpu().numpy(), rtol=0.1)).bool()
                sky_mask_for_depth = sky_mask
                if sky_mask_for_depth.shape != edge_mask.shape:
                    sky_mask_for_depth = F.interpolate(
                        sky_mask_for_depth[None, None].float(),
                        size=edge_mask.shape,
                        mode="nearest",
                    )[0, 0].bool()
                full_mask = (sky_mask_for_depth | edge_mask).to(device)
                max_d = torch.quantile(full_depth["distance"][~full_mask], q=0.99).item()
                full_depth["distance"] = torch.clip(full_depth["distance"], 0, max_d)
                if args.contract is not None and meta_info.get("scene_type") == "outdoor":
                    contract_distance = torch.median(full_depth["distance"].reshape(-1), dim=0)[0].item() * args.contract
                    contract_mask = full_depth["distance"] > contract_distance
                    full_depth["distance"][contract_mask] = (2 * contract_distance) - (contract_distance ** 2 / (full_depth["distance"][contract_mask] + 1e-6))
            with timer.track("[IO] Save panorama depth"):
                torch.save(full_depth, f"{scene_path}/render_results/full_depth_prediction.pt")
        edge_mask = torch.from_numpy(utils3d.numpy.depth_edge(full_depth["distance"].cpu().numpy(), rtol=0.1)).bool()
        full_depth["distance"] = full_depth["distance"].to(device)
        full_depth["rays"] = full_depth["rays"].to(device)
        sky_mask_for_depth = sky_mask
        if sky_mask_for_depth.shape != edge_mask.shape:
            sky_mask_for_depth = F.interpolate(
                sky_mask_for_depth[None, None].float(),
                size=edge_mask.shape,
                mode="nearest",
            )[0, 0].bool()
        full_mask = (sky_mask_for_depth | edge_mask).to(device)
        global_median_depth = torch.median(full_depth["distance"][~full_mask]).item()

        # get global points
        if not os.path.exists(f"{scene_path}/render_results/global_pcd.ply"):
            with timer.track("Get panorama pointcloud"):
                depth_h, depth_w = full_depth["distance"].shape
                pcd_img = full_img
                if pcd_img.size != (depth_w, depth_h):
                    pcd_img = pcd_img.resize((depth_w, depth_h), resample=Image.Resampling.BICUBIC)
                global_pcd = convert_rgbd2pcd_panorama(
                    rgb=torch.tensor(np.array(pcd_img) / 255, dtype=torch.float32),
                    distance=full_depth["distance"],
                    rays=full_depth["rays"],
                    excluded_region_mask=full_mask,
                    dropout_pcd=False
                )
            with timer.track("[IO] Save panorama pointcloud"):
                global_pcd.export(f"{scene_path}/render_results/global_pcd.ply")
        else:
            global_pcd = trimesh.load(f"{scene_path}/render_results/global_pcd.ply")

        # Global Mesh
        if not os.path.exists(f"{scene_path}/render_results/global_mesh.ply"):
            with timer.track("Get panorama mesh"):
                mesh_h, mesh_w = 960, 1920
                img_resized = full_img.resize((mesh_w, mesh_h), resample=Image.Resampling.BICUBIC)
                depth_resized = F.interpolate(full_depth['distance'][None, None], size=(mesh_h, mesh_w), mode='nearest')[0, 0]
                rays_resized = F.interpolate(full_depth['rays'].permute(2, 0, 1)[None], size=(mesh_h, mesh_w), mode='bilinear')[0].permute(1, 2, 0)
                mask_resized = F.interpolate(sky_mask.float()[None, None], size=(mesh_h, mesh_w), mode='nearest')[0, 0].bool()
                global_mesh = convert_rgbd2mesh_panorama(
                    rgb=torch.tensor(np.array(img_resized) / 255, dtype=torch.float32),
                    distance=depth_resized.to(device),
                    rays=rays_resized.to(device),
                    excluded_region_mask=mask_resized.to(device),
                    device=device
                )
            with timer.track("[IO] Save panorama mesh"):
                o3d.io.write_triangle_mesh(f"{scene_path}/render_results/global_mesh.ply", global_mesh, compressed=True)
        else:
            global_mesh = o3d.io.read_triangle_mesh(f"{scene_path}/render_results/global_mesh.ply")

        # save normals
        with timer.track("Get panorama normal"):
            normal_map, _ = erp_distance_ray_to_normal(
                full_depth["distance"].cpu().numpy(),
                full_depth["rays"].cpu().numpy(),
                smooth_sigma=0.5,
                facing_camera=True  # Keep normals facing the camera.
            )
            normal_map = normal_map[~full_mask.cpu().numpy()]
        with timer.track("[IO] Save panorama normal"):
            np.save(f"{scene_path}/render_results/global_normal.npy", normal_map)

        # save sky points
        if not os.path.exists(f"{scene_path}/render_results/sky_pcd.ply"):
            with timer.track("Get sky pointcloud"):
                sky_depth = full_depth["distance"].clone()
                sky_depth[sky_mask] = sky_depth.max()
                sky_depth = smooth_sky_depth_boundary(sky_depth, sky_mask.to(device), transition_width=100, method="mean")
                sky_pcd = convert_rgbd2pcd_panorama(
                    rgb=torch.tensor(np.array(full_img) / 255, dtype=torch.float32),
                    distance=sky_depth,
                    rays=full_depth["rays"],
                    excluded_region_mask=~sky_mask.to(device),
                    dropout_pcd=True
                )
            with timer.track("[IO] Save sky pointcloud"):
                sky_pcd.export(f"{scene_path}/render_results/sky_pcd.ply")

        # resize the shortest side to splitted_resolution
        image_h = args.splitted_resolution
        image_w = int(round(np.tan(np.deg2rad(args.fov_x / 2)) / np.tan(np.deg2rad(args.fov_y / 2)) * image_h))
        image_h, image_w = adjust_image_size(image_h, image_w)

        print(f"Image size: {image_h}x{image_w}")

        # Sampling polar views
        with timer.track("Sampling polar views"):
            polar_points = [np.array([-1, 0, 1.0], dtype=np.float32), np.array([-1, 0, -1.0], dtype=np.float32),
                            np.array([0.1, 0, -1.0], dtype=np.float32), np.array([0.1, 0, 1.0], dtype=np.float32)]
            direct_points = polar_points.copy()
            rot_deg = 90
            N_view = int(360 / rot_deg)
            for polar_point in polar_points:
                for i in range(1, N_view):
                    direct_points.append(rotate_around_z_axis(polar_point.reshape(1, 3), rot_deg * i)[0])
            direct_points = np.stack(direct_points, axis=0)
            intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(args.fov_x), fov_y=np.deg2rad(args.fov_y))
            splitted_intrinsics = [intrinsics] * len(direct_points)
            splitted_extrinsics = utils3d.numpy.extrinsics_look_at(np.array([0, 0, 0]), direct_points, np.array([0, 0, 1])).astype(np.float32)

            # build polar bank
            splitted_images = split_panorama_image(np.array(full_img), splitted_extrinsics, splitted_intrinsics, h=image_h, w=image_w, interp=cv2.INTER_AREA)
            splitted_depths = split_panorama_depth(np.array(full_depth["distance"].cpu()), splitted_extrinsics, splitted_intrinsics, h=image_h, w=image_w, distance_to_depth=True)
            splitted_masks = split_panorama_depth(~np.array(full_mask.cpu()), splitted_extrinsics, splitted_intrinsics, h=image_h, w=image_w)

        # save polar set
        save_tasks = []
        with timer.track("[IO] Save polar views"):
            os.makedirs(f"{scene_path}/render_results/polar_bank/images", exist_ok=True)
            os.makedirs(f"{scene_path}/render_results/polar_bank/depths", exist_ok=True)
            bank_cameras = {}
            for i in range(len(splitted_images)):
                fname = str(i).zfill(4)
                splitted_image = Image.fromarray(splitted_images[i])
                depth = splitted_depths[i]
                depth_mask = splitted_masks[i].bool()
                depth[~depth_mask] = 0
                depth = depth[0]
                K = splitted_intrinsics[i].copy()
                K[0] *= image_w
                K[1] *= image_h
                bank_cameras[fname] = {"intrinsic": K.tolist(), "extrinsic": splitted_extrinsics[i].tolist()}

                # Prepare data on the main thread before parallel IO.
                save_tasks.append(('image', splitted_image, f"{scene_path}/render_results/polar_bank/images/{fname}.png"))
                save_tasks.append(('depth', depth.cpu().numpy(), f"{scene_path}/render_results/polar_bank/depths/{fname}.png"))

            if save_tasks:
                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = [
                        executor.submit(save_image if task[0] == 'image' else save_depth, task[1], task[2])
                        for task in save_tasks
                    ]
                    for future in futures:
                        future.result()

            with open(f"{scene_path}/render_results/polar_bank/cameras.json", "w") as w:
                json.dump(bank_cameras, w)

        # Sampling center view, and build memory bank for in this case
        with timer.track("Sampling panorama views"):
            start_points = [np.array([-1, 0, 0], dtype=np.float32), np.array([-1, 0, 0.5], dtype=np.float32), np.array([-1, 0, -0.5], dtype=np.float32)]
            direct_points = start_points.copy()
            mid_indices = [0]
            rot_deg = 40
            N_view = int(360 / rot_deg)
            for start_point in start_points:
                for i in range(1, N_view):
                    direct_points.append(rotate_around_z_axis(start_point.reshape(1, 3), rot_deg * i)[0])
                    if start_point[2] == 0:
                        mid_indices.append(len(direct_points) - 1)
            direct_points = np.stack(direct_points, axis=0)
            intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(args.fov_x), fov_y=np.deg2rad(args.fov_y))
            splitted_intrinsics = [intrinsics] * len(direct_points)
            splitted_extrinsics = utils3d.numpy.extrinsics_look_at(np.array([0, 0, 0]), direct_points, np.array([0, 0, 1])).astype(np.float32)

            # build memory bank
            splitted_images = split_panorama_image(np.array(full_img), splitted_extrinsics, splitted_intrinsics, h=image_h, w=image_w, interp=cv2.INTER_AREA)
            splitted_depths = split_panorama_depth(np.array(full_depth["distance"].cpu()), splitted_extrinsics, splitted_intrinsics, h=image_h, w=image_w, distance_to_depth=True)
            splitted_masks = split_panorama_depth(~np.array(full_mask.cpu()), splitted_extrinsics, splitted_intrinsics, h=image_h, w=image_w)

        save_tasks = []
        with timer.track("[IO] Save panorama views"):
            # save support set
            os.makedirs(f"{scene_path}/render_results/pano_bank/images", exist_ok=True)
            os.makedirs(f"{scene_path}/render_results/pano_bank/depths", exist_ok=True)
            bank_cameras = {}
            for i in range(len(splitted_images)):
                fname = str(i).zfill(4)
                splitted_image = Image.fromarray(splitted_images[i])
                depth = splitted_depths[i]
                depth_mask = splitted_masks[i].bool()
                depth[~depth_mask] = 0
                depth = depth[0]
                K = splitted_intrinsics[i].copy()
                K[0] *= image_w
                K[1] *= image_h
                bank_cameras[fname] = {"intrinsic": K.tolist(), "extrinsic": splitted_extrinsics[i].tolist()}

                save_tasks.append(('image', splitted_image, f"{scene_path}/render_results/pano_bank/images/{fname}.png"))
                save_tasks.append(('depth', depth.cpu().numpy(), f"{scene_path}/render_results/pano_bank/depths/{fname}.png"))

            if save_tasks:
                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = [
                        executor.submit(save_image if task[0] == 'image' else save_depth, task[1], task[2])
                        for task in save_tasks
                    ]
                    for future in futures:
                        future.result()

            with open(f"{scene_path}/render_results/pano_bank/cameras.json", "w") as w:
                json.dump(bank_cameras, w)

        # only mid indices are used for camera control
        mid_indices = set(mid_indices)
        keep_indices = [i for i in range(len(splitted_images)) if i in mid_indices]
        splitted_images = [splitted_images[i] for i in keep_indices]
        splitted_depths = [splitted_depths[i] for i in keep_indices]
        splitted_masks = [splitted_masks[i] for i in keep_indices]
        splitted_intrinsics = [splitted_intrinsics[i] for i in keep_indices]
        splitted_extrinsics = splitted_extrinsics[keep_indices]
        direct_points = [direct_points[i] for i in keep_indices]

        scene = trimesh.Scene()
        kdtree = cKDTree(global_pcd.vertices)

        with timer.track("Split panorama views"):
            # Keep the panorama front view as the first split view.
            new_start_points = np.array([-1, 0, 0], dtype=np.float32)
            direct_points = [new_start_points]
            rot_deg = 360 / args.split_view_num
            for i in range(1, args.split_view_num):
                direct_points.append(rotate_around_z_axis(new_start_points.reshape(1, 3), rot_deg * i)[0])
            direct_points = np.stack(direct_points, axis=0)
            intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(args.fov_x), fov_y=np.deg2rad(args.fov_y))
            splitted_intrinsics = [intrinsics] * len(direct_points)
            splitted_extrinsics = utils3d.numpy.extrinsics_look_at(np.array([0, 0, 0]), direct_points, np.array([0, 0, 1])).astype(np.float32)

            splitted_images = split_panorama_image(np.array(full_img), splitted_extrinsics, splitted_intrinsics, h=image_h, w=image_w, interp=cv2.INTER_AREA)
            splitted_depths = split_panorama_depth(np.array(full_depth["distance"].cpu()), splitted_extrinsics, splitted_intrinsics, h=image_h, w=image_w, distance_to_depth=True)
            splitted_masks = split_panorama_depth(~np.array(full_mask.cpu()), splitted_extrinsics, splitted_intrinsics, h=image_h, w=image_w)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(global_mesh.vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(global_mesh.triangles))

        # Prepare per-view data before planning.
        with timer.track("Prepare all views data"):
            view_data_list = []
            io_save_tasks = []

            for i in range(len(splitted_images)):
                if not args.skip_exist and os.path.exists(f"{scene_path}/render_results/view{i}"):
                    shutil.rmtree(f"{scene_path}/render_results/view{i}")

                c2w_start = np.linalg.inv(np.array(splitted_extrinsics[i]))
                K = splitted_intrinsics[i].copy()
                K[0, :] *= image_w
                K[1, :] *= image_h

                depth = splitted_depths[i]
                depth_mask = splitted_masks[i].bool()
                if depth_mask.sum() == 0:
                    print(f"Error View {i} of {scene_path} has no valid depth")
                    continue

                median_depth = torch.median(depth[depth_mask]).item()
                if not np.isfinite(median_depth):
                    print(f"Error View {i} of {scene_path} has no valid depth")
                    continue

                os.makedirs(f"{scene_path}/render_results/view{i}", exist_ok=True)

                splitted_image = Image.fromarray(splitted_images[i])

                projected_points, projected_colors, projected_uv = get_view_point_from_panorama_point(
                    global_pcd, splitted_extrinsics[i], splitted_intrinsics[i], image_h, image_w
                )
                projected_pcd = trimesh.PointCloud(vertices=projected_points, colors=projected_colors)
                point_mask = np.zeros((image_h, image_w), dtype=np.uint8)
                point_mask[projected_uv[:, 1], projected_uv[:, 0]] = 255
                point_mask_img = Image.fromarray(point_mask)

                view_data_list.append({
                    'view_i': i,
                    'c2w_start': c2w_start,
                    'K': K,
                    'median_depth': median_depth,
                })

                io_save_tasks.append((i, scene_path, projected_pcd, point_mask_img, projected_uv, splitted_image))

        if io_save_tasks:
            with timer.track("[IO] Save all views initial data (parallel)"):
                with ThreadPoolExecutor(max_workers=min(len(io_save_tasks), 16)) as io_executor:
                    io_futures = [io_executor.submit(save_view_initial_data, task) for task in io_save_tasks]
                    for future in as_completed(io_futures):
                        future.result()

        with timer.track("Plan regular trajectories"):
            total_trajectories = 0
            for view_data in tqdm(view_data_list, desc="Planning regular views"):
                view_i = view_data['view_i']
                c2w_start = view_data['c2w_start']
                K = view_data['K']
                median_depth = view_data['median_depth']

                for trajectory_i, move in enumerate(camera_candidates_near):
                    camera_path = f"{scene_path}/render_results/view{view_i}/traj{trajectory_i}/camera.json"
                    if args.skip_exist and os.path.exists(camera_path):
                        continue

                    c2ws_next, obs_iteration = get_c2w(
                        c2w_start.copy(), move, median_depth,
                        air_bound=median_depth * 0.5,
                        n_inter=args.nframe - 1,
                        kdtree=kdtree,
                        mesh=mesh,
                        distance_threshold=args.distance_threshold,
                        local_rank=0,
                        obs_decay=args.obs_decay,
                        obs_limit=args.obs_iteration_limit
                    )
                    total_trajectories += 1

                    if obs_iteration > args.obs_iteration_limit:
                        print(f"Too many collisions in view{view_i}/traj{trajectory_i}, ignore...")
                        continue

                    os.makedirs(f"{scene_path}/render_results/view{view_i}/traj{trajectory_i}", exist_ok=True)

                    c2ws_next = np.concatenate([c2w_start[None], c2ws_next], axis=0)

                    for c2w in c2ws_next:
                        add_scene_cam(scene, c2w, CAM_COLORS[trajectory_i % len(CAM_COLORS)], None,
                                      args.splitted_resolution * 0.5, imsize=[image_w, image_h],
                                      screen_width=median_depth * 0.15)

                    w2cs = np.linalg.inv(c2ws_next)
                    Ks = np.array([K] * w2cs.shape[0])
                    camera_info = {
                        "extrinsic": w2cs.tolist(),
                        "intrinsic": Ks.tolist(),
                        "width": image_w,
                        "height": image_h,
                        "type": move["name"],
                        "rotation_deg": np.sum(np.abs(move['rotation'])) * (args.obs_decay ** obs_iteration)
                    }

                    with open(f"{scene_path}/render_results/view{view_i}/traj{trajectory_i}/camera.json", "w") as write:
                        json.dump(camera_info, write, indent=2)
            print(f"Total trajectory tasks: {total_trajectories}")

        point = trimesh.PointCloud(vertices=np.array([0, 0, 0]).reshape(1, 3),
                                   colors=np.array([255, 255, 255]).reshape(1, 3))
        scene.add_geometry(point)
        scene.export(f"{scene_path}/render_results/cameras.glb")

        # ======================================== Stage2: Navmesh Trajectory Generation ========================================
        if args.apply_nav_traj:
            print(f"Stage2: Start navigation trajectory generation for scene: {scene_path.split('/')[-1]}")
            camera_dir = f"{scene_path}/camera_trajectory"
            os.makedirs(camera_dir, exist_ok=True)

            # VLM & SAM3
            segmentation_data = []
            unique_objects = []
            if not (args.skip_exist and os.path.exists(os.path.join(scene_path, "objects.json"))):
                try:
                    base64_image = pil_image_to_base64(full_img)
                    messages = [
                        {"role": "system", "content": "You are a robot navigation assistant."},
                        {"role": "user", "content": [
                            {"type": "text", "text": get_navigation_instruction(args.force_vlm)},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        ]}
                    ]
                    print(f"Qwen3-VL labeling for {scene_path}...")
                    with timer.track("Qwen3-VL labeling objects"):
                        response = client.chat.completions.create(model=MODEL_NAME, messages=messages, max_tokens=1024, temperature=0.0, seed=1024)
                        clean_text = response.choices[0].message.content.strip().replace('[', '').replace(']', '').replace('"', '').replace("'", "").replace("```json", "").replace("```", "").replace("-", "_")
                        unique_objects = deduplicate_ordered([item.strip() for item in clean_text.split(',') if item.strip()])
                    with open(os.path.join(scene_path, "objects.json"), "w") as f:
                        json.dump(unique_objects, f, indent=4)
                except Exception as e:
                    rank0_log(f"  VLM Error: {e}", "ERROR")
            else:
                unique_objects = json.load(open(os.path.join(scene_path, "objects.json"), "r"))

            # processing object segmentation and masking
            if unique_objects:
                img_w, img_h = full_img.size
                occupancy_map = np.zeros((img_h, img_w), dtype=bool)
                valid_masks_vis = []
                valid_labels_vis = []
                valid_directions_vis = []
                seen_pairs = set()

                print(f"SAM3 process for {scene_path}...")
                for i in range(0, len(unique_objects), SAM_BATCH_SIZE):
                    batch_objects = unique_objects[i: i + SAM_BATCH_SIZE]
                    batch_objects = [obj for obj in batch_objects if len(obj.split()) < 8 and obj.lower() not in ("sun")]
                    batch_images = [full_img] * len(batch_objects)
                    if len(batch_objects) == 0:
                        continue
                    with timer.track("SAM3 segmentation"):
                        inputs = sam3_processor(images=batch_images, text=batch_objects, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = sam3_model(**inputs)
                        results = sam3_processor.post_process_instance_segmentation(outputs, threshold=0.4, mask_threshold=0.5, target_sizes=[full_img.size[::-1]] * len(batch_objects))

                    no_cluster_num = 0
                    with timer.track("Processing object masks (filtering)"):
                        batch_candidates = []
                        for j, res in enumerate(results):
                            current_obj_name = batch_objects[j]
                            masks = res.get("masks", None)
                            scores = res.get("scores", None)
                            if masks is None or len(masks) == 0:
                                continue

                            is_door = "door" in current_obj_name.lower() or "gate" in current_obj_name.lower()
                            limit = 6 if is_door else MAX_MASK_COUNT

                            if len(masks) >= limit:
                                is_cluster_obj = True
                            else:
                                is_cluster_obj = False

                            masks_np = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.array(masks)
                            scores_np = scores.detach().cpu().numpy() if scores is not None else [0.0] * len(masks)
                            if masks_np.ndim == 2: masks_np = masks_np[None, ...]

                            for m_k, mask in enumerate(masks_np):
                                area = np.sum(mask)
                                if area == 0:
                                    continue

                                # Drop masks that are too low or too high in the panorama.
                                mask_upper = np.min(np.where(mask == 1)[0])
                                mask_lower = np.max(np.where(mask == 1)[0])
                                if mask_upper > mask.shape[0] * 0.6 or mask_lower < mask.shape[0] * 0.4:
                                    print(f"Mask low ~ high from {current_obj_name}: ({mask_upper / mask.shape[0]}~{mask_lower / mask.shape[0]}), skipping")
                                    continue

                                mask_left_bound = np.min(np.where(mask == 1)[1])
                                mask_right_bound = np.max(np.where(mask == 1)[1])
                                if (mask_right_bound - mask_left_bound) > mask.shape[1] * 0.75:
                                    print(f"Mask width from {current_obj_name} is too wide: ({(mask_right_bound - mask_left_bound) / mask.shape[1]}), skipping")
                                    continue

                                batch_candidates.append({
                                    "mask": mask,
                                    "area": area,
                                    "label": current_obj_name,
                                    "score": float(scores_np[m_k]),
                                    "is_cluster_obj": is_cluster_obj
                                })
                                if not is_cluster_obj:
                                    no_cluster_num += 1

                        # Prefer individual objects when the batch has non-cluster candidates.
                        if no_cluster_num >= 1:
                            print(f"Ignore cluster objects because of too many objects, {len(batch_candidates)} -> {no_cluster_num}.")
                            batch_candidates = [cand for cand in batch_candidates if not cand['is_cluster_obj']]

                        batch_candidates.sort(key=lambda x: x['area'], reverse=True)

                        for cand in batch_candidates:
                            mask = cand['mask']
                            label = cand['label']
                            area = cand['area']

                            intersection = np.logical_and(mask, occupancy_map)
                            intersection_area = np.sum(intersection)
                            overlap_ratio = intersection_area / area
                            print(f"Mask Area {label}: overlap_ratio {overlap_ratio}.")
                            if overlap_ratio > 0.5:
                                continue

                            center_2d = find_robust_center(mask, full_depth["distance"])
                            if center_2d is None:
                                continue

                            direction_label, bearing = get_bearing_and_direction(center_2d[0], width_origin)
                            if (label, direction_label) in seen_pairs:
                                continue

                            occupancy_map = np.logical_or(occupancy_map, mask)
                            seen_pairs.add((label, direction_label))

                            point_3d, depth_val = project_center_to_3d(
                                center_2d,
                                full_depth["distance"],
                                full_depth["rays"],
                                mask=mask,
                                std_threshold=5.0
                            )

                            (left_3d, left_2d), (right_3d, right_2d) = get_mask_edge_points_3d(mask, full_depth["distance"], full_depth["rays"])
                            bbox_scale, _, _ = get_max_size_center(mask, full_depth["distance"], full_depth["rays"])

                            if left_2d is not None and right_2d is not None:
                                mid_x = (left_2d[0] + right_2d[0]) / 2
                                edge_center_direction, edge_center_bearing = get_bearing_and_direction(mid_x, width_origin)
                            elif left_2d is not None:
                                edge_center_direction, edge_center_bearing = get_bearing_and_direction(left_2d[0], width_origin)
                            elif right_2d is not None:
                                edge_center_direction, edge_center_bearing = get_bearing_and_direction(right_2d[0], width_origin)
                            else:
                                # Fall back to the robust center when edge points are missing.
                                edge_center_direction, edge_center_bearing = direction_label, bearing

                            segmentation_data.append({
                                "id": len(segmentation_data),
                                "label": label,
                                "score": float(cand['score']),
                                "scale_3d": float(bbox_scale),
                                "center_point_2d": center_2d.tolist() if isinstance(center_2d, np.ndarray) else center_2d,
                                "direction": direction_label,
                                "bearing_angle": float(bearing),
                                "center_point_3d": point_3d.tolist() if isinstance(point_3d, np.ndarray) else point_3d,
                                "depth_distance": float(depth_val),
                                "mask_area": mask.mean(),
                                "left_point_3d": left_3d,
                                "right_point_3d": right_3d,
                                "edge_center_direction": edge_center_direction,
                                "edge_center_bearing": float(edge_center_bearing)
                            })

                            valid_masks_vis.append(mask)
                            valid_labels_vis.append(label)
                            valid_directions_vis.append(direction_label)

                with timer.track("Processing object masks (ranking)"):
                    segmentation_data, _ = get_topk_seg_data(segmentation_data, topk=999)  # Sort only.
                with open(os.path.join(camera_dir, "target_camera.json"), "w") as f:
                    json.dump(segmentation_data, f, indent=4)

                if valid_masks_vis:
                    vis_path = f"{scene_path}/render_results/segmentation_vis.png"
                    save_visualization(full_img, np.array(valid_masks_vis), valid_labels_vis, valid_directions_vis, vis_path)
                    print(f"Saved visualization to {vis_path}")

                with timer.track("[IO] Save combined_with_markers.ply"):
                    if segmentation_data:
                        create_and_save_combined_pcd(global_pcd, segmentation_data, os.path.join(scene_path, "render_results", "combined_with_markers.ply"))

            # NavMesh & Paths
            try:
                is_outdoor = (meta_info.get("scene_type") == "outdoor")
                process_single_scene(scene_path, scene_path.split('/')[-1], global_mesh, args, segmentation_data, global_median_depth, is_outdoor=is_outdoor, timer=timer)
            except Exception as e:
                rank0_log(f"  Navmesh Error: {e}", "ERROR")

            # Re-load some info
            scene = trimesh.Scene()
            # Load Camera Intrinsics
            temp_list = glob(f"{scene_path}/render_results/view*/traj*/camera.json")
            if not temp_list:
                print("Camera files are not found! You should run the stage1 of 'traj_generate.py' at first.")
                continue
            else:
                ref_camera_info = json.load(open(temp_list[0]))
                image_h, image_w = ref_camera_info["height"], ref_camera_info["width"]
                K = np.array(ref_camera_info["intrinsic"])[0]

            seg_data = []
            seg_labels = {}
            wonder_camera_data = {}
            reconstruct_data = []
            surround_data = []
            trajectory_i = 0

            seg_path = os.path.join(camera_dir, "target_camera.json")

            # Load segmentation data and clear stale camera paths.
            if os.path.exists(seg_path):
                try:
                    with open(seg_path, "r") as f:
                        seg_data = json.load(f)
                    for idx, obj in enumerate(seg_data):
                        seg_labels[idx] = obj.get("label", "unknown")
                    with open(f"{scene_path}/navmesh/reconstruct_pairs.json", "r") as f:
                        reconstruct_data = json.load(f)
                    for item in reconstruct_data:
                        if 'camera_path' in item:
                            del item['camera_path']
                    surround_data = copy.deepcopy(seg_data)
                    for item in surround_data:
                        if 'camera_path' in item:
                            del item['camera_path']

                except Exception as e:
                    rank0_log(f"Warning: Failed to load segmentation labels: {e}", "ERROR")

            # Keep this render order: target/surround -> exploration -> reconstruct.
            render_tasks = [
                ("surround", f"{scene_path}/navmesh/surround/paths.json", "target"),
                ("exploration", f"{scene_path}/navmesh/exploration/paths.json", "wonder"),
                ("reconstruct", f"{scene_path}/navmesh/reconstruct/paths.json", "reconstruct"),
            ]

            for task_name, json_path, out_prefix in render_tasks:

                if not os.path.exists(json_path):
                    continue

                raw_paths_data = json.load(open(json_path))

                if not raw_paths_data:
                    continue

                path_list = []
                for p in raw_paths_data:
                    if p is None:
                        path_list.append(None)
                        continue
                    path_list.append(np.array(p))

                if task_name == "reconstruct":
                    assert len(path_list) == len(reconstruct_data)

                wonder_direction_label = "Unknown"
                origin_paths = []
                processed_c2ws = []
                path_vis_list = []
                folder_names = []
                processed_seg_list = []

                with timer.track("Per-path post-processing (get c2ws from path)"):
                    for i, path_data in enumerate(path_list):
                        target_center_3d = None
                        if path_data is None:
                            continue

                        current_path = path_data
                        current_path_vis = current_path

                        if task_name == "target":
                            if i < len(seg_data):
                                center_pt = seg_data[i].get("center_point_3d")
                                if center_pt:
                                    target_center_3d = np.array(center_pt)
                        elif task_name == "reconstruct":
                            target_pt = reconstruct_data[i].get("target_pt")
                            if target_pt:
                                target_center_3d = np.array(target_pt)
                        elif task_name == "surround":
                            if i < len(seg_data):
                                center_pt = seg_data[i].get("center_point_3d")
                                if center_pt:
                                    target_center_3d = np.array(center_pt)
                        else:
                            # Exploration: wonder0, wonder1...
                            if len(path_data) > 1:
                                # Use the path tangent on the Z-up XY plane.
                                look_ahead_idx = len(path_data) - 1

                                p_start = path_data[0]
                                p_end = path_data[look_ahead_idx]

                                dx = p_end[0] - p_start[0]
                                dy = p_end[1] - p_start[1]

                                # Panorama forward is treated as +X.
                                bearing = math.degrees(math.atan2(dy, -dx))

                                if -45 <= bearing < 45:
                                    wonder_direction_label = "Front"
                                elif 45 <= bearing < 135:
                                    wonder_direction_label = "Right"
                                elif -135 <= bearing < -45:
                                    wonder_direction_label = "Left"
                                else:
                                    wonder_direction_label = "Back"

                                print(f"Path {i} Bearing: {bearing:.2f}°, Direction: {wonder_direction_label}")

                        # Clamp paths to the ground plane before trajectory processing.
                        current_path_flat = current_path.copy()
                        current_path_flat[:, 2] = np.maximum(current_path_flat[:, 2], 0)

                        if task_name == "reconstruct" and len(current_path_flat) < 5:
                            print(f"Task {task_name}, Trajectory {i} is too short (discard).")
                            continue

                        if task_name == "exploration":
                            move_threshold = args.move_dist * global_median_depth
                            # Trim the final exploration point before post-processing.
                            current_path_flat = current_path_flat[:-1]
                        else:
                            move_threshold = args.move_dist * global_median_depth * 1.5
                        c2ws_batch = process_trajectories(
                            [current_path_flat],
                            move_threshold,
                            args.nframe,
                            smoothing=0.2 if task_name == "reconstruct" else 0.5,
                            world_up=np.array([0, 0, 1]),
                            look_at_target=target_center_3d,
                            is_recon=(task_name == "reconstruct")
                        )

                        if len(c2ws_batch) == 0:
                            print(f"Task {task_name}, Trajectory {i} is failed.")
                            c2ws = None
                        else:
                            c2ws = c2ws_batch[0]

                        if task_name == "target" or task_name == "surround":
                            obj_label = seg_labels.get(i, "unknown").strip().replace(" ", "_")
                            folder_name = f"{out_prefix}_{obj_label}_{i}"
                        else:
                            folder_name = f"{out_prefix}_{i}"

                        if c2ws is not None:
                            if task_name == "exploration":
                                origin_paths.append(current_path_flat)
                            elif task_name == "reconstruct":
                                processed_seg_list.append(reconstruct_data[i])
                            else:
                                processed_seg_list.append(seg_data[i])
                            processed_c2ws.append(c2ws)
                            folder_names.append(folder_name)
                            path_vis_list.append(current_path_vis)

                if len(processed_c2ws) == 0:
                    print(f"No valid trajectories found for {task_name}")
                    continue

                # Filter duplicate or low-diversity trajectories.
                with timer.track("Filtering trajectories"):
                    if task_name == "exploration":
                        processed_paths = np.array(processed_c2ws)
                        processed_paths = processed_paths[:, :, :3, -1]
                        filtered_indices = filter_and_select_diverse_trajectories(
                            processed_paths,
                            origin_paths,
                            start_point=np.array([0, 0, 0]),
                            max_k=args.wonder_topk,
                            overlap_ratio=0.5,
                            dist_threshold=0.1,
                            sample_points=50,
                            min_angle_thresh=np.deg2rad(args.min_angle_threshold),
                            min_length_ratio=0.2,
                            length_weight=0.3,
                            diversity_weight=0.7
                        )
                        filtered_indices = filtered_indices[:args.wonder_topk]
                        topk_paths = [origin_paths[i] for i in filtered_indices]
                        folder_names = [folder_names[i] for i in filtered_indices]
                        path_vis_list = [path_vis_list[i] for i in filtered_indices]
                        processed_c2ws = process_trajectories(topk_paths, move_threshold, args.nframe, smoothing=0.5, world_up=np.array([0, 0, 1]))
                    else:
                        processed_paths = np.array(processed_c2ws)
                        path_sim_matrix = compute_trajectory_similarity_matrix(processed_paths, pos_scale=median_depth, rot_scale_deg=20.0, weights=(0.75, 0.25))
                        path_sim_matrix[np.triu_indices(path_sim_matrix.shape[0])] = 0
                        filtered_indices = np.where(np.max(path_sim_matrix, axis=1) < args.traj_sim_threshold)[0]
                        processed_c2ws = [processed_c2ws[i] for i in filtered_indices]
                        folder_names = [folder_names[i] for i in filtered_indices]
                        path_vis_list = [path_vis_list[i] for i in filtered_indices]
                        processed_seg_list = [processed_seg_list[i] for i in filtered_indices]
                        print(f"Task {task_name}: Filtered {processed_paths.shape[0]} -> {len(processed_c2ws)} trajectories")
                        if task_name == "reconstruct" and len(processed_c2ws) > args.recon_topk:
                            print(f"Task {task_name}: Using FPS to select {args.recon_topk} diverse trajectories")
                            processed_seg_list, fps_indices = select_reconstruct_via_fps(processed_seg_list, args.recon_topk)
                            processed_c2ws = [processed_c2ws[i] for i in fps_indices]
                            folder_names = [folder_names[i] for i in fps_indices]
                            path_vis_list = [path_vis_list[i] for i in fps_indices]
                        elif len(processed_c2ws) > args.recon_topk:
                            print(f"Task {task_name}: Filtered {len(processed_c2ws)} -> {args.recon_topk} trajectories with topk filtering")
                            processed_seg_list, sorted_indices = get_topk_seg_data(processed_seg_list, args.recon_topk)
                            processed_c2ws = [processed_c2ws[i] for i in sorted_indices]
                            folder_names = [folder_names[i] for i in sorted_indices]
                            path_vis_list = [path_vis_list[i] for i in sorted_indices]

                assert len(processed_c2ws) == len(path_vis_list) == len(folder_names)
                for c2ws, folder_name, current_path_vis in zip(processed_c2ws, folder_names, path_vis_list):
                    c2ws[0, :3, 3] = 0

                    processed_path = c2ws[:, :3, -1]
                    w2cs = np.linalg.inv(c2ws)

                    K_pano = K.astype(np.float64)
                    K_pano[0, :] /= image_w
                    K_pano[1, :] /= image_h
                    splitted_images = split_panorama_image(np.array(full_img), w2cs[0:1], np.array([K_pano]), h=image_h, w=image_w, interp=cv2.INTER_AREA)

                    out_dir = f"{scene_path}/render_results/{folder_name}/traj0"
                    os.makedirs(out_dir, exist_ok=True)

                    visualize_comparison(current_path_vis, processed_path, idx=i, save_path=f"{out_dir}/traj_vis.png")

                    start_image_pil = Image.fromarray(splitted_images[0])
                    start_image_pil.save(f"{scene_path}/render_results/{folder_name}/start_frame.png")

                    camera_info = {
                        "id": i,
                        "type": task_name,
                        "width": image_w,
                        "height": image_h,
                        "intrinsic": [K.tolist()] * len(w2cs),
                        "extrinsic": w2cs.tolist()
                    }

                    with open(f"{out_dir}/camera.json", "w") as write:
                        json.dump(camera_info, write, indent=2)

                    for c2w in c2ws:
                        add_scene_cam(scene, c2w, CAM_COLORS[trajectory_i % len(CAM_COLORS)], None, image_h * 0.5,
                                      imsize=[image_w, image_h], screen_width=median_depth * 0.15)
                    trajectory_i += 1

                    if task_name == "target":
                        if i < len(seg_data):
                            seg_data[i]['camera_path'] = camera_info
                    elif task_name == "surround":
                        if i < len(surround_data):
                            surround_data[i]['camera_path'] = camera_info
                    elif task_name == "reconstruct":
                        if i < len(reconstruct_data):
                            reconstruct_data[i]['camera_path'] = camera_info
                    elif task_name == "exploration":
                        camera_info["direction"] = wonder_direction_label
                        wonder_camera_data[i] = camera_info

                    # Build optional upward rotation routes.
                    if args.apply_up_route and task_name in ("target", "surround", "exploration"):
                        up_rot_deg = 45
                        obs_decay = 0.65
                        obs_iteration = 0
                        min_distance = 0
                        max_xy_angle = 0
                        distance_threshold = 0.1
                        success_aerial = True
                        with timer.track("Plan aerial pose"):
                            while (min_distance < distance_threshold or max_xy_angle > 75) and obs_iteration < 4:
                                if obs_iteration > 0:
                                    if min_distance < distance_threshold:
                                        rank0_log(f"Obstruction is detected in aerial routes min distance: {min_distance} reduce the rot_deg {up_rot_deg}->{up_rot_deg * obs_decay}")
                                    elif max_xy_angle > 75:
                                        rank0_log(f"Abnormal is detected in aerial routes max xy angle: {max_xy_angle} reduce the rot_deg {up_rot_deg}->{up_rot_deg * obs_decay}")
                                    up_rot_deg = up_rot_deg * obs_decay

                                rise_move = {"type": "normal", "backward-forward": 0, "left-right": 0, "rotation": [0, -up_rot_deg], "name": "up-rotation"}
                                c2w0 = np.linalg.inv(w2cs[0])

                                c2ws_next, obs_iteration_inner = get_c2w(c2w0, rise_move, median_depth, air_bound=median_depth * 0.5, n_inter=args.nframe // 2,
                                                                         kdtree=kdtree, distance_threshold=distance_threshold, local_rank=0, obs_decay=obs_decay)
                                up_rot_deg = up_rot_deg * (obs_decay ** obs_iteration_inner)

                                if up_rot_deg < 15:
                                    rank0_log(f"Too many collisions are detected in the whole aerial routes, ignore...", "WARNING")
                                    success_aerial = False
                                    break

                                c2ws_rise = np.concatenate([c2w0[None], c2ws_next], axis=0)
                                c2ws_ = np.linalg.inv(w2cs)
                                offsets = c2ws_[1:, :3, 3] - c2ws_[0:1, :3, 3]
                                c2w_Rs = c2ws_[1:, :3, :3]
                                c2w_rise_next = np.eye(4, dtype=np.float32)
                                c2w_rise_next[:3, 3] = c2ws_rise[-1, :3, 3]
                                c2w_rise_next = np.tile(c2w_rise_next[None], [offsets.shape[0], 1, 1])  # [N-1,4,4]
                                c2w_rise_next[:, :3, 3] += offsets
                                rise_R = np.array([[1, 0, 0],
                                                   [0, np.cos(-np.deg2rad(up_rot_deg)), -np.sin(-np.deg2rad(up_rot_deg))],
                                                   [0, np.sin(-np.deg2rad(up_rot_deg)), np.cos(-np.deg2rad(up_rot_deg))]], dtype=np.float32)
                                c2w_rise_next[:, :3, :3] = c2w_Rs @ rise_R[None]
                                c2ws_rise = np.concatenate([c2ws_rise, c2w_rise_next], axis=0)

                                if c2ws_rise.shape[0] < args.nframe - 1:
                                    c2ws_ = interpolate_poses(c2ws_rise, args.nframe - 1)
                                else:
                                    c2ws_ = c2ws_rise.copy()
                                query_points = c2ws_[:, :3, 3]
                                distances, _ = kdtree.query(query_points, k=5)
                                distances = distances.mean(axis=1)
                                min_distance = distances.min()
                                c2ws_R = c2ws_[:, :3, :3]
                                xy_angles = compute_lookat_xy_angle(c2ws_R)
                                max_xy_angle = xy_angles.max()
                                obs_iteration += 1

                        if not success_aerial:
                            continue

                        c2ws_rise = interpolate_poses(c2ws_rise[1:], M=args.nframe - 1)
                        c2ws_rise = np.concatenate([c2w0[None], c2ws_rise], axis=0)
                        w2cs_rise = np.linalg.inv(c2ws_rise)

                        camera_info_rise = {
                            "id": i,
                            "type": task_name,
                            "width": image_w,
                            "height": image_h,
                            "intrinsic": [K.tolist()] * len(w2cs_rise),
                            "extrinsic": w2cs_rise.tolist(),
                            "rotation_deg": float(up_rot_deg),
                            "max_xy_angle": float(max_xy_angle),
                        }

                        os.makedirs(f"{scene_path}/render_results/{folder_name}/traj1", exist_ok=True)
                        with open(f"{scene_path}/render_results/{folder_name}/traj1/camera.json", "w") as write:
                            json.dump(camera_info_rise, write, indent=2)

                        for c2w in c2ws_rise:
                            add_scene_cam(scene, c2w, CAM_COLORS[trajectory_i % len(CAM_COLORS)], None, image_h * 0.5,
                                          imsize=[image_w, image_h], screen_width=median_depth * 0.15)
                        trajectory_i += 1

                    # Build optional reconstruct-aware eloop iteration.
                    if args.apply_recon_iteration and task_name == "reconstruct":
                        with timer.track("Plan eloop iteration"):
                            eloop_move = {"type": "eloop", "radius_x": args.eloop_dist, "radius_y": args.eloop_dist}
                            c2w0 = c2ws[-1].copy()
                            w2c0 = np.linalg.inv(c2w0)
                            # Render depth from the target view to estimate local scale.
                            _, guided_depth = point_rendering(K=torch.from_numpy(K[None]).to(device=device, dtype=torch.float32),
                                                              w2cs=torch.from_numpy(w2c0[None]).to(device=device, dtype=torch.float32),
                                                              points=global_pcd.vertices, colors=torch.zeros((global_pcd.vertices.shape[0], 3), device=device, dtype=torch.float32),
                                                              h=image_h, w=image_w, render_radius=0.008, points_per_pixel=8,
                                                              device=device, background_color=[0, 0, 0], return_depth=True)
                            guided_depth = guided_depth[0, 0]
                            guided_depth[guided_depth == -1] = 0
                            median_depth = torch.median(guided_depth[guided_depth > 0]).item()

                            c2ws_eloop, obs_iteration = get_c2w(
                                c2w0.copy(), eloop_move, median_depth,
                                air_bound=median_depth * 0.5,
                                n_inter=args.nframe - 1,
                                kdtree=kdtree,
                                mesh=mesh,
                                distance_threshold=args.distance_threshold,
                                local_rank=0,
                                obs_decay=args.obs_decay,
                                obs_limit=args.obs_iteration_limit
                            )

                            c2ws_eloop = interpolate_poses(c2ws_eloop[1:], M=args.nframe - 1)
                            c2ws_eloop = np.concatenate([c2w0[None], c2ws_eloop], axis=0)
                            w2cs_eloop = np.linalg.inv(c2ws_eloop)

                            camera_info_eloop = {
                                "id": i,
                                "type": task_name,
                                "width": image_w,
                                "height": image_h,
                                "intrinsic": [K.tolist()] * len(w2cs_eloop),
                                "extrinsic": w2cs_eloop.tolist(),
                                "eloop_rotation": 0.5 * ((args.obs_decay) ** obs_iteration)
                            }

                            os.makedirs(f"{scene_path}/render_results/{folder_name}/traj1", exist_ok=True)
                            with open(f"{scene_path}/render_results/{folder_name}/traj1/camera.json", "w") as write:
                                json.dump(camera_info_eloop, write, indent=2)

                            for c2w in c2ws_eloop:
                                add_scene_cam(scene, c2w, CAM_COLORS[trajectory_i % len(CAM_COLORS)], None, image_h * 0.5,
                                              imsize=[image_w, image_h], screen_width=median_depth * 0.15)
                            trajectory_i += 1

            # Save camera visualization.
            point = trimesh.PointCloud(vertices=np.array([0, 0, 0]).reshape(1, 3),
                                       colors=np.array([255, 255, 255]).reshape(1, 3))
            scene.add_geometry(point)
            scene.export(f"{scene_path}/render_results/cameras_navi.glb")

        timer.summary()
