import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import trimesh
import utils3d
from PIL import Image
from moge.model.v2 import MoGeModel
from tqdm import tqdm

from src.general_utils import rank0_log, Timer, load_video, save_16bit_png_depth
from src.panorama_utils import split_panorama_image, split_panorama_depth, rotate_around_z_axis

timer = Timer()

def save_io(fname, frame_img, depth_path, normal_map, w2c, K, output_path):
    img_path = f"{output_path}/images/{fname}.png"
    depth_target_path = f"{output_path}/depths/{fname}.png"
    normal_path = f"{output_path}/normals/{fname}.png"

    # Save the RGB frame.
    frame_img.save(img_path)
    if depth_path is not None:
        shutil.copy(depth_path, depth_target_path)

    if normal_map is not None:
        Image.fromarray(((normal_map + 1.0) / 2.0 * 255.0).astype(np.uint8)).save(normal_path)

    return fname, w2c.tolist(), K.tolist()


def gather_and_merge_cameras(local_cameras: dict, world_size: int, rank: int) -> dict:
    """
    Gather camera dictionaries from all ranks and merge them on rank 0.

    Args:
        local_cameras: Camera dictionary for the current rank.
        world_size: Total process count.
        rank: Current process rank.

    Returns:
        Merged camera dictionary. Only rank 0 receives the merged result.
    """
    gather_list = [None] * world_size
    dist.all_gather_object(gather_list, local_cameras)

    if rank == 0:
        merged_cameras = {}
        for cameras in gather_list:
            merged_cameras.update(cameras)
        return merged_cameras
    return {}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', '-r', type=str, required=True, help='Path to the root folder containing pano and aligned folders')
    parser.add_argument('--out_name', '-o', type=str, default='gs_data', help='Output folder name under gs_data')
    parser.add_argument('--custom_out_name', type=str, default=None, help='Output folder name under gs_data')
    parser.add_argument('--result_name', '-n', type=str, default='worldstereo-memory-dmd', help='Result video name')
    parser.add_argument('--skip_start_frame', '-s', action='store_true', help='Whether to skip the first frame in the video')
    parser.add_argument('--no_aerial', action='store_true', help='Whether to skip aerial trajectories')
    parser.add_argument('--interval', '-i', type=int, default=1, help='Frame interval for processing video')
    parser.add_argument('--pano_name', '-p', type=str, default='panorama', help='Panorama image name without extension')
    parser.add_argument('--save_normal', action='store_true', help='Whether to generate and save normal maps')
    parser.add_argument('--high_res', action='store_true', help='Whether to use high resolution pano')
    parser.add_argument('--split_sky', action='store_true', help='Save sky pointcloud as a separate sky_points.ply (sky still participates in downsampling)')
    parser.add_argument('--split_align', action='store_true', help='Save aligned_pcd as a separate align_points.ply (for align protection during training)')
    parser.add_argument('--regen_pano_polar', action='store_true', help='Force regenerate panorama/polar images from full panorama instead of copying from pano_bank/polar_bank')
    parser.add_argument('--scene_name', type=str, default=None, help='Only process the specified scene under root_path (e.g. "scene_001")')
    parser.add_argument('--pano_density_mult', type=int, default=2, help='Panorama azimuth density multiplier (1=original 27, 4=108 views)')
    parser.add_argument('--polar_up_density_mult', type=int, default=4, help='Polar upper azimuth density multiplier (1=original 8, up to 8=64 views)')
    parser.add_argument('--polar_down_density_mult', type=int, default=1, help='Polar lower azimuth density multiplier (1=original 8, 2=16 views)')

    args = parser.parse_args()
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
    )

    fov_x, fov_y = 120, 90

    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    moge_model.eval()

    if args.scene_name:
        scene_list = [os.path.join(args.root_path, args.scene_name)]
    elif os.path.exists(f"{args.root_path}/render_results"):
        scene_list = [args.root_path]
    else:
        scene_list = glob(f"{args.root_path}/*")
    scene_list.sort()

    for scene_path in tqdm(scene_list, disable=rank != 0):
        result_name = args.result_name

        required_files = [
            f"{scene_path}/render_results/generation_bank_{result_name}/global_pcd.ply",
            f"{scene_path}/render_results/generation_bank_{result_name}/aligned_pcd.ply",
        ]
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            rank0_log(f"Skipping {scene_path}: missing files {missing}")
            dist.barrier()
            continue

        pano_image_path = f"{scene_path}/{args.pano_name}.png"
        if not os.path.exists(pano_image_path):
            pano_image_path = f"{scene_path}/{args.pano_name.replace('_sr', '')}.png"

        if args.custom_out_name is not None:
            output_path = args.custom_out_name
        else:
            output_path = f"{scene_path}/{args.out_name}"

        pano_bank_path = f"{scene_path}/render_results/pano_bank"
        polar_bank_path = f"{scene_path}/render_results/polar_bank"
        dense_pano = args.pano_density_mult > 1
        dense_polar = args.polar_up_density_mult > 1 or args.polar_down_density_mult > 1
        need_regen = args.regen_pano_polar or dense_pano or dense_polar or not os.path.exists(pano_bank_path) or not os.path.exists(polar_bank_path)

        if args.high_res or need_regen:
            full_img = Image.open(pano_image_path)
            if args.high_res and full_img.size[1] > 1920:
                full_img = full_img.resize((3840, 1920), resample=Image.Resampling.BICUBIC)
            depth_pt_path = f"{scene_path}/render_results/full_depth_prediction.pt"
            sky_mask_path = f"{scene_path}/render_results/sky_mask.png"
            if os.path.exists(depth_pt_path) and os.path.exists(sky_mask_path):
                full_depth = torch.load(depth_pt_path, weights_only=False)
                sky_mask = np.array(Image.open(sky_mask_path)) / 255
                sky_mask = ~torch.from_numpy(sky_mask).bool()
                edge_mask = torch.from_numpy(utils3d.numpy.depth_edge(full_depth["distance"].cpu().numpy(), rtol=0.1)).bool()
                full_mask = (sky_mask | edge_mask).to(device)
            else:
                rank0_log(f"Warning: full_depth_prediction.pt or sky_mask.png not found, depth/mask will be skipped for pano/polar regen")
                full_depth, sky_mask, edge_mask, full_mask = None, None, None, None
        else:
            full_img, full_depth, sky_mask, edge_mask, full_mask = None, None, None, None, None

        if os.path.exists(output_path) and rank == 0:
            shutil.rmtree(output_path, ignore_errors=True)
            if os.path.exists(output_path):
                import time
                time.sleep(2)
                shutil.rmtree(output_path, ignore_errors=True)
        if rank == 0:
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(f"{output_path}/images", exist_ok=True)
            os.makedirs(f"{output_path}/depths", exist_ok=True)
            if args.save_normal:
                os.makedirs(f"{output_path}/normals", exist_ok=True)
        dist.barrier()


        if rank == 0:
            with timer.track("[IO] Loading/Downsampling pointcloud"):
                global_pcd = trimesh.load(f"{scene_path}/render_results/generation_bank_{result_name}/global_pcd.ply")
                global_points = global_pcd.vertices
                global_rgbs = global_pcd.colors

                if global_points.shape[0] > 3_000_000:
                    rank0_log(f"Downsampling global pointcloud from {global_points.shape[0]} to 3_000_000...")
                    rdv_indices = np.random.choice(global_points.shape[0], 3_000_000, replace=False)
                    global_points = global_points[rdv_indices]
                    global_rgbs = global_rgbs[rdv_indices]

                extra_pcd = trimesh.load(f"{scene_path}/render_results/generation_bank_{result_name}/aligned_pcd.ply")
                extra_points = extra_pcd.vertices
                extra_rgbs = extra_pcd.colors

                sky_pcd = None

                if os.path.exists(f"{scene_path}/render_results/generation_bank_{result_name}/sky_pcd.ply"):
                    sky_pcd = trimesh.load(f"{scene_path}/render_results/generation_bank_{result_name}/sky_pcd.ply")
                    if hasattr(sky_pcd, "vertices"):
                        if sky_pcd.vertices.shape[0] > 300_000:
                            rank0_log(f"Downsampling sky pointcloud from {sky_pcd.vertices.shape[0]} to 300_000...")
                            rdv_indices = np.random.choice(sky_pcd.vertices.shape[0], 300_000, replace=False)
                            sky_pcd.vertices = sky_pcd.vertices[rdv_indices]
                            sky_pcd.colors = sky_pcd.colors[rdv_indices]
                    else:
                        rank0_log("No sky pointcloud is detected!")

                if args.split_align:
                    rank0_log(f"Keeping {extra_points.shape[0]} align points separate (--split_align)")
                else:
                    global_points = np.concatenate([extra_points, global_points])
                    global_rgbs = np.concatenate([extra_rgbs, global_rgbs])

        if rank == 0:
            with timer.track("[IO] Resaving processed pointcloud"):
                rank0_log("Saving pointclouds...")

                if sky_pcd is not None and hasattr(sky_pcd, "vertices"):
                    if args.split_sky:
                        _ = sky_pcd.export(f"{output_path}/sky_points.ply")
                        rank0_log(f"Saved {sky_pcd.vertices.shape[0]} sky points to sky_points.ply")
                    else:
                        global_points = np.concatenate([sky_pcd.vertices, global_points], axis=0)
                        global_rgbs = np.concatenate([sky_pcd.colors, global_rgbs], axis=0)
                _ = trimesh.PointCloud(vertices=global_points, colors=global_rgbs).export(f"{output_path}/points.ply")
                rank0_log(f"Saved {global_points.shape[0]} points to points.ply")

                if args.split_align:
                    align_pcd = trimesh.PointCloud(vertices=extra_points, colors=extra_rgbs)
                    _ = align_pcd.export(f"{output_path}/align_points.ply")
                    rank0_log(f"Saved {extra_points.shape[0]} align points to align_points.ply")

                rank0_log("Saving over...")

        dist.barrier()

        # Each rank keeps only the cameras it writes locally.
        save_cameras = {}
        img_width, img_height = None, None
        to_tensor = transforms.ToTensor()

        with timer.track("[IO] Search all video paths"):
            if args.no_aerial:
                    video_paths = (glob(f"{scene_path}/render_results/view*/*/{result_name}_result.mp4") +
                               glob(f"{scene_path}/render_results/target*/traj0/{result_name}_result.mp4") +
                               glob(f"{scene_path}/render_results/reconstruct*/traj0/{result_name}_result.mp4") +
                               glob(f"{scene_path}/render_results/wonder*/traj0/{result_name}_result.mp4"))
            else:
                video_paths = (glob(f"{scene_path}/render_results/view*/*/{result_name}_result.mp4") +
                               glob(f"{scene_path}/render_results/target*/*/{result_name}_result.mp4") +
                               glob(f"{scene_path}/render_results/reconstruct*/*/{result_name}_result.mp4") +
                               glob(f"{scene_path}/render_results/wonder*/*/{result_name}_result.mp4"))

        video_paths = sorted(video_paths, key=lambda x: (x.split("/")[-3], x.split("/")[-2]))
        video_paths = video_paths[rank::world_size]

        with timer.track("[IO] Loading & Processing video frames"):
            for video_path in tqdm(video_paths, desc="Loading & Processing video...", disable=rank != 0):
                view_id, traj_id = video_path.split("/")[-3], video_path.split("/")[-2]
                video_dir = os.path.dirname(video_path)
                view_dir = os.path.dirname(video_dir)
                frames = load_video(video_path)

                if img_width is None:
                    img_width, img_height = frames[0].size

                with open(os.path.join(video_dir, "camera.json"), "r") as f:
                    render_camera = json.load(f)
                w2cs = np.array(render_camera["extrinsic"])
                Ks = np.array(render_camera["intrinsic"])

                if w2cs.shape[0] != len(frames):
                    if w2cs.shape[0] == 81 and len(frames) == 21:
                        rank0_log(f"Downsampling cameras from {w2cs.shape[0]} to {len(frames)}")
                        w2cs = w2cs[0::4]
                        Ks = Ks[0::4]
                    else:
                        raise ValueError(f"Frame/camera mismatch: {len(frames)} vs {w2cs.shape[0]}")

                indices = np.arange(len(frames))
                frames = frames[::args.interval]
                w2cs = w2cs[::args.interval]
                Ks = Ks[::args.interval]
                indices = indices[::args.interval]

                replace_start_frame = not args.skip_start_frame and not (view_id.startswith("reconstruct_") and traj_id == "traj1")
                if replace_start_frame:
                    start_frame = Image.open(os.path.join(view_dir, "start_frame.png"))
                    frames[0] = start_frame

                with ThreadPoolExecutor(max_workers=12) as executor:
                    futures = []
                    for i in tqdm(range(len(frames)), disable=rank != 0):
                        if frames[i].size[0] != img_width or frames[i].size[1] != img_height:
                            frames[i] = frames[i].resize((img_width, img_height), Image.LANCZOS)

                        memory_bank_path = f"{scene_path}/render_results/generation_bank_{result_name}/{view_id}/{traj_id}"
                        depth_path = f"{memory_bank_path}/depths/{indices[i]:04d}.png"
                        if not os.path.exists(depth_path):
                            depth_path = None

                        if args.save_normal:
                            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                                moge_prediction = moge_model.infer(to_tensor(frames[i])[None].to(device))
                                normal_map = moge_prediction["normal"][0].cpu().numpy()  # [H,W,3], from -1 to 1
                        else:
                            normal_map = None

                        fname = f"{view_id}-{traj_id}_{indices[i]:06d}"

                        fut = executor.submit(save_io, fname, frames[i], depth_path, normal_map, w2cs[i], Ks[i], output_path)
                        futures.append(fut)

                    for fut in as_completed(futures):
                        fname, extrinsic, intrinsic = fut.result()
                        save_cameras[fname] = {
                            "extrinsic": extrinsic,
                            "intrinsic": intrinsic
                        }
        rank0_log(f"Saving video frames over..., width: {img_width}, height: {img_height}")

        # Saving pano images
        rank0_log("Processing Panorama images...")
        regen_pano = args.regen_pano_polar or dense_pano or not os.path.exists(pano_bank_path)
        with timer.track("[IO] Processing Panorama images"):
            if args.high_res or regen_pano:
                out_h = img_height * 2 if args.high_res else img_height
                out_w = img_width * 2 if args.high_res else img_width
                pano_mult = args.pano_density_mult
                pano_rot_deg = 40.0 / pano_mult
                pano_n_az = int(360 / pano_rot_deg)
                pano_layer_starts = [
                    np.array([-1, 0, 0], dtype=np.float32),
                    np.array([-1, 0, 0.5], dtype=np.float32),
                    np.array([-1, 0, -0.5], dtype=np.float32),
                ]
                direct_points = []
                pano_view_meta = []
                for layer_idx, start_point in enumerate(pano_layer_starts):
                    for az_idx in range(pano_n_az):
                        if az_idx == 0:
                            direct_points.append(start_point)
                        else:
                            direct_points.append(rotate_around_z_axis(start_point.reshape(1, 3), pano_rot_deg * az_idx)[0])
                        pano_view_meta.append((layer_idx, az_idx))
                direct_points = np.stack(direct_points, axis=0)
                rank0_log(f"Panorama: {len(pano_layer_starts)} layers x {pano_n_az} azimuths (rot_deg={pano_rot_deg}) = {len(direct_points)} views")

                intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(fov_x), fov_y=np.deg2rad(fov_y))
                splitted_intrinsics = [intrinsics] * len(direct_points)
                splitted_extrinsics = utils3d.numpy.extrinsics_look_at(np.array([0, 0, 0]), direct_points, np.array([0, 0, 1])).astype(np.float32)

                splitted_images = split_panorama_image(np.array(full_img), splitted_extrinsics, splitted_intrinsics, h=out_h, w=out_w, interp=cv2.INTER_LINEAR)
                if full_depth is not None and full_mask is not None:
                    splitted_depths = split_panorama_depth(np.array(full_depth["distance"].cpu()), splitted_extrinsics, splitted_intrinsics, h=out_h, w=out_w, distance_to_depth=True)
                    splitted_masks = split_panorama_depth(~np.array(full_mask.cpu()), splitted_extrinsics, splitted_intrinsics, h=out_h, w=out_w)
                else:
                    splitted_depths = None
                    splitted_masks = None
                rank_indices = np.arange(len(splitted_images))[rank::world_size]

                for i in tqdm(rank_indices, disable=rank != 0):
                    layer_idx, az_idx = pano_view_meta[i]
                    fname = f"panorama_L{layer_idx:02d}_A{az_idx:04d}"

                    splitted_image = Image.fromarray(splitted_images[i])
                    K = splitted_intrinsics[i].copy()
                    K[0] *= out_w
                    K[1] *= out_h
                    save_cameras[fname] = {
                        "intrinsic": K.tolist(),
                        "extrinsic": splitted_extrinsics[i].tolist(),
                        "source_type": "panorama",
                        "layer_idx": layer_idx,
                        "azimuth_idx": az_idx,
                    }

                    splitted_image.save(f"{output_path}/images/{fname}.png")
                    if splitted_depths is not None:
                        depth = splitted_depths[i]
                        depth_mask = splitted_masks[i].bool()
                        depth[~depth_mask] = 0
                        depth = depth[0]
                        save_16bit_png_depth(depth, f"{output_path}/depths/{fname}.png")

                    if args.save_normal:
                        frame = splitted_image
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                            moge_prediction = moge_model.infer(to_tensor(frame)[None].to(device))
                            normal_map = moge_prediction["normal"][0].cpu().numpy()
                            Image.fromarray(((normal_map + 1.0) / 2.0 * 255.0).astype(np.uint8)).save(f"{output_path}/normals/{fname}.png")
            else:
                splitted_pano_list = glob(f"{pano_bank_path}/images/*.png")
                splitted_pano_list.sort()
                splitted_pano_list = splitted_pano_list[rank::world_size]
                with open(f"{pano_bank_path}/cameras.json", "r") as f:
                    pano_cameras = json.load(f)
                for i in tqdm(range(len(splitted_pano_list)), disable=rank != 0):
                    orig_fname = splitted_pano_list[i].split('/')[-1].split('.')[0]
                    fname = f"panorama_{orig_fname}"

                    save_cameras[fname] = {
                        "extrinsic": pano_cameras[orig_fname]['extrinsic'],
                        "intrinsic": pano_cameras[orig_fname]['intrinsic']
                    }
                    shutil.copy(splitted_pano_list[i], f"{output_path}/images/{fname}.png")
                    depth_path = f"{pano_bank_path}/depths/{orig_fname}.png"
                    shutil.copy(depth_path, f"{output_path}/depths/{fname}.png")

                    if args.save_normal:
                        frame = Image.open(splitted_pano_list[i])
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                            moge_prediction = moge_model.infer(to_tensor(frame)[None].to(device))
                            normal_map = moge_prediction["normal"][0].cpu().numpy()
                            Image.fromarray(((normal_map + 1.0) / 2.0 * 255.0).astype(np.uint8)).save(f"{output_path}/normals/{fname}.png")

        rank0_log("Saving Panorama frames over...")

        # Save polar images split into upper (sky) and lower (ground) views.
        rank0_log("Processing Polar images...")
        regen_polar = args.regen_pano_polar or dense_polar or not os.path.exists(polar_bank_path)
        with timer.track("[IO] Processing Polar images"):
            if args.high_res or regen_polar:
                out_h = img_height * 2 if args.high_res else img_height
                out_w = img_width * 2 if args.high_res else img_width
                intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(fov_x), fov_y=np.deg2rad(fov_y))

                # --- Polar Upper (sky) ---
                up_mult = args.polar_up_density_mult
                up_rot_deg = 90.0 / up_mult if up_mult <= 2 else 90.0 / (up_mult // 2)
                up_n_az = int(360 / up_rot_deg)
                use_extra_up_layers = (up_mult > 2)
                polar_up_base = [
                    np.array([-1, 0, 1.0], dtype=np.float32),    # 45 deg
                    np.array([0.1, 0, 1.0], dtype=np.float32),   # ~84 deg
                ]
                polar_up_extra = [
                    np.array([-1, 0, 2.14], dtype=np.float32),   # ~65 deg
                    np.array([-1, 0, 4.70], dtype=np.float32),   # ~78 deg
                ]
                if use_extra_up_layers:
                    polar_up_layers = [polar_up_base[0], polar_up_extra[0], polar_up_extra[1], polar_up_base[1]]
                else:
                    polar_up_layers = polar_up_base

                up_direct_points = []
                up_view_meta = []
                for layer_idx, start_point in enumerate(polar_up_layers):
                    for az_idx in range(up_n_az):
                        if az_idx == 0:
                            up_direct_points.append(start_point)
                        else:
                            up_direct_points.append(rotate_around_z_axis(start_point.reshape(1, 3), up_rot_deg * az_idx)[0])
                        up_view_meta.append(("polar_up", layer_idx, az_idx))
                rank0_log(f"Polar upper: {len(polar_up_layers)} layers x {up_n_az} azimuths (rot_deg={up_rot_deg}) = {len(up_direct_points)} views")

                # --- Polar Lower (ground) ---
                down_mult = args.polar_down_density_mult
                down_rot_deg = 90.0 / down_mult
                down_n_az = int(360 / down_rot_deg)
                polar_down_layers = [
                    np.array([-1, 0, -1.0], dtype=np.float32),   # -45 deg
                    np.array([0.1, 0, -1.0], dtype=np.float32),  # ~-84 deg
                ]

                down_direct_points = []
                down_view_meta = []
                for layer_idx, start_point in enumerate(polar_down_layers):
                    for az_idx in range(down_n_az):
                        if az_idx == 0:
                            down_direct_points.append(start_point)
                        else:
                            down_direct_points.append(rotate_around_z_axis(start_point.reshape(1, 3), down_rot_deg * az_idx)[0])
                        down_view_meta.append(("polar_down", layer_idx, az_idx))
                rank0_log(f"Polar lower: {len(polar_down_layers)} layers x {down_n_az} azimuths (rot_deg={down_rot_deg}) = {len(down_direct_points)} views")

                all_polar_points = up_direct_points + down_direct_points
                all_polar_meta = up_view_meta + down_view_meta
                direct_points = np.stack(all_polar_points, axis=0)
                splitted_intrinsics = [intrinsics] * len(direct_points)
                splitted_extrinsics = utils3d.numpy.extrinsics_look_at(np.array([0, 0, 0]), direct_points, np.array([0, 0, 1])).astype(np.float32)

                splitted_images = split_panorama_image(np.array(full_img), splitted_extrinsics, splitted_intrinsics, h=out_h, w=out_w, interp=cv2.INTER_LINEAR)
                if full_depth is not None and full_mask is not None:
                    splitted_depths = split_panorama_depth(np.array(full_depth["distance"].cpu()), splitted_extrinsics, splitted_intrinsics, h=out_h, w=out_w, distance_to_depth=True)
                    splitted_masks = split_panorama_depth(~np.array(full_mask.cpu()), splitted_extrinsics, splitted_intrinsics, h=out_h, w=out_w)
                else:
                    splitted_depths = None
                    splitted_masks = None
                rank_indices = np.arange(len(splitted_images))[rank::world_size]

                for i in tqdm(rank_indices, disable=rank != 0):
                    source_type, layer_idx, az_idx = all_polar_meta[i]
                    fname = f"{source_type}_L{layer_idx:02d}_A{az_idx:04d}"

                    splitted_image = Image.fromarray(splitted_images[i])
                    K = splitted_intrinsics[i].copy()
                    K[0] *= out_w
                    K[1] *= out_h
                    save_cameras[fname] = {
                        "intrinsic": K.tolist(),
                        "extrinsic": splitted_extrinsics[i].tolist(),
                        "source_type": source_type,
                        "layer_idx": layer_idx,
                        "azimuth_idx": az_idx,
                    }

                    splitted_image.save(f"{output_path}/images/{fname}.png")
                    if splitted_depths is not None:
                        depth = splitted_depths[i]
                        depth_mask = splitted_masks[i].bool()
                        depth[~depth_mask] = 0
                        depth = depth[0]
                        save_16bit_png_depth(depth, f"{output_path}/depths/{fname}.png")

                    if args.save_normal:
                        frame = splitted_image
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                            moge_prediction = moge_model.infer(to_tensor(frame)[None].to(device))
                            normal_map = moge_prediction["normal"][0].cpu().numpy()
                            Image.fromarray(((normal_map + 1.0) / 2.0 * 255.0).astype(np.uint8)).save(f"{output_path}/normals/{fname}.png")
            else:
                splitted_pano_list = glob(f"{polar_bank_path}/images/*.png")
                splitted_pano_list.sort()
                splitted_pano_list = splitted_pano_list[rank::world_size]
                with open(f"{polar_bank_path}/cameras.json", "r") as f:
                    pano_cameras = json.load(f)

                for i in tqdm(range(len(splitted_pano_list)), disable=rank != 0):
                    orig_fname = splitted_pano_list[i].split('/')[-1].split('.')[0]
                    fname = f"polar_{orig_fname}"

                    save_cameras[fname] = {
                        "extrinsic": pano_cameras[orig_fname]['extrinsic'],
                        "intrinsic": pano_cameras[orig_fname]['intrinsic']
                    }
                    shutil.copy(splitted_pano_list[i], f"{output_path}/images/{fname}.png")
                    depth_path = f"{polar_bank_path}/depths/{orig_fname}.png"
                    shutil.copy(depth_path, f"{output_path}/depths/{fname}.png")

        rank0_log("Saving Polar frames over...")

        # Gather all per-rank camera metadata and write it from rank 0.
        with timer.track("[Gather] Gathering cameras from all ranks"):
            all_sizes = [None] * world_size
            dist.all_gather_object(all_sizes, (img_width, img_height))
            final_width, final_height = None, None
            for w, h in all_sizes:
                if w is not None:
                    final_width, final_height = w, h
                    break

            merged_cameras = gather_and_merge_cameras(save_cameras, world_size, rank)

            if rank == 0:
                merged_cameras["width"] = final_width
                merged_cameras["height"] = final_height
                with open(f"{output_path}/cameras.json", "w") as w:
                    json.dump(merged_cameras, w, indent=2)
                rank0_log(f"Saved {len(merged_cameras) - 2} cameras to cameras.json")  # -2 for width/height

        dist.barrier()

        if rank == 0:
            shutil.copy(f"{scene_path}/meta_info.json", f"{output_path}/meta_info.json")

        if rank == 0:
            timer.summary()

    dist.destroy_process_group()
