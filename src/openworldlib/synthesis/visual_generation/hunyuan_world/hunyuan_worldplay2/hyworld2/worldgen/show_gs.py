import argparse
import json
import math
import os
import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import viser
from gsplat.rendering import rasterization
from gs.utils import depth_to_normal
from plyfile import PlyData
from tqdm import tqdm

import nerfview
from nerfview import apply_float_colormap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--none_sh_degree", action='store_true')
    parser.add_argument("--ckpt", type=str, default=None, help="path to the .pt file")
    parser.add_argument("--port", type=int, default=443, help="port for the viewer server")
    parser.add_argument("--backend", type=str, default="gsplat", choices=["gsplat", "gsplat_legacy", "inria"])
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"
    if args.ckpt is None:
        raise ValueError("--ckpt is required")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    torch.manual_seed(42)
    device = "cuda"
    up_direction = None
    facing_direction = None
    center_point = None

    if "rank*" in args.ckpt or args.ckpt.endswith(".pt"):
        ckpt_paths = sorted(glob(args.ckpt)) if "rank*" in args.ckpt else [args.ckpt]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoints matched: {args.ckpt}")

        splat_chunks = {"means": [], "quats": [], "scales": [], "opacities": [], "sh0": [], "shN": []}
        for ckpt_path in tqdm(ckpt_paths, desc="Loading checkpoints..."):
            ckpt_all = torch.load(ckpt_path, map_location=device, weights_only=False)
            up_direction = ckpt_all.get("up_direction", up_direction)
            facing_direction = ckpt_all.get("facing_direction", facing_direction)
            center_point = ckpt_all.get("center_point", center_point)

            ckpt = ckpt_all["splats"]
            splat_chunks["means"].append(ckpt["means"])
            splat_chunks["quats"].append(F.normalize(ckpt["quats"], p=2, dim=-1))
            splat_chunks["scales"].append(torch.exp(ckpt["scales"]))
            splat_chunks["opacities"].append(torch.sigmoid(ckpt["opacities"]))
            splat_chunks["sh0"].append(ckpt["sh0"])
            splat_chunks["shN"].append(ckpt["shN"])

        means = torch.cat(splat_chunks["means"], dim=0)
        quats = torch.cat(splat_chunks["quats"], dim=0)
        scales = torch.cat(splat_chunks["scales"], dim=0)
        opacities = torch.cat(splat_chunks["opacities"], dim=0)
        sh0 = torch.cat(splat_chunks["sh0"], dim=0)
        shN = torch.cat(splat_chunks["shN"], dim=0)
    elif args.ckpt.endswith(".ply"):
        with open(os.path.join(os.path.dirname(args.ckpt), "position_meta_info.json"), "r") as f:
            meta_info = json.load(f)
        up_direction = np.array(meta_info["up_direction"])
        facing_direction = np.array(meta_info["facing_direction"])
        center_point = np.array(meta_info["center_point"])

        print(f"[load_ply] Reading {args.ckpt} ...")
        plydata = PlyData.read(args.ckpt)
        vertex = plydata['vertex']
        n_points = len(vertex.data)
        print(f"[load_ply] Number of points: {n_points}")

        # 1. Positions: means (N, 3).
        means = torch.tensor(np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1), dtype=torch.float32, device=device)

        # 2. Quaternions: quats (N, 4), stored in PLY as rot_0~rot_3 (wxyz).
        quats = torch.tensor(np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=-1), dtype=torch.float32, device=device)
        quats = F.normalize(quats, p=2, dim=-1)

        # 3. Scales: scales (N, 3), stored in log space.
        scales = torch.tensor(np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1), dtype=torch.float32, device=device)
        scales = torch.exp(scales)

        # 4. Opacities: opacities (N,), stored as logits before sigmoid.
        opacities = torch.tensor(np.array(vertex['opacity']), dtype=torch.float32, device=device)
        opacities = torch.sigmoid(opacities)

        # 5. SH coefficients: DC (sh0) + remaining bands (shN).
        # sh0: f_dc_0, f_dc_1, f_dc_2 -> (N, 1, 3).
        sh_dc = torch.tensor(np.stack([
            vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']
        ], axis=-1), dtype=torch.float32, device=device)  # (N, 3)
        sh0 = sh_dc.unsqueeze(1)  # (N, 1, 3)

        # shN: f_rest_* -> (N, C, 3).
        rest_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith('f_rest_')],
            key=lambda x: int(x.split('_')[-1])
        )

        if rest_names:
            sh_rest_flat = torch.tensor(np.stack(
                [vertex[name] for name in rest_names], axis=-1
            ), dtype=torch.float32, device=device)  # (N, num_rest)

            # Standard 3DGS stores f_rest flattened as (C * 3); reshape to (N, C, 3).
            num_rest_coeffs = len(rest_names)
            assert num_rest_coeffs % 3 == 0, \
                f"Invalid PLY: f_rest count {num_rest_coeffs} is not divisible by 3."
            num_sh_rest = num_rest_coeffs // 3
            shN = sh_rest_flat.reshape(n_points, num_sh_rest, 3)  # (N, C, 3)
        else:
            shN = torch.zeros(n_points, 0, 3, dtype=torch.float32, device=device)
    else:
        raise NotImplementedError(f"Unsupported checkpoint format: {args.ckpt}")

    colors = torch.cat([sh0, shN], dim=-2)
    if args.none_sh_degree:
        sh_degree = None
    else:
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    # repeat the scene into a grid (to mimic a large-scale setting)
    repeats = args.scene_grid
    gridx, gridy = torch.meshgrid(
        [
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        ],
        indexing="ij",
    )

    grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(-1, 3)
    means = means[None, :, :] + grid[:, None, :]
    means = means.reshape(-1, 3)
    quats = quats.repeat(repeats ** 2, 1)
    scales = scales.repeat(repeats ** 2, 1)
    colors = colors.repeat(repeats ** 2, 1, 1)
    if sh_degree is None:
        colors = colors[:, 0]
    opacities = opacities.repeat(repeats ** 2)
    print("Number of Gaussians:", len(means))

    render_state = {"mode": "RGB"}

    if args.backend == "gsplat":
        rasterization_fn = rasterization
    elif args.backend == "gsplat_legacy":
        from gsplat import rasterization_legacy_wrapper
        rasterization_fn = rasterization_legacy_wrapper
    elif args.backend == "inria":
        from gsplat import rasterization_inria_wrapper
        rasterization_fn = rasterization_inria_wrapper
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(
            camera_state: nerfview.CameraState, render_tab_state: nerfview.RenderTabState
    ):
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K([width, height])
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        current_mode = render_state["mode"]
        if current_mode not in {"RGB", "Depth", "Normal"}:
            raise ValueError(f"Unknown render mode: {current_mode}")

        raster_kwargs = {
            "sh_degree": sh_degree,
            "render_mode": "RGB" if current_mode == "RGB" else "ED",
            "radius_clip": 3,
        }
        if current_mode == "RGB":
            raster_kwargs["backgrounds"] = torch.zeros((3,), device=device)

        render_colors, _, _ = rasterization_fn(
            means, quats, scales, opacities, colors, viewmat[None], K[None], width, height, **raster_kwargs
        )

        if current_mode == "RGB":
            render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        elif current_mode == "Depth":
            depth = render_colors[0, ..., 0:1]  # (H, W, 1)
            near_plane = depth.min()
            far_plane = depth.max()
            depth_normalized = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_normalized = torch.clamp(depth_normalized, 0, 1)
            render_rgbs = apply_float_colormap(depth_normalized, "turbo").cpu().numpy()
        else:
            depth = render_colors[0, ..., 0]  # (H, W)
            normal_vis = depth_to_normal(depth, K)  # (H, W, 3)
            render_rgbs = normal_vis.cpu().numpy()

        return render_rgbs


    # Set up the Viser server.
    server = viser.ViserServer(port=args.port, verbose=False)

    server.initial_camera.up = up_direction if up_direction is not None else (-1, 0, 0)
    server.scene.set_up_direction(up_direction if up_direction is not None else "-x")
    server.initial_camera.look_at = facing_direction if facing_direction is not None else (0.0, 0.0, -1.0)
    server.initial_camera.position = center_point if center_point is not None else (0, 0, 0)
    server.initial_camera.fov = np.deg2rad(70)

    # Add GUI controls.
    with server.gui.add_folder("Render Settings"):
        render_mode_dropdown = server.gui.add_dropdown(
            "Render Mode",
            options=["RGB", "Depth", "Normal"],
            initial_value="RGB",
        )


    @render_mode_dropdown.on_update
    def _(_) -> None:
        render_state["mode"] = render_mode_dropdown.value
        print(f"Render mode changed to: {render_state['mode']}")


    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )

    print("Viewer running... Ctrl+C to exit.")
    print("Available render modes: RGB, Depth, Normal")
    time.sleep(100000)