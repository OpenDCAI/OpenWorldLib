import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import trimesh
from PIL import Image
from diffusers.utils import export_to_video
from tqdm import tqdm

from src.general_utils import set_seed, Timer, rank0_log
from src.pointcloud import multi_gpu_point_rendering
from src.vlm_utils import get_traj_caption

os.environ["TOKENIZERS_PARALLELISM"] = "false"
timer = Timer()

LLM_ADDR = "localhost"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
LLM_PORT = 8000


def caption_single_video(args):
    """Caption one rendered video."""
    render_path, llm_addr, llm_port, model_name = args
    output_path = render_path.replace('/render.mp4', '/traj_caption.json')

    try:
        traj_caption = get_traj_caption(llm_addr, llm_port, model_name, render_path)
        with open(output_path, "w") as write:
            json.dump({"prompt": traj_caption}, write, indent=2)
        return render_path, True, None
    except Exception as e:
        return render_path, False, str(e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", default=None, type=str, help="target path")
    parser.add_argument("--seed", default=1024, type=int, help="random seed for reproducibility")
    # Multi-node sharding params.
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

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    device_num = torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
    )
    set_seed(args.seed)

    scene_list = [args.target_path] if os.path.exists(f"{args.target_path}/panorama.png") else glob(f"{args.target_path}/*")
    scene_list.sort()
    scene_list = scene_list[args.node_rank::args.node_size]

    for scene_path in tqdm(scene_list):

        # Load all pre-defined trajectories.
        traj_list = (glob(f"{scene_path}/render_results/view*/traj*") +
                     glob(f"{scene_path}/render_results/target*/traj*") +
                     glob(f"{scene_path}/render_results/wonder*/traj*") +
                     glob(f"{scene_path}/render_results/reconstruct*/traj*"))

        rank0_log(f"Get {len(traj_list)} trajectories at all for {scene_path}")

        # Load the global point cloud once per scene.
        with timer.track("[IO] Loading point cloud for rendering"):
            global_pcd = trimesh.load(f"{scene_path}/render_results/global_pcd.ply")

        for traj_path in tqdm(traj_list, desc="Rendering Trajectories...", disable=rank != 0):
            if not os.path.exists(f"{traj_path}/camera.json"):
                continue

            with open(f"{traj_path}/camera.json", "r") as f:
                camera_info = json.load(f)
            view_id, traj_id = traj_path.split('/')[-2], traj_path.split('/')[-1]
            image_path = f"{scene_path}/render_results/{view_id}/start_frame.png"
            splitted_image = Image.open(image_path)
            image_w, image_h = splitted_image.size

            Ks = torch.tensor(np.array(camera_info["intrinsic"]), dtype=torch.float32)
            w2cs = torch.tensor(np.array(camera_info["extrinsic"]), dtype=torch.float32)

            dist.barrier()

            # Render the trajectory with multi-GPU point splatting.
            with timer.track("Multi-GPU point rendering"):
                replace_first_frame = not (view_id.startswith("reconstruct_") and traj_id == "traj1")
                pcd_renders, pcd_mask = multi_gpu_point_rendering(image=splitted_image, Ks=Ks, w2cs=w2cs,
                                                                  render_points=global_pcd.vertices,
                                                                  render_colors=global_pcd.colors[:, :3] / 255 * 2 - 1,  # [-1~1]
                                                                  image_h=image_h, image_w=image_w,
                                                                  device=device, device_num=device_num,
                                                                  render_radius=0.008, points_per_pixel=20,
                                                                  slice_size=4, local_rank=local_rank, replace_first_frame=replace_first_frame)

            dist.barrier()

            pcd_renders = pcd_renders.to(torch.float32)
            to_pil = transforms.ToPILImage()
            render_video = [to_pil((frame + 1) / 2) for frame in pcd_renders]
            mask_video = [to_pil(mask) for mask in pcd_mask]

            if rank == 0:
                with timer.track("[IO] Save rendered results"):
                    export_to_video(render_video, f"{scene_path}/render_results/{view_id}/{traj_id}/render.mp4", fps=16)
                    export_to_video(mask_video, f"{scene_path}/render_results/{view_id}/{traj_id}/render_mask.mp4", fps=16)

            dist.barrier()

        # Caption rendered trajectories with concurrent vLLM requests.
        if rank == 0:
            total_render_list = glob(f"{scene_path}/render_results/*/traj*/render.mp4")
            total_render_list = [path for path in total_render_list if not (path.split("/")[-3].startswith("reconstruct_") and path.split("/")[-2] == "traj1")]

            if total_render_list:
                with timer.track("vllm Qwen3-VL trajectory caption (parallel)"):
                    tasks = [(path, LLM_ADDR, LLM_PORT, MODEL_NAME) for path in total_render_list]
                    with ThreadPoolExecutor(max_workers=min(len(tasks), 32)) as executor:
                        futures = [executor.submit(caption_single_video, task) for task in tasks]
                        for future in tqdm(as_completed(futures), total=len(futures), desc="VLLM Captioning..."):
                            render_path, success, error = future.result()
                            if not success:
                                rank0_log(f"Failed: {render_path}, Error: {error}", "ERROR")

            # Iteration trajectories reuse the traj0 caption.
            for render_path in glob(f"{scene_path}/render_results/reconstruct_*/traj1/render.mp4"):
                shutil.copy(render_path.replace("traj1", "traj0").replace("render.mp4", "traj_caption.json"),
                            render_path.replace("render.mp4", "traj_caption.json"))

        dist.barrier()

        if rank == 0:
            timer.summary()
