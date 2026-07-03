import argparse
import gc
import json
import os
from glob import glob

import imagesize
import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils import export_to_video
from moge.model.v2 import MoGeModel
from torch.distributed.device_mesh import init_device_mesh
from tqdm import tqdm
from transformers import Sam3VideoModel, Sam3VideoProcessor

from models.worldstereo_wrapper import WorldStereo
from src.data_utils import sort_trajs, load_mutli_traj_dataset
from src.general_utils import set_seed, load_video, rank0_log, Timer
from src.retrieval_wm import PanoramaMemoryBank
from src.sp_utils.parallel_states import initialize_parallel_state

os.environ["TOKENIZERS_PARALLELISM"] = "false"
timer = Timer()

SAM3_REPO_ID = "facebook/sam3"
MOGE_ID = "Ruicheng/moge-2-vitl-normal"

if __name__ == '__main__':
    # == parse configs ==
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="worldstereo-memory-dmd", choices=["worldstereo-memory", "worldstereo-memory-dmd"],
                        help="Model type (e.g., 'worldstereo-memory', 'worldstereo-memory-dmd')")
    parser.add_argument("--target_path", default=None, type=str, help="target path")
    parser.add_argument("--align_nframe", default=8, type=int, help="align downsample nframe")
    parser.add_argument("--max_reference", default=8, type=int, help="max reference number")
    parser.add_argument("--downsampled_pts", default=2_000_000, type=int, help="Downsampled points number")
    parser.add_argument("--kb_anomaly_percentile", default=90, type=float, help="alignment anoamly percentile")
    parser.add_argument("--pcd_nb_neighbors", default=10, type=int, help="pointcloud filtering number of neighbors")
    parser.add_argument("--pcd_std_ratio", default=2.0, type=float, help="pointcloud filtering std ratio")
    parser.add_argument("--local_files_only", action="store_true", help="If True, avoid downloading the file and return the path to the local cached file if it exists.")
    parser.add_argument("--fsdp", action="store_true", help="Enable FSDP model sharding")
    parser.add_argument("--skip_exist", action="store_true", help="skip existing videos")
    parser.add_argument("--seed", default=1024, type=int, help="Random seed")

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
    device_num = torch.cuda.device_count()
    mesh_size = (world_size // device_num, device_num)
    mesh_dims = ("rep", "shard")
    device_mesh = init_device_mesh("cuda", mesh_size, mesh_dim_names=mesh_dims)

    # == init logger ==
    rank0_log(f"World size: {world_size}")

    # == init SP ==
    parallel_dims = initialize_parallel_state(sp=world_size)
    sp_enabled = parallel_dims.sp_enabled
    sp_size = parallel_dims.sp if sp_enabled else 1
    sp_rank = parallel_dims.sp_rank if sp_enabled else 0
    data_rank = dist.get_rank() // sp_size
    data_world_size = dist.get_world_size() // sp_size
    global_seed = args.seed + data_rank
    set_seed(global_seed)
    print(f"Global rank:{dist.get_rank()}, Local rank:{local_rank}, SP_rank:{sp_rank}, SP_group:{data_rank}, seed:{global_seed}.")

    # == setup models ==
    # Note: FP8 quantization is done INSIDE init_wan_from_cfg, BEFORE FSDP sharding
    moge_model = MoGeModel.from_pretrained(MOGE_ID).to(device)
    sam3_model = Sam3VideoModel.from_pretrained(SAM3_REPO_ID).to(device, dtype=torch.bfloat16)
    sam3_processor = Sam3VideoProcessor.from_pretrained(SAM3_REPO_ID)
    rank0_log("Model init over...")

    # reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(torch.float)

    # == Video Generation Inference ==
    worldstereo = WorldStereo.from_pretrained(
        "hanshanxue/WorldStereo",
        subfolder=args.model_type,
        local_files_only=args.local_files_only,
        sp_world_size=sp_size,
        fsdp=args.fsdp,
        device_mesh=device_mesh,
        device=device,
    )
    dist.barrier()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Auto-select autocast precision: prefer bf16, then fp16, fall back to fp32 (disable autocast)
    if torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif torch.cuda.get_device_capability(device)[0] >= 7:  # fp16 requires SM >= 70
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None  # no half-precision support, fall back to fp32
    rank0_log(f"Autocast dtype: {autocast_dtype if autocast_dtype else 'disabled (fp32)'}")

    # load data
    if os.path.exists(f"{args.target_path}/panorama.png"):
        scene_list = [args.target_path]  # single path VLM inference
    else:
        scene_list = glob(f"{args.target_path}/*")
    scene_list.sort()
    rank0_log(f"Building dataset. {len(scene_list)} scenes found.")

    # == evaluation ==
    with torch.no_grad():
        for scene in tqdm(scene_list):
            scene_name = os.path.basename(scene)
            rank0_log(f"Processing scene {scene_name}.")
            scene_type = json.load(open(f"{scene}/meta_info.json"))["scene_type"]

            # Generation order: (view*_up-->left-->right)-->wonder0,1,2...-->iter*
            with timer.track("Sorting trajectories"):
                render_list = sort_trajs(f"{scene}/render_results")

            rank0_log(f"Scene {scene.split('/')[-1]}: {len(render_list)} renderings found.")

            if os.path.exists(f"{scene}/render_results/generation_bank_{args.model_type}/aligned_pcd.ply") and args.skip_exist:
                rank0_log(f"Scene {scene.split('/')[-1]}: aligned_pcd.ply exists, skip.")
                continue

            width, height = imagesize.get(f"{'/'.join(render_list[0].split('/')[:-2])}/start_frame.png")
            rank0_log("Enable memory control, initializing memory bank.")
            with timer.track("[IO] Memory Bank Initialization"):
                memory_bank = PanoramaMemoryBank(root_path=scene, image_width=width, image_height=height, device=device, nframe=worldstereo.cfg.nframe,
                                                 max_reference=args.max_reference, align_nframe=args.align_nframe, rank=sp_rank, world_size=sp_size, moge_model=moge_model,
                                                 sam3_model=sam3_model, sam3_processor=sam3_processor, results_name=args.model_type, valid_threshold=0.15, pts_num=args.downsampled_pts,
                                                 kb_anomaly_percentile=args.kb_anomaly_percentile, pcd_nb_neighbors=args.pcd_nb_neighbors, pcd_std_ratio=args.pcd_std_ratio)

            for render_path in render_list:
                with timer.track("[IO] Loading cameras"):
                    view_id, traj_id = render_path.split('/')[-3], render_path.split('/')[-2]
                    rank0_log(f"Scene {scene_name}: view: {view_id}, traj: {traj_id}.")

                    target_cameras = json.load(open(f"{scene}/render_results/{view_id}/{traj_id}/camera.json"))
                    tar_w2cs = torch.from_numpy(np.array(target_cameras["extrinsic"])).to(dtype=torch.float32, device=device)
                    tar_Ks = torch.from_numpy(np.array(target_cameras["intrinsic"])).to(dtype=torch.float32, device=device)

                    if args.skip_exist and os.path.exists(f"{scene}/render_results/{view_id}/{traj_id}/{args.model_type}_result.mp4"):
                        if memory_bank is not None:  # Only update the memory bank
                            gen_frames = load_video(f"{scene}/render_results/{view_id}/{traj_id}/{args.model_type}_result.mp4")
                            memory_bank.update_memory(gen_frames=gen_frames, tar_w2cs_full=tar_w2cs, tar_Ks_full=tar_Ks, view_id=view_id, traj_id=traj_id)
                        continue

                # All ranks run retrieval; sequence-parallel rendering happens inside.
                with timer.track("Memory Retrieval"):
                    retrieved_frames, ref_index, ref_index_dict, ref_w2cs, _ = memory_bank.retrieval(tar_w2cs, tar_Ks, view_id=view_id, traj_id=traj_id)
                    combined_frames = retrieved_frames / 255
                if rank == 0:  # Rank 0 saves retrieval results
                    with timer.track("[IO] Save Memory retrieval results"):
                        os.makedirs(f"{scene}/render_results/{view_id}/{traj_id}/memory_inputs", exist_ok=True)
                        export_to_video(combined_frames, f"{scene}/render_results/{view_id}/{traj_id}/memory_inputs/{args.model_type}.mp4", fps=16)
                        if ref_index_dict is not None:
                            with open(f"{scene}/render_results/{view_id}/{traj_id}/memory_inputs/{args.model_type}_ref_index.json", "w") as w:
                                json.dump(ref_index_dict, w, indent=2)
                        if ref_w2cs is not None:
                            ref_w2cs = ref_w2cs.cpu().numpy().tolist()
                            with open(f"{scene}/render_results/{view_id}/{traj_id}/memory_inputs/{args.model_type}_ref_w2cs.json", "w") as w:
                                json.dump(ref_w2cs, w, indent=2)

                dist.barrier()
                with timer.track("[IO] Loading meta inputs"):
                    meta_data = load_mutli_traj_dataset(cfg=worldstereo.cfg, input_path=f"{scene}/render_results", output_path=f"{scene}/render_results",
                                                        view_id=view_id, traj_id=traj_id, device=device, ref_index=ref_index, model_type=args.model_type, task_type="panorama")

                # ==== Pipline Inputs ====
                pipeline_kwargs = {k: v for k, v in meta_data.items() if v is not None}
                pipeline_kwargs.update(
                    negative_prompt=worldstereo.cfg.get("negative_prompt", ""),
                    generator=generator,
                    output_type="pt",
                    latent_cond_mode=worldstereo.cfg.latent_cond_mode,
                )

                if args.model_type == "worldstereo-memory-dmd":
                    pipeline_kwargs["mode"] = "test"
                else:
                    pipeline_kwargs["guidance_scale"] = 5.0

                # pipeline inference
                with timer.track("Video Model Inference"), torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    output = worldstereo.pipeline(**pipeline_kwargs).frames[0].float()

                gc.collect()
                torch.cuda.empty_cache()

                if dist.get_rank() % sp_size == 0:
                    with timer.track("[IO] Save Results"):
                        # [f,c,h,w]->[f,h,w,c]
                        output = output.permute(0, 2, 3, 1).cpu().numpy()
                        export_to_video(output, f"{scene}/render_results/{view_id}/{traj_id}/{args.model_type}_result.mp4", fps=16)
                dist.barrier()

                # update memory bank
                if memory_bank is not None:
                    with timer.track("[IO] Reload results for memory update* (need to be optimized)"):
                        gen_frames = load_video(f"{scene}/render_results/{view_id}/{traj_id}/{args.model_type}_result.mp4")
                    memory_bank.update_memory(gen_frames=gen_frames, tar_w2cs_full=tar_w2cs, tar_Ks_full=tar_Ks, view_id=view_id, traj_id=traj_id)
                dist.barrier()

            if memory_bank is not None:
                with timer.track("Run World Mirror"):
                    memory_bank.apply_worldmirror(skip_exist=True)
                dist.barrier()

                with timer.track("Memory bank Alignment"):
                    memory_bank.alignment(debug_mode=False)
                dist.barrier()

                # memory bank over, export pcd
                with timer.track("[IO] Save final aligned pointcloud (update memory)"):
                    memory_bank.export_pcd(f"{memory_bank.root_path}/render_results/generation_bank_{args.model_type}", N_points=args.downsampled_pts)
                dist.barrier()

            if rank == 0:
                timer.summary()

    if dist.is_initialized():
        dist.destroy_process_group()
