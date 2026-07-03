# World Generation Module

This module implements the **World Generation** pipeline of [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) — transforming a single panorama into a high-fidelity, navigable 3D world (3DGS / mesh).

It covers the last three stages of the full HY-World 2.0 pipeline:

> *Panorama Generation* (HY-Pano 2.0) &rarr; **Trajectory Planning** (WorldNav) &rarr; **World Expansion** (WorldStereo 2.0) &rarr; **World Composition** (WorldMirror 2.0 + 3DGS Learning)

<p align="center">
  <img src="../../assets/overview.png" width="95%">
</p>

## Pipeline Overview

| Stage | Script | Description |
|-------|--------|-------------|
| 1. Trajectory Planning | `traj_generate.py` | VLM-guided camera trajectory planning with obstacle-aware navigation (WorldNav) |
| 2. Trajectory Rendering | `traj_render.py` | Multi-GPU point-cloud rendering along planned trajectories + VLM captioning |
| 3. World Expansion | `video_gen.py` | WorldStereo 2.0 diffusion model generates photorealistic keyframes with memory-guided consistency |
| 4. GS Data Preparation | `gen_gs_data.py` | Extracts frames, aligned depth, normals, and camera parameters for 3DGS training |
| 5. 3DGS Training | `world_gs_trainer.py` | Gaussian Splatting optimization with depth/normal/mask regularization (custom gsplat backend) |
| Viewer | `show_gs.py` | Interactive browser-based 3DGS viewer (viser + nerfview) |

## Quick Start

### Prerequisites

- CUDA 12.8, Python 3.11+
- &ge;4 GPUs recommended (tested with 8× H20)
- A running [vLLM](https://vllm.ai/) server hosting a VLM (e.g. Qwen3-VL-8B) for trajectory planning (stages 1 & 2). You need to obtain `LLM_ADDR`, `LLM_PORT`, and `LLM_NAME` from your vLLM deployment and pass them as `--llm_addr`, `--llm_port`, `--llm_name` to `traj_generate.py` and `traj_render.py`. Example:

  ```bash
  # Launch vLLM server (on a separate GPU group or machine)
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-VL-8B-Instruct \
      --served-model-name Qwen/Qwen3-VL-8B-Instruct \
      --port 8000 \
      --host 0.0.0.0 \
      --tensor-parallel-size 8 \
      --pipeline-parallel-size 1 \
      --max-model-len 32768 \
      --trust-remote-code \
      --gpu-memory-utilization 0.80
  ```

- Model checkpoints for WorldStereo 2.0 (see [Model Zoo](../../README.md#-model-zoo), and codes will download weights automatically)

### Installation

Please follow the root installation guide in [HY-World 2.0 Get Started](../../README.md#-get-started).

<details>
<summary><b>Installation Notes</b></summary>

- `submodules/hunyuan-worldplay2-gsplat-maskgaussian` is our modified version of [gsplat](https://github.com/nerfstudio-project/gsplat) that integrates MaskGaussian for adaptive probabilistic Gaussian pruning during 3DGS training. `submodules/hunyuan-worldplay2-navmesh` needs [recastnavigation](https://github.com/recastnavigation/recastnavigation) for NavMesh-based path planning. The OpenWorldLib setup script downloads external GLM/Recast dependencies and compiles both packages from source.

</details>

### Running the Full Pipeline

All stages share a common `--target_path` (scene directory) that accumulates intermediate results:

```bash
TARGET_PATH=/path/to/your/scene       # ../../examples/worldgen/case000
RESULT_DIR=/path/to/output
LLM_ADDR=0.0.0.0        # vLLM server address
LLM_PORT=8000             # vLLM server port
LLM_NAME=Qwen/Qwen3-VL-8B-Instruct  # Model name served by vLLM

# Stage 1: Trajectory Planning (single GPU)
python traj_generate.py --target_path $TARGET_PATH \
    --llm_addr $LLM_ADDR --llm_port $LLM_PORT --llm_name $LLM_NAME \
    --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm

# Stage 2: Trajectory Rendering (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 traj_render.py \
    --target_path $TARGET_PATH \
    --llm_addr $LLM_ADDR --llm_port $LLM_PORT --llm_name $LLM_NAME

# Stage 3: World Expansion - Keyframe Generation (multi-GPU + FSDP)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 video_gen.py \
    --target_path $TARGET_PATH --fsdp

# Stage 4: Build GS Training Data (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 gen_gs_data.py \
    --root_path $TARGET_PATH --save_normal --split_sky

# Stage 5: 3DGS Training (x8 GPUs)
# Note: If using fewer GPUs, increase max_steps and strategy steps proportionally:
# x4 GPUs: max_steps 2000; x2 GPUs: max_steps 4000; x1 GPU: max_steps 8000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m world_gs_trainer default \
    --data_dir $TARGET_PATH/gs_data --result_dir $RESULT_DIR \
    --max_steps 1500 --save_steps 1500 --eval_steps 1500 --ply_steps 1500 \
    --save_ply --convert_to_spz --disable_video \
    --use_scale_regularization --antialiased \
    --depth_loss --normal_loss --sky_depth_from_pcd \
    --use_mask_gaussian --mask_export_stochastic \
    --no-mask-export-anchor-protection --use_anchor_protection --export_mesh \
    --strategy.refine-start-iter 150 --strategy.refine-stop-iter 750 \
    --strategy.refine-every 100 --strategy.refine-scale2d-stop-iter 750 \
    --strategy.reset-every 99990 --strategy.grow-grad2d 0.0001 --strategy.prune-scale3d 0.1

# Viewer: visualize the trained 3DGS
python show_gs.py --port 8081 --gpu_id 0 --ckpt "$RESULT_DIR/ckpts/ckpt_1499_rank*.pt"
```

## Architecture

### WorldNav (Trajectory Planning)

Implemented in `traj_generate.py` + `src/navi_utils.py`:

- Uses VLM (Qwen3-VL) to identify interesting targets
- Plans diverse trajectories: surround, exploration, reconstruction, and aerial routes
- Obstacle-aware path planning with iterative refinement
- SAM3 semantic segmentation to guide navigation

### WorldStereo 2.0 (World Expansion Model)

Located in `models/`, WorldStereo 2.0 is a diffusion-based video generation model that expands a panoramic scene along camera trajectories:

- **WorldStereoModel** — Transformer backbone extending WanTransformer3DModel with camera embeddings and ControlNet conditioning on point-cloud renders
- **PanoramaMemoryBank** — Retrieval-based memory that maintains cross-trajectory consistency via reference panorama injection
- **Distribution Matching Distillation (DMD)** — four-step inference mode for efficient generation

Model variants:
- `worldstereo-memory` — Full multi-step inference with memory
- `worldstereo-memory-dmd` — DMD-accelerated four-step inference (default, recommended)

### World Composition (3DGS Training)

Implemented in `world_gs_trainer.py` + `gs/`:

- Uses `gsplat_maskgaussian` (our custom gsplat fork with integrated MaskGaussian) for differentiable rasterization and adaptive probabilistic Gaussian pruning
- Depth, normal, and LPIPS loss regularization for high-quality geometry
- Supports DefaultStrategy and MCMCStrategy for Gaussian densification
- Sky-aware training with separate sky point clouds and sky-depth-from-PCD
- Geometry-aware point cloud downsampling
- Exports to `.ply`, `.spz` (compressed), and mesh (TSDF fusion) formats

## Data Layout

Each scene directory follows this structure (produced incrementally by stages 1–5):

```
<scene_dir>/
├── panorama.png                        # Input 360° panorama
├── meta_info.json                      # Scene metadata
├── objects.json                        # Detected objects from VLM
│
├── navmesh/                            # Stage 1: NavMesh & trajectory planning
│
├── render_results/                     # Stage 2–3: Rendering & generation results
│   ├── global_pcd.ply                  # Global point cloud
│   ├── global_mesh.ply                 # Global mesh
│   ├── global_normal.npy               # Global normal map
│   ├── full_depth_prediction.pt        # Full-scene depth prediction
│   ├── sky_pcd.ply                     # Sky point cloud
│   │
│   ├── view{N}/                        # Per-viewpoint (N=0,1,2,...)
│   │   ├── start_frame.png            # Starting frame
│   │   └── traj{M}/                   # Per-trajectory (M=0,1,2)
│   │       ├── camera.json            # Camera parameters
│   │       ├── render.mp4             # Point-cloud rendered video (Stage 2)
│   │       ├── render_mask.mp4        # Render mask video
│   │       ├── traj_caption.json      # VLM caption
│   │       ├── worldstereo-memory-dmd_result.mp4  # Generated video (Stage 3)
│   │       └── memory_inputs/         # Memory bank inputs for generation
│   ├── target_*/                       # Target-based trajectories (same structure as view)
│   ├── wonder_*/                       # Exploration trajectories
│   └── reconstruct_*/                  # Reconstruction trajectories
│   │
│   └── generation_bank_worldstereo-memory-dmd/  # Accumulated generation results
│
├── gs_data/                            # Stage 4: GS training data
│
└── <result_dir>/                       # Stage 5: 3DGS output (separate path)
```

For full benchmark results, refer to the [technical report](https://arxiv.org/abs/2604.14268).

## Related Projects

- [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) — Full project (this repository)
- [WorldMirror 2.0](../../hyworld2/worldrecon/) — Feed-forward 3D reconstruction from multi-view images/videos
- [WorldStereo](https://github.com/FuchengSu/WorldStereo) — Previous version (open-source preview)
- [HunyuanWorld 1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) — Previous panorama generation

## Citation

```bibtex
@article{hy2026hy,
  title={HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds},
  author={HY-World, Team and Cao, Chenjie and Zuo, Xuhui and Wang, Zhenwei and Zhang, Yisu and Wu, Junta and Liu, Zhenyang and Gong, Yuning and Liu, Yang and Yuan, Bo and others},
  journal={arXiv preprint arXiv:2604.14268},
  year={2026}
}
```
