import os
import sys
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from diffusers.utils import export_to_video


from sceneflow.pipelines.lingbot_world.pipeline_lingbot_world import LingBotPipeline
from sceneflow.synthesis.visual_generation.lingbot.lingbot_world.distributed.util import init_distributed_group

# === Configuration ===
IMAGE_PATH = "./data/test_case1/ref_image.png"
PRETRAINED_MODEL_PATH = "robbyant/lingbot-world-base-cam"
DEFAULT_PROMPT = "A charming medieval village with cobblestone streets, thatched-roof houses."
RESIZE_H = 480
RESIZE_W = 832
FRAMES_PER_TURN = 81

AVAILABLE_INTERACTIONS = [
    "forward", "backward", "left", "right", 
    "up", "down", 
    "camera_l", "camera_r", "camera_up", "camera_down"
]

def broadcast_input(prompt_text, rank):
    """
    In DDP mode, only Rank 0 accepts input, then broadcasts to other processes.
    """
    if rank == 0:
        data = [prompt_text]
    else:
        data = [None]
    
    if dist.is_initialized():
        dist.broadcast_object_list(data, src=0)
    
    return data[0]

def main():
    # === 1. Distributed Setup ===
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        ulysses_size = world_size
        if ulysses_size > 1:
            init_distributed_group()
    else:
        ulysses_size = 1

    # === 2. Load Model & Pipeline ===
    if rank == 0:
        print(f"Loading pipeline from {PRETRAINED_MODEL_PATH}...")
        
    pipeline = LingBotPipeline.from_pretrained(
        synthesis_model_path=PRETRAINED_MODEL_PATH,
        task="i2v-A14B",
        device=f"cuda:{local_rank}",
        rank=rank,
        t5_fsdp=(world_size > 1),
        dit_fsdp=(world_size > 1),
        ulysses_size=ulysses_size,
        t5_cpu=False,
        offload_model=False
    )

    # === 3. Load Initial Image ===
    if not os.path.exists(IMAGE_PATH):
        if rank == 0: print(f"Error: Image not found at {IMAGE_PATH}")
        return
        
    input_image = Image.open(IMAGE_PATH).convert("RGB")
    
    # === 4. Interactive Loop ===
    if rank == 0:
        print("\n" + "="*40)
        print("LingBot World - Interactive Stream")
        print("="*40)
        print("Available actions:", ", ".join(AVAILABLE_INTERACTIONS))
        print("Tips:")
        print("  - Input format: 'action1, action2' (e.g., 'forward, camera_l')")
        print("  - You can also optionally specify a prompt by typing 'p:your prompt | actions'")
        print("  - Input 'q' or 'exit' to stop.")
        print("-" * 40)

    turn_idx = 0
    
    while True:
        # --- A. User Input (Rank 0 only, then broadcast) ---
        user_input_str = ""
        if rank == 0:
            user_input_str = input(f"\n[Turn {turn_idx}] Enter commands: ").strip()
        
        # Broadcast input to all ranks so they stay in sync
        user_input_str = broadcast_input(user_input_str, rank)

        if user_input_str.lower() in ['q', 'exit', 'quit']:
            if rank == 0: print("Stopping interaction loop...")
            break
        
        if not user_input_str:
            continue

        # --- B. Parse Input ---
        current_prompt = DEFAULT_PROMPT
        action_part = user_input_str
        
        if "p:" in user_input_str and "|" in user_input_str:
            parts = user_input_str.split("|")
            prompt_part = parts[0].strip()
            action_part = parts[1].strip()
            if prompt_part.startswith("p:"):
                current_prompt = prompt_part[2:].strip()
        
        raw_actions = [s.strip() for s in action_part.split(',') if s.strip()]
        
        # Validate actions
        valid_actions = []
        for act in raw_actions:
            if act in AVAILABLE_INTERACTIONS:
                valid_actions.append(act)
            else:
                if rank == 0: print(f"Warning: Ignoring invalid action '{act}'")
        
        if not valid_actions:
            if rank == 0: print("No valid actions provided. Try again.")
            continue

        if rank == 0:
            print(f"Processing Turn {turn_idx}: Actions={valid_actions}, Prompt='{current_prompt[:30]}...'")

        # --- C. Run Stream Generation ---
        start_img = input_image if turn_idx == 0 else None
        
        interaction_signal = {
            "prompt": current_prompt,
            "action_list": valid_actions
        }

        video_chunk = pipeline.stream(
            interaction_signal=interaction_signal,
            initial_image=start_img,
            num_output_frames=FRAMES_PER_TURN,
            resize_H=RESIZE_H,
            resize_W=RESIZE_W,
            seed=42 + turn_idx
        )

        turn_idx += 1

    # === 5. Export Final Video ===
    if rank == 0:
        print("Concatenating all frames...")
        all_frames_numpy = pipeline.memory_module.all_frames 
        # all_frames is a list of numpy arrays [T, H, W, C]
        
        if len(all_frames_numpy) > 0:
            full_video = np.concatenate(all_frames_numpy, axis=0)
            
            output_filename = "lingbot_stream_demo.mp4"
            print(f"Exporting video to {output_filename}, Total frames: {full_video.shape[0]}")
            export_to_video(full_video, output_filename, fps=16)
        else:
            print("No frames generated.")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()