import os
import sys
import torch
import torch.distributed as dist
from PIL import Image
from diffusers.utils import export_to_video

from sceneflow.pipelines.lingbot_world.pipeline_lingbot_world import LingBotPipeline
from sceneflow.synthesis.visual_generation.lingbot.lingbot_world.distributed.util import init_distributed_group


image_path = "./data/test_case1/ref_image.png"
pretrained_model_path = "robbyant/lingbot-world-base-cam"
input_image = Image.open(image_path).convert("RGB")
prompt = "A charming medieval village with cobblestone streets, thatched-roof houses."

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


pipeline = LingBotPipeline.from_pretrained(
    synthesis_model_path=pretrained_model_path,
    task="i2v-A14B",
    device=f"cuda:{local_rank}",
    rank=rank,
    t5_fsdp=(world_size > 1),
    dit_fsdp=(world_size > 1),
    ulysses_size=ulysses_size,
    t5_cpu=False,
    offload_model=False
)

action_commands = ["backward", "camera_l"] 
output_video = pipeline(
    input_image=input_image,
    num_output_frames=81,
    interaction_signal={
        "prompt": prompt,
        "action_list": action_commands,
    },
    resize_H=480,
    resize_W=832,
    seed=42
)

if rank == 0 and output_video is not None:
    export_to_video(output_video, "lingbot_command_demo.mp4", fps=16)
    print("Done! Video saved.")

if dist.is_initialized():
    dist.destroy_process_group()