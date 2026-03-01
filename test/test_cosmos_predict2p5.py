import logging
import imageio
import warnings
import numpy as np
from pathlib import Path

import torch

from sceneflow.pipelines.cosmos.pipeline_cosmos_predict2p5 import CosmosPredict2p5Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)


"""
Huggingface token is required to download cosmos-series model
You can also skip download by specifying local ckpts path
"""
token = "hf_xxxxxxxxxxxxxx"
transformer_name_or_path = "nvidia/Cosmos-Predict2.5"
text_encoder_name_or_path = "nvidia/Cosmos-Reason1-7B"
vae_name_or_path = "Wan-AI/Wan2.1-T2V-1.3B"

pipeline = CosmosPredict2p5Pipeline.from_pretrained(
    transformer_name_or_path=transformer_name_or_path,
    text_encoder_name_or_path=text_encoder_name_or_path,
    vae_name_or_path=vae_name_or_path,
    token=token,
    task="img2world",
    device="cuda",
    dtype=torch.bfloat16
)

# Set default negative prompt
pipeline.set_negative_prompt()

prompt = (
    "A robotic arm, primarily white with black joints and cables, is shown in a clean, modern indoor "
    "setting with a white tabletop. The arm, equipped with a gripper holding a small, light green pitcher, "
    "is positioned above a clear glass containing a reddish-brown liquid and a spoon. The robotic arm is "
    "in the process of pouring a transparent liquid into the glass. To the left of the pitcher, there is "
    "an opened jar with a similar reddish-brown substance visible through its transparent body. In the background, "
    "a vase with white flowers and a brown couch are partially visible, adding to the contemporary ambiance. The "
    "lighting is bright, casting soft shadows on the table. The robotic arm's movements are smooth and controlled, "
    "demonstrating precision in its task. As the video progresses, the robotic arm completes the pour, leaving the "
    "glass half-filled with the reddish-brown liquid. The jar remains untouched throughout the sequence, and the "
    "spoon inside the glass remains stationary. The other robotic arm on the right side also stays stationary "
    "throughout the video. The final frame captures the robotic arm with the pitcher finishing the pour, with the "
    "glass now filled to a higher level, while the pitcher is slightly tilted but still held securely by the gripper."
)
image_path = "data/test_case2/robot_arm.png"
save_path = "data/test_case2/cosmos_predict2p5_call.mp4"

output_video = pipeline(
    prompt=prompt,
    image_path=image_path,
    output_type='np',  # Optional[str] = 'pt', 'pil', 'np' ...
    num_inference_steps=35,
)[0]  # shape: (T, H, W, C)

output_video = (np.clip(output_video, 0, 1) * 255).astype(np.uint8)
imageio.mimsave(save_path, output_video, fps=28)
