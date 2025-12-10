import os
import sys
import torch
import imageio
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from sceneflow.pipelines.wonder_journey.pipeline import WonderJourneyPipeline

def main():
    input_image_path = "sceneflow/representations/assets/00_alice_1.png" 
    if not os.path.exists(input_image_path):
        os.makedirs("assets", exist_ok=True)
        Image.new('RGB', (512, 512), color='blue').save(input_image_path)
    start_image = Image.open(input_image_path).convert("RGB").resize((512, 512))
    
    prompt = "A futuristic cyberpunk city"
    
    interactions = [
        {"type": "movement", "content": "straight", "frames": 30},
        {"type": "movement", "content": "turn_right", "frames": 30}
    ]

    # 加载 Pipeline 
    pipe = WonderJourneyPipeline.from_pretrained(
        pretrained_model_path='sceneflow/representations/models/wonder_journey/pretrained_models',
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Running Pipeline...")
    output_frames = pipe(
        initial_image=start_image,
        prompt=prompt,
        num_frames=60,
        interactions=interactions,
        enable_visibility_check=True 
    )

    print("Saving...")
    frames_np = [np.array(f) for f in output_frames]
    imageio.mimsave("test_final_result.mp4", frames_np, fps=20)

if __name__ == "__main__":
    main()