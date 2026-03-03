import os
import sys
import numpy as np
import imageio
from sceneflow.pipelines.kling.pipeline_astra import AstraPipeline


def export_to_video(video_frames, output_video_path, fps=20):
    frames_np = [np.array(frame) for frame in video_frames]
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in frames_np:
            writer.append_data(frame)


def test():
    astra_path = "EvanEternal/Astra" 
    wan_model_path = "Wan-AI/Wan2.1-T2V-1.3B" 

    image_path = "./data/test_case1/ref_image.png"
    interaction = {
        "prompt": "A cozy snowy fairy-tale village with thatched cottages covered in thick snow.",
        "direction": ["forward", "camera_l", "camera_r"] # interaction list: forward, backward, left, right, forward_left, s_curve
    }

    output_path = "astra_test.mp4"

    print("Initializing Astra Pipeline...")
    pipeline = AstraPipeline.from_pretrained(
        model_path=astra_path,
        required_components={"wan_model_path": wan_model_path},
        device="cuda",
        moe_num_experts=3 # ensure match with the default setting
    )

    print("Running Inference...")
    video_frames = pipeline(
        image_path=image_path,
        interactions=interaction
    )

    export_to_video(video_frames, output_path, fps=20)
    print(f"Test finished! Video saved to {output_path}")

if __name__ == "__main__":
    test()
