import imageio, os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sceneflow.pipelines.kling.pipeline_recammaster import ReCamMasterPipeline


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()

pretrained_model_path = "Wan-AI/Wan2.1-T2V-1.3B"
recammaster_ckpt_path = "KlingTeam/ReCamMaster-Wan2.1"

video_path = "./data/test_video_case1/talking_man.mp4"
interaction = [100, 100, 0, 0, 30]  # dx, dy, dz, theta_x, theta_z
textual_prompt = """
A man in a black suit and a green shirt is standing in a kitchen, engaging in a conversation. 
He appears to be expressing himself with hand gestures, possibly emphasizing a point or 
explaining something. The kitchen is cluttered with various items, including a refrigerator, 
cabinets, and a stove. The man's expressions and gestures suggest he is explaining something 
important or reacting to a situation. The main subject is a man wearing a black suit and a 
green shirt. He is positioned centrally in the frame and is making hand gestures with his 
right hand. His facial expressions change throughout the video, indicating he is speaking or 
reacting to something. The man's movements are primarily focused on his hand gestures and facial 
expressions. He occasionally shifts his body slightly, but his primary actions involve moving
his right hand and changing his facial expressions. The background remains static throughout the video.
"""

pipeline = ReCamMasterPipeline.from_pretrained(model_path=recammaster_ckpt_path,
                                               required_components={"wan_model_path": pretrained_model_path})

output_video = pipeline(interaction, video_path, textual_prompt)
save_video(output_video, "./recammaster_output.mp4", fps=30, quality=5)
