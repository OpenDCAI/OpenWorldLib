import torch
import imageio
from diffusers.utils import export_to_video
from pathlib import Path


def infer_matrix_game2_pipeline(pipe, input_image, interaction_signal, output_path=None, fps=None, **kwargs):
    num_output_frames = len(interaction_signal) * 12
    output_video = pipe(
        images=input_image,
        num_frames=num_output_frames,
        interactions=interaction_signal,
        visualize_ops=False
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = fps if fps is not None else 12
        export_to_video(output_video, str(output_path), fps=fps)
    return output_video


def infer_hunyuan_game_craft_pipeline(pipe, input_image, interaction_signal, output_path=None, fps=None, **kwargs):
    num_output_frames = len(interaction_signal) * 12
    input_interactions = []
    for signal in interaction_signal:
        if signal in pipe.operators.interaction_template:
            input_interactions.append(signal)
    output_video = pipe(
        images=input_image,
        num_frames=num_output_frames,
        interactions=input_interactions,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = fps if fps is not None else 24
        imageio.mimsave(str(output_path), output_video, fps=fps, quality=8)
    return output_video


def infer_lingbot_world_pipeline(pipe, input_image, interaction_signal, output_path=None, fps=None, **kwargs):
    num_output_frames = len(interaction_signal) * 36 + 1
    output_video = pipe(
    images=input_image,
    num_frames=num_output_frames,
    interactions=interaction_signal
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = fps if fps is not None else 16
        export_to_video(output_video, str(output_path), fps=fps)
    return output_video


def infer_matrix_game3_pipeline(pipe, input_image, interaction_signal, output_path=None, fps=None, prompt=None, **kwargs):
    """
    Matrix-Game-3 wrapper inference.
    Upstream generates and saves mp4; we return the saved path.
    """
    output_path = Path(output_path) if output_path is not None else None
    save_name = output_path.stem if output_path is not None else "matrix_game_3_demo"
    output_dir = str(output_path.parent) if output_path is not None else None
    prompt_text = prompt or "A first-person view interactive scene."

    video_path = pipe(
        images=input_image,
        interactions=interaction_signal,
        prompt=prompt_text,
        output_dir=output_dir,
        save_name=save_name,
    )

    # If caller requested a specific output file name that isn't mp4, keep behavior simple:
    # - if it's .mp4, we already generated it at that exact name
    # - otherwise, return the actual generated mp4 path
    return video_path

def infer_wan2p2_pipeline(pipe, prompt, image_path=None, size="1280*704", output_path=None, fps=None):
    output_video = pipe(
        prompt=prompt,
        image_path=image_path,
        size=size,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = fps if fps is not None else 12

        if isinstance(output_video, torch.Tensor):
            from openworldlib.memories.visual_synthesis.wan.wan_2p2_memeory import tensor_frame_to_pil
            
            if output_video.ndim == 4:
                video_frames = []
                for t in range(output_video.shape[1]):
                    frame = output_video[:, t, :, :]
                    pil_frame = tensor_frame_to_pil(frame)
                    video_frames.append(pil_frame)
                export_to_video(video_frames, str(output_path), fps=fps)
    return output_video


def infer_qwen2p5_omni_pipeline(pipe, prompt, image_path=None, video_path=None):
    response = pipe(
        text=prompt,
        images=[image_path] if image_path else [],  # reference image
        videos=[video_path] if video_path else [],  # generated video
        max_new_tokens=1024
    )

    if isinstance(response, list):
        response_text = response[0] if response else ""
    else:
        response_text = str(response)
    return response_text


def infer_spirit_v1p5_pipeline(pipe, images, raw_state, task, robot_type="Franka", return_all_steps=True):
    """
    VLA inference function for Spirit-v1.5 pipeline.
    
    Args:
        pipe: SpiritV1p5Pipeline instance
        images: dict of camera images, e.g., {"cam_high": PIL.Image, "cam_left_wrist": PIL.Image}
        raw_state: robot state observation
        task: task description string
        robot_type: robot type, default "Franka"
        return_all_steps: whether to return all action steps
        
    Returns:
        actions: predicted actions (list if return_all_steps=True, else single action)
    """
    actions = pipe(
        images=images,
        raw_state=raw_state,
        task=task,
        robot_type=robot_type,
        return_all_steps=return_all_steps,
    )
    return actions


def infer_cosmos_predict2p5_pipeline(pipe, prompt, input_image, output_path=None, fps=None):
    output_video = pipe(
        prompt=prompt,
        images=input_image,
        output_type='np',
        num_inference_steps=35,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = fps if fps is not None else 16
        export_to_video(output_video, str(output_path), fps=fps)
    return output_video


def infer_cosmos3_pipeline(pipe, prompt, output_path, **kwargs):
    return pipe(prompt=prompt, output_path=output_path, **kwargs)


def infer_gamma_world_pipeline(pipe, output_dir, **kwargs):
    return pipe(output_dir=output_dir, **kwargs)


def infer_solaris_pipeline(pipe, output_dir, eval_num_samples=1, **kwargs):
    return pipe(output_dir=output_dir, eval_num_samples=eval_num_samples, **kwargs)


def infer_ctrl_world_pipeline(pipe, interaction_signal="ddcu", **kwargs):
    return pipe(interactions=interaction_signal, **kwargs)


def infer_fantasy_world_pipeline(pipe, image_path, camera_json_path, prompt, output_dir, **kwargs):
    return pipe(
        image_path=image_path,
        camera_json_path=camera_json_path,
        prompt=prompt,
        output_dir=output_dir,
        **kwargs,
    )


def infer_hunyuan_worldplay2_pipeline(pipe, input_path, output_dir, **kwargs):
    return pipe(input_path=input_path, output_dir=output_dir, **kwargs)


def infer_memflow_pipeline(pipe, prompt, output_dir, **kwargs):
    return pipe(prompt=prompt, output_dir=output_dir, **kwargs)


video_gen_pipe_infer = {
    "matrix-game2": infer_matrix_game2_pipeline,
    "matrix-game3": infer_matrix_game3_pipeline,
    "matrix-game-3": infer_matrix_game3_pipeline,
    "wan2p2": infer_wan2p2_pipeline,
    "hunyuan-game-craft": infer_hunyuan_game_craft_pipeline,
    "lingbot-world": infer_lingbot_world_pipeline,
    "cosmos-predict2p5": infer_cosmos_predict2p5_pipeline,
    "cosmos3": infer_cosmos3_pipeline,
    "gamma-world": infer_gamma_world_pipeline,
    "solaris": infer_solaris_pipeline,
    "memflow": infer_memflow_pipeline,
}

reasoning_pipe_infer = {
    "qwen2p5-omni": infer_qwen2p5_omni_pipeline,
}

three_dim_pipe_infer = {
    "fantasy-world": infer_fantasy_world_pipeline,
    "hunyuan-worldplay2": infer_hunyuan_worldplay2_pipeline,
    "hy-world-2": infer_hunyuan_worldplay2_pipeline,
}

vla_pipe_infer = {
    "spirit-v1p5": infer_spirit_v1p5_pipeline,
    "ctrl-world": infer_ctrl_world_pipeline,
}
