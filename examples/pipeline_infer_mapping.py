import torch
import imageio
from diffusers.utils import export_to_video
from pathlib import Path


def infer_matrix_game2_pipeline(pipe, input_image, interaction_signal, output_path=None, fps=None):
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


def infer_hunyuan_game_craft_pipeline(pipe, input_image, interaction_signal, output_path=None, fps=None):
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


def infer_wan2p2_pipeline(pipe, prompt, input_image=None, size="1280*704", output_path=None, fps=None):
    output_video = pipe(
        prompt=prompt,
        size=size,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = fps if fps is not None else 12

        if isinstance(output_video, torch.Tensor):
            from sceneflow.memories.visual_synthesis.wan.wan_2p2_memeory import tensor_frame_to_pil
            
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


video_gen_pipe_infer = {
    "matrix-game2": infer_matrix_game2_pipeline,
    "wan2p2": infer_wan2p2_pipeline,
    "hunyuan-game-craft": infer_hunyuan_game_craft_pipeline,
}

reasoning_pipe_infer = {
    "qwen2p5-omni": infer_qwen2p5_omni_pipeline,
}

three_dim_pipe_infer = {

}
