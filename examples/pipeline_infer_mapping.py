import torch
from diffusers.utils import export_to_video
from pathlib import Path


def infer_matrix_game2_pipeline(pipe, input_image, interaction_signal, output_path=None, fps=None):
    num_output_frames = len(interaction_signal) * 12
    output_video = pipe(
        input_image=input_image,
        num_output_frames=num_output_frames,
        interaction_signal=interaction_signal,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = fps if fps is not None else 12
        export_to_video(output_video, str(output_path), fps=fps)
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


video_gen_pipe_infer = {
    "matrix-game2": infer_matrix_game2_pipeline,
    "wan2p2": infer_wan2p2_pipeline,
}

reasoning_pipe_infer = {

}

three_dim_pipe_infer = {

}
