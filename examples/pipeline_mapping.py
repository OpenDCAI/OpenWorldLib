def load_matrix_game2_pipeline(model_path, device):
    from sceneflow.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
    return MatrixGame2Pipeline.from_pretrained(
            synthesis_model_path=model_path,
            mode="universal",
            device=device,
            )

def load_qwen2p5_omni_pipeline(model_path, device):
    from sceneflow.pipelines.qwen.pipeline_qwen2p5_omni import Qwen2p5OmniPipeline
    return Qwen2p5OmniPipeline.from_pretrained(
            pretrained_model_path=model_path,
            use_audio_in_video=False,
            device=device,
            )

def load_spirit_v1p5_pipeline(model_path, device, norm_stats_path=None):
    from sceneflow.pipelines.spirit_ai.pipeline_spirit_v1p5 import SpiritV1p5Pipeline
    from pathlib import Path
    
    # 如果未提供 norm_stats_path，尝试在模型目录下查找 norm_stats.json
    if norm_stats_path is None:
        model_dir = Path(model_path)
        if model_dir.is_dir():
            norm_stats_file = model_dir / "norm_stats.json"
            if norm_stats_file.exists():
                norm_stats_path = str(norm_stats_file)
    
    return SpiritV1p5Pipeline.from_pretrained(
            pretrained_model_path=model_path,
            norm_stats_path=norm_stats_path,
            device=device,
            use_bf16=True,
            )


## utilize lazy loader to load different tasks pipeline
video_gen_pipe = {
    "matrix-game2": load_matrix_game2_pipeline,
}

reasoning_pipe = {
    "qwen2p5omni": load_qwen2p5_omni_pipeline,
}

vla_pipe = {
    "spirit-v1p5": load_spirit_v1p5_pipeline,
}
