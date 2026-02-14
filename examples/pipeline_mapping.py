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


## utilize lazy loader to load different tasks pipeline
ALL_PIPELINES = {
    "matrix-game2": load_matrix_game2_pipeline,
    "qwen2p5omni": load_qwen2p5_omni_pipeline,
}
