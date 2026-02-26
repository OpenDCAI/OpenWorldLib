def load_matrix_game2_pipeline(model_path, device):
    from sceneflow.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
    return MatrixGame2Pipeline.from_pretrained(
            synthesis_model_path=model_path,
            mode="universal",
            device=device,
            )

def load_hunyuan_game_craft_pipeline(model_path, device):
    from sceneflow.pipelines.hunyuan_world.pipeline_hunyuan_game_craft import HunyuanGameCraftPipeline
    return HunyuanGameCraftPipeline.from_pretrained(
            synthesis_model_path=model_path,
            device=device,
            )

def load_qwen2p5_omni_pipeline(model_path, device):
    from sceneflow.pipelines.qwen.pipeline_qwen2p5_omni import Qwen2p5OmniPipeline
    return Qwen2p5OmniPipeline.from_pretrained(
            pretrained_model_path=model_path,
            use_audio_in_video=False,
            device=device,
            )

def load_wan2p2_pipeline(model_path, device):
    from sceneflow.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline
    return Wan2p2Pipeline.from_pretrained(
            synthesis_model_path=model_path,
            task="ti2v-5B",
            )


## utilize lazy loader to load different tasks pipeline
video_gen_pipe = {
    "matrix-game2": load_matrix_game2_pipeline,
    "wan2p2": load_wan2p2_pipeline,
    "hunyuan-game-craft": load_hunyuan_game_craft_pipeline,
}

reasoning_pipe = {
    "qwen2p5omni": load_qwen2p5_omni_pipeline,
}

three_dim_pipe = {

}
