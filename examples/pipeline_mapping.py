## video_gen_pipeline
from sceneflow.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
from sceneflow.pipelines.qwen.pipeline_qwen2p5_omni import Qwen2p5OmniPipeline

## utilize lazy loader to load different tasks pipeline
video_gen_pipe = {
    "matrix-game2": MatrixGame2Pipeline,
}

reasoning_pipe = {
    "qwen2p5omni": Qwen2p5OmniPipeline,
}

three_dim_pipe = {}
