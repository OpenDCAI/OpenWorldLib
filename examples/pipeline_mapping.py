## video_gen_pipeline
from sceneflow.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline

## utilize lazy loader to load different tasks pipeline
video_gen_pipe = {
    "matrix-game2": MatrixGame2Pipeline,
}

reasoning_pipe = {}

three_dim_pipe = {}
