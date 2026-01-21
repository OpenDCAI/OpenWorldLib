import sys
from pathlib import Path
sys.path.append("..")

from sceneflow.pipelines.thor.pipeline_ai2thor import Ai2ThorPipeline
from sceneflow.representations.simulation_environment.thor.ai2thor_representation import Ai2ThorRepresentation
from sceneflow.operators.ai2thor_operator import Ai2ThorOperator

EXEC = "./data/test_sim_env_case1/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917"

rep = Ai2ThorRepresentation(
    executable_path=EXEC,
    scene="FloorPlan1",
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    width=300,
    height=300,
)

op = Ai2ThorOperator(grid_size=0.25, rotate_deg=90, look_deg=30)

pipe = Ai2ThorPipeline(operators=op, representation=rep)
pipe.run_interactive(
    output_dir="./thor_record",
    fps=10,
    save_frames=True,
    max_steps=None,       # 不限步数，按 ESC 退出
)
