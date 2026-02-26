import sys
import json
from pathlib import Path
sys.path.append("..")

from sceneflow.pipelines.thor.pipeline_ai2thor import Ai2ThorPipeline

# 测试用 policy：不基于 obs 做决策，仅按顺序回放 JSON 中的高层动作 token，
# 每个 step 返回一个 token（forward / camera / interact 等），用于验证 agent 接线与 pipeline 流程。
def load_json_policy(path):
    data = json.load(open(path))
    tokens = data["tokens"]
    i = 0

    def policy(obs):
        nonlocal i
        if i >= len(tokens):
            return []
        t = tokens[i]
        i += 1
        return [t]

    return policy


policy = load_json_policy("./data/test_sim_policy_case1/thor/test.json")

# 请从QUALITY_SETTINGS选择quality: {"DONOTUSE": 0, "High": 5, "High WebGL": 8, "Low": 2, "Medium": 3, "MediumCloseFitShadows": 4, "Ultra": 7, "Very High": 6, "Very Low": 1}
rep_cfg = dict(
    executable_path="submodules/ai2thor/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917",
    quality="Ultra",                      
    scene="FloorPlan1",
    visibilityDistance=1.5,
    gridSize=0.05,
    rotateStepDegrees=90,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    width=600,
    height=600,
)

op_cfg = dict(
    grid_size=0.05,
    rotate_deg=90,
    look_deg=5,
    camera_yaw_deg=3.0,
    human_window_size=600,
)

pipe = Ai2ThorPipeline.from_pretrained(rep_cfg=rep_cfg, op_cfg=op_cfg)

results = pipe(
    fps=10,
    max_steps=60,         # action steps limit
    max_timesteps=100,    # tick/video frames limit (optional) 最大时长（秒） = max_timesteps / fps
    include_depth=False,
    include_instance=False,
    record_frames=True,
    record_actions=True,
    policy=policy,         # None -> human control
)

save_info = pipe.save_results(
    results=results,
    output_dir="./output/ai2thor_smoke",
    fps=10,                # keep consistent with run fps (optional but recommended)
    save_frames=False,
)
print(save_info)
