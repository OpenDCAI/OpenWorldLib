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

rep_cfg = dict(
    scene="FloorPlan1",
    visibilityDistance=1.5,
    gridSize=0.05,
    rotateStepDegrees=90,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    width=300,
    height=300,
)

op_cfg = dict(
    grid_size=0.05,
    rotate_deg=90,
    look_deg=5,
    camera_yaw_deg=3.0,
)

pipe = Ai2ThorPipeline.from_pretrained(rep_cfg=rep_cfg, op_cfg=op_cfg)

results = pipe(
    fps=10,
    max_steps=1000,
    include_depth=False,
    include_instance=False,
    window_name="thor_smoke",
    show_window=True,
    record_frames=True,
    record_actions=True,
    policy=policy
)

save_info = pipe.save_results(
    results=results,
    output_dir="./output/ai2thor_smoke",
    save_frames=False,  
)
print(save_info)
