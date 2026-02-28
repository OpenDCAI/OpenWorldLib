import sys
import json
sys.path.append("..")

from sceneflow.pipelines.simworld.pipeline_simworld import SimWorldPipeline

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

policy1 = load_json_policy("./data/test_simulate_case1/test_policy/simworld/test1.json")
policy2 = load_json_policy("./data/test_simulate_case1/test_policy/simworld/test2.json")
policy3 = load_json_policy("./data/test_simulate_case1/test_policy/simworld/test3.json")

# 三个 agent 生成在 (0,0) (200,0) (400,0)
# 所有物体生成在 y=1500 以外，确保不干扰 agent
# 如需更改仿真环境 请参考 data/test_simulate_case1/simworld_data/default.yaml 中的配置项，详情参考： https://simworld.readthedocs.io/en/latest/
rep_cfg = dict(
    ip="127.0.0.1",
    port=9000,
    resolution=(640, 480),
    camera_id=0,                # camera 0 = 全局视角
    num_agents=3,
    agent_model_path="/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C",
    spawn_position=[(0, 0), (200, 0), (400, 0)],
    spawn_direction=[(0, 1), (0, 1), (0, 1)],
    ue_manager_path="/Game/TrafficSystem/UE_Manager.UE_Manager_C",
    interact_radius=150.0,

    spawn_objects=[
        # ── 静态道具（y=1500，远离三个 agent）──
        {
            "object_name": "BP_Mug_C_1",
            "model_path": "/Game/InteractableAsset/Cup/BP_Mug.BP_Mug_C",
            "position": (0, 1500, 0),
            "direction": (0, 0, 0),
        },
        {
            "object_name": "BP_Interactable_Box_C_1",
            "model_path": "/Game/InteractableAsset/Box/BP_Interactable_Box.BP_Interactable_Box_C",
            "position": (200, 1500, 0),
            "direction": (0, 0, 0),
        },
        # ── 载具（y=2000）──
        {
            "object_name": "BP_VehicleBase_Destruction_C_1",
            "model_path": "/Game/Interactable_Vehicle/Blueprint/BP_VehicleBase_Destruction.BP_VehicleBase_Destruction_C",
            "position": (400, 2000, 0),
            "direction": (0, 0, 0),
        },
        # ── Robot Dog（x=-1500，侧面远离）──
        {
            "object_name": "Demo_RobotDog",
            "type": "dog",
            "model_path": "/Game/Robot_Dog/Blueprint/BP_SpotRobot.BP_SpotRobot_C",
            "position": (-1500, 0, 0),
            "direction": (0, 0, 0),
        },
        # ── NPC 行人（x=1500，侧面远离）──
        {
            "object_name": "NPC_Pedestrian_1",
            "type": "pedestrian",
            "model_path": "/Game/TrafficSystem/Pedestrian/Base_Pedestrian.Base_Pedestrian_C",
            "position": (1500, 0, 0),
            "direction": (0, 1),
        },
    ],

    # ── Scooter（y=-1500，agent 背后远离）──
    spawn_scooter={
        "position": (200, -1500),
        "direction": (0, 1),
        "model_path": "/Game/ScooterAssets/Blueprints/BP_Scooter_Pawn.BP_Scooter_Pawn_C",
    },
)

op_cfg = dict(
    rotate_angle=90.0,
    step_duration=1.0,
    default_speed=200.0,
    enable_pick_up=True,
    enable_scooter=True,
    enable_vehicle=True,
    enable_social=True,
    enable_path=True,

    scooter_throttle=0.7,
    scooter_brake=1.0,
    scooter_steer=0.5,

    keyboard_mapping={
        "w": "forward",
        "s": "backward",
        "a": "left",
        "d": "right",
        "j": "camera_l",
        "l": "camera_r",
        "p": "pick_up",
        "r": "drop_object",
        "f": "sit",
        "g": "stand",
        "x": "stop_action",
        "c": "stop",
        "z": "rescan",
        "v": "get_on_scooter",
        "b": "get_off_scooter",
        "n": "enter_vehicle",
        "m": "exit_vehicle",
        "1": "argue",
        "2": "discuss",
        "3": "listen",
        "4": "wave_to_dog",
        "5": "directing_path",
        "h": "follow_path",
    },

    human_window_name="SimWorld Human Control",
    human_window_size=800,
    draw_crosshair=True,
)

pipe = SimWorldPipeline.from_pretrained(rep_cfg=rep_cfg, op_cfg=op_cfg)

results = pipe(
    policies=[policy1, policy2, policy3],       # policy 为 None 转为键盘控制
    fps=10,
    max_steps=1000,
    max_time=30,                                # 最多仿真 30 秒，避免 policy token 不够用时无限等待
    include_image=True,
    record_frames=True,
    record_actions=True,
    record_positions=True,
    record_collisions=True,
)

save_info = pipe.save_results(
    results=results,
    output_dir="./output/multiagent_test",
    fps=10,
    save_video=True,
    save_actions=True,
    save_meta=True,
    save_positions=True,
    save_collisions=True,
)

print(f"Output directory: {save_info['output_dir']}")
print(f"Global video:     {save_info['global_video_path']}")
for name, vpath in save_info.get("agent_video_paths", {}).items():
    print(f"{name} video:  {vpath}")
