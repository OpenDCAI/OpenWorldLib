import sys
from pathlib import Path
sys.path.append("..")

from sceneflow.pipelines.thor.pipeline_ai2thor import Ai2ThorPipeline
from sceneflow.representations.simulation_environment.thor.ai2thor_representation import Ai2ThorRepresentation
from sceneflow.operators.ai2thor_operator import Ai2ThorOperator

# 模拟agent策略：先粗暴复现人类log的导航动作，然后闭环搜索冰箱开门、拿蛋、放蛋、关门
def make_replay_then_closedloop_policy(
    *,
    # --------- 粗 replay（把人类 log 那段“走到附近”复现出来）---------
    n_turn_left=14,
    n_look_down=8,
    n_turn_right=11,
    n_right1=5,
    n_forward=6,
    n_right2=5,
    n_left=2,

    # --------- 闭环搜索上限（防死循环）---------
    max_find_fridge_open=120,
    max_find_egg=240,
    max_find_fridge_put=180,
    max_find_fridge_close=160,

    # --------- 扫视策略参数 ---------
    scan_down_every=6,      # 每隔几步 lookdown 一下
    scan_span=4,            # 左右扫切换周期（越大扫得越慢）
    prefer_horizon_for_egg=True,   # 找 egg 时多往下

    # --------- 走位抖动（“差一点点角度/距离”的情况）---------
    enable_position_jitter=True,
    jitter_every=25,

    # --------- 拿起后视角调整（关键）---------
    n_lookup_after_pick=6,   # 每次 5°，6 次≈30°

    # --------- 放回后收尾动作 ---------
    n_backup_after_close=4,  # 关门后后退几步
):
    # prefix 脚本（纯导航，不 interact）
    prefix = []
    prefix += ["camera_l"] * int(n_turn_left)
    prefix += ["camera_down"] * int(n_look_down)
    prefix += ["camera_r"] * int(n_turn_right)
    prefix += ["right"] * int(n_right1)
    prefix += ["forward"] * int(n_forward)
    prefix += ["right"] * int(n_right2)
    prefix += ["left"] * int(n_left)

    st = {
        "phase": 0,   # 0 prefix, 1 open, 2 pick, 25 lookup, 3 put, 35 close, 4 backup, 5 done
        "i": 0,
        "t": 0,
        "j": 0,
        "u": 0,       # lookup counter
        "b": 0,       # backup counter
        "last_tok": None,
    }

    def focus(obs):
        f = obs.get("focus", None)
        return f if isinstance(f, dict) else None

    def typ(f):
        return str((f or {}).get("objectType", "")).lower()

    def is_fridge_openable(f):
        return bool(f) and typ(f) == "fridge" and bool(f.get("openable", False))

    def is_fridge_receptacle(f):
        return bool(f) and typ(f) == "fridge" and bool(f.get("receptacle", False))

    def is_egg_pickupable(f):
        return bool(f) and typ(f) == "egg" and bool(f.get("pickupable", False))

    def scan_token(t):
        # 左右扫 + 偶尔向下
        if t % int(scan_down_every) == 0:
            return "camera_down"
        return "camera_l" if ((t // int(scan_span)) % 2 == 0) else "camera_r"

    def tiny_jitter(j):
        seq = ["forward", "backward", "right", "left"]
        return seq[j % len(seq)]

    def policy(obs):
        f = focus(obs)

        # phase 0: 跑 prefix 到大概位置
        if st["phase"] == 0:
            if st["i"] < len(prefix):
                tok = prefix[st["i"]]
                st["i"] += 1
                st["last_tok"] = tok
                return [tok]
            st["phase"] = 1
            st["t"] = 0
            return ["camera_l"]

        # phase 1: 找到 Fridge(openable) -> interact 打开
        if st["phase"] == 1:
            if is_fridge_openable(f):
                st["phase"] = 2
                st["t"] = 0
                st["last_tok"] = "interact"
                return ["interact"]  # OpenObject
            st["t"] += 1
            if st["t"] > int(max_find_fridge_open):
                return []
            tok = scan_token(st["t"])
            st["last_tok"] = tok
            return [tok]

        # phase 2: 找到 Egg(pickupable) -> interact 拿起
        if st["phase"] == 2:
            if is_egg_pickupable(f):
                st["phase"] = 25
                st["u"] = 0
                st["last_tok"] = "interact"
                return ["interact"]  # PickupObject

            st["t"] += 1
            if st["t"] > int(max_find_egg):
                return []

            if prefer_horizon_for_egg and (st["t"] % 3 == 0):
                st["last_tok"] = "camera_down"
                return ["camera_down"]

            if enable_position_jitter and (st["t"] % int(jitter_every) == 0):
                tok = tiny_jitter(st["j"])
                st["j"] += 1
                st["last_tok"] = tok
                return [tok]

            tok = scan_token(st["t"])
            st["last_tok"] = tok
            return [tok]

        # phase 25: 拿起后强制抬头（关键：更容易看到冰箱内部/门）
        if st["phase"] == 25:
            if st["u"] < int(n_lookup_after_pick):
                st["u"] += 1
                st["last_tok"] = "camera_up"
                return ["camera_up"]
            st["phase"] = 3
            st["t"] = 0
            return ["camera_l"]

        # phase 3: 找到 Fridge(receptacle) -> interact 放回
        if st["phase"] == 3:
            if is_fridge_receptacle(f):
                st["phase"] = 35
                st["t"] = 0
                st["last_tok"] = "interact"
                return ["interact"]  # PutObject

            st["t"] += 1
            if st["t"] > int(max_find_fridge_put):
                # 实在找不到也尝试放一次
                st["phase"] = 35
                st["t"] = 0
                st["last_tok"] = "interact"
                return ["interact"]

            if enable_position_jitter and (st["t"] % (int(jitter_every) + 10) == 0):
                tok = tiny_jitter(st["j"])
                st["j"] += 1
                st["last_tok"] = tok
                return [tok]

            tok = scan_token(st["t"])
            st["last_tok"] = tok
            return [tok]

        # phase 35: 放回后，重新对准 Fridge(openable) -> interact 关门
        if st["phase"] == 35:
            if is_fridge_openable(f):
                st["phase"] = 4
                st["b"] = 0
                st["last_tok"] = "interact"
                return ["interact"]  # CloseObject
            st["t"] += 1
            if st["t"] > int(max_find_fridge_close):
                # 找不到也试一次（有时 focus 仍在 fridge 上）
                st["phase"] = 4
                st["b"] = 0
                st["last_tok"] = "interact"
                return ["interact"]
            tok = scan_token(st["t"])
            st["last_tok"] = tok
            return [tok]

        # phase 4: 关门后后退 n 步
        if st["phase"] == 4:
            if st["b"] < int(n_backup_after_close):
                st["b"] += 1
                st["last_tok"] = "backward"
                return ["backward"]
            st["phase"] = 5
            return []

        # done
        return []

    return policy

EXEC = "./thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917" # 请修改为你的 ai2thor 可执行文件路径

rep = Ai2ThorRepresentation(
    executable_path=EXEC,
    scene="FloorPlan1",
    visibilityDistance=1.5,
    gridSize=0.05,
    rotateStepDegrees=90,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    width=300,
    height=300,
)

op = Ai2ThorOperator(
    grid_size=0.05,
    rotate_deg=90,
    look_deg=5,
    camera_yaw_deg=3.0,
    rot_step_pixels=2.0,
    max_yaw_per_tick=12,
)

policy = make_replay_then_closedloop_policy()

pipeline = Ai2ThorPipeline(operators=op, representation=rep)
pipeline(
    output_dir="./thor_record",
    fps=60,
    save_frames=True,
    max_steps=None,
    policy=policy
)
