from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np

from ..pipeline_utils import PipelineABC
from ...operators.simworld_operator import SimWorldOperator
from ...representations.simulation_environment.simworld.simworld_representation import SimWorldRepresentation
from ...memories.simulation_environment.simworld.simworld_memory import SimWorldMemory


class SimWorldPipeline(PipelineABC):
    def __init__(
        self,
        operators: Optional[SimWorldOperator] = None,
        representation: Optional[SimWorldRepresentation] = None,
        memory_module: Optional[SimWorldMemory] = None,
    ):
        super().__init__()
        self.operators = operators
        self.representation = representation
        self.memory_module = memory_module

    @classmethod
    def from_pretrained(
        cls,
        *,
        operators: Optional[SimWorldOperator] = None,
        representation: Optional[SimWorldRepresentation] = None,
        memory_module: Optional[SimWorldMemory] = None,
        op_cfg: Optional[Dict[str, Any]] = None,
        rep_cfg: Optional[Dict[str, Any]] = None,
        mem_cfg: Optional[Dict[str, Any]] = None,
    ) -> "SimWorldPipeline":
        if representation is None:
            representation = SimWorldRepresentation(**({} if rep_cfg is None else dict(rep_cfg)))
        if operators is None:
            operators = SimWorldOperator(**({} if op_cfg is None else dict(op_cfg)))
        if memory_module is None:
            memory_module = SimWorldMemory(**({} if mem_cfg is None else dict(mem_cfg)))
        return cls(operators=operators, representation=representation, memory_module=memory_module)

    def process(
        self,
        obs: Optional[Dict[str, Any]],
        *,
        policy: Optional[Callable[[Dict[str, Any]], Union[List[str], str, None]]] = None,
        use_human_control: bool = False,
        tick_idx: int = 0,
        action_idx: int = 0,
        mode: str = "step",
    ) -> Dict[str, Any]:
        if self.operators is None:
            raise ValueError("operators is None")

        if mode == "close":
            if use_human_control:
                self.operators.get_interaction({"type": "close_human_control"})
            self.operators.close_all_windows()
            return {}

        assert obs is not None, "obs must not be None when mode='step'"

        self.operators.update_counters(tick_idx, action_idx)
        last_percep = self.operators.process_perception(obs)

        if use_human_control:
            self.operators.get_interaction(
                {"type": "human_control", "frame": obs.get("rgb", None)}
            )
        else:
            if policy is not None:
                out = policy(obs)
                if out is not None and self.operators.check_interaction(out):
                    self.operators.get_interaction(out)

        hist = self.operators.get_interaction_history()
        tokens: List[str] = (
            hist[-1]
            if (isinstance(hist, list) and len(hist) > 0 and isinstance(hist[-1], list))
            else []
        )

        should_quit = use_human_control and ("quit" in tokens)

        nearest_object = obs.get("nearest_object", None)
        if nearest_object is None:
            nearby = obs.get("nearby_objects", [])
            if nearby:
                nearest_object = nearby[0]["object_name"]

        actions: List[Dict[str, Any]] = self.operators.process_interaction(
            use_step_mode=True,
            object_name=nearest_object,
            vehicle_name=nearest_object,
            scooter_id=obs.get("scooter_id", None),
        )

        return {
            "last_percep": last_percep,
            "actions": actions,
            "tokens": tokens,
            "should_quit": should_quit,
            "nearest_object": nearest_object,
            "nearby_objects": obs.get("nearby_objects", []),
        }

    def __call__(
        self,
        *,
        policy: Optional[Callable[[Dict[str, Any]], Union[List[str], str, None]]] = None,
        policies: Optional[List[Callable[[Dict[str, Any]], Union[List[str], str, None]]]] = None,
        fps: int = 10,
        max_steps: Optional[int] = None,
        max_time: Optional[float] = None,

        include_image: bool = True,
        use_multicam: bool = False,

        record_frames: bool = True,
        record_actions: bool = True,
        record_positions: bool = True,
        record_collisions: bool = False,
    ) -> Dict[str, Any]:
        if self.representation is None:
            raise ValueError("representation is None")
        if self.memory_module is None:
            self.memory_module = SimWorldMemory()

        mem = self.memory_module
        use_human_control = (policy is None and policies is None)

        mem.manage(action="reset")
        mem.manage(action="set_meta", meta={
            "fps": int(fps),
            "include_image": bool(include_image),
            "use_multicam": bool(use_multicam),
            "max_steps": None if max_steps is None else int(max_steps),
            "max_time": None if max_time is None else float(max_time),
        })

        obs = self.representation.get_representation({
            "mode": "init",
            "include_image": include_image,
            "use_multicam": use_multicam,
        })

        tick_idx: int = 0
        action_idx: int = 0
        tick_dt = 1.0 / float(fps)
        next_time = time.time()
        start_time = time.time()

        try:
            while True:
                if max_steps is not None and action_idx >= int(max_steps):
                    break
                if max_time is not None and (time.time() - start_time) >= float(max_time):
                    break

                now = time.time()
                if now < next_time:
                    time.sleep(min(0.001, next_time - now))
                    continue
                next_time += tick_dt

                obs = self.representation.get_representation({
                    "mode": "observe",
                    "include_image": include_image,
                    "use_multicam": use_multicam,
                })

                # ---- memory: 记录帧 ----
                if record_frames:
                    frame = obs.get("rgb", None)
                    if isinstance(frame, np.ndarray):
                        mem.record({"rgb": frame}, metadata={
                            "type": "image", "tick": int(tick_idx), "agent_idx": 0,
                        })
                    for i, agent_obs_item in enumerate(obs.get("all_agents", [])):
                        rgb = agent_obs_item.get("rgb", None)
                        if isinstance(rgb, np.ndarray):
                            mem.record(
                                {"rgb": rgb},
                                metadata={
                                    "type": "image",
                                    "tick": int(tick_idx),
                                    "agent_idx": i + 1,
                                    "agent_name": agent_obs_item["name"],
                                    "camera_id": agent_obs_item["camera_id"],
                                }
                            )

                # ---- memory: 记录位置 ----
                if record_positions:
                    for i, agent_obs_item in enumerate(obs.get("all_agents", [])):
                        pos = agent_obs_item.get("position", None)
                        if pos is not None:
                            mem.record(pos, metadata={
                                "type": "other",
                                "subtype": "position",
                                "tick": int(tick_idx),
                                "agent_idx": i + 1,
                                "agent_name": agent_obs_item["name"],
                            })

                # ---- memory: 记录碰撞 ----
                if record_collisions:
                    collision = obs.get("collision", None)
                    if collision is not None:
                        mem.record(collision, metadata={
                            "type": "other", "subtype": "collision",
                            "tick": int(tick_idx), "agent_idx": 0,
                        })
                    for i, agent_obs_item in enumerate(obs.get("all_agents", [])):
                        collision = agent_obs_item.get("collision", None)
                        if collision is not None:
                            mem.record(collision, metadata={
                                "type": "other",
                                "subtype": "collision",
                                "tick": int(tick_idx),
                                "agent_idx": i + 1,
                                "agent_name": agent_obs_item["name"],
                            })

                # ---- 多 agent policy 处理 ----
                _policies = policies if policies is not None else ([policy] if policy is not None else [None])

                all_tokens: List[str] = []
                should_quit = False

                for agent_idx, _policy in enumerate(_policies):
                    if agent_idx == 0:
                        agent_obs = obs
                    else:
                        all_agents = obs.get("all_agents", [])
                        agent_obs = all_agents[agent_idx - 1] if (agent_idx - 1) < len(all_agents) else obs

                    output_dict = self.process(
                        agent_obs,
                        policy=_policy,
                        use_human_control=use_human_control and agent_idx == 0,
                        tick_idx=tick_idx,
                        action_idx=action_idx,
                        mode="step",
                    )

                    if output_dict["should_quit"]:
                        should_quit = True

                    if agent_idx == 0:
                        all_tokens = output_dict["tokens"]

                    if output_dict["actions"]:
                        for a in output_dict["actions"]:
                            obs_after = self.representation.get_representation({
                                "mode": "step",
                                "action": a,
                                "agent_idx": agent_idx,
                                "include_image": include_image,
                                "use_multicam": use_multicam,
                            })

                            if record_actions:
                                mem.record(a, metadata={
                                    "type": "action",
                                    "tokens": list(output_dict["tokens"]),
                                    "action": a,
                                    "tick": int(tick_idx),
                                    "step": int(action_idx),
                                    "agent_idx": agent_idx + 1,
                                    "action_success": obs_after.get("action_success", None),
                                    "position": obs_after.get("position", None),
                                })

                            action_idx += 1
                            if max_steps is not None and action_idx >= int(max_steps):
                                break

                if should_quit:
                    break

                tokens = all_tokens
                tick_idx += 1

        finally:
            try:
                self.process(None, use_human_control=use_human_control, mode="close")
            except Exception:
                pass
            try:
                self.representation.get_representation({"mode": "close"})
            except Exception:
                pass

        export = mem.process(None, target_format="export")
        if isinstance(export, dict):
            if "meta" not in export or export["meta"] is None:
                export["meta"] = {}
            export["meta"]["tick_count"] = int(tick_idx)
            export["meta"]["action_count"] = int(action_idx)
            export["meta"]["elapsed_time"] = round(time.time() - start_time, 2)

        return {
            "export": export,
            "memory": mem,
            "tick_count": tick_idx,
            "action_count": action_idx,
            "elapsed_time": round(time.time() - start_time, 2),
        }

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str,
        *,
        fps: int = 10,
        save_video: bool = True,
        save_actions: bool = True,
        save_meta: bool = True,
        save_frames: bool = False,
        save_positions: bool = True,
        save_collisions: bool = False,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        export = results.get("export", None)
        if not isinstance(export, dict):
            raise ValueError("results['export'] is missing. Call pipeline(...) first.")

        # 归一化 agent_* key 为 int
        for key in ("agent_actions", "agent_positions", "agent_collisions"):
            raw_dict = export.get(key, {})
            if raw_dict and not isinstance(next(iter(raw_dict)), int):
                export[key] = {int(k): v for k, v in raw_dict.items()}

        # ── meta ──
        meta_path = os.path.join(output_dir, "meta.json")
        if save_meta:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(export.get("meta", {}), f, ensure_ascii=False, indent=2)

        mem = results.get("memory", None)

        # ── 按 agent_idx 收集帧 ──
        agent_frames: Dict[int, list] = defaultdict(list)
        if (save_video or save_frames) and mem is not None:
            for item in mem.select({"type": "image"}):
                raw_idx = item.get("metadata", {}).get("agent_idx", 0)
                content = item.get("content", None)
                rgb = content.get("rgb") if isinstance(content, dict) else content
                if isinstance(rgb, np.ndarray):
                    agent_frames[raw_idx].append(rgb)

        # ── agent_idx → 文件夹名映射 ──
        # raw_idx=0  → global/  （只存视频/帧，不存 actions/positions/collisions）
        # raw_idx=1+ → agent0/, agent1/, ...
        all_raw_indices = set(agent_frames.keys())
        for key in ("agent_actions", "agent_positions", "agent_collisions"):
            all_raw_indices.update(export.get(key, {}).keys())

        agent_raw_indices = sorted(i for i in all_raw_indices if i != 0)
        raw_to_folder = {0: "global"}
        for seq, raw in enumerate(agent_raw_indices):
            raw_to_folder[raw] = f"agent{seq}"

        global_video_path = None
        agent_video_paths: Dict[str, str] = {}
        agent_dirs: Dict[str, str] = {}

        # ── 全局文件夹（camera 0）── 只保存视频/帧
        global_dir = os.path.join(output_dir, "global")
        os.makedirs(global_dir, exist_ok=True)
        agent_dirs["global"] = global_dir

        if save_video and agent_frames.get(0):
            frames = agent_frames[0]
            vpath = os.path.join(global_dir, "video.avi")
            h, w = frames[0].shape[:2]
            vw = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (w, h))
            if vw.isOpened():
                for fr in frames:
                    vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                vw.release()
            global_video_path = vpath

        if save_frames and agent_frames.get(0):
            frames_dir = os.path.join(global_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            for i, fr in enumerate(agent_frames[0]):
                cv2.imwrite(
                    os.path.join(frames_dir, f"{i:06d}.jpg"),
                    cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                )

        # ── 各 agent 文件夹（raw_idx != 0）── 保存视频 + actions + positions + collisions
        for raw_idx in agent_raw_indices:
            folder_name = raw_to_folder[raw_idx]
            agent_dir = os.path.join(output_dir, folder_name)
            os.makedirs(agent_dir, exist_ok=True)
            agent_dirs[folder_name] = agent_dir

            if save_video and agent_frames.get(raw_idx):
                frames = agent_frames[raw_idx]
                vpath = os.path.join(agent_dir, "video.avi")
                h, w = frames[0].shape[:2]
                vw = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (w, h))
                if vw.isOpened():
                    for fr in frames:
                        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                    vw.release()
                agent_video_paths[folder_name] = vpath

            if save_frames and agent_frames.get(raw_idx):
                frames_dir = os.path.join(agent_dir, "frames")
                os.makedirs(frames_dir, exist_ok=True)
                for i, fr in enumerate(agent_frames[raw_idx]):
                    cv2.imwrite(
                        os.path.join(frames_dir, f"{i:06d}.jpg"),
                        cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                    )

            if save_actions:
                acts = export.get("agent_actions", {}).get(raw_idx, [])
                if acts:
                    apath = os.path.join(agent_dir, "actions.jsonl")
                    with open(apath, "w", encoding="utf-8") as f:
                        for a in acts:
                            f.write(json.dumps(a, ensure_ascii=False) + "\n")

            if save_positions:
                positions = export.get("agent_positions", {}).get(raw_idx, [])
                if positions:
                    ppath = os.path.join(agent_dir, "positions.jsonl")
                    with open(ppath, "w", encoding="utf-8") as f:
                        for p in positions:
                            f.write(json.dumps(p, ensure_ascii=False) + "\n")

            if save_collisions:
                collisions = export.get("agent_collisions", {}).get(raw_idx, [])
                if collisions:
                    cpath = os.path.join(agent_dir, "collisions.jsonl")
                    with open(cpath, "w", encoding="utf-8") as f:
                        for c in collisions:
                            f.write(json.dumps(c, ensure_ascii=False) + "\n")

        return {
            "output_dir": output_dir,
            "global_dir": global_dir,
            "agent_dirs": agent_dirs,
            "global_video_path": global_video_path,
            "agent_video_paths": agent_video_paths,
            "meta_path": meta_path if save_meta else None,
        }
        