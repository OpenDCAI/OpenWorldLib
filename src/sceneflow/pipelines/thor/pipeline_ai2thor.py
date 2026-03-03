from __future__ import annotations

import json
import os
import time
import numpy as np
import cv2
from typing import Any, Callable, Dict, List, Optional, Union

from ..pipeline_utils import PipelineABC
from ...operators.ai2thor_operator import Ai2ThorOperator
from ...representations.simulation_environment.thor.ai2thor_representation import Ai2ThorRepresentation
from ...memories.simulation_environment.thor.ai2thor_memory import Ai2ThorMemory


class Ai2ThorPipeline(PipelineABC):
    def __init__(
        self,
        operators: Optional[Ai2ThorOperator] = None,
        representation: Optional[Ai2ThorRepresentation] = None,
        memory_module: Optional[Ai2ThorMemory] = None,
    ):
        super().__init__()
        self.operators = operators
        self.representation = representation
        self.memory_module = memory_module

    @classmethod
    def from_pretrained(
        cls,
        *,
        operators: Optional[Ai2ThorOperator] = None,
        representation: Optional[Ai2ThorRepresentation] = None,
        memory_module: Optional[Ai2ThorMemory] = None,
        op_cfg: Optional[Dict[str, Any]] = None,
        rep_cfg: Optional[Dict[str, Any]] = None,
        mem_cfg: Optional[Dict[str, Any]] = None,
    ) -> "Ai2ThorPipeline":
        if representation is None:
            representation = Ai2ThorRepresentation(**({} if rep_cfg is None else dict(rep_cfg)))
        if operators is None:
            operators = Ai2ThorOperator(**({} if op_cfg is None else dict(op_cfg)))
        if memory_module is None:
            memory_module = Ai2ThorMemory(**({} if mem_cfg is None else dict(mem_cfg)))
        return cls(operators=operators, representation=representation, memory_module=memory_module)

    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: Ai2ThorPipeline._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [Ai2ThorPipeline._to_serializable(v) for v in obj]
        # LazyInstanceSegmentationMasks 等懒加载对象，趁 controller 还活着立即物化
        if hasattr(obj, "items"):
            try:
                return {k: Ai2ThorPipeline._to_serializable(v) for k, v in obj.items()}
            except Exception:
                pass
        if hasattr(obj, "__iter__"):
            try:
                return [Ai2ThorPipeline._to_serializable(v) for v in obj]
            except Exception:
                pass
        return None  # 实在无法序列化的 fallback
    
    def process(
        self,
        obs: Dict[str, Any],
        *,
        policy: Optional[Callable[[Dict[str, Any]], Union[List[str], str, None]]] = None,
        use_human_control: bool = False,
    ) -> Dict[str, Any]:
        """
        Process perception and interaction using operators ONLY.
        No representation calls, no memory recording.
        """
        if self.operators is None:
            raise ValueError("operators is None")

        # 1) Process perception (operator)
        last_percep = self.operators.process_perception(obs)

        # 2) Get interaction (operator)
        if use_human_control:
            self.operators.get_interaction({"type": "human_control", "frame": obs.get("frame", None)})
        else:
            out = policy(obs) if policy is not None else None
            if out is not None and self.operators.check_interaction(out):
                self.operators.get_interaction(out)

        # 3) Read latest tokens (operator)
        hist = self.operators.get_interaction_history()
        tokens: List[str] = hist[-1] if (isinstance(hist, list) and len(hist) > 0 and isinstance(hist[-1], list)) else []

        # 4) Check for quit
        should_quit = use_human_control and ("quit" in tokens)

        # 5) Process interaction to get actions (operator)
        actions: List[Dict[str, Any]] = self.operators.process_interaction()

        return {
            "last_percep": last_percep,
            "actions": actions,
            "tokens": tokens,
            "should_quit": should_quit,
        }

    def __call__(
        self,
        *,
        policy: Optional[Callable[[Dict[str, Any]], Union[List[str], str, None]]] = None,
        fps: int = 20,
        max_steps: Optional[int] = None,
        max_timesteps: Optional[int] = None,
        include_depth: bool = False,
        include_instance: bool = False,
        focus_check_visible: bool = False,
        record_frames: bool = True,
        record_actions: bool = True,
        record_depth: bool = False,
        record_instance: bool = False,
        record_instance_payload: bool = False,
        slim_focus: bool = True,
    ) -> Dict[str, Any]:
        if self.representation is None:
            raise ValueError("representation is None")
        if self.memory_module is None:
            self.memory_module = Ai2ThorMemory()

        mem = self.memory_module
        mem.manage(action="reset")
        mem.manage(action="set_meta", meta={
            "fps": int(fps),
            "include_depth": bool(include_depth),
            "include_instance": bool(include_instance),
            "focus_check_visible": bool(focus_check_visible),
            "max_steps": None if max_steps is None else int(max_steps),
            "max_timesteps": None if max_timesteps is None else int(max_timesteps),
        })

        use_human_control = (policy is None)

        # Initialize representation
        obs = self.representation.get_representation({
            "mode": "init",
            "include_depth": include_depth,
            "include_instance": include_instance,
            "attach_focus": True,
            "focus_check_visible": focus_check_visible,
        })

        frame0 = obs.get("frame", None)
        if not isinstance(frame0, np.ndarray):
            raise RuntimeError("No frame from AI2-THOR (event.frame is None).")

        tick_idx: int = 0
        action_idx: int = 0

        tick_dt = 1.0 / float(fps)
        next_time = time.time()

        try:
            while True:
                if max_steps is not None and action_idx >= int(max_steps):
                    break
                if max_timesteps is not None and tick_idx >= int(max_timesteps):
                    break

                now = time.time()
                if now < next_time:
                    time.sleep(min(0.001, next_time - now))
                    continue
                next_time += tick_dt

                # Get observation from representation
                obs = self.representation.get_representation({
                    "mode": "observe",
                    "include_depth": include_depth,
                    "include_instance": include_instance,
                    "attach_focus": True,
                    "focus_check_visible": focus_check_visible,
                })

                # Record frames to memory
                if record_frames:
                    frame = obs.get("frame", None)
                    if isinstance(frame, np.ndarray):
                        payload = {"frame": frame}
                        if include_depth:
                            payload["depth_frame"] = obs.get("depth_frame", None)
                        if include_instance:
                            payload["instance_segmentation_frame"] = obs.get("instance_segmentation_frame", None)
                            payload["instance_masks"] = obs.get("instance_masks", None)
                            payload["instance_detections2D"] = obs.get("instance_detections2D", None)
                        mem.record(payload, metadata={"type": "image", "tick": int(tick_idx), "sceneName": obs.get("sceneName", "")})

                if include_depth and record_depth:
                    d = obs.get("depth_frame", None)
                    if isinstance(d, np.ndarray):
                        mem.record(d, metadata={"type": "image", "subtype": "depth", "tick": int(tick_idx)})

                if include_instance and record_instance:
                    inst = obs.get("instance_segmentation_frame", None)
                    if isinstance(inst, np.ndarray):
                        mem.record(inst, metadata={"type": "image", "subtype": "instance", "tick": int(tick_idx)})

                if include_instance and record_instance_payload:
                    payload = self._to_serializable({
                        "instance_masks": obs.get("instance_masks", None),
                        "instance_detections2D": obs.get("instance_detections2D", None),
                    })
                    mem.record(payload, metadata={"type": "other", "subtype": "instance_payload", "tick": int(tick_idx)})

                output_dict = self.process(
                    obs,
                    policy=policy,
                    use_human_control=use_human_control,
                )

                # Check for quit
                if output_dict["should_quit"]:
                    break

                actions = output_dict["actions"]
                tokens = output_dict["tokens"]

                # Execute actions via representation
                if actions:
                    for a in actions:
                        obs_after = self.representation.get_representation({
                            "mode": "step",
                            "action": a,
                            "include_depth": include_depth,
                            "include_instance": include_instance,
                            "attach_focus": True,
                            "focus_check_visible": focus_check_visible,
                        })

                        if record_actions:
                            focus_obj = None
                            focus = obs_after.get("focus", None)
                            if isinstance(focus, dict):
                                focus_obj = focus.get("object", None)

                            if slim_focus and isinstance(focus_obj, dict):
                                focus_obj = {
                                    # 原有字段
                                    "objectId": focus_obj.get("objectId"),
                                    "objectType": focus_obj.get("objectType", ""),
                                    "pickupable": focus_obj.get("pickupable", False),
                                    "openable": focus_obj.get("openable", False),
                                    "isOpen": focus_obj.get("isOpen", False),
                                    "toggleable": focus_obj.get("toggleable", False),
                                    "isToggled": focus_obj.get("isToggled", False),
                                    "receptacle": focus_obj.get("receptacle", False),
                                    "visible": focus_obj.get("visible", False),
                                    "distance": focus_obj.get("distance", None),
                                    # 新增：物品状态字段
                                    "cookable": focus_obj.get("cookable", False),
                                    "isCooked": focus_obj.get("isCooked", False),
                                    "canFillWithLiquid": focus_obj.get("canFillWithLiquid", False),
                                    "isFilledWithLiquid": focus_obj.get("isFilledWithLiquid", False),
                                    "dirtyable": focus_obj.get("dirtyable", False),
                                    "isDirty": focus_obj.get("isDirty", False),
                                    "sliceable": focus_obj.get("sliceable", False),
                                    "isSliced": focus_obj.get("isSliced", False),
                                    "breakable": focus_obj.get("breakable", False),
                                    "isBroken": focus_obj.get("isBroken", False),
                                    "canBeUsedUp": focus_obj.get("canBeUsedUp", False),
                                    "isUsedUp": focus_obj.get("isUsedUp", False),
                                }

                            mem.record(a, metadata={
                                "type": "action",
                                "tokens": list(tokens),
                                "action": a,
                                "tick": int(tick_idx),
                                "step": int(action_idx),
                                "lastActionSuccess": obs_after.get("lastActionSuccess", None),
                                "errorMessage": obs_after.get("errorMessage", ""),
                                "sceneName": obs_after.get("sceneName", ""),
                                "agent": obs_after.get("agent", {}),
                                "focus": focus_obj,
                                "inventory": obs_after.get("inventory", None),
                            })

                        action_idx += 1
                        if max_steps is not None and action_idx >= int(max_steps):
                            break

                tick_idx += 1

        finally:
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

        return {
            "export": export,
            "memory": mem,
            "tick_count": tick_idx,
            "action_count": action_idx
        }

    # ---------------- User manual saving (save ALL) ----------------
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str,
        *,
        fps: int = 20,
        save_video: bool = True,
        save_actions: bool = True,
        save_meta: bool = True,
        save_frames: bool = False,
        save_depth: bool = True,
        save_instance: bool = True,
        save_instance_payloads: bool = True,
    ):
        os.makedirs(output_dir, exist_ok=True)
        export = results.get("export", None)
        if not isinstance(export, dict):
            raise ValueError("results['export'] is missing. Call pipeline(...) first.")

        # 1) meta
        meta_path = os.path.join(output_dir, "meta.json")
        if save_meta:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(export.get("meta", {}), f, ensure_ascii=False, indent=2)

        # 2) actions
        actions_path = os.path.join(output_dir, "actions.jsonl")
        if save_actions:
            with open(actions_path, "w", encoding="utf-8") as f:
                for a in export.get("actions", []):
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")

        # 3) video (RGB -> BGR)
        video_path = os.path.join(output_dir, "video.avi")
        frames = export.get("frames_rgb", [])
        if save_video and len(frames) > 0:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            vw = cv2.VideoWriter(video_path, fourcc, float(fps), (w, h))
            if not vw.isOpened():
                raise RuntimeError(f"VideoWriter failed to open: {video_path}")
            for fr in frames:
                bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                vw.write(bgr)
            vw.release()

        # 4) optional: rgb frames
        frames_dir = None
        if save_frames and len(frames) > 0:
            frames_dir = os.path.join(output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            for i, fr in enumerate(frames):
                bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(frames_dir, f"{i:06d}.jpg"), bgr)

        # 5) depth
        depth_dir = None
        depth_frames = export.get("depth_frames", [])
        if save_depth and len(depth_frames) > 0:
            depth_dir = os.path.join(output_dir, "depth")
            os.makedirs(depth_dir, exist_ok=True)
            for i, d in enumerate(depth_frames):
                np.save(os.path.join(depth_dir, f"{i:06d}.npy"), d)

        # 6) instance segmentation frame
        instance_dir = None
        inst_frames = export.get("instance_segmentation_frames", [])
        if save_instance and len(inst_frames) > 0:
            instance_dir = os.path.join(output_dir, "instance_segmentation")
            os.makedirs(instance_dir, exist_ok=True)
            for i, inst in enumerate(inst_frames):
                cv2.imwrite(os.path.join(instance_dir, f"{i:06d}.png"), inst)

        # 7) instance payloads (masks/det2d)
        instance_payloads_path = None
        payloads = export.get("instance_payloads", [])
        if save_instance_payloads and len(payloads) > 0:
            instance_payloads_path = os.path.join(output_dir, "instance_payloads.jsonl")
            with open(instance_payloads_path, "w", encoding="utf-8") as f:
                for p in payloads:
                    f.write(json.dumps(self._to_serializable(p), ensure_ascii=False) + "\n")

        return {
            "output_dir": output_dir,
            "video_path": video_path if save_video else None,
            "actions_path": actions_path if save_actions else None,
            "meta_path": meta_path if save_meta else None,
            "frames_dir": frames_dir,
            "depth_dir": depth_dir,
            "instance_dir": instance_dir,
            "instance_payloads_path": instance_payloads_path,
        }
        