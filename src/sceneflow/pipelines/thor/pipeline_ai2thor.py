from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set

import cv2
import numpy as np
import json

from ..pipeline_utils import PipelineABC
from ...operators.ai2thor_operator import Ai2ThorOperator
from ...representations.simulation_environment.thor.ai2thor_representation import Ai2ThorRepresentation
from ...memories.simulation_environment.thor.ai2thor_memory import Ai2ThorMemory


class _InputState:
    def __init__(self):
        self.pressed: Set[str] = set()
        self.quit: bool = False
        self.save_snapshot: bool = False


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


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

    def _start_keyboard_listener(self, state: _InputState):
        try:
            from pynput import keyboard
        except Exception as e:
            raise RuntimeError("pynput is required for keyboard. Install with: pip install pynput") from e

        def on_press(key):
            if key == keyboard.Key.esc:
                state.quit = True
                return False
            try:
                k = key.char.lower()
                if k == "p":
                    state.save_snapshot = True
                state.pressed.add(k)
            except Exception:
                pass

        def on_release(key):
            try:
                k = key.char.lower()
                state.pressed.discard(k)
            except Exception:
                pass

        kb = keyboard.Listener(on_press=on_press, on_release=on_release)
        kb.start()
        return kb

    @staticmethod
    def _draw_crosshair(img: np.ndarray, cx: int, cy: int, size: int = 10, thickness: int = 1) -> None:
        cv2.line(img, (cx - size, cy), (cx + size, cy), (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy - size), (cx, cy + size), (255, 255, 255), thickness, cv2.LINE_AA)

    def _inputs_to_tokens(
        self,
        pressed_keys: Set[str],
        mouse: Dict[str, Any],
        *,
        rot_step_pixels: float,
        look_step_pixels: float,
        max_yaw_per_tick: int,
        max_pitch_per_tick: int,
    ) -> List[str]:
        tokens: List[str] = []

        if "w" in pressed_keys:
            tokens.append("forward")
        elif "s" in pressed_keys:
            tokens.append("backward")
        elif "a" in pressed_keys:
            tokens.append("left")
        elif "d" in pressed_keys:
            tokens.append("right")

        dx = float(mouse.get("dx", 0.0))
        dy = float(mouse.get("dy", 0.0))

        deadzone = 0.5
        if abs(dx) < deadzone:
            dx = 0.0
        if abs(dy) < deadzone:
            dy = 0.0

        if abs(dx) >= float(rot_step_pixels):
            n = min(int(abs(dx) // float(rot_step_pixels)), int(max_yaw_per_tick))
            if n > 0:
                tok = "camera_r" if dx > 0 else "camera_l"
                tokens.extend([tok] * n)
                mouse["dx"] = dx - (_sign(dx) * n * float(rot_step_pixels))
        else:
            mouse["dx"] = dx

        if abs(dy) >= float(look_step_pixels):
            n = min(int(abs(dy) // float(look_step_pixels)), int(max_pitch_per_tick))
            if n > 0:
                tok = "camera_up" if dy < 0 else "camera_down"
                tokens.extend([tok] * n)
                mouse["dy"] = dy - (_sign(dy) * n * float(look_step_pixels))
        else:
            mouse["dy"] = dy

        return tokens

    def process(
        self,
        *,
        policy: Optional[Callable[[Dict[str, Any]], List[str]]] = None,
        fps: int = 20,
        max_steps: Optional[int] = 200,          # ticks
        max_actions: Optional[int] = None,

        include_depth: bool = False,
        include_instance: bool = False,

        window_name: str = "thor",
        show_window: bool = True,

        rot_step_pixels: float = 6.0,
        look_step_pixels: float = 8.0,
        max_yaw_per_tick: int = 3,
        max_pitch_per_tick: int = 2,

        overlay_crosshair: bool = True,
        overlay_error: bool = True,
        overlay_focus: bool = True,
        focus_check_visible: bool = False,

        record_frames: bool = True,
        record_actions: bool = True,
        record_depth: bool = False,            # 可选：把 depth 当 image(subtype=depth) 存入 memory
        record_instance: bool = False,         # 可选：把 instance seg 当 image(subtype=instance) 存入 memory
        record_instance_payload: bool = False, # 可选：把 masks/detections 当 other(subtype=instance_payload)
        slim_focus: bool = True,
    ) -> Dict[str, Any]:
        if self.representation is None:
            raise ValueError("representation is None")
        if self.operators is None:
            raise ValueError("operators is None")
        if self.memory_module is None:
            self.memory_module = Ai2ThorMemory()

        mem = self.memory_module

        mem.manage(action="reset")
        mem.manage(action="set_meta", meta={
            "fps": int(fps),
            "window_name": str(window_name),
            "include_depth": bool(include_depth),
            "include_instance": bool(include_instance),
            "focus_check_visible": bool(focus_check_visible),
        })

        tick_idx: int = 0
        action_idx: int = 0

        # ---- init env (ONLY get_representation) ----
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

        if show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        mouse: Dict[str, Any] = {"last": None, "dx": 0.0, "dy": 0.0, "click": False}

        def on_mouse(evt, x, y, flags, param):
            if evt == cv2.EVENT_MOUSEMOVE:
                if mouse["last"] is None:
                    mouse["last"] = (x, y)
                    return
                lx, ly = mouse["last"]
                mouse["dx"] += (x - lx)
                mouse["dy"] += (y - ly)
                mouse["last"] = (x, y)
            elif evt in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                mouse["last"] = (x, y)
            if evt == cv2.EVENT_LBUTTONUP:
                mouse["click"] = True

        if show_window and policy is None:
            cv2.setMouseCallback(window_name, on_mouse)

        state = _InputState()
        kb_listener = None
        if policy is None:
            kb_listener = self._start_keyboard_listener(state)

        tick_dt = 1.0 / float(fps)
        next_time = time.time()

        last_percep: Dict[str, Any] = {}

        try:
            while True:
                # quit
                if show_window:
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        state.quit = True
                if state.quit:
                    break

                if max_steps is not None and tick_idx >= int(max_steps):
                    break
                if max_actions is not None and action_idx >= int(max_actions):
                    break

                now = time.time()
                if now < next_time:
                    time.sleep(min(0.001, next_time - now))
                    continue
                next_time += tick_dt

                # observe (ONLY get_representation)
                obs = self.representation.get_representation({
                    "mode": "observe",
                    "include_depth": include_depth,
                    "include_instance": include_instance,
                    "attach_focus": True,
                    "focus_check_visible": focus_check_visible,
                })

                # perception update (ONLY BaseOperator template)
                last_percep = self.operators.process_perception(obs)

                frame = obs.get("frame", None)
                frame_bgr = None
                if isinstance(frame, np.ndarray):
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    if overlay_crosshair:
                        h, w = frame_bgr.shape[:2]
                        self._draw_crosshair(frame_bgr, w // 2, h // 2)

                    if overlay_focus:
                        fid = last_percep.get("focus_object_id", None)
                        ftype = last_percep.get("focus_object_type", "")
                        txt = f"FOCUS: {ftype} | {str(fid)[:48]}" if fid else "FOCUS: None"
                        cv2.putText(frame_bgr, txt, (10, 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                    if overlay_error:
                        if not obs.get("lastActionSuccess", True):
                            err = str(obs.get("errorMessage", "") or "")
                            if err:
                                cv2.putText(frame_bgr, f"ERR: {err[:80]}", (10, 44),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                    if show_window:
                        cv2.imshow(window_name, frame_bgr)

                # record frames
                if record_frames and isinstance(frame, np.ndarray):
                    payload = {"frame": frame}

                    if include_depth:
                        payload["depth_frame"] = obs.get("depth_frame", None)

                    if include_instance:
                        payload["instance_segmentation_frame"] = obs.get("instance_segmentation_frame", None)
                        payload["instance_masks"] = obs.get("instance_masks", None)
                        payload["instance_detections2D"] = obs.get("instance_detections2D", None)

                    mem.record(
                        payload,
                        metadata={
                            "type": "image",
                            "tick": int(tick_idx),
                            "sceneName": obs.get("sceneName", ""),
                        },
                    )

                if include_depth and record_depth:
                    d = obs.get("depth_frame", None)
                    if isinstance(d, np.ndarray):
                        mem.record(d, metadata={
                            "type": "image",
                            "subtype": "depth",
                            "tick": int(tick_idx),
                        })

                if include_instance and record_instance:
                    inst = obs.get("instance_segmentation_frame", None)
                    if isinstance(inst, np.ndarray):
                        mem.record(inst, metadata={
                            "type": "image",
                            "subtype": "instance",
                            "tick": int(tick_idx),
                        })

                if include_instance and record_instance_payload:
                    payload = {
                        "instance_masks": obs.get("instance_masks", None),
                        "instance_detections2D": obs.get("instance_detections2D", None),
                    }
                    mem.record(payload, metadata={
                        "type": "other",
                        "subtype": "instance_payload",
                        "tick": int(tick_idx),
                    })

                # optional: snapshot key = manual jpg save (still allowed)
                if policy is None and state.save_snapshot and show_window and frame_bgr is not None:
                    state.save_snapshot = False
                    snap = {"frame_bgr": frame_bgr.copy(), "tick": int(tick_idx)}
                    mem.record(snap, metadata={"type": "other", "subtype": "snapshot", "tick": int(tick_idx)})

                # decide tokens
                tokens: List[str] = []
                mode = "agent" if policy is not None else "human"

                if policy is None:
                    if mouse.get("click", False):
                        mouse["click"] = False
                        tokens.append("interact")
                    tokens.extend(self._inputs_to_tokens(
                        pressed_keys=state.pressed,
                        mouse=mouse,
                        rot_step_pixels=rot_step_pixels,
                        look_step_pixels=look_step_pixels,
                        max_yaw_per_tick=max_yaw_per_tick,
                        max_pitch_per_tick=max_pitch_per_tick,
                    ))
                else:
                    out = policy(obs)
                    tokens = [str(t) for t in out] if isinstance(out, list) else []

                # tokens -> actions (ONLY BaseOperator template)
                actions: List[Dict[str, Any]] = []
                raycast = None
                if tokens:
                    mem.record(list(tokens), metadata={
                        "type": "other",
                        "subtype": "interaction",
                        "tick": int(tick_idx),
                        "mode": mode,
                    })

                    self.operators.get_interaction(tokens)

                    if "interact" in tokens:
                        q = self.representation.get_representation({
                            "mode": "query",
                            "query": "raycast",
                            "x": 0.5,
                            "y": 0.78,
                        })
                        ar = q.get("actionReturn", None)
                        if isinstance(ar, dict) and all(k in ar for k in ("x", "y", "z")):
                            raycast = {"x": float(ar["x"]), "y": float(ar["y"]), "z": float(ar["z"])}

                    actions = self.operators.process_interaction(raycast=raycast)
                else:
                    actions = []

                # execute actions (ONLY get_representation)
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

                        # slim focus (来自 obs_after["focus"]["object"])
                        focus_obj = None
                        focus = obs_after.get("focus", None)
                        if isinstance(focus, dict):
                            focus_obj = focus.get("object", None)

                        if slim_focus and isinstance(focus_obj, dict):
                            focus_obj = {
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
                            }

                        if record_actions:
                            mem.record(a, metadata={
                                "type": "action",
                                "mode": mode,
                                "tokens": list(tokens),
                                "action": a,
                                "tick": int(tick_idx),
                                "lastActionSuccess": obs_after.get("lastActionSuccess", None),
                                "errorMessage": obs_after.get("errorMessage", ""),
                                "sceneName": obs_after.get("sceneName", ""),
                                "agent": obs_after.get("agent", {}),
                                "focus": focus_obj,
                                "inventory": obs_after.get("inventory", None),
                            })
                            action_idx += 1

                tick_idx += 1

        finally:
            try:
                if kb_listener is not None:
                    kb_listener.stop()
            except Exception:
                pass
            try:
                if show_window:
                    cv2.destroyAllWindows()
            except Exception:
                pass
            try:
                if self.representation is not None:
                    self.representation.get_representation({"mode": "close"})
            except Exception:
                pass

        export = mem.process(None, target_format="export")

        return {
            "export": export,
            "memory": mem,
        }

    def __call__(self, *args, **kwds):
        return self.process(*args, **kwds)

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
        save_frames: bool = False,            # 可选：逐帧 rgb
        save_depth: bool = True,              # 深度图，可选
        save_instance: bool = True,           # 分割图
        save_instance_payloads: bool = True,  # masks/det2d json
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
                # 常见 depth 是 float32，保存成 npy 
                np.save(os.path.join(depth_dir, f"{i:06d}.npy"), d)

        # 6) instance segmentation frame
        instance_dir = None
        inst_frames = export.get("instance_segmentation_frames", [])
        if save_instance and len(inst_frames) > 0:
            instance_dir = os.path.join(output_dir, "instance_segmentation")
            os.makedirs(instance_dir, exist_ok=True)
            for i, inst in enumerate(inst_frames):
                # 可能是彩色或label图；用 png 保存
                cv2.imwrite(os.path.join(instance_dir, f"{i:06d}.png"), inst)

        # 7) instance payloads (masks/det2d)
        instance_payloads_path = None
        payloads = export.get("instance_payloads", [])
        if save_instance_payloads and len(payloads) > 0:
            instance_payloads_path = os.path.join(output_dir, "instance_payloads.jsonl")
            with open(instance_payloads_path, "w", encoding="utf-8") as f:
                for p in payloads:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

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
