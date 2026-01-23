from __future__ import annotations

import os
import json
import time
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from ...operators.ai2thor_operator import Ai2ThorOperator
from ...representations.simulation_environment.thor.ai2thor_representation import Ai2ThorRepresentation


class _InputState:
    """Keyboard state only (mouse handled by OpenCV window)."""
    def __init__(self):
        self.pressed = set()
        self.quit = False
        self.save_snapshot = False


class Ai2ThorPipeline:
    """
    Unified pipeline.

    Modes:
    - Human (policy=None):
        W/A/S/D move; mouse move -> look; LMB up -> click_to_action()
        ESC quits, P snapshot, Q (in cv2 window) quits.
    - MLLM/Agent (policy!=None):
        policy(obs)->tokens; tokens action space:
          forward/backward/left/right/camera_l/camera_r/camera_up/camera_down/interact
        Q quits in cv2 window.

    Outputs:
      - video.avi
      - actions.jsonl
      - frames/*.jpg (optional)
    """

    def __init__(self, operators: Ai2ThorOperator, representation: Ai2ThorRepresentation):
        self.operators = operators
        self.representation = representation

    @staticmethod
    def _ensure_dir(p: str) -> None:
        os.makedirs(p, exist_ok=True)

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
    def _rgb_to_bgr_uint8(frame_rgb: np.ndarray, w: int, h: int) -> np.ndarray:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if frame_bgr.shape[1] != w or frame_bgr.shape[0] != h:
            frame_bgr = cv2.resize(frame_bgr, (w, h))
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)
        return frame_bgr

    @staticmethod
    def _draw_crosshair(img: np.ndarray, cx: int, cy: int, size: int = 10, thickness: int = 1) -> None:
        cv2.line(img, (cx - size, cy), (cx + size, cy), (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy - size), (cx, cy + size), (255, 255, 255), thickness, cv2.LINE_AA)

    def run(
        self,
        output_dir: str,
        *,
        policy: Optional[Callable[[Dict[str, Any]], List[str]]] = None,  # None=human；否则=agent(token)模式

        fps: int = 20,                               # 渲染/执行节奏（影响控制灵敏度 & 视频帧率）
        max_steps: Optional[int] = None,             # 最大 step 数；None=不限（一般按 q/esc 退出）

        save_frames: bool = False,                   # 是否额外保存逐帧 jpg（调试用，体积大）
        save_frame_every: int = 3,                   # 每 N step 存一张（越小越密）

        include_depth: bool = False,                 # obs 里是否带 depth（开了会更慢/更占存储）
        include_instance: bool = False,              # obs 里是否带 instance seg（开了会更慢）

        flush_every: int = 30,                       # actions.jsonl 每 N step flush（防崩溃丢日志）
        window_name: str = "thor",                   # OpenCV 窗口名（多实例时区分）

        rot_step_pixels: float = 6.0,                # 人类模式：鼠标 dx 达到多少像素触发一次 yaw
        look_step_pixels: float = 8.0,               # 人类模式：鼠标 dy 达到多少像素触发一次 pitch

        overlay_focus: bool = True,                  # 画面叠加 focus 信息（闭环 debug 很有用）
        overlay_error: bool = True,                  # 画面叠加 lastActionSuccess/errorMessage
        overlay_crosshair: bool = True,              # 画面中心准星（方便对准 focus）
    ) -> None:

        self._ensure_dir(output_dir)
        frames_dir = os.path.join(output_dir, "frames")
        if save_frames:
            self._ensure_dir(frames_dir)

        # ---- init environment ----
        self.representation.controller_init()
        controller = self.representation.controller
        assert controller is not None
        event = controller.last_event

        first_frame = getattr(event, "frame", None)
        if first_frame is None:
            raise RuntimeError("No frame from AI2-THOR (event.frame is None).")

        h, w = first_frame.shape[:2]

        # ---- OpenCV window ----
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # ---- Human-mode mouse state ----
        mouse: Dict[str, Any] = {
            "last": None,
            "dx": 0.0,
            "dy": 0.0,
            "click": False,
            "click_time": 0.0,
        }

        def on_mouse(evt, x, y, flags, param):
            # accumulate dx/dy on move
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

            # click interaction
            if evt == cv2.EVENT_LBUTTONUP:
                mouse["click"] = True
                mouse["click_time"] = time.time()

        # Only set mouse callback in human mode (optional, but keeps things clean)
        if policy is None:
            cv2.setMouseCallback(window_name, on_mouse)

        # ---- video writer ----
        video_path = os.path.join(output_dir, "video.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter failed to open: {video_path}")

        # ---- logs ----
        log_f = open(os.path.join(output_dir, "actions.jsonl"), "w", encoding="utf-8")

        def _log(step: int, mode: str, action: Dict[str, Any], ev_after: Any, extra: Dict[str, Any]):
            md = getattr(ev_after, "metadata", {}) or {}
            record = {
                "step": step,
                "time": time.time(),
                "mode": mode,  # "human" or "agent"
                "action": action,
                "lastActionSuccess": md.get("lastActionSuccess", None),
                "errorMessage": md.get("errorMessage", ""),
                "sceneName": md.get("sceneName", ""),
                "agent": md.get("agent", {}),
                "actionReturn": md.get("actionReturn", None),
                **extra,
            }
            log_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # ---- keyboard listener (human mode only) ----
        state = _InputState()
        kb_listener = None
        if policy is None:
            kb_listener = self._start_keyboard_listener(state)

        step_idx = 0
        tick = 1.0 / float(fps)
        next_time = time.time()

        try:
            while True:
                # cv2 window key
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    state.quit = True

                if state.quit:
                    break
                if max_steps is not None and step_idx >= int(max_steps):
                    break

                now = time.time()
                if now < next_time:
                    time.sleep(min(0.001, next_time - now))
                    continue
                next_time += tick

                # ---- update focus each tick ----
                self.operators.update_focus(self.representation, event)

                # ---- get obs + frame ----
                obs = self.representation.get_representation(
                    event, include_depth=include_depth, include_instance=include_instance
                )
                
                # 把 focus 信息也给 policy（闭环需要）
                obs["focus"] = self.operators.get_focus_info_for_overlay()  # 可能是 None
                obs["agent"] = (getattr(event, "metadata", {}) or {}).get("agent", {})
                
                frame = obs.get("frame", None)
                if frame is None:
                    step_idx += 1
                    continue

                frame_bgr = self._rgb_to_bgr_uint8(frame, w, h)

                # ---- overlays ----
                if overlay_crosshair:
                    self._draw_crosshair(frame_bgr, w // 2, h // 2, size=10, thickness=1)

                if overlay_focus:
                    info = self.operators.get_focus_info_for_overlay()
                    if info is not None:
                        t1 = f'FOCUS: {info.get("objectType","")} | {str(info.get("objectId",""))[:48]}'
                        t2 = (
                            f'pickup:{info.get("pickupable")} open:{info.get("openable")} '
                            f'toggle:{info.get("toggleable")} recept:{info.get("receptacle")} dist:{info.get("distance")}'
                        )
                        cv2.putText(frame_bgr, t1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame_bgr, t2, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame_bgr, "FOCUS: None", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                if overlay_error:
                    md_now = getattr(event, "metadata", {}) or {}
                    if not md_now.get("lastActionSuccess", True):
                        err = str(md_now.get("errorMessage", "") or "")
                        if err:
                            cv2.putText(frame_bgr, f"ERR: {err[:80]}", (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                # ---- show + record ----
                cv2.imshow(window_name, frame_bgr)
                writer.write(frame_bgr)

                # ---- snapshot (human mode) ----
                if policy is None and state.save_snapshot:
                    snap_path = os.path.join(output_dir, f"snapshot_{int(time.time())}.jpg")
                    cv2.imwrite(snap_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    state.save_snapshot = False

                # ---- decide actions ----
                actions: List[Dict[str, Any]] = []
                mode = "agent" if policy is not None else "human"
                extra_for_log: Dict[str, Any] = {}

                if policy is None:
                    # 1) click action
                    if mouse.get("click", False):
                        mouse["click"] = False
                        a_click = self.operators.click_to_action(self.representation, event)
                        if a_click is not None:
                            actions.append(a_click)
                            extra_for_log = {
                                "kind": "click",
                                "pressed": sorted(list(state.pressed)),
                                "mouse": {
                                    "dx": float(mouse.get("dx", 0.0)),
                                    "dy": float(mouse.get("dy", 0.0)),
                                },
                            }

                    # 2) nav actions
                    nav_actions = self.operators.inputs_to_actions(
                        pressed_keys=state.pressed,
                        mouse=mouse,
                        rot_step_pixels=rot_step_pixels,
                        look_step_pixels=look_step_pixels,
                    )
                    if nav_actions:
                        # if we already had click, keep kind mixed; else kind nav
                        if "kind" not in extra_for_log:
                            extra_for_log = {
                                "kind": "nav",
                                "pressed": sorted(list(state.pressed)),
                                "mouse": {
                                    "dx": float(mouse.get("dx", 0.0)),
                                    "dy": float(mouse.get("dy", 0.0)),
                                },
                            }
                        actions.extend(nav_actions)

                else:
                    # Agent mode: policy(obs)->tokens -> operator.process_interaction(...)
                    tokens = policy(obs)
                    if not isinstance(tokens, list):
                        tokens = []
                    tokens = [str(t) for t in tokens]

                    # put tokens into operator buffer then translate to actions
                    self.operators.get_interaction(tokens)
                    actions = self.operators.process_interaction(self.representation, event)

                    extra_for_log = {"tokens": tokens}

                # ---- execute actions ----
                if not actions:
                    step_idx += 1
                    continue

                for a in actions:
                    event = self.representation.step(a)
                    obs2 = self.representation.get_representation(
                        event, include_depth=include_depth, include_instance=include_instance
                    )

                    # if look failed, clear dy (human mode safety)
                    if policy is None and a.get("action") in ("LookUp", "LookDown") and not obs2.get("lastActionSuccess", True):
                        mouse["dy"] = 0.0

                    _log(step_idx, mode, a, event, extra_for_log)

                    if step_idx % max(1, flush_every) == 0:
                        log_f.flush()

                    if save_frames and (step_idx % max(1, save_frame_every) == 0):
                        f2 = obs2.get("frame", None)
                        if f2 is not None:
                            f2b = self._rgb_to_bgr_uint8(f2, w, h)
                            cv2.imwrite(
                                os.path.join(frames_dir, f"{step_idx:06d}.jpg"),
                                f2b,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                            )

                    step_idx += 1

        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            try:
                if kb_listener is not None:
                    kb_listener.stop()
            except Exception:
                pass
            try:
                log_f.close()
            except Exception:
                pass
            try:
                writer.release()
            except Exception:
                pass
            try:
                self.representation.close()
            except Exception:
                pass
            cv2.destroyAllWindows()

    def __call__(self, output_dir: str, **kwargs) -> None:
        return self.run(output_dir=output_dir, **kwargs)
