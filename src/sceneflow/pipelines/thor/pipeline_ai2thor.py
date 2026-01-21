from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional

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
    OpenCV-interactive pipeline.

    Controls:
    - W/A/S/D: move (operator decides mapping -> AI2-THOR actions)
    - Hold mouse button + drag:
        dx -> RotateLeft/RotateRight (via operator)
        dy -> LookUp/LookDown (via operator)
    - ESC: quit
    - P: snapshot
    - Q: quit (in OpenCV window)

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

    def run_interactive(
        self,
        output_dir: str,
        fps: int = 20,
        save_frames: bool = False,
        max_steps: Optional[int] = None,
        include_depth: bool = False,
        include_instance: bool = False,
        save_frame_every: int = 3,
        flush_every: int = 30,
        window_name: str = "thor",
        rot_step_pixels: float = 6.0,
        look_step_pixels: float = 8.0,
        mouse_button: str = "lmb",  # "lmb" or "rmb"
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

        # ---- OpenCV window + mouse callback ----
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        mouse: Dict[str, Any] = {
            "down": False,
            "last": None,
            "dx": 0.0,
            "dy": 0.0,
        }

        if mouse_button.lower() not in ("lmb", "rmb"):
            raise ValueError("mouse_button must be 'lmb' or 'rmb'")

        def _is_pressed(flags: int) -> bool:
            if mouse_button.lower() == "lmb":
                return bool(flags & cv2.EVENT_FLAG_LBUTTON)
            return bool(flags & cv2.EVENT_FLAG_RBUTTON)

        def on_mouse(evt, x, y, flags, param):
            down = _is_pressed(flags)
            mouse["down"] = down

            if evt == cv2.EVENT_MOUSEMOVE and down:
                if mouse["last"] is None:
                    mouse["last"] = (x, y)
                    return
                lx, ly = mouse["last"]
                mouse["dx"] += (x - lx)
                mouse["dy"] += (y - ly)
                mouse["last"] = (x, y)
            else:
                mouse["last"] = None  # 松开/非拖动时，清掉 last 防跳变

        cv2.setMouseCallback(window_name, on_mouse)

        # ---- video writer ----
        video_path = os.path.join(output_dir, "video.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter failed to open: {video_path}")

        # ---- logs ----
        log_f = open(os.path.join(output_dir, "actions.jsonl"), "w", encoding="utf-8")

        # ---- keyboard listener ----
        state = _InputState()
        kb_listener = self._start_keyboard_listener(state)

        step_idx = 0
        tick = 1.0 / float(fps)
        next_time = time.time()

        try:
            while True:
                # 必须要 waitKey(1) 才会派发鼠标事件；也支持窗口内按 q 退出
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    state.quit = True

                if state.quit:
                    break
                if max_steps is not None and step_idx >= max_steps:
                    break

                now = time.time()
                if now < next_time:
                    time.sleep(min(0.001, next_time - now))
                    continue
                next_time += tick

                # ---- always render current frame to window/video ----
                obs = self.representation.get_representation(
                    event, include_depth=include_depth, include_instance=include_instance
                )
                frame = obs["frame"]
                if frame is None:
                    continue
                frame_bgr = self._rgb_to_bgr_uint8(frame, w, h)

                cv2.imshow(window_name, frame_bgr)
                writer.write(frame_bgr)

                # snapshot（不依赖 action）
                if state.save_snapshot:
                    snap_path = os.path.join(output_dir, f"snapshot_{int(time.time())}.jpg")
                    cv2.imwrite(
                        snap_path,
                        frame_bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                    state.save_snapshot = False

                # ---- inputs -> actions (DONE IN OPERATOR) ----
                actions = self.operators.inputs_to_actions(
                    pressed_keys=state.pressed,
                    mouse=mouse,
                    rot_step_pixels=rot_step_pixels,
                    look_step_pixels=look_step_pixels,
                )

                # no action this tick -> continue
                if len(actions) == 0:
                    step_idx += 1
                    continue

                for a in actions:
                    event = self.representation.step(a)
                    obs2 = self.representation.get_representation(
                        event, include_depth=include_depth, include_instance=include_instance
                    )

                    # 可选：LookUp/Down 失败就清 dy，避免一直撞 pitch 上限抖动
                    if a.get("action") in ("LookUp", "LookDown") and not obs2.get("lastActionSuccess", True):
                        mouse["dy"] = 0.0

                    md = getattr(event, "metadata", {}) or {}
                    record = {
                        "step": step_idx,
                        "time": time.time(),
                        # 记录原始输入 + 本次执行动作
                        "pressed": sorted(list(state.pressed)),
                        "mouse": {
                            "down": bool(mouse.get("down", False)),
                            "dx": float(mouse.get("dx", 0.0)),
                            "dy": float(mouse.get("dy", 0.0)),
                        },
                        "action": a,
                        "lastActionSuccess": obs2.get("lastActionSuccess"),
                        "errorMessage": obs2.get("errorMessage", ""),
                        "sceneName": obs2.get("sceneName", ""),
                        "agent": obs2.get("agent", {}),
                        "actionReturn": md.get("actionReturn", None),
                    }
                    log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if step_idx % max(1, flush_every) == 0:
                        log_f.flush()

                    # optional save frame（用当前 obs2 的 frame）
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

    def __call__(
        self,
        output_dir: str,
        fps: int = 20,
        save_frames: bool = False,
        max_steps: Optional[int] = None,
        include_depth: bool = False,
        include_instance: bool = False,
        save_frame_every: int = 3,
        flush_every: int = 30,
        window_name: str = "thor",
        rot_step_pixels: float = 6.0,
        look_step_pixels: float = 8.0,
        mouse_button: str = "lmb",
    ) -> None:
        return self.run_interactive(
            output_dir=output_dir,
            fps=fps,
            save_frames=save_frames,
            max_steps=max_steps,
            include_depth=include_depth,
            include_instance=include_instance,
            save_frame_every=save_frame_every,
            flush_every=flush_every,
            window_name=window_name,
            rot_step_pixels=rot_step_pixels,
            look_step_pixels=look_step_pixels,
            mouse_button=mouse_button,
        )
