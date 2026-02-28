from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import time
import cv2
import numpy as np

from .base_operator import BaseOperator

ActionSpec = Dict[str, Any]
Token = str


class SimWorldOperator(BaseOperator):
    def __init__(
        self,
        operation_types: Optional[List[str]] = None,
        interaction_template: Optional[List[str]] = None,

        # 移动参数
        default_speed: float = 200.0,
        step_duration: float = 1,       # human control 每步持续秒数
        rotate_angle: float = 90.0,       # 转向角度（度）
        camera_rotate_angle: float = 30.0, # 相机旋转角度（度）

        # Scooter 驾驶参数
        scooter_throttle: float = 0.7,    # 油门强度 [0, 1]
        scooter_brake: float = 1.0,       # 刹车强度 [0, 1]
        scooter_steer: float = 0.5,       # 转向强度 [0, 1]

        # 功能开关
        enable_pick_up: bool = True,
        enable_scooter: bool = True,
        enable_vehicle: bool = True,
        enable_social: bool = True,
        enable_path: bool = True,

        # 键盘映射
        keyboard_mapping: Optional[Dict[str, str]] = None,

        # Human control UI
        human_window_name: str = "SimWorld Human Control",
        human_window_size: int = 600,
        draw_crosshair: bool = True,

        # Debug 观测窗口（纯显示）
        show_debug_window: bool = False,
        debug_window_name: str = "SimWorld Observation",
        debug_window_size: Optional[int] = None,
    ):
        super().__init__(operation_types=[] if operation_types is None else operation_types)

        if operation_types is None:
            operation_types = ["action_instruction"]
        self.operation_types = operation_types

        # 默认完整 token 模板
        default_template = self._build_default_template(
            enable_pick_up, enable_scooter, enable_vehicle,
            enable_social, enable_path,
        )
        if interaction_template is None:
            interaction_template = default_template
        else:
            it = list(interaction_template)
            for basic in ["forward", "stop"]:
                if basic not in it:
                    it.append(basic)
            interaction_template = it

        self.interaction_template = interaction_template
        self.interaction_template_init()

        # 参数
        self.default_speed = float(default_speed)
        self.step_duration = float(step_duration)
        self.rotate_angle = float(rotate_angle)
        self.camera_rotate_angle = float(camera_rotate_angle)

        # Scooter 驾驶参数
        self.scooter_throttle = float(scooter_throttle)
        self.scooter_brake = float(scooter_brake)
        self.scooter_steer = float(scooter_steer)

        self.enable_pick_up = bool(enable_pick_up)
        self.enable_scooter = bool(enable_scooter)
        self.enable_vehicle = bool(enable_vehicle)
        self.enable_social = bool(enable_social)
        self.enable_path = bool(enable_path)
        self._last_action_time: Dict[str, float] = {}
        self._action_debounce_tokens: set = {
            "get_on_scooter", "get_off_scooter",
            "enter_vehicle", "exit_vehicle",
        }
        self._action_debounce_interval: float = 2.0  # 秒，下车动作冷却时间

        # 键盘映射
        if keyboard_mapping is None:
            keyboard_mapping = self._build_default_keyboard_mapping()
        self.keyboard_mapping = keyboard_mapping

        # Human control UI
        self.human_window_name = str(human_window_name)
        self.human_window_size = int(human_window_size)
        self.draw_crosshair = bool(draw_crosshair)
        self._human_window_inited = False

        # Debug 观测窗口
        self.show_debug_window = bool(show_debug_window)
        self.debug_window_name = str(debug_window_name)
        self.debug_window_size = debug_window_size
        self._debug_window_inited = False

        # 感知状态（由 process_perception 更新）
        self._position: Optional[Dict[str, float]] = None
        self._direction: Optional[Dict[str, float]] = None
        self._yaw: Optional[float] = None
        self._collision: Optional[Dict[str, int]] = None
        self._has_object: bool = False
        self._on_scooter: bool = False
        self._in_vehicle: bool = False

        # 调试计数
        self._tick_idx: int = 0
        self._action_idx: int = 0

        # 避免重复处理历史
        self._last_processed_history_len: int = 0

        # 当前步进模式
        self._current_use_step_mode: bool = True

    @staticmethod
    def _build_default_template(
        enable_pick_up: bool,
        enable_scooter: bool,
        enable_vehicle: bool,
        enable_social: bool,
        enable_path: bool,
    ) -> List[str]:
        tpl = [
            # 移动
            "forward", "backward", "left", "right",
            "set_speed",
            # 相机旋转
            "camera_l", "camera_r",
            # 姿态
            "sit", "stand", 
            "stop_action", "stop", "rescan",
        ]
        if enable_pick_up:
            tpl += ["pick_up", "drop_object"]
        if enable_scooter:
            tpl += ["get_on_scooter", "get_off_scooter"]
        if enable_vehicle:
            tpl += ["enter_vehicle", "exit_vehicle"]
        if enable_social:
            tpl += ["argue", "discuss", "listen", "wave_to_dog", "directing_path"]
        if enable_path:
            tpl += ["follow_path", "set_path"]
        return tpl

    def _build_default_keyboard_mapping(self) -> Dict[str, str]:
        mapping = {
            # 移动
            "w": "forward",
            "s": "backward",
            "a": "left",
            "d": "right",
            # 相机旋转
            "j": "camera_l",
            "l": "camera_r",
            # 姿态
            "f": "sit",
            "g": "stand",
            "x": "stop_action",
            "c": "stop",
            # 物品
            "p": "pick_up",
            "r": "drop_object",
            "z": "rescan",
        }
        if self.enable_scooter:
            mapping["v"] = "get_on_scooter"
            mapping["b"] = "get_off_scooter"
        if self.enable_vehicle:
            mapping["n"] = "enter_vehicle"
            mapping["m"] = "exit_vehicle"
        if self.enable_social:
            mapping["1"] = "argue"
            mapping["2"] = "discuss"
            mapping["3"] = "listen"
            mapping["4"] = "wave_to_dog"
            mapping["5"] = "directing_path"
        if self.enable_path:
            mapping["h"] = "follow_path"
        return mapping

    def _ensure_human_window(self) -> None:
        if self._human_window_inited:
            return
        cv2.namedWindow(self.human_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.human_window_name, self.human_window_size, self.human_window_size)
        self._human_window_inited = True

    def _close_human_window(self) -> None:
        try:
            cv2.destroyWindow(self.human_window_name)
        except Exception:
            pass
        self._human_window_inited = False

    def _render_and_poll_key(self, frame_rgb: np.ndarray) -> int:
        self._ensure_human_window()
        display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if self.draw_crosshair:
            h, w = display.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.line(display, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 2)
            cv2.line(display, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

        # 在画面上显示当前驾驶模式
        mode_text = ""
        if self._on_scooter:
            mode_text = "[SCOOTER MODE] W/S=油门/刹车  A/D=转向  B=下车"
        elif self._in_vehicle:
            mode_text = "[PASSENGER MODE] M=下车"
        if mode_text:
            cv2.putText(display, mode_text, (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(self.human_window_name, display)
        return cv2.waitKey(1)

    def _ensure_debug_window(self) -> None:
        if self._debug_window_inited:
            return
        cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
        if self.debug_window_size is not None:
            cv2.resizeWindow(self.debug_window_name, self.debug_window_size, self.debug_window_size)
        self._debug_window_inited = True

    def _close_debug_window(self) -> None:
        try:
            cv2.destroyWindow(self.debug_window_name)
        except Exception:
            pass
        self._debug_window_inited = False

    def _render_debug_frame(self, frame_rgb: np.ndarray) -> None:
        self._ensure_debug_window()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        pos = self._position or {}
        txt = (
            f"Tick:{self._tick_idx}  Action:{self._action_idx}  "
            f"Pos:({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f})"
        )
        cv2.putText(frame_bgr, txt, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.debug_window_name, frame_bgr)
        cv2.waitKey(1)

    def close_all_windows(self) -> None:
        """关闭所有窗口（由 process(mode='close') 统一调用）。"""
        self._close_human_window()
        self._close_debug_window()

    def update_counters(self, tick_idx: int, action_idx: int) -> None:
        """同步 Pipeline 的 tick/action 计数，供调试窗口叠加显示。"""
        self._tick_idx = int(tick_idx)
        self._action_idx = int(action_idx)

    def process_perception(self, obs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        更新感知状态，并按需渲染调试窗口。

        obs 字段说明（来自 SimWorldRepresentation）：
        position   : {"x": float, "y": float}
        direction  : {"x": float, "y": float}
        yaw        : float
        collision  : {"human": int, "object": int, "building": int, "vehicle": int}
        rgb        : np.ndarray（可选）
        has_object : bool（可选，UE 侧返回）
        on_scooter : bool（可选）状态变化时打印 [OP] Mode changed → SCOOTER/HUMANOID
        in_vehicle : bool（可选）状态变化时打印 [OP] Mode changed → VEHICLE/HUMANOID
        scooter_id : int（可选）当前 scooter id，透传给 get_off_scooter
        """
        self._position = obs.get("position", None)
        self._direction = obs.get("direction", None)
        self._yaw = obs.get("yaw", None)
        self._collision = obs.get("collision", None)

        # 从 obs 更新持有/骑行/乘车状态（如果 Representation 提供）
        if "has_object" in obs:
            self._has_object = bool(obs["has_object"])
        if "on_scooter" in obs:
            prev = self._on_scooter
            self._on_scooter = bool(obs["on_scooter"])
            if prev != self._on_scooter:
                mode = "SCOOTER" if self._on_scooter else "HUMANOID"
                print(f"[OP] Mode changed → {mode}")
        if "in_vehicle" in obs:
            prev = self._in_vehicle
            self._in_vehicle = bool(obs["in_vehicle"])
            if prev != self._in_vehicle:
                mode = "VEHICLE" if self._in_vehicle else "HUMANOID"
                print(f"[OP] Mode changed → {mode}")

        if self.show_debug_window:
            frame = obs.get("rgb", None)
            if isinstance(frame, np.ndarray):
                self._render_debug_frame(frame)

        return {
            "position": self._position,
            "direction": self._direction,
            "yaw": self._yaw,
            "collision": self._collision,
            "has_object": self._has_object,
            "on_scooter": self._on_scooter,
            "in_vehicle": self._in_vehicle,
        }

    def check_interaction(self, interaction: Any) -> bool:
        if interaction is None:
            return False
        if isinstance(interaction, list) and len(interaction) == 0:
            return False
        if isinstance(interaction, dict):
            if interaction.get("type") in ("human_control", "close_human_control"):
                return True
        if isinstance(interaction, int):
            return True
        if isinstance(interaction, str):
            return interaction == "quit" or interaction in self.interaction_template
        if isinstance(interaction, list):
            for tok in interaction:
                if not isinstance(tok, str):
                    return False
                if tok != "quit" and tok not in self.interaction_template:
                    return False
            return True
        return False

    def get_interaction(
        self,
        interaction: Union[Token, List[Token], Dict[str, Any], int],
    ) -> None:
        if not self.check_interaction(interaction):
            raise ValueError(f"Invalid interaction: {interaction}")

        # Human UI driver
        if isinstance(interaction, dict) and interaction.get("type") == "human_control":
            frame = interaction.get("frame", None)
            if not isinstance(frame, np.ndarray):
                return
            keycode = self._render_and_poll_key(frame)
            self.get_interaction(int(keycode))
            return

        if isinstance(interaction, dict) and interaction.get("type") == "close_human_control":
            self._close_human_window()
            return

        # cv2 keycode
        if isinstance(interaction, int):
            keycode = interaction & 0xFF
            if keycode == 255:
                return
            if keycode == 27:           # ESC → quit
                self._push(["quit"])
                return
            try:
                ch = chr(keycode)
            except Exception:
                return
            if ch in ("q", "Q"):
                self._push(["quit"])
                return
            mapped_ch = " " if keycode == 32 else ch.lower()
            if mapped_ch in self.keyboard_mapping:
                tok = self.keyboard_mapping[mapped_ch]
                if tok not in self.interaction_template:
                    raise ValueError(f"{tok!r} not in interaction_template")
                
                # 上下车类 token 防抖
                if tok in self._action_debounce_tokens:
                    now = time.time()
                    last = self._last_action_time.get(tok, 0.0)
                    if now - last < self._action_debounce_interval:
                        return  # 冷却中，忽略
                    self._last_action_time[tok] = now
                
                self._push([tok])
            return

        # Agent token(s)
        tokens = [interaction] if isinstance(interaction, str) else list(interaction)
        if len(tokens) == 1 and tokens[0] == "quit":
            self._push(["quit"])
            return

        # 对 debounce token 做防抖（与键盘路径一致）
        filtered = []
        now = time.time()
        for tok in tokens:
            if tok in self._action_debounce_tokens:
                last = self._last_action_time.get(tok, 0.0)
                if now - last < self._action_debounce_interval:
                    continue  # 冷却中，跳过
                self._last_action_time[tok] = now
            filtered.append(tok)

        if filtered:
            self._push(filtered)

    def _push(self, tokens: List[str]) -> None:
        """把 token 列表同时写入 current_interaction 和 interaction_history。"""
        self.current_interaction.append(tokens)
        self.interaction_history.append(tokens)

    def process_interaction(
        self,
        use_step_mode: bool = True,
        **kwargs,
    ) -> List[ActionSpec]:
        """
        将 interaction_history 中最新的交互转换为 ActionSpec 列表。

        Args:
            use_step_mode:
                True  — step_forward(duration)，UE 自动停止后续动作，适合精细控制
                False — start_move，UE 持续执行直到收到 stop_current_action
            **kwargs:
                object_name  : str  — pick_up / interact 目标物体名
                scooter_id   : str  — get_off_scooter 目标 scooter
                vehicle_name : str  — enter_vehicle / exit_vehicle 目标载具
                speed        : float — set_speed 目标速度
                path         : str  — set_path 路径字符串（"x1,y1;x2,y2;..."）
                argue_type   : int  — argue 类型 [0,1]
                discuss_type : int  — discuss 类型 [0,1]
        """
        current_history_len = len(self.interaction_history)
        if current_history_len == self._last_processed_history_len:
            return []
        if not self.current_interaction:
            return []

        self._current_use_step_mode = bool(use_step_mode)

        now_interaction = self.current_interaction[-1]
        actions: List[ActionSpec] = []
        for tok in now_interaction:
            actions.extend(self._token_to_actions(tok, **kwargs))

        self._last_processed_history_len = current_history_len
        return actions

    def _token_to_actions(self, token: str, **kwargs) -> List[ActionSpec]:
        # ==================== 移动（根据当前模式分叉）====================
        if token == "forward":
            if self._on_scooter:
                return [{"type": "scooter_control",
                        "throttle": self.scooter_throttle, "brake": 0.0, "steering": 0.0}]
            if self._in_vehicle:
                return [{"type": "vehicle_control",
                        "throttle": self.scooter_throttle, "brake": 0.0, "steering": 0.0}]
            return [{"type": "step_forward", "duration": self.step_duration, "direction": 0}]

        if token == "backward":
            if self._on_scooter:
                return [{"type": "scooter_control",
                        "throttle": 0.0, "brake": self.scooter_brake, "steering": 0.0}]
            if self._in_vehicle:
                return [{"type": "vehicle_control",
                        "throttle": 0.0, "brake": self.scooter_brake, "steering": 0.0}]
            return [{"type": "step_forward", "duration": self.step_duration, "direction": 1}]

        if token == "left":
            if self._on_scooter:
                return [{"type": "scooter_control",
                        "throttle": self.scooter_throttle, "brake": 0.0,
                        "steering": -self.scooter_steer}]
            if self._in_vehicle:
                return [{"type": "vehicle_control",
                        "throttle": self.scooter_throttle, "brake": 0.0,
                        "steering": -self.scooter_steer}]
            return [
                {"type": "rotate", "angle": self.rotate_angle, "direction": "left"},
                {"type": "step_forward", "duration": self.step_duration, "direction": 0},
            ]

        if token == "right":
            if self._on_scooter:
                return [{"type": "scooter_control",
                        "throttle": self.scooter_throttle, "brake": 0.0,
                        "steering": self.scooter_steer}]
            if self._in_vehicle:
                return [{"type": "vehicle_control",
                        "throttle": self.scooter_throttle, "brake": 0.0,
                        "steering": self.scooter_steer}]
            return [
                {"type": "rotate", "angle": self.rotate_angle, "direction": "right"},
                {"type": "step_forward", "duration": self.step_duration, "direction": 0},
            ]

        if token == "stop":
            if self._on_scooter:
                return [{"type": "scooter_control",
                        "throttle": 0.0, "brake": self.scooter_brake, "steering": 0.0}]
            if self._in_vehicle:
                return [{"type": "vehicle_control",
                        "throttle": 0.0, "brake": 1.0, "steering": 0.0}]
            return [{"type": "stop"}]

        if token == "set_speed":
            speed = float(kwargs.get("speed", self.default_speed))
            return [{"type": "set_speed", "speed": speed}]

        # ==================== 相机旋转 ====================
        if token == "camera_l":
            return [{"type": "rotate", "angle": self.camera_rotate_angle, "direction": "left"}]

        if token == "camera_r":
            return [{"type": "rotate", "angle": self.camera_rotate_angle, "direction": "right"}]

        # ==================== 姿态 ====================
        if token == "sit":
            return [{"type": "sit_down"}]

        if token == "stand":
            return [{"type": "stand_up"}]

        if token == "stop_action":
            return [{"type": "stop_current_action"}]

        # ==================== 物品 ====================
        if token == "pick_up":
            if not self.enable_pick_up:
                return []
            object_name = kwargs.get("object_name", None)
            if object_name is None:
                return []
            return [{"type": "pick_up", "object_name": object_name}]

        if token == "drop_object":
            return [{"type": "drop_object"}]

        # ==================== 骑行 ====================
        if token == "get_on_scooter":
            if not self.enable_scooter:
                return []
            # 上车后状态由 Representation 的 obs 反馈，process_perception 同步
            return [{"type": "get_on_scooter"}]

        if token == "get_off_scooter":
            if not self.enable_scooter:
                return []
            return [{"type": "get_off_scooter", "scooter_id": kwargs.get("scooter_id", None)}]
        
        # ==================== 载具 ====================
        if token == "enter_vehicle":
            if not self.enable_vehicle:
                return []
            vehicle_name = kwargs.get("vehicle_name", None)
            if vehicle_name is None:
                return []
            return [{"type": "enter_vehicle", "vehicle_name": vehicle_name}]

        if token == "exit_vehicle":
            if not self.enable_vehicle:
                return []
            vehicle_name = kwargs.get("vehicle_name", None)
            if vehicle_name is None:
                return []
            return [{"type": "exit_vehicle", "vehicle_name": vehicle_name}]

        # ==================== 社交 ====================
        if token == "argue":
            if not self.enable_social:
                return []
            argue_type = int(kwargs.get("argue_type", 0))
            return [{"type": "argue", "argue_type": argue_type}]

        if token == "discuss":
            if not self.enable_social:
                return []
            discuss_type = int(kwargs.get("discuss_type", 0))
            return [{"type": "discuss", "discuss_type": discuss_type}]

        if token == "listen":
            if not self.enable_social:
                return []
            return [{"type": "listen"}]

        if token == "wave_to_dog":
            if not self.enable_social:
                return []
            return [{"type": "wave_to_dog"}]

        if token == "directing_path":
            if not self.enable_social:
                return []
            return [{"type": "directing_path"}]

        # ==================== 路径 ====================
        if token == "follow_path":
            if not self.enable_path:
                return []
            return [{"type": "follow_path"}]

        if token == "set_path":
            if not self.enable_path:
                return []
            path = kwargs.get("path", None)
            if path is None:
                return []
            return [{"type": "set_path", "path": path}]
        
        if token == "rescan":
            return [{"type": "rescan_objects"}]

        if token == "quit":
            return []

        raise ValueError(f"Unknown token: {token!r}")
    