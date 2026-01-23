from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .base_operator import BaseOperator

ActionSpec = Dict[str, Any]
Token = Union[str, Dict[str, Any]]


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


class Ai2ThorOperator(BaseOperator):
    """
    人类模式：
    - 键盘：w/a/s/d -> MoveAhead/MoveLeft/MoveBack/MoveRight
    - 鼠标：dx -> RotateLeft/RotateRight, dy -> LookUp/LookDown
    - LMB click：调用 click_to_action()

    MLLM/Token 模式：
    - tokens: forward/left/right/backward/camera_*/interact
    - interact = “模拟一次左键”，复用 click_to_action() 的完整交互逻辑
    """
    def __init__(
        self,
        operation_types: Optional[List[str]] = None,
        interaction_template: Optional[List[str]] = None,

        # ---- Thor 原子动作尺度 ----
        grid_size: Optional[float] = None,        # 移动步长（米），影响 forward/backward 精细度
        rotate_deg: Optional[float] = None,       # 离散旋转角度（一般给 agent 用，如 90°）
        look_deg: float = 30.0,                   # 单次抬头/低头角度（度）

        camera_yaw_deg: Optional[float] = None,   # camera_l / camera_r 的 yaw 角度（扫描粒度，越小越精细）

        # ---- 人类鼠标输入解析 ----
        rot_step_pixels: float = 6.0,             # 鼠标 dx 达到多少像素触发一次旋转
        look_step_pixels: float = 8.0,            # 鼠标 dy 达到多少像素触发一次抬头/低头
        max_yaw_per_tick: int = 3,                # 单帧最大旋转次数（限制角速度）
        max_pitch_per_tick: int = 2,              # 单帧最大俯仰次数

        # ---- interact 行为策略 ----
        interact_check_visible: bool = False,     # focus 是否要求 visible（False 更鲁棒）
        pickup_force_action: bool = False,        # PickupObject 是否强制
        open_force_action: bool = False,          # Open/CloseObject 是否强制
        toggle_force_action: bool = False,        # ToggleObject 是否强制
        put_force_action: bool = False,           # PutObject 是否强制
        open_openness: float = 1.0,               # OpenObject 打开程度（1.0=全开，利于放物）
    ):

        super().__init__()

        if operation_types is None:
            operation_types = ["action_instruction"]

        default_template = [
            "forward",
            "left",
            "right",
            "backward",
            "camera_l",
            "camera_r",
            "camera_up",
            "camera_down",
            "interact",
        ]

        if interaction_template is None:
            interaction_template = default_template
        else:
            # 兼容：用户传了旧模板（没带 interact）时自动补上
            it = list(interaction_template)
            if "interact" not in it:
                it = it + ["interact"]

            expected = set(default_template)
            got = set(it)
            if expected != got:
                raise ValueError(
                    f"interaction_template must contain exactly tokens: {sorted(list(expected))}, got: {sorted(list(got))}"
                )
            # 按 default_template 顺序排序，保证一致性
            interaction_template = [t for t in default_template if t in it]

        self.interaction_template = interaction_template
        self.interaction_template_init()

        self.operation_types = operation_types
        self.grid_size = grid_size
        self.rotate_deg = rotate_deg
        self.look_deg = float(look_deg)

        if camera_yaw_deg is not None:
            self.camera_yaw_deg = float(camera_yaw_deg)
        elif rotate_deg is not None:
            self.camera_yaw_deg = float(rotate_deg)
        else:
            self.camera_yaw_deg = 15.0

        self.rot_step_pixels = float(rot_step_pixels)
        self.look_step_pixels = float(look_step_pixels)
        self.max_yaw_per_tick = int(max_yaw_per_tick)
        self.max_pitch_per_tick = int(max_pitch_per_tick)

        # focus 状态
        self.focus_object_id: Optional[str] = None
        self.focus_object_meta: Optional[Dict[str, Any]] = None

        # 策略参数
        self.interact_check_visible = bool(interact_check_visible)
        self.pickup_force_action = bool(pickup_force_action)
        self.open_force_action = bool(open_force_action)
        self.toggle_force_action = bool(toggle_force_action)
        self.put_force_action = bool(put_force_action)
        self.open_openness = float(open_openness)

    # ---------------- 人类输入：移动 + 视角 ----------------
    def inputs_to_actions(
        self,
        pressed_keys: "set[str]",
        mouse: Dict[str, Any],
        *,
        rot_step_pixels: Optional[float] = None,
        look_step_pixels: Optional[float] = None,
        max_yaw_per_tick: Optional[int] = None,
        max_pitch_per_tick: Optional[int] = None,
    ) -> List[ActionSpec]:
        rot_step_pixels = float(self.rot_step_pixels if rot_step_pixels is None else rot_step_pixels)
        look_step_pixels = float(self.look_step_pixels if look_step_pixels is None else look_step_pixels)
        max_yaw_per_tick = int(self.max_yaw_per_tick if max_yaw_per_tick is None else max_yaw_per_tick)
        max_pitch_per_tick = int(self.max_pitch_per_tick if max_pitch_per_tick is None else max_pitch_per_tick)

        actions: List[ActionSpec] = []

        if "w" in pressed_keys:
            actions.extend(self._move("MoveAhead"))
        elif "s" in pressed_keys:
            actions.extend(self._move("MoveBack"))
        elif "a" in pressed_keys:
            actions.extend(self._move("MoveLeft"))
        elif "d" in pressed_keys:
            actions.extend(self._move("MoveRight"))

        dx = float(mouse.get("dx", 0.0))
        dy = float(mouse.get("dy", 0.0))

        deadzone = 0.5
        if abs(dx) < deadzone:
            dx = 0.0
        if abs(dy) < deadzone:
            dy = 0.0

        if abs(dx) >= rot_step_pixels:
            n = min(int(abs(dx) // rot_step_pixels), max_yaw_per_tick)
            if n > 0:
                rot_action = "RotateRight" if dx > 0 else "RotateLeft"
                for _ in range(n):
                    actions.extend(self._rotate(rot_action, degrees=self.camera_yaw_deg))
                mouse["dx"] = dx - (_sign(dx) * n * rot_step_pixels)
        else:
            mouse["dx"] = dx

        if abs(dy) >= look_step_pixels:
            n = min(int(abs(dy) // look_step_pixels), max_pitch_per_tick)
            if n > 0:
                look_action = "LookUp" if dy < 0 else "LookDown"
                for _ in range(n):
                    actions.append({"action": look_action, "degrees": float(self.look_deg)})
                mouse["dy"] = dy - (_sign(dy) * n * look_step_pixels)
        else:
            mouse["dy"] = dy

        return actions

    # ---------------- focus 更新（每帧） ----------------
    def update_focus(self, representation, event: Any) -> None:
        oid = representation.get_focus_object(event, checkVisible=self.interact_check_visible)
        self.focus_object_id = oid
        self.focus_object_meta = None
        if oid is None:
            return
        self.focus_object_meta = representation.get_object_meta(event, oid)

    # ---------------- 单击 -> action（复用） ----------------
    def click_to_action(self, representation, event: Any) -> Optional[ActionSpec]:
        oid = self.focus_object_id
        obj = self.focus_object_meta
        if oid is None or obj is None:
            return None

        def _is_floor(oid_: str, obj_: Dict[str, Any]) -> bool:
            obj_type = str(obj_.get("objectType", "") or "")
            if obj_type.lower() == "floor":
                return True
            s = str(oid_)
            return s.startswith("Floor|") or ("Floor|" in s)

        def _is_ok_place_coord(coord: Any) -> bool:
            return isinstance(coord, dict) and all(k in coord for k in ("x", "y", "z"))

        def _agent_pose(md: Dict[str, Any]) -> Dict[str, Any]:
            return (md.get("agent", {}) or {})

        def _push_point_away_if_too_close(coord: Dict[str, Any], md: Dict[str, Any]) -> Dict[str, Any]:
            agent = _agent_pose(md)
            pos = (agent.get("position", {}) or {})
            rot = (agent.get("rotation", {}) or {})
            ax = float(pos.get("x", 0.0))
            az = float(pos.get("z", 0.0))
            yaw_deg = float(rot.get("y", 0.0))

            tx = float(coord["x"])
            tz = float(coord["z"])
            dx = tx - ax
            dz = tz - az
            dist = (dx * dx + dz * dz) ** 0.5

            min_dist = 0.65
            if dist >= min_dist:
                return coord

            import math
            yaw = math.radians(yaw_deg)
            fx = math.sin(yaw)
            fz = math.cos(yaw)
            return {"x": ax + fx * min_dist, "y": float(coord["y"]), "z": az + fz * min_dist}

        def _get_aabb(o: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            aabb = o.get("axisAlignedBoundingBox", None)
            if not isinstance(aabb, dict):
                return None
            c = aabb.get("center", None)
            s = aabb.get("size", None)
            if not (isinstance(c, dict) and isinstance(s, dict)):
                return None
            if not all(k in c for k in ("x", "y", "z")):
                return None
            if not all(k in s for k in ("x", "y", "z")):
                return None
            return aabb

        def _aabb_top_y(aabb: Dict[str, Any]) -> float:
            cy = float(aabb["center"]["y"])
            sy = float(aabb["size"]["y"])
            return cy + 0.5 * sy

        md = getattr(event, "metadata", {}) or {}
        has_in_hand = representation.agent_has_in_hand(event)

        # 手上有东西：放置/丢弃
        if has_in_hand:
            held_id = representation.held_object_id(event)
            if held_id is None:
                return {"action": "DropHandObject"}

            if (not _is_floor(oid, obj)) and bool(obj.get("receptacle", False)) and bool(obj.get("visible", True)):
                return {
                    "action": "PutObject",
                    "objectId": oid,
                    "forceAction": bool(self.put_force_action),
                    "placeStationary": True,
                }

            if (not _is_floor(oid, obj)) and bool(obj.get("visible", True)):
                aabb_tgt = _get_aabb(obj)
                if aabb_tgt is not None:
                    tx = float(aabb_tgt["center"]["x"])
                    tz = float(aabb_tgt["center"]["z"])
                    top_y = _aabb_top_y(aabb_tgt)

                    held_meta = representation.get_object_meta(event, held_id)
                    add_h = 0.03
                    if isinstance(held_meta, dict):
                        aabb_h = _get_aabb(held_meta)
                        if aabb_h is not None:
                            add_h += 0.5 * float(aabb_h["size"]["y"])

                    return {
                        "action": "PlaceObjectAtPoint",
                        "objectId": held_id,
                        "position": {"x": tx, "y": float(top_y + add_h), "z": tz},
                        "forceKinematic": False,
                    }

            coord = representation.get_coordinate_from_raycast(0.5, 0.78)
            if _is_ok_place_coord(coord):
                coord2 = _push_point_away_if_too_close(coord, md)
                return {
                    "action": "PlaceObjectAtPoint",
                    "objectId": held_id,
                    "position": {"x": float(coord2["x"]), "y": float(coord2["y"]), "z": float(coord2["z"])},
                    "forceKinematic": True,
                }

            return {"action": "DropHandObject"}

        # 手空：拾取/开关
        if bool(obj.get("pickupable", False)) and bool(obj.get("visible", True)):
            return {"action": "PickupObject", "objectId": oid, "forceAction": bool(self.pickup_force_action)}

        if bool(obj.get("openable", False)) and bool(obj.get("visible", True)):
            if bool(obj.get("isOpen", False)):
                return {"action": "CloseObject", "objectId": oid, "forceAction": bool(self.open_force_action)}
            return {
                "action": "OpenObject",
                "objectId": oid,
                "openness": float(self.open_openness),
                "forceAction": bool(self.open_force_action),
            }

        if bool(obj.get("toggleable", False)) and bool(obj.get("visible", True)):
            if bool(obj.get("isToggled", False)):
                return {"action": "ToggleObjectOff", "objectId": oid, "forceAction": bool(self.toggle_force_action)}
            return {"action": "ToggleObjectOn", "objectId": oid, "forceAction": bool(self.toggle_force_action)}

        return None

    # overlay info
    def get_focus_info_for_overlay(self) -> Optional[Dict[str, Any]]:
        if self.focus_object_id is None or self.focus_object_meta is None:
            return None
        o = self.focus_object_meta
        return {
            "objectId": self.focus_object_id,
            "objectType": o.get("objectType", ""),
            "pickupable": bool(o.get("pickupable", False)),
            "openable": bool(o.get("openable", False)),
            "isOpen": bool(o.get("isOpen", False)),
            "toggleable": bool(o.get("toggleable", False)),
            "isToggled": bool(o.get("isToggled", False)),
            "receptacle": bool(o.get("receptacle", False)),
            "visible": bool(o.get("visible", False)),
            "distance": o.get("distance", None),
        }

    # ---------------- token 相关 ----------------
    def check_interaction(self, interaction: Token) -> bool:
        if not isinstance(interaction, str):
            raise TypeError(f"Only str tokens are allowed, got: {type(interaction)}")
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template: {self.interaction_template}")
        return True

    def get_interaction(self, interaction: Union[Token, List[Token]]) -> None:
        if not isinstance(interaction, list):
            interaction = [interaction]
        for act in interaction:
            self.check_interaction(act)
        self.current_interaction.append(interaction)

    def _move(self, action_name: str) -> List[ActionSpec]:
        a: ActionSpec = {"action": action_name}
        if self.grid_size is not None:
            a["moveMagnitude"] = float(self.grid_size)
        return [a]

    def _rotate(self, action_name: str, degrees: Optional[float] = None) -> List[ActionSpec]:
        a: ActionSpec = {"action": action_name}
        deg = degrees if degrees is not None else self.rotate_deg
        if deg is not None:
            a["degrees"] = float(deg)
        return [a]

    def _token_to_actions(
        self,
        token: str,
        *,
        representation: Any = None,
        event: Any = None,
    ) -> List[ActionSpec]:
        if token == "forward":
            return self._move("MoveAhead")
        if token == "backward":
            return self._move("MoveBack")
        if token == "left":
            return self._move("MoveLeft")
        if token == "right":
            return self._move("MoveRight")

        if token == "camera_l":
            return self._rotate("RotateLeft", degrees=self.camera_yaw_deg)
        if token == "camera_r":
            return self._rotate("RotateRight", degrees=self.camera_yaw_deg)

        if token == "camera_up":
            return [{"action": "LookUp", "degrees": float(self.look_deg)}]
        if token == "camera_down":
            return [{"action": "LookDown", "degrees": float(self.look_deg)}]

        if token == "interact":
            if representation is None or event is None:
                return []
            # 确保 focus 最新
            self.update_focus(representation, event)
            a = self.click_to_action(representation, event)
            return [] if a is None else [a]

        raise ValueError(f"Unknown token: {token}")

    def process_interaction(self, representation: Any = None, event: Any = None) -> List[ActionSpec]:
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")

        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)

        actions: List[ActionSpec] = []
        for tok in now_interaction:
            actions.extend(self._token_to_actions(tok, representation=representation, event=event))
        return actions
