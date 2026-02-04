from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .base_operator import BaseOperator

ActionSpec = Dict[str, Any]
Token = str


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


class Ai2ThorOperator(BaseOperator):
    def __init__(
        self,
        operation_types: Optional[List[str]] = None,
        interaction_template: Optional[List[str]] = None,
        # Thor 原子动作尺度
        grid_size: Optional[float] = None,
        rotate_deg: Optional[float] = None,
        look_deg: float = 30.0,
        camera_yaw_deg: Optional[float] = None,
        # interact 策略
        interact_check_visible: bool = False,
        pickup_force_action: bool = False,
        open_force_action: bool = False,
        toggle_force_action: bool = False,
        put_force_action: bool = False,
        open_openness: float = 1.0,
    ):
        super().__init__(operation_types=[] if operation_types is None else operation_types)

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
            it = list(interaction_template)
            if "interact" not in it:
                it = it + ["interact"]
            expected = set(default_template)
            got = set(it)
            if expected != got:
                raise ValueError(
                    f"interaction_template must contain exactly tokens: {sorted(list(expected))}, got: {sorted(list(got))}"
                )
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

        # “瞬时感知状态”（不存 history）
        self._focus_object_id: Optional[str] = None
        self._focus_object_meta: Optional[Dict[str, Any]] = None
        self._inventory_has_in_hand: bool = False
        self._held_object_id: Optional[str] = None

        self.interact_check_visible = bool(interact_check_visible)
        self.pickup_force_action = bool(pickup_force_action)
        self.open_force_action = bool(open_force_action)
        self.toggle_force_action = bool(toggle_force_action)
        self.put_force_action = bool(put_force_action)
        self.open_openness = float(open_openness)

    # ============ BaseOperator 模版：感知处理 ============
    def process_perception(self, obs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        更新 focus / inventory 等瞬时状态
        - 不记录历史（交给 memory）
        - 返回轻量 summary（给 pipeline overlay / log 用）
        """
        focus = obs.get("focus", None)
        if isinstance(focus, dict):
            self._focus_object_id = focus.get("objectId", None)
            self._focus_object_meta = focus.get("object", None)
        else:
            self._focus_object_id = None
            self._focus_object_meta = None

        inv = obs.get("inventory", None)
        if isinstance(inv, dict):
            self._inventory_has_in_hand = bool(inv.get("has_in_hand", False))
            self._held_object_id = inv.get("held_object_id", None)
        else:
            self._inventory_has_in_hand = False
            self._held_object_id = None

        focus_type = (self._focus_object_meta or {}).get("objectType", "") if self._focus_object_meta else ""
        summary = {
            "focus_object_id": self._focus_object_id,
            "focus_object_type": focus_type,
            "has_in_hand": self._inventory_has_in_hand,
            "held_object_id": self._held_object_id,
        }
        return summary

    # ============ BaseOperator 模版：交互输入 ============
    def check_interaction(self, interaction: Token) -> bool:
        if not isinstance(interaction, str):
            raise TypeError(f"Only str tokens are allowed, got: {type(interaction)}")
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template: {self.interaction_template}")
        return True

    def get_interaction(self, interaction: Union[Token, List[Token]]):
        if not isinstance(interaction, list):
            interaction = [interaction]
        for act in interaction:
            self.check_interaction(act)
        self.current_interaction.append(interaction)

    # ============ token -> ActionSpec ============
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

    def _token_to_actions(self, token: str, **kwargs) -> List[ActionSpec]:
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
            a = self._decide_interact_action(**kwargs)
            return [] if a is None else [a]

        raise ValueError(f"Unknown token: {token}")

    def _decide_interact_action(self, **kwargs) -> Optional[ActionSpec]:
        oid = self._focus_object_id
        obj = self._focus_object_meta
        if oid is None or obj is None:
            return None

        def _is_floor(oid_: str, obj_: Dict[str, Any]) -> bool:
            obj_type = str(obj_.get("objectType", "") or "")
            if obj_type.lower() == "floor":
                return True
            s = str(oid_)
            return s.startswith("Floor|") or ("Floor|" in s)

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

        has_in_hand = bool(self._inventory_has_in_hand)

        # 手上有东西：放置/丢弃
        if has_in_hand:
            held_id = self._held_object_id
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
                    add_h = 0.06
                    return {
                        "action": "PlaceObjectAtPoint",
                        "objectId": held_id,
                        "position": {"x": tx, "y": float(top_y + add_h), "z": tz},
                        "forceKinematic": False,
                    }

            raycast = kwargs.get("raycast", None)
            if isinstance(raycast, dict) and all(k in raycast for k in ("x", "y", "z")):
                return {
                    "action": "PlaceObjectAtPoint",
                    "objectId": held_id,
                    "position": {"x": float(raycast["x"]), "y": float(raycast["y"]), "z": float(raycast["z"])},
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

    # ============ BaseOperator 模版：处理 interaction ============
    def process_interaction(self, **kwargs) -> List[ActionSpec]:
        """
        只把 token 转成 actions
        不写 interaction_history（history 交给 memory）
        """
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]

        actions: List[ActionSpec] = []
        for tok in now_interaction:
            actions.extend(self._token_to_actions(tok, **kwargs))
        return actions
