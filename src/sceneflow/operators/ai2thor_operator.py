from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np

from .base_operator import BaseOperator

ActionSpec = Dict[str, Any]
Token = str

_VALID_LIQUIDS = {"water", "coffee", "wine"}

class Ai2ThorOperator(BaseOperator):
    def __init__(
        self,
        operation_types: Optional[List[str]] = None,
        interaction_template: Optional[List[str]] = None,

        # 移动参数
        grid_size: Optional[float] = None,
        rotate_deg: Optional[float] = None,
        look_deg: float = 30.0,
        camera_yaw_deg: Optional[float] = None,

        # 核心交互参数 
        interact_check_visible: bool = False,
        pickup_force_action: bool = False,
        open_force_action: bool = False,
        toggle_force_action: bool = False,
        put_force_action: bool = False,
        open_openness: float = 1.0,

        # 物品状态开关 
        enable_slice: bool = True,
        enable_break: bool = True,
        enable_cook: bool = True,
        enable_fill: bool = True,
        enable_empty: bool = True,
        enable_clean: bool = True,
        enable_dirty: bool = False,       # 默认关：Agent 一般不主动弄脏
        enable_use_up: bool = True,

        # 投掷 
        enable_throw: bool = True,
        throw_magnitude: float = 150.0,

        # 姿态 
        enable_crouch: bool = False,      # 默认关：多数任务不需要

        # 填充液体默认类型 
        fill_liquid: str = "water",       # "water" | "coffee" | "wine"

        # 场景管理开关（高权限，谨慎开启
        enable_remove_object: bool = False,
        enable_disable_object: bool = False,
        enable_enable_object: bool = False,
        enable_set_object_pose: bool = False,
        set_pose_target: Optional[Dict[str, float]] = None,   # {"x":…,"y":…,"z":…}

        # 物理属性
        enable_set_mass: bool = False,
        set_mass_value: float = 1.0,
        set_drag_value: float = 1.0,
        set_angular_drag_value: float = 0.05,

        # 温度衰减
        enable_set_temp_decay: bool = False,
        enable_toggle_temp_decay: bool = False,
        temp_decay_time: float = 10.0,    # 秒
        temp_decay_object_type: Optional[str] = None,  # None = 全局

        # 放置策略 
        use_precise_placement: bool = False,
        placement_height_offset: float = 0.06,

        # 人机控制 UI
        keyboard_mapping: Optional[Dict[str, str]] = None,
        human_window_name: str = "AI2-THOR Human Control",
        human_window_size: int = 600,
        draw_crosshair: bool = True,
    ):
        super().__init__(operation_types=[] if operation_types is None else operation_types)

        if operation_types is None:
            operation_types = ["action_instruction"]
        self.operation_types = operation_types

        # 移动
        self.grid_size = grid_size
        self.rotate_deg = rotate_deg
        self.look_deg = float(look_deg)
        self.camera_yaw_deg = float(camera_yaw_deg if camera_yaw_deg is not None
                                    else (rotate_deg if rotate_deg is not None else 15.0))

        # 内部状态
        self._focus_object_id: Optional[str] = None
        self._focus_object_meta: Optional[Dict[str, Any]] = None
        self._inventory_has_in_hand: bool = False
        self._held_object_id: Optional[str] = None

        # 核心交互
        self.interact_check_visible = bool(interact_check_visible)
        self.pickup_force_action = bool(pickup_force_action)
        self.open_force_action = bool(open_force_action)
        self.toggle_force_action = bool(toggle_force_action)
        self.put_force_action = bool(put_force_action)
        self.open_openness = max(0.0, min(1.0, float(open_openness)))

        # 物品状态
        self.enable_slice = bool(enable_slice)
        self.enable_break = bool(enable_break)
        self.enable_cook = bool(enable_cook)
        self.enable_fill = bool(enable_fill)
        self.enable_empty = bool(enable_empty)
        self.enable_clean = bool(enable_clean)
        self.enable_dirty = bool(enable_dirty)
        self.enable_use_up = bool(enable_use_up)
        
        assert fill_liquid in _VALID_LIQUIDS, f"fill_liquid must be one of {_VALID_LIQUIDS}"
        self.fill_liquid = str(fill_liquid)

        # 投掷
        self.enable_throw = bool(enable_throw)
        self.throw_magnitude = float(throw_magnitude)

        # 姿态
        self.enable_crouch = bool(enable_crouch)

        # 放置策略
        self.use_precise_placement = bool(use_precise_placement)
        self.placement_height_offset = float(placement_height_offset)

        # 场景管理
        self.enable_remove_object = bool(enable_remove_object)
        self.enable_disable_object = bool(enable_disable_object)
        self.enable_enable_object = bool(enable_enable_object)
        self.enable_set_object_pose = bool(enable_set_object_pose)
        self.set_pose_target = set_pose_target  # None 时 set_object_pose 无效

        # 物理属性
        self.enable_set_mass = bool(enable_set_mass)
        assert set_mass_value > 0, "set_mass_value must be > 0"
        assert set_drag_value > 0, "set_drag_value must be > 0"
        assert set_angular_drag_value > 0, "set_angular_drag_value must be > 0"
        self.set_mass_value = float(set_mass_value)
        self.set_drag_value = float(set_drag_value)
        self.set_angular_drag_value = float(set_angular_drag_value)

        # 温度衰减
        self.enable_set_temp_decay = bool(enable_set_temp_decay)
        self.enable_toggle_temp_decay = bool(enable_toggle_temp_decay)
        self.temp_decay_time = float(temp_decay_time)
        self.temp_decay_object_type = temp_decay_object_type

        # 交互模板 & 键盘映射
        self.interaction_template = self._build_interaction_template(interaction_template)
        if keyboard_mapping is None:
            keyboard_mapping = self._build_default_keyboard_mapping()
        self.keyboard_mapping = keyboard_mapping

        self.interaction_template_init()

        # 人机 UI
        self.human_window_name = str(human_window_name)
        self.human_window_size = int(human_window_size)
        self.draw_crosshair = bool(draw_crosshair)
        self._human_window_inited = False
        self._last_processed_history_len = 0

    def _build_interaction_template(self, custom: Optional[List[str]] = None) -> List[str]:
        if custom is not None:
            return list(custom)

        t = [
            # 移动
            "forward", "backward", "left", "right",
            # 镜头
            "camera_l", "camera_r", "camera_up", "camera_down",
            # 核心
            "interact", "drop",
        ]

        # 物品状态
        if self.enable_throw:           t.append("throw")
        if self.enable_slice:           t.append("slice")
        if self.enable_break:           t.append("break")
        if self.enable_cook:            t.append("cook")
        if self.enable_fill:            t.append("fill")
        if self.enable_empty:           t.append("empty")
        if self.enable_clean:           t.append("clean")
        if self.enable_dirty:           t.append("dirty")
        if self.enable_use_up:          t.append("use_up")

        # 姿态
        if self.enable_crouch:          t += ["crouch", "stand"]

        # 场景管理（高权限）
        if self.enable_remove_object:   t.append("remove_object")
        if self.enable_disable_object:  t.append("disable_object")
        if self.enable_enable_object:   t.append("enable_object")
        if self.enable_set_object_pose: t.append("set_object_pose")

        # 物理属性
        if self.enable_set_mass:        t.append("set_mass")

        # 温度衰减
        if self.enable_set_temp_decay:  t.append("set_temp_decay")
        if self.enable_toggle_temp_decay: t.append("toggle_temp_decay")

        return t

    def _build_default_keyboard_mapping(self) -> Dict[str, str]:
        m: Dict[str, str] = {
            # 移动
            "w": "forward", "s": "backward",
            "a": "left",    "d": "right",
            # 镜头
            "i": "camera_up",  "k": "camera_down",
            "j": "camera_l",   "l": "camera_r",
            # 核心
            "e": "interact",
            "g": "drop",
        }
        if self.enable_throw:           m["t"] = "throw"
        if self.enable_slice:           m["c"] = "slice"        # C = Cut
        if self.enable_break:           m["b"] = "break"        # B = Break
        if self.enable_cook:            m["o"] = "cook"         # O = cook
        if self.enable_fill:            m["f"] = "fill"         # F = Fill
        if self.enable_empty:           m["v"] = "empty"        # V = empty
        if self.enable_clean:           m["n"] = "clean"        # N = clean
        if self.enable_dirty:           m["m"] = "dirty"        # M (dirty)
        if self.enable_use_up:          m["u"] = "use_up"       # U = Use up

        if self.enable_crouch:          m["z"] = "crouch"
        if self.enable_crouch:          m["x"] = "stand"

        return m

    def process_perception(self, obs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
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

        focus_type = (self._focus_object_meta or {}).get("objectType", "")
        return {
            "focus_object_id": self._focus_object_id,
            "focus_object_type": focus_type,
            "has_in_hand": self._inventory_has_in_hand,
            "held_object_id": self._held_object_id,
        }

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
        cv2.imshow(self.human_window_name, display)
        return cv2.waitKey(1)

    def _is_agent_format(self, interaction: Any) -> bool:
        if isinstance(interaction, str):
            return True
        if isinstance(interaction, list) and len(interaction) > 0:
            return isinstance(interaction[0], str)
        return False

    def _is_human_event_format(self, interaction: Any) -> bool:
        if isinstance(interaction, dict):
            return "type" in interaction
        if isinstance(interaction, list) and len(interaction) > 0:
            return isinstance(interaction[0], dict) and "type" in interaction[0]
        return False

    def get_interaction(
        self,
        interaction: Union[Token, List[Token], Dict[str, Any], List[Dict[str, Any]], int],
    ):
        if interaction is None:
            return
        if isinstance(interaction, list) and len(interaction) == 0:
            return

        if isinstance(interaction, dict) and interaction.get("type") == "human_control":
            frame = interaction.get("frame", None)
            if not isinstance(frame, np.ndarray):
                return
            keycode = self._render_and_poll_key(frame)
            return self.get_interaction(int(keycode))

        if isinstance(interaction, dict) and interaction.get("type") == "close_human_control":
            self._close_human_window()
            return

        if isinstance(interaction, int):
            keycode = interaction & 0xFF
            if keycode in (255,):
                return
            if keycode == 27:                          # ESC -> quit
                self._push_tokens(["quit"])
                return
            try:
                ch = chr(keycode)
            except Exception:
                return
            if ch.lower() in ("q",):
                self._push_tokens(["quit"])
                return
            ch = ch.lower()
            if ch in self.keyboard_mapping:
                tok = self.keyboard_mapping[ch]
                self._validate_and_push([tok])
            return

        if self._is_human_event_format(interaction):
            events = interaction if isinstance(interaction, list) else [interaction]
            tokens: List[str] = []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                if ev.get("type") == "keyboard":
                    key = str(ev.get("key", "")).lower()
                    if key == "quit":
                        self._push_tokens(["quit"])
                        return
                    if key in self.keyboard_mapping:
                        tok = self.keyboard_mapping[key]
                        if tok not in tokens:
                            tokens.append(tok)
            if tokens:
                self._validate_and_push(tokens)
            return

        # agent token(s)
        if self._is_agent_format(interaction):
            toks = interaction if isinstance(interaction, list) else [interaction]
            if len(toks) == 1 and isinstance(toks[0], str) and toks[0].lower() == "quit":
                self._push_tokens(["quit"])
                return
            self._validate_and_push(list(toks))
            return

        raise ValueError(f"Unknown interaction format: {type(interaction)} | {interaction}")

    def _validate_and_push(self, toks: List[str]) -> None:
        for tok in toks:
            if not isinstance(tok, str):
                raise TypeError(f"Token must be str, got {type(tok)}")
            if tok not in self.interaction_template:
                raise ValueError(f"Token '{tok}' not in template: {self.interaction_template}")
        self._push_tokens(toks)

    def _push_tokens(self, toks: List[str]) -> None:
        self.current_interaction.append(toks)
        self.interaction_history.append(toks)

    def check_interaction(self, interaction: Any) -> bool:
        if interaction is None:
            return False
        if isinstance(interaction, list) and len(interaction) == 0:
            return False
        if isinstance(interaction, dict) and interaction.get("type") in ("human_control", "close_human_control"):
            return True
        if isinstance(interaction, int):
            return True
        if self._is_agent_format(interaction):
            toks = interaction if isinstance(interaction, list) else [interaction]
            for tok in toks:
                if not isinstance(tok, str):
                    return False
                if tok == "quit":
                    continue
                if tok not in self.interaction_template:
                    return False
            return True
        if self._is_human_event_format(interaction):
            return True
        return False

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

    def _token_to_actions(self, token: str, **kwargs) -> List[ActionSpec]:  # noqa: C901
        # 移动
        if token == "forward":      return self._move("MoveAhead")
        if token == "backward":     return self._move("MoveBack")
        if token == "left":         return self._move("MoveLeft")
        if token == "right":        return self._move("MoveRight")

        # 镜头 
        if token == "camera_l":     return self._rotate("RotateLeft",  degrees=self.camera_yaw_deg)
        if token == "camera_r":     return self._rotate("RotateRight", degrees=self.camera_yaw_deg)
        if token == "camera_up":    return [{"action": "LookUp",   "degrees": float(self.look_deg)}]
        if token == "camera_down":  return [{"action": "LookDown", "degrees": float(self.look_deg)}]

        # 姿态
        if token == "crouch":       return [{"action": "Crouch"}]
        if token == "stand":        return [{"action": "Stand"}]

        # 核心交互
        if token == "interact":
            a = self._decide_interact_action(**kwargs)
            return [] if a is None else [a]

        if token == "drop":
            return [{"action": "DropHandObject", "forceAction": True}]

        if token == "throw" and self.enable_throw:
            if not self._inventory_has_in_hand:
                return []
            return [{"action": "ThrowObject",
                     "moveMagnitude": float(self.throw_magnitude),
                     "forceAction": True}]

        # 物品状态
        oid = self._focus_object_id
        target = self._held_object_id if self._inventory_has_in_hand else oid

        if token == "slice" and self.enable_slice:
            if oid is None: return []
            if not self._inventory_has_in_hand: return []   # 需手持刀具
            return [{"action": "SliceObject", "objectId": oid, "forceAction": False}]

        if token == "break" and self.enable_break:
            if oid is None: return []
            return [{"action": "BreakObject", "objectId": oid, "forceAction": False}]

        if token == "cook" and self.enable_cook:
            if target is None: return []
            return [{"action": "CookObject", "objectId": target, "forceAction": False}]

        if token == "fill" and self.enable_fill:
            if target is None: return []
            return [{"action": "FillObjectWithLiquid",
                     "objectId": target,
                     "fillLiquid": self.fill_liquid,
                     "forceAction": False}]

        if token == "empty" and self.enable_empty:
            if target is None: return []
            return [{"action": "EmptyLiquidFromObject", "objectId": target, "forceAction": False}]

        if token == "clean" and self.enable_clean:
            if target is None: return []
            return [{"action": "CleanObject", "objectId": target, "forceAction": False}]

        if token == "dirty" and self.enable_dirty:
            if target is None: return []
            return [{"action": "DirtyObject", "objectId": target, "forceAction": False}]

        if token == "use_up" and self.enable_use_up:
            if target is None: return []
            return [{"action": "UseUpObject", "objectId": target, "forceAction": False}]

        # 场景管理
        if token == "remove_object" and self.enable_remove_object:
            if oid is None: return []
            return [{"action": "RemoveFromScene", "objectId": oid}]

        if token == "disable_object" and self.enable_disable_object:
            if oid is None: return []
            return [{"action": "DisableObject", "objectId": oid}]

        if token == "enable_object" and self.enable_enable_object:
            eid = getattr(self, "_last_disabled_id", None)
            if eid is None: return []
            return [{"action": "EnableObject", "objectId": eid}]

        if token == "set_object_pose" and self.enable_set_object_pose:
            # 改用 PlaceObjectAtPoint，避免 SetObjectPoses 清除场景内所有其他可移动物体
            if oid is None or self.set_pose_target is None: return []
            return [{"action": "PlaceObjectAtPoint",
                     "objectId": oid,
                     "position": self.set_pose_target}]

        # 物理属性
        if token == "set_mass" and self.enable_set_mass:
            if oid is None: return []
            return [{"action": "SetMassProperties",
                     "objectId": oid,
                     "mass": float(self.set_mass_value),
                     "drag": float(self.set_drag_value),
                     "angularDrag": float(self.set_angular_drag_value)}]

        # 温度衰减
        if token == "set_temp_decay" and self.enable_set_temp_decay:
            if self.temp_decay_object_type is not None:
                return [{"action": "SetRoomTempDecayTimeForType",
                         "objectType": self.temp_decay_object_type,
                         "TimeUntilRoomTemp": float(self.temp_decay_time)}]
            return [{"action": "SetGlobalRoomTempDecayTime",
                     "TimeUntilRoomTemp": float(self.temp_decay_time)}]

        if token == "toggle_temp_decay" and self.enable_toggle_temp_decay:
            cur = getattr(self, "_temp_decay_enabled", True)
            self._temp_decay_enabled = not cur
            return [{"action": "SetDecayTemperatureBool",
                     "allowDecayTemperature": self._temp_decay_enabled}]

        if token == "quit":
            return []

        raise ValueError(f"Unknown or disabled token: '{token}'")

    def _decide_interact_action(self, **kwargs) -> Optional[ActionSpec]:  # noqa: C901
        """
        优先级顺序（手中无物时）：
          Pickupable → Openable → Toggleable → Fillable(空) →
          Cookable(未熟) → Dirtyable(脏→清洁) → Sliceable → Breakable →
          UsedUp(未用完)
        
        手中有物时：智能放置（PutObject 或 PlaceObjectAtPoint）
        """
        oid = self._focus_object_id
        obj = self._focus_object_meta
        if oid is None or obj is None:
            return None

        has_in_hand = bool(self._inventory_has_in_hand)
        held_id = self._held_object_id

        # 手中有物 → 智能放置
        if has_in_hand:
            if held_id is None:
                return {"action": "DropHandObject", "forceAction": True}

            # 目标不是容器时，fallback 到 drop
            if not bool((obj or {}).get("receptacle", False)):
                return {"action": "DropHandObject", "forceAction": True}

            # 检测特殊物体（裂开/切片/破损 → 用精确放置更稳）
            is_special = any(s in str(held_id) for s in ("Cracked", "Broken", "Sliced"))

            if self.use_precise_placement or is_special:
                pos = obj.get("position", None)
                aabb = obj.get("axisAlignedBoundingBox", None)
                if isinstance(pos, dict) and all(k in pos for k in ("x", "y", "z")):
                    tx, tz = float(pos["x"]), float(pos["z"])
                    if isinstance(aabb, dict):
                        c = aabb.get("center", {})
                        s = aabb.get("size", {})
                        if isinstance(c, dict) and isinstance(s, dict):
                            ty = float(c.get("y", pos["y"])) + 0.5 * float(s.get("y", 0.1)) + self.placement_height_offset
                        else:
                            ty = float(pos["y"]) + self.placement_height_offset
                    else:
                        ty = float(pos["y"]) + self.placement_height_offset
                    # PlaceObjectAtPoint 只接受 objectId 和 position，无 forceKinematic
                    return {"action": "PlaceObjectAtPoint",
                            "objectId": held_id,
                            "position": {"x": tx, "y": ty, "z": tz}}

            # 默认：PutObject（目标已确认为 receptacle）
            return {
                "action": "PutObject",
                "objectId": oid,
                "forceAction": bool(self.put_force_action),
                "placeStationary": True,
            }

        # 手中无物 → 按优先级决策 
        # 1. Pickupable
        if bool(obj.get("pickupable", False)):
            return {"action": "PickupObject",
                    "objectId": oid,
                    "forceAction": bool(self.pickup_force_action)}

        # 2. Openable
        if bool(obj.get("openable", False)):
            if bool(obj.get("isOpen", False)):
                return {"action": "CloseObject", "objectId": oid, "forceAction": bool(self.open_force_action)}
            return {"action": "OpenObject", "objectId": oid,
                    "openness": float(self.open_openness),
                    "forceAction": bool(self.open_force_action)}

        # 3. Toggleable
        if bool(obj.get("toggleable", False)):
            if bool(obj.get("isToggled", False)):
                return {"action": "ToggleObjectOff", "objectId": oid, "forceAction": bool(self.toggle_force_action)}
            return {"action": "ToggleObjectOn",  "objectId": oid, "forceAction": bool(self.toggle_force_action)}

        # 4. Fillable（空才填充）
        if self.enable_fill and bool(obj.get("canFillWithLiquid", False)) and not bool(obj.get("isFilledWithLiquid", False)):
            return {"action": "FillObjectWithLiquid",
                    "objectId": oid,
                    "fillLiquid": self.fill_liquid,
                    "forceAction": False}

        # 5. Cookable（未熟才烹饪）
        if self.enable_cook and bool(obj.get("cookable", False)) and not bool(obj.get("isCooked", False)):
            return {"action": "CookObject", "objectId": oid, "forceAction": False}

        # 6. Dirtyable（脏 → 清洁；interact 默认清洁，不主动弄脏）
        if self.enable_clean and bool(obj.get("dirtyable", False)) and bool(obj.get("isDirty", False)):
            return {"action": "CleanObject", "objectId": oid, "forceAction": False}

        # 7. Sliceable（未切）
        if self.enable_slice and bool(obj.get("sliceable", False)) and not bool(obj.get("isSliced", False)):
            if not self._inventory_has_in_hand:
                return None  # 没有刀具，interact 不处理
            return {"action": "SliceObject", "objectId": oid, "forceAction": False}

        # 8. Breakable（未破）
        if self.enable_break and bool(obj.get("breakable", False)) and not bool(obj.get("isBroken", False)):
            return {"action": "BreakObject", "objectId": oid, "forceAction": False}

        # 9. UsedUp（未用完）
        if self.enable_use_up and bool(obj.get("canBeUsedUp", False)) and not bool(obj.get("isUsedUp", False)):
            return {"action": "UseUpObject", "objectId": oid, "forceAction": False}

        return None

    def _track_side_effects(self, actions: List[ActionSpec]) -> None:
        """记录 disable_object 的目标 ID，供后续 enable_object 使用"""
        for a in actions:
            if a.get("action") == "DisableObject":
                self._last_disabled_id = a.get("objectId", None)

    def process_interaction(self, **kwargs) -> List[ActionSpec]:
        current_len = len(self.interaction_history)
        if current_len == self._last_processed_history_len:
            return []
        if len(self.current_interaction) == 0:
            return []

        now_tokens = self.current_interaction[-1]
        actions: List[ActionSpec] = []
        for tok in now_tokens:
            actions.extend(self._token_to_actions(tok, **kwargs))

        self._track_side_effects(actions)

        self._last_processed_history_len = current_len
        return actions
