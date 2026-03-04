from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np

from .base_operator import BaseOperator

ActionSpec = Dict[str, Any]
Token = str


class Ai2ThorOperator(BaseOperator):
    def __init__(
        self,
        operation_types: Optional[List[str]] = None,
        interaction_template: Optional[List[str]] = None,
        
        # Movement parameters
        grid_size: Optional[float] = None,
        rotate_deg: Optional[float] = None,
        look_deg: float = 30.0,
        camera_yaw_deg: Optional[float] = None,
        
        # Interaction parameters
        interact_check_visible: bool = False,
        pickup_force_action: bool = False,
        open_force_action: bool = False,
        toggle_force_action: bool = False,
        put_force_action: bool = False,
        open_openness: float = 1.0,
        
        # Extended interactions (only slice & break)
        enable_slice: bool = True,
        enable_break: bool = True,
        
        # Keyboard mapping
        keyboard_mapping: Optional[Dict[str, str]] = None,

        # Human control UI
        human_window_name: str = "AI2-THOR Human Control",
        human_window_size: int = 600,
        draw_crosshair: bool = True,
    ):
        super().__init__(operation_types=[] if operation_types is None else operation_types)

        if operation_types is None:
            operation_types = ["action_instruction"]

        self.operation_types = operation_types
        
        # Movement parameters
        self.grid_size = grid_size
        self.rotate_deg = rotate_deg
        self.look_deg = float(look_deg)

        if camera_yaw_deg is not None:
            self.camera_yaw_deg = float(camera_yaw_deg)
        elif rotate_deg is not None:
            self.camera_yaw_deg = float(rotate_deg)
        else:
            self.camera_yaw_deg = 15.0

        # Interaction state
        self._focus_object_id: Optional[str] = None
        self._focus_object_meta: Optional[Dict[str, Any]] = None
        self._inventory_has_in_hand: bool = False
        self._held_object_id: Optional[str] = None

        # Interaction parameters
        self.interact_check_visible = bool(interact_check_visible)
        self.pickup_force_action = bool(pickup_force_action)
        self.open_force_action = bool(open_force_action)
        self.toggle_force_action = bool(toggle_force_action)
        self.put_force_action = bool(put_force_action)
        self.open_openness = float(open_openness)
        
        # Placement strategy
        self.use_precise_placement = False  # 如果设为 True，使用坐标精确放置
        self.placement_height_offset = 0.06  # 放置时的高度偏移
    
        # Extended interactions
        self.enable_slice = enable_slice
        self.enable_break = enable_break

        # Build interaction template
        self.interaction_template = self._build_interaction_template(interaction_template)
        self.interaction_template_init()

        # Keyboard mapping
        if keyboard_mapping is None:
            keyboard_mapping = self._build_default_keyboard_mapping()
        self.keyboard_mapping = keyboard_mapping

        # Human UI
        self.human_window_name = str(human_window_name)
        self.human_window_size = int(human_window_size)
        self.draw_crosshair = bool(draw_crosshair)
        self._human_window_inited = False
        
        # Track processed history
        self._last_processed_history_len = 0

    def _build_interaction_template(self, custom_template: Optional[List[str]] = None) -> List[str]:
        """构建支持的 tokens"""
        if custom_template is not None:
            return list(custom_template)
        
        template = [
            # Movement
            "forward", "backward", "left", "right",
            
            # Camera
            "camera_l", "camera_r", "camera_up", "camera_down",
            
            # Core interaction
            "interact",  # 智能交互（拾取/放置/开关等）
            "drop",      # 强制丢弃手中物体
        ]
        
        # Extended interactions
        if self.enable_slice:
            template.append("slice")
        if self.enable_break:
            template.append("break")
        
        return template
    
    def _build_default_keyboard_mapping(self) -> Dict[str, str]:
        """构建默认键盘映射"""
        mapping = {
            # Movement
            "w": "forward",
            "s": "backward",
            "a": "left",
            "d": "right",
            
            # Camera
            "i": "camera_up",
            "k": "camera_down",
            "j": "camera_l",
            "l": "camera_r",
            
            # Core interaction
            "e": "interact",  # 智能交互
            "g": "drop",      # 强制丢弃（G = Give up / Ground）
        }
        
        # Extended interactions
        if self.enable_slice:
            mapping["c"] = "slice"  # C = Cut
        if self.enable_break:
            mapping["b"] = "break"  # B = Break
        
        return mapping

    def process_perception(self, obs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """处理感知信息，更新内部状态"""
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
        return {
            "focus_object_id": self._focus_object_id,
            "focus_object_type": focus_type,
            "has_in_hand": self._inventory_has_in_hand,
            "held_object_id": self._held_object_id,
        }

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
        """渲染画面并获取按键输入
        准星仅作为视觉辅助，帮助人类操作时对准物体
        实际交互由 AI2-THOR 的 GetObjectInFrame 决定
        """
        self._ensure_human_window()

        display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if self.draw_crosshair:
            h, w = display.shape[:2]
            cx, cy = w // 2, h // 2
            # 绘制准星（纯视觉辅助，与 GetObjectInFrame(0.5, 0.5) 对应）
            cv2.line(display, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 2)
            cv2.line(display, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

        cv2.imshow(self.human_window_name, display)
        return cv2.waitKey(1)

    def get_interaction(self, interaction: Union[Token, List[Token], Dict[str, Any], List[Dict[str, Any]], int]):
        """获取交互输入"""
        if interaction is None:
            return
        if isinstance(interaction, list) and len(interaction) == 0:
            return
        
        # Human UI driver
        if isinstance(interaction, dict) and interaction.get("type") == "human_control":
            frame = interaction.get("frame", None)
            if not isinstance(frame, np.ndarray):
                return
            keycode = self._render_and_poll_key(frame)
            return self.get_interaction(int(keycode))

        # Close UI
        if isinstance(interaction, dict) and interaction.get("type") == "close_human_control":
            self._close_human_window()
            return

        # CV2 keycode
        if isinstance(interaction, int):
            keycode = interaction & 0xFF

            if keycode in (255,):
                return

            # ESC -> quit
            if keycode == 27:
                toks = ["quit"]
                self.current_interaction.append(toks)
                self.interaction_history.append(toks)
                return

            try:
                ch = chr(keycode)
            except Exception:
                return

            if ch in ("q", "Q"):
                toks = ["quit"]
                self.current_interaction.append(toks)
                self.interaction_history.append(toks)
                return

            ch = ch.lower()
            if ch in self.keyboard_mapping:
                tok = self.keyboard_mapping[ch]
                if tok not in self.interaction_template:
                    raise ValueError(f"{tok} not in template: {self.interaction_template}")
                toks = [tok]
                self.current_interaction.append(toks)
                self.interaction_history.append(toks)
            return

        # Human event format
        if self._is_human_event_format(interaction):
            events = interaction if isinstance(interaction, list) else [interaction]
            tokens: List[str] = []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                if ev.get("type") == "keyboard":
                    key = str(ev.get("key", "")).lower()
                    if key == "quit":
                        toks = ["quit"]
                        self.current_interaction.append(toks)
                        self.interaction_history.append(toks)
                        return
                    if key in self.keyboard_mapping:
                        tok = self.keyboard_mapping[key]
                        if tok not in tokens:
                            tokens.append(tok)

            if tokens:
                for tok in tokens:
                    if tok not in self.interaction_template:
                        raise ValueError(f"{tok} not in template: {self.interaction_template}")
                self.current_interaction.append(tokens)
                self.interaction_history.append(tokens)
            return

        # Agent token(s)
        if self._is_agent_format(interaction):
            toks = interaction if isinstance(interaction, list) else [interaction]

            if len(toks) == 1 and isinstance(toks[0], str) and toks[0].lower() == "quit":
                q = ["quit"]
                self.current_interaction.append(q)
                self.interaction_history.append(q)
                return

            for token in toks:
                if not isinstance(token, str):
                    raise TypeError(f"Agent token must be str, got {type(token)}")
                if token not in self.interaction_template:
                    raise ValueError(f"{token} not in template: {self.interaction_template}")

            self.current_interaction.append(list(toks))
            self.interaction_history.append(list(toks))
            return

        raise ValueError(f"Unknown interaction format: {type(interaction)} | {interaction}")

    def check_interaction(self, interaction):
        """检查交互是否有效"""
        if interaction is None:
            return False
        if isinstance(interaction, list) and len(interaction) == 0:
            return False
        
        if isinstance(interaction, dict):
            if interaction.get("type") in ("human_control", "close_human_control"):
                return True
        
        if isinstance(interaction, int):
            return True
        
        if self._is_agent_format(interaction):
            toks = interaction if isinstance(interaction, list) else [interaction]
            for token in toks:
                if not isinstance(token, str):
                    return False
                if token == "quit":
                    continue
                if token not in self.interaction_template:
                    return False
            return True
        
        if self._is_human_event_format(interaction):
            return True
        
        return False

    # ================= Token -> Action =================
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
        """将 token 转换为 AI2-THOR 动作"""
        # === Movement ===
        if token == "forward":
            return self._move("MoveAhead")
        if token == "backward":
            return self._move("MoveBack")
        if token == "left":
            return self._move("MoveLeft")
        if token == "right":
            return self._move("MoveRight")

        # === Camera ===
        if token == "camera_l":
            return self._rotate("RotateLeft", degrees=self.camera_yaw_deg)
        if token == "camera_r":
            return self._rotate("RotateRight", degrees=self.camera_yaw_deg)
        if token == "camera_up":
            return [{"action": "LookUp", "degrees": float(self.look_deg)}]
        if token == "camera_down":
            return [{"action": "LookDown", "degrees": float(self.look_deg)}]

        # === Core Interaction ===
        if token == "interact":
            a = self._decide_interact_action(**kwargs)
            return [] if a is None else [a]
        
        if token == "drop":
            # 强制丢弃手中物体
            return [{"action": "DropHandObject", "forceAction": True}]

        # === Extended Interactions ===
        oid = self._focus_object_id
        
        if token == "slice" and self.enable_slice:
            if oid is None:
                return []
            return [{"action": "SliceObject", "objectId": oid, "forceAction": False}]
        
        if token == "break" and self.enable_break:
            if oid is None:
                return []
            return [{"action": "BreakObject", "objectId": oid, "forceAction": False}]
        
        if token == "quit":
            return []

        raise ValueError(f"Unknown or disabled token: {token}")

    def _decide_interact_action(self, **kwargs) -> Optional[ActionSpec]:
        """
        智能交互：支持两种放置策略
        
        放置策略：
        1. PutObject (默认): AI2-THOR 自动放置
           - 会自动判断：容器/表面/地面
           - 参数：forceAction 控制是否忽略某些限制
        
        2. PlaceObjectAtPoint (精确): 基于物体坐标放置
           - 精确可控，适合需要准确位置的场景
           - 启用方式：self.use_precise_placement = True
        
        其他交互：
        - 可拾取 -> Pickup
        - 可开关 -> Open/Close
        - 可切片 -> Slice
        - 可破坏 -> Break
        - 可切换 -> Toggle
        """
        oid = self._focus_object_id
        obj = self._focus_object_meta
        if oid is None or obj is None:
            return None

        has_in_hand = bool(self._inventory_has_in_hand)
        held_id = self._held_object_id

        # === 手里有物体 -> 智能放置 ===
        if has_in_hand:
            if held_id is None:
                # 安全 fallback
                return {"action": "DropHandObject", "forceAction": True}
            
            # 检测特殊物体（如 EggCracked）
            is_special_object = (
                "Cracked" in str(held_id) or 
                "Broken" in str(held_id) or
                "Sliced" in str(held_id)
            )
            
            # 策略1: 精确坐标放置（适合需要准确位置的 Agent）
            if self.use_precise_placement or is_special_object:
                # 对于特殊物体，强制使用精确放置
                pos = obj.get("position", None)
                aabb = obj.get("axisAlignedBoundingBox", None)
                
                if isinstance(pos, dict) and all(k in pos for k in ("x", "y", "z")):
                    # 放在物体上方
                    target_x = float(pos["x"])
                    target_z = float(pos["z"])
                    
                    # 计算放置高度
                    if isinstance(aabb, dict):
                        center = aabb.get("center", {})
                        size = aabb.get("size", {})
                        if isinstance(center, dict) and isinstance(size, dict):
                            cy = float(center.get("y", pos.get("y", 0)))
                            sy = float(size.get("y", 0.1))
                            target_y = cy + 0.5 * sy + self.placement_height_offset
                        else:
                            target_y = float(pos["y"]) + self.placement_height_offset
                    else:
                        target_y = float(pos["y"]) + self.placement_height_offset
                    
                    return {
                        "action": "PlaceObjectAtPoint",
                        "objectId": held_id,
                        "position": {
                            "x": target_x,
                            "y": target_y,
                            "z": target_z,
                        },
                        "forceKinematic": False,  # 允许物理模拟
                    }
            
            # 策略2: PutObject（默认，用于普通物体）
            # 关键修正：区分容器和表面
            action = {
                "action": "PutObject",
                "objectId": oid,  # 🔑 明确指定目标物体 ID
                "forceAction": bool(self.put_force_action),
            }
            
            # 如果目标不是容器，才使用屏幕坐标
            if not bool(obj.get("receptacle", False)):
                action["x"] = 0.5
                action["y"] = 0.5
                action["placeStationary"] = True
            
            return action

        # === 手里没物体 ===
        # 1. Pickupable
        if bool(obj.get("pickupable", False)) and bool(obj.get("visible", True)):
            return {
                "action": "PickupObject",
                "objectId": oid,
                "forceAction": bool(self.pickup_force_action)
            }

        # 2. Openable
        if bool(obj.get("openable", False)) and bool(obj.get("visible", True)):
            if bool(obj.get("isOpen", False)):
                return {
                    "action": "CloseObject",
                    "objectId": oid,
                    "forceAction": bool(self.open_force_action)
                }
            return {
                "action": "OpenObject",
                "objectId": oid,
                "openness": float(self.open_openness),
                "forceAction": bool(self.open_force_action)
            }

        # 3. Sliceable
        if self.enable_slice and bool(obj.get("sliceable", False)) and not bool(obj.get("isSliced", False)):
            return {"action": "SliceObject", "objectId": oid, "forceAction": False}
        
        # 4. Breakable
        if self.enable_break and bool(obj.get("breakable", False)) and not bool(obj.get("isBroken", False)):
            return {"action": "BreakObject", "objectId": oid, "forceAction": False}

        # 5. Toggleable
        if bool(obj.get("toggleable", False)) and bool(obj.get("visible", True)):
            if bool(obj.get("isToggled", False)):
                return {
                    "action": "ToggleObjectOff",
                    "objectId": oid,
                    "forceAction": bool(self.toggle_force_action)
                }
            return {
                "action": "ToggleObjectOn",
                "objectId": oid,
                "forceAction": bool(self.toggle_force_action)
            }

        return None

    def process_interaction(self, **kwargs) -> List[ActionSpec]:
        """将交互历史转换为动作序列"""
        current_history_len = len(self.interaction_history)
        
        if current_history_len == self._last_processed_history_len:
            return []
        
        if len(self.current_interaction) == 0:
            return []
        
        now_interaction = self.current_interaction[-1]
        actions: List[ActionSpec] = []
        for tok in now_interaction:
            actions.extend(self._token_to_actions(tok, **kwargs))
        
        self._last_processed_history_len = current_history_len
        return actions
    