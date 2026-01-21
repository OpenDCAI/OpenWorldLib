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
    交互输入目标：
    - 键盘：w/a/s/d 直接 -> MoveAhead/MoveLeft/MoveBack/MoveRight（均相对当前朝向）
    - 鼠标：按住拖拽
        dx -> RotateLeft/RotateRight（yaw）
        dy -> LookUp/LookDown（pitch）

    同时保留 8-token（forward/left/...）模式，方便 LLM 复用。
    """

    def __init__(
        self,
        # LLM/脚本 token 模式：仍然支持 8 token
        operation_types: Optional[List[str]] = None,
        interaction_template: Optional[List[str]] = None,

        # thor 参数
        grid_size: Optional[float] = None,
        rotate_deg: Optional[float] = None,
        look_deg: float = 30.0,
        camera_yaw_deg: Optional[float] = None,

        # 交互解析参数（鼠标像素阈值等）
        rot_step_pixels: float = 6.0,
        look_step_pixels: float = 8.0,
        max_yaw_per_tick: int = 3,
        max_pitch_per_tick: int = 2,
    ):
        super().__init__()

        if operation_types is None:
            operation_types = ["action_instruction"]

        # 保留你原先的 8-token（给 LLM 用）
        default_template = [
            "forward",
            "left",
            "right",
            "backward",
            "camera_l",
            "camera_r",
            "camera_up",
            "camera_down",
        ]
        if interaction_template is None:
            interaction_template = default_template
        else:
            if list(interaction_template) != default_template:
                raise ValueError(
                    f"interaction_template must be exactly {default_template}, got: {interaction_template}"
                )

        self.interaction_template = interaction_template
        self.interaction_template_init()

        self.operation_types = operation_types
        self.grid_size = grid_size
        self.rotate_deg = rotate_deg
        self.look_deg = float(look_deg)

        # yaw 步长：优先显式传入，其次 rotate_deg，否则 15
        if camera_yaw_deg is not None:
            self.camera_yaw_deg = float(camera_yaw_deg)
        elif rotate_deg is not None:
            self.camera_yaw_deg = float(rotate_deg)
        else:
            self.camera_yaw_deg = 15.0

        # 交互解析参数
        self.rot_step_pixels = float(rot_step_pixels)
        self.look_step_pixels = float(look_step_pixels)
        self.max_yaw_per_tick = int(max_yaw_per_tick)
        self.max_pitch_per_tick = int(max_pitch_per_tick)

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
        """
        pressed_keys: 例如 {'w','a'}，来自 pynput state.pressed
        mouse: dict，至少包含：
            - down: bool
            - dx: float (累计水平拖动像素)
            - dy: float (累计垂直拖动像素)

        返回：可直接喂给 representation.step 的 ActionSpec list
        注意：会就地消耗 mouse['dx']/mouse['dy']
        """

        rot_step_pixels = float(self.rot_step_pixels if rot_step_pixels is None else rot_step_pixels)
        look_step_pixels = float(self.look_step_pixels if look_step_pixels is None else look_step_pixels)
        max_yaw_per_tick = int(self.max_yaw_per_tick if max_yaw_per_tick is None else max_yaw_per_tick)
        max_pitch_per_tick = int(self.max_pitch_per_tick if max_pitch_per_tick is None else max_pitch_per_tick)

        actions: List[ActionSpec] = []

        # ---- keyboard: 每 tick 只做一个移动 ----
        # 这些移动在 AI2-THOR 中都是“相对当前朝向”的局部坐标移动
        if "w" in pressed_keys:
            actions.extend(self._move("MoveAhead"))
        elif "s" in pressed_keys:
            actions.extend(self._move("MoveBack"))
        elif "a" in pressed_keys:
            actions.extend(self._move("MoveLeft"))
        elif "d" in pressed_keys:
            actions.extend(self._move("MoveRight"))

        # ---- mouse look: only while button held ----
        if not bool(mouse.get("down", False)):
            return actions

        dx = float(mouse.get("dx", 0.0))
        dy = float(mouse.get("dy", 0.0))

        # dx -> yaw rotate
        if abs(dx) >= rot_step_pixels:
            n = min(int(abs(dx) // rot_step_pixels), max_yaw_per_tick)
            if n > 0:
                rot_action = "RotateRight" if dx > 0 else "RotateLeft"
                for _ in range(n):
                    actions.extend(self._rotate(rot_action, degrees=self.camera_yaw_deg))
                mouse["dx"] = dx - (_sign(dx) * n * rot_step_pixels)

        # dy -> pitch look
        if abs(dy) >= look_step_pixels:
            n = min(int(abs(dy) // look_step_pixels), max_pitch_per_tick)
            if n > 0:
                # dy < 0：鼠标往上拖 -> 抬头 LookUp
                look_action = "LookUp" if dy < 0 else "LookDown"
                for _ in range(n):
                    actions.append({"action": look_action, "degrees": float(self.look_deg)})
                mouse["dy"] = dy - (_sign(dy) * n * look_step_pixels)

        return actions

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

    def _token_to_actions(self, token: str) -> List[ActionSpec]:
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

        raise ValueError(f"Unknown token: {token}")

    def process_interaction(self) -> List[ActionSpec]:
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")

        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)

        actions: List[ActionSpec] = []
        for tok in now_interaction:
            actions.extend(self._token_to_actions(tok))
        return actions
