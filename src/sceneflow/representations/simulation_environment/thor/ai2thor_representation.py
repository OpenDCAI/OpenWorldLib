from __future__ import annotations

from typing import Any, Dict, Optional, Union, List

from ...base_representation import BaseRepresentation

from .ai2thor.controller import Controller
from .ai2thor.platform import CloudRendering


class Ai2ThorRepresentation(BaseRepresentation):
    def __init__(
        self,
        executable_path: Optional[str] = None,
        quality: Optional[str] = None,
        scene: str = "FloorPlan212",
        visibilityDistance: float = 1.5,
        gridSize: float = 0.25,
        rotateStepDegrees: int = 90,
        width: int = 300,
        height: int = 300,
        fieldOfView: int = 90,
        renderDepthImage: bool = False,
        renderInstanceSegmentation: bool = False,
        headless: bool = False,
        snapToGrid: bool = True,
        agentMode: str = "default",
    ):
        super().__init__()

        self.executable_path = executable_path                              # ai2thor可执行文件路径
        self.quality = quality                                              # 渲染质量
        self.scene = scene                                                  # 默认场景
        self.visibilityDistance = visibilityDistance                        # 可见距离
        self.gridSize = gridSize                                            # 移动距离
        self.rotateStepDegrees = rotateStepDegrees                          # 旋转角度
        self.width = width                                                  # 可视图像宽度
        self.height = height                                                # 图像高度
        self.fieldOfView = fieldOfView                                      # 视野角度
        self.renderDepthImage = renderDepthImage                            # 是否渲染深度图
        self.renderInstanceSegmentation = renderInstanceSegmentation        # 是否渲染实例分割图
        self.headless = headless                                            # 是否无头模式
        self.snapToGrid = snapToGrid                                        # 是否贴合网格移动
        self.agentMode = agentMode                                          # agent模式

        self.controller: Optional[Controller] = None                        # Controller实例,初始为None
        self._last_event: Any = None

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str = "", device=None, **kwargs):
        # ai2thor 不是权重模型;保持接口一致即可
        return cls(**kwargs)

    @staticmethod
    def _get_inventory_objects(md: Dict[str, Any]) -> List[Dict[str, Any]]:
        inv_top = md.get("inventoryObjects", None)
        if isinstance(inv_top, list):
            return inv_top
        agent = md.get("agent", {}) or {}
        inv_agent = agent.get("inventoryObjects", None)
        if isinstance(inv_agent, list):
            return inv_agent
        return []

    def _ensure_controller(self) -> None:
        """内部使用:保证 controller 已初始化。"""
        if self.controller is not None:
            return

        kwargs: Dict[str, Any] = dict(
            agentMode=self.agentMode,
            visibilityDistance=float(self.visibilityDistance),
            quality=self.quality,
            scene=self.scene,
            gridSize=float(self.gridSize),
            snapToGrid=bool(self.snapToGrid),
            rotateStepDegrees=int(self.rotateStepDegrees),
            renderDepthImage=bool(self.renderDepthImage),
            renderInstanceSegmentation=bool(self.renderInstanceSegmentation),
            width=int(self.width),
            height=int(self.height),
            fieldOfView=int(self.fieldOfView),
        )

        if self.executable_path is not None:
            kwargs["local_executable_path"] = self.executable_path

        if self.headless:
            kwargs["platform"] = CloudRendering

        self.controller = Controller(**kwargs)
        self._last_event = self.controller.last_event

    def _close(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
        self._last_event = None

    def _event_to_obs(
        self,
        event: Any,
        *,
        include_depth: bool = False,
        include_instance: bool = False,
        attach_focus: bool = False,
        focus_check_visible: bool = False,
    ) -> Dict[str, Any]:
        md = getattr(event, "metadata", {}) or {}
        obs: Dict[str, Any] = {
            "frame": getattr(event, "frame", None),
            "agent": md.get("agent", {}),
            "objects": md.get("objects", []),
            "sceneName": md.get("sceneName", ""),
            "lastAction": md.get("lastAction", ""),
            "lastActionSuccess": md.get("lastActionSuccess", None),
            "errorMessage": md.get("errorMessage", ""),
            "isSceneAtRest": md.get("isSceneAtRest", None),
            "actionReturn": md.get("actionReturn", None),
        }

        if include_depth:
            obs["depth_frame"] = getattr(event, "depth_frame", None)

        if include_instance:
            obs["instance_segmentation_frame"] = getattr(event, "instance_segmentation_frame", None)
            obs["instance_masks"] = getattr(event, "instance_masks", None)
            obs["instance_detections2D"] = getattr(event, "instance_detections2D", None)

        if attach_focus:
            # 通过环境 Query 得到 focus
            focus_id = self._get_object_in_frame(0.5, 0.5, checkVisible=focus_check_visible)
            focus_meta = self._get_object_meta(md, focus_id) if focus_id is not None else None
            obs["focus"] = {
                "objectId": focus_id,
                "object": focus_meta,
            }

            # inventory 简化信息(供 operator interact 使用)
            inv = self._get_inventory_objects(md)
            obs["inventory"] = {
                "has_in_hand": len(inv) > 0,
                "held_object_id": (inv[0].get("objectId") if len(inv) > 0 else None),
            }

        return obs

    def _get_object_meta(self, md: Dict[str, Any], object_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if object_id is None:
            return None
        for o in md.get("objects", []):
            if o.get("objectId") == object_id:
                return o
        return None

    def _get_object_in_frame(self, x: float, y: float, checkVisible: bool = False) -> Optional[str]:
        if self.controller is None:
            return None
        q = self.controller.step(action="GetObjectInFrame", x=float(x), y=float(y), checkVisible=bool(checkVisible))
        self._last_event = q
        return q.metadata.get("actionReturn", None)

    def _get_coordinate_from_raycast(self, x: float, y: float):
        if self.controller is None:
            return None
        q = self.controller.step(action="GetCoordinateFromRaycast", x=float(x), y=float(y))
        self._last_event = q
        return q.metadata.get("actionReturn", None)

    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        BaseRepresentation 模版接口

        data 约定:
        - {"mode": "init"|"reset"|"close"|"observe"}
        - {"mode":"step", "action": <str|dict>}
        - {"mode":"query", "query":"focus"|"raycast", ...}

        常用 kwargs:
        - include_depth/include_instance
        - attach_focus/focus_check_visible(把 focus + inventory 附到 obs 里,供 operator 用)
        """
        if not isinstance(data, dict):
            raise TypeError(f"data must be dict, got {type(data)}")

        mode = str(data.get("mode", "observe"))

        include_depth = bool(data.get("include_depth", False))
        include_instance = bool(data.get("include_instance", False))
        attach_focus = bool(data.get("attach_focus", False))
        focus_check_visible = bool(data.get("focus_check_visible", False))

        if mode == "init":
            self._ensure_controller()
            assert self.controller is not None
            self._last_event = self.controller.last_event
            return self._event_to_obs(
                self._last_event,
                include_depth=include_depth,
                include_instance=include_instance,
                attach_focus=attach_focus,
                focus_check_visible=focus_check_visible,
            )

        if mode == "reset":
            self._ensure_controller()
            assert self.controller is not None
            scene = data.get("scene", None) or self.scene
            ev = self.controller.reset(scene=scene)
            self._last_event = ev
            return self._event_to_obs(
                ev,
                include_depth=include_depth,
                include_instance=include_instance,
                attach_focus=attach_focus,
                focus_check_visible=focus_check_visible,
            )

        if mode == "close":
            self._close()
            return {"closed": True}

        if mode == "step":
            self._ensure_controller()
            assert self.controller is not None
            action = data.get("action", None)
            if action is None:
                raise ValueError("mode=step requires data['action']")
            if isinstance(action, str):
                ev = self.controller.step(action)
            elif isinstance(action, dict):
                ev = self.controller.step(**action)
            else:
                raise TypeError(f"Unsupported action type: {type(action)}")
            self._last_event = ev
            return self._event_to_obs(
                ev,
                include_depth=include_depth,
                include_instance=include_instance,
                attach_focus=attach_focus,
                focus_check_visible=focus_check_visible,
            )

        if mode == "query":
            # query 也通过 get_representation 走
            self._ensure_controller()
            query = str(data.get("query", ""))
            if query == "raycast":
                coord = self._get_coordinate_from_raycast(float(data.get("x", 0.5)), float(data.get("y", 0.5)))
                return {"actionReturn": coord}
            if query == "focus":
                oid = self._get_object_in_frame(float(data.get("x", 0.5)), float(data.get("y", 0.5)),
                                                checkVisible=bool(data.get("checkVisible", False)))
                md = getattr(self._last_event, "metadata", {}) or {}
                return {"actionReturn": oid, "object": self._get_object_meta(md, oid)}
            return {"error": f"Unknown query: {query}"}

        # observe: 不 step,仅把 last_event 转成 obs
        self._ensure_controller()
        if self._last_event is None and self.controller is not None:
            self._last_event = self.controller.last_event
        return self._event_to_obs(
            self._last_event,
            include_depth=include_depth,
            include_instance=include_instance,
            attach_focus=attach_focus,
            focus_check_visible=focus_check_visible,
        )
    