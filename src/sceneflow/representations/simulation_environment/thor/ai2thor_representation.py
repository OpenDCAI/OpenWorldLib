from __future__ import annotations

from typing import Any, Dict, Optional, Union, List

from .ai2thor.controller import Controller
from .ai2thor.platform import CloudRendering


class Ai2ThorRepresentation:
    """
    - controller_init(): 初始化 Controller
    - step(): 执行动作，返回 Event
    - get_representation(): 从 Event 里取 frame / metadata 等
    """

    def __init__(
        self,
        executable_path: Optional[str] = None,
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
        self.executable_path = executable_path                              # 可选，ai2thor可执行文件路径
        self.scene = scene                                                  # 默认场景，具体可以在https://ai2thor.allenai.org/ithor/documentation/scenes查看详情
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

        self.controller: Optional[Controller] = None                        # Controller实例，初始为None

    def controller_init(self) -> None:
        kwargs: Dict[str, Any] = dict(
            agentMode=self.agentMode,
            visibilityDistance=float(self.visibilityDistance),
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

    def reset(self, scene: Optional[str] = None, **kwargs) -> Any:
        if self.controller is None:
            raise RuntimeError("Controller not initialized. Call controller_init() first.")
        if scene is None:
            scene = self.scene
        return self.controller.reset(scene=scene, **kwargs)

    def step(self, action: Union[str, Dict[str, Any]]) -> Any:
        if self.controller is None:
            raise RuntimeError("Controller not initialized. Call controller_init() first.")
        if isinstance(action, str):
            return self.controller.step(action)
        if isinstance(action, dict):
            return self.controller.step(**action)
        raise TypeError(f"Unsupported action type: {type(action)}")

    def get_representation(
        self,
        event: Any,
        include_depth: bool = False,
        include_instance: bool = False,
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
        }

        if include_depth:
            obs["depth_frame"] = getattr(event, "depth_frame", None)

        if include_instance:
            obs["instance_segmentation_frame"] = getattr(event, "instance_segmentation_frame", None)
            obs["instance_masks"] = getattr(event, "instance_masks", None)
            obs["instance_detections2D"] = getattr(event, "instance_detections2D", None)

        return obs

    def get_object_in_frame(self, x: float, y: float, checkVisible: bool = False) -> Optional[str]:
        if self.controller is None:
            raise RuntimeError("Controller not initialized. Call controller_init() first.")
        q = self.controller.step(action="GetObjectInFrame", x=float(x), y=float(y), checkVisible=bool(checkVisible))
        return q.metadata.get("actionReturn", None)

    def get_coordinate_from_raycast(self, x: float, y: float):
        if self.controller is None:
            raise RuntimeError("Controller not initialized. Call controller_init() first.")
        q = self.controller.step(action="GetCoordinateFromRaycast", x=float(x), y=float(y))
        return q.metadata.get("actionReturn", None)

    # ===== NEW: inventory helper =====
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

    def agent_has_in_hand(self, event: Any) -> bool:
        md = getattr(event, "metadata", {}) or {}
        inv = self._get_inventory_objects(md)
        return len(inv) > 0

    def held_object_id(self, event: Any) -> Optional[str]:
        md = getattr(event, "metadata", {}) or {}
        inv = self._get_inventory_objects(md)
        if len(inv) == 0:
            return None
        return inv[0].get("objectId", None)

    # ===== NEW: object meta helper =====
    def get_object_meta(self, event: Any, object_id: str) -> Optional[Dict[str, Any]]:
        md = getattr(event, "metadata", {}) or {}
        for o in md.get("objects", []):
            if o.get("objectId") == object_id:
                return o
        return None

    # ===== NEW: focus helper (center) =====
    def get_focus_object(self, event: Any, checkVisible: bool = False) -> Optional[str]:
        return self.get_object_in_frame(0.5, 0.5, checkVisible=checkVisible)
    
    def get_focus_coordinate(self) -> Optional[Dict[str, float]]:
        """Raycast at center crosshair (0.5, 0.5)."""
        coord = self.get_coordinate_from_raycast(0.5, 0.5)
        if not isinstance(coord, dict):
            return None
        if not all(k in coord for k in ("x", "y", "z")):
            return None
        return {"x": float(coord["x"]), "y": float(coord["y"]), "z": float(coord["z"])}

    def close(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
