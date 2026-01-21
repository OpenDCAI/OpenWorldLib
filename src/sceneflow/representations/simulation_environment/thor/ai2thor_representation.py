from __future__ import annotations

from typing import Any, Dict, Optional, Union

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
        scene: str = "FloorPlan212",                  # 默认场景，共120个场景，详细可见：https://ai2thor.allenai.org/ithor/documentation/scenes
        visibilityDistance: float = 1.5,              # 可见距离，单位米
        gridSize: float = 0.25,                       # 移动距离，单位米
        rotateStepDegrees: int = 90,                  # 旋转角度，单位度
        width: int = 300,                             # 画面宽度，单位像素
        height: int = 300,                            # 画面高度，单位像素
        fieldOfView: int = 90,                        # 视野，单位度
        renderDepthImage: bool = False,               # 是否渲染深度图
        renderInstanceSegmentation: bool = False,     # 是否渲染实例分割图
        headless: bool = False,                       # 是否无头模式（不弹 Unity 窗口）
        snapToGrid: bool = True,                      # 决定智能体在执行任何移动动作（如“前进”和“完全传送”）后是否将其位置对齐到网格点。网格点之间的间距由gridSize决定。设置为False可允许对角线移动。
        agentMode: str = "default",                   # 代理模式，默认"default"
    ):
        self.executable_path = executable_path
        self.scene = scene
        self.visibilityDistance = visibilityDistance
        self.gridSize = gridSize
        self.rotateStepDegrees = rotateStepDegrees
        self.width = width
        self.height = height
        self.fieldOfView = fieldOfView
        self.renderDepthImage = renderDepthImage
        self.renderInstanceSegmentation = renderInstanceSegmentation
        self.headless = headless
        self.snapToGrid = snapToGrid
        self.agentMode = agentMode

        self.controller: Optional[Controller] = None

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

        # Headless/off-screen（不弹 Unity 窗口），那些云服务器平台需要此参数
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

    def close(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
