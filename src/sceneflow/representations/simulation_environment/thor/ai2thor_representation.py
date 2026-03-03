from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

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

    def _simple_result(self, ev: Any) -> Dict[str, Any]:
        """只需要成功/失败信息时的通用返回格式"""
        md = getattr(ev, "metadata", {}) or {}
        return {
            "lastActionSuccess": md.get("lastActionSuccess", None),
            "errorMessage": md.get("errorMessage", ""),
        }

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
            focus_id = self._get_object_in_frame(0.5, 0.5, checkVisible=focus_check_visible)
            focus_meta = self._get_object_meta(md, focus_id) if focus_id is not None else None
            obs["focus"] = {"objectId": focus_id, "object": focus_meta}
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

    def _get_coordinate_from_raycast(self, x: float, y: float) -> Any:
        if self.controller is None:
            return None
        q = self.controller.step(action="GetCoordinateFromRaycast", x=float(x), y=float(y))
        self._last_event = q
        return q.metadata.get("actionReturn", None)

    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
        if not isinstance(data, dict):
            raise TypeError(f"data must be dict, got {type(data)}")

        mode = str(data.get("mode", "observe"))

        # 公共渲染参数（仅 observe/step/init/reset/teleport 系会用到）
        include_depth     = bool(data.get("include_depth", False))
        include_instance  = bool(data.get("include_instance", False))
        attach_focus      = bool(data.get("attach_focus", False))
        focus_check_visible = bool(data.get("focus_check_visible", False))

        if mode == "init":
            self._ensure_controller()
            self._last_event = self.controller.last_event
            return self._event_to_obs(
                self._last_event,
                include_depth=include_depth, include_instance=include_instance,
                attach_focus=attach_focus, focus_check_visible=focus_check_visible,
            )

        if mode == "reset":
            self._ensure_controller()
            scene = data.get("scene", None) or self.scene
            ev = self.controller.reset(scene=scene)
            self._last_event = ev
            return self._event_to_obs(
                ev,
                include_depth=include_depth, include_instance=include_instance,
                attach_focus=attach_focus, focus_check_visible=focus_check_visible,
            )

        if mode == "close":
            self._close()
            return {"closed": True}

        if mode == "step":
            self._ensure_controller()
            action = data.get("action", None)
            if action is None:
                raise ValueError("mode=step requires data['action']")
            ev = self.controller.step(action) if isinstance(action, str) else self.controller.step(**action)
            self._last_event = ev
            return self._event_to_obs(
                ev,
                include_depth=include_depth, include_instance=include_instance,
                attach_focus=attach_focus, focus_check_visible=focus_check_visible,
            )

        if mode == "query":
            self._ensure_controller()
            query = str(data.get("query", ""))
            if query == "raycast":
                coord = self._get_coordinate_from_raycast(float(data.get("x", 0.5)), float(data.get("y", 0.5)))
                return {"actionReturn": coord}
            if query == "focus":
                oid = self._get_object_in_frame(
                    float(data.get("x", 0.5)), float(data.get("y", 0.5)),
                    checkVisible=bool(data.get("checkVisible", False)),
                )
                md = getattr(self._last_event, "metadata", {}) or {}
                return {"actionReturn": oid, "object": self._get_object_meta(md, oid)}
            return {"error": f"Unknown query: {query}"}

        if mode == "teleport":
            self._ensure_controller()
            tp_kwargs: Dict[str, Any] = {}
            for k in ("position", "rotation", "horizon", "standing"):
                if k in data:
                    tp_kwargs[k] = data[k]
            ev = self.controller.step(action="Teleport", **tp_kwargs)
            self._last_event = ev
            return self._event_to_obs(
                ev,
                include_depth=include_depth, include_instance=include_instance,
                attach_focus=attach_focus, focus_check_visible=focus_check_visible,
            )

        if mode == "teleport_full":
            # 所有字段必填
            self._ensure_controller()
            ev = self.controller.step(
                action="TeleportFull",
                position=data["position"],
                rotation=data["rotation"],
                horizon=data["horizon"],
                standing=data["standing"],
            )
            self._last_event = ev
            return self._event_to_obs(
                ev,
                include_depth=include_depth, include_instance=include_instance,
                attach_focus=attach_focus, focus_check_visible=focus_check_visible,
            )

        if mode == "get_reachable_positions":
            self._ensure_controller()
            ev = self.controller.step(action="GetReachablePositions")
            self._last_event = ev
            return {"positions": ev.metadata.get("actionReturn", [])}

        if mode == "set_object_poses":
            # data["objectPoses"]: list[{"objectName":…, "position":…, "rotation":…}]
            # 注意：未在列表中指定的 moveable/pickupable 物体会被移除
            self._ensure_controller()
            ev = self.controller.step(
                action="SetObjectPoses",
                objectPoses=data["objectPoses"],
            )
            self._last_event = ev
            return self._simple_result(ev)

        if mode == "remove_object":
            # data["objectId"]: str — 永久删除，场景 reset 前不可恢复
            self._ensure_controller()
            ev = self.controller.step(action="RemoveFromScene", objectId=data["objectId"])
            self._last_event = ev
            return self._simple_result(ev)

        if mode == "disable_object":
            # data["objectId"]: str — 隐藏但不删除，可用 enable_object 恢复
            self._ensure_controller()
            ev = self.controller.step(action="DisableObject", objectId=data["objectId"])
            self._last_event = ev
            return self._simple_result(ev)

        if mode == "enable_object":
            # data["objectId"]: str — 恢复被 disable 的物体到原始位置
            self._ensure_controller()
            ev = self.controller.step(action="EnableObject", objectId=data["objectId"])
            self._last_event = ev
            return self._simple_result(ev)

        if mode == "randomize_materials":
            # 可选参数：useTrainMaterials / useValMaterials / useTestMaterials / inRoomTypes
            self._ensure_controller()
            kwargs = {k: data[k] for k in (
                "useTrainMaterials", "useValMaterials", "useTestMaterials", "inRoomTypes"
            ) if k in data}
            ev = self.controller.step(action="RandomizeMaterials", **kwargs)
            self._last_event = ev
            return self._simple_result(ev)

        if mode == "randomize_lighting":
            # 可选参数：brightness / randomizeColor / hue / saturation / synchronized
            self._ensure_controller()
            kwargs = {k: data[k] for k in (
                "brightness", "randomizeColor", "hue", "saturation", "synchronized"
            ) if k in data}
            ev = self.controller.step(action="RandomizeLighting", **kwargs)
            self._last_event = ev
            return self._simple_result(ev)

        if mode == "randomize_colors":
            self._ensure_controller()
            ev = self.controller.step(action="RandomizeColors")
            self._last_event = ev
            return self._simple_result(ev)

        if mode == "initial_random_spawn":
            # 可选参数见官方文档（https://ai2thor.allenai.org/ithor/documentation）：randomSeed / forceVisible / numPlacementAttempts /
            # placeStationary / numDuplicatesOfType / excludedReceptacles / excludedObjectIds
            self._ensure_controller()
            kwargs = {k: data[k] for k in (
                "randomSeed", "forceVisible", "numPlacementAttempts",
                "placeStationary", "numDuplicatesOfType",
                "excludedReceptacles", "excludedObjectIds",
            ) if k in data}
            ev = self.controller.step(action="InitialRandomSpawn", **kwargs)
            self._last_event = ev
            # InitialRandomSpawn 后 objectId 会重新计算，把新 objects 一并返回
            md = getattr(ev, "metadata", {}) or {}
            return {
                **self._simple_result(ev),
                "objects": md.get("objects", []),
            }

        if mode == "set_mass":
            # 必填：objectId / mass / drag / angularDrag
            self._ensure_controller()
            ev = self.controller.step(
                action="SetMassProperties",
                objectId=data["objectId"],
                mass=float(data["mass"]),
                drag=float(data.get("drag", 1.0)),
                angularDrag=float(data.get("angularDrag", 0.05)),
            )
            self._last_event = ev
            return self._simple_result(ev)

        if mode == "set_temp_decay":
            # 按类型设定衰减时间
            # 必填：objectType / TimeUntilRoomTemp
            self._ensure_controller()
            ev = self.controller.step(
                action="SetRoomTempDecayTimeForType",
                objectType=data["objectType"],
                TimeUntilRoomTemp=float(data["TimeUntilRoomTemp"]),
            )
            self._last_event = ev
            return self._simple_result(ev)

        if mode == "set_global_temp_decay":
            # 全局衰减时间，必填：TimeUntilRoomTemp
            self._ensure_controller()
            ev = self.controller.step(
                action="SetGlobalRoomTempDecayTime",
                TimeUntilRoomTemp=float(data["TimeUntilRoomTemp"]),
            )
            self._last_event = ev
            return self._simple_result(ev)

        if mode == "set_decay_bool":
            # 开关温度衰减，必填：allowDecayTemperature (bool)
            self._ensure_controller()
            ev = self.controller.step(
                action="SetDecayTemperatureBool",
                allowDecayTemperature=bool(data.get("allowDecayTemperature", True)),
            )
            self._last_event = ev
            return self._simple_result(ev)

        if mode == "set_object_state":
            self._ensure_controller()
            oid = data["objectId"]
            results: List[Dict[str, Any]] = []

            action_map = [
                ("cook",    lambda: self.controller.step(action="CookObject",            objectId=oid, forceAction=True)),
                ("fill",    lambda: self.controller.step(action="FillObjectWithLiquid",  objectId=oid, fillLiquid=str(data["fill"]), forceAction=True)),
                ("empty",   lambda: self.controller.step(action="EmptyLiquidFromObject", objectId=oid, forceAction=True)),
                ("clean",   lambda: self.controller.step(action="CleanObject",           objectId=oid, forceAction=True)),
                ("dirty",   lambda: self.controller.step(action="DirtyObject",           objectId=oid, forceAction=True)),
                ("use_up",  lambda: self.controller.step(action="UseUpObject",           objectId=oid, forceAction=True)),
                ("break_",  lambda: self.controller.step(action="BreakObject",           objectId=oid, forceAction=True)),
                ("slice_",  lambda: self.controller.step(action="SliceObject",           objectId=oid, forceAction=True)),
            ]

            for key, fn in action_map:
                if key not in data:
                    continue
                # fill 的值是液体类型字符串，其余是 bool
                if key == "fill" and not isinstance(data[key], str):
                    continue
                if key != "fill" and not bool(data[key]):
                    continue
                ev = fn()
                self._last_event = ev
                results.append({"action": key, **self._simple_result(ev)})

            return {"results": results}

        self._ensure_controller()
        if self._last_event is None and self.controller is not None:
            self._last_event = self.controller.last_event
        return self._event_to_obs(
            self._last_event,
            include_depth=include_depth, include_instance=include_instance,
            attach_focus=attach_focus, focus_check_visible=focus_check_visible,
        )
