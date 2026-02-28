from __future__ import annotations

import json
import math
import re
import threading
import cv2
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from ...base_representation import BaseRepresentation

from .simworld.config import Config
from .simworld.communicator.unrealcv import UnrealCV
from .simworld.communicator.communicator import Communicator
from .simworld.agent.humanoid import Humanoid
from .simworld.agent.pedestrian import Pedestrian
from .simworld.utils.vector import Vector
from .simworld.traffic.controller.traffic_controller import TrafficController


class SimWorldRepresentation(BaseRepresentation):
    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 9000,
        resolution: tuple = (1280, 720),
        camera_id: int = 1,
        camera_ids: Optional[List[int]] = None,
        num_agents: int = 1,
        agent_model_path: str = "/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C",
        spawn_position: tuple = (0, 0),
        spawn_direction: tuple = (0, 1),
        ue_manager_path: str = "/Game/TrafficSystem/UE_Manager.UE_Manager_C",
        scooter_model_path: str = "/Game/ScooterAssets/Blueprints/BP_Scooter_Pawn.BP_Scooter_Pawn_C",
        traffic_light_model_path: str = "/Game/city_props/BP/props/street_light/BP_street_light.BP_street_light_C",
        pedestrian_light_model_path: str = "/Game/city_props/BP/props/street_light/BP_street_light_ped.BP_street_light_ped_C",
        pedestrian_model_path: str = "/Game/TrafficSystem/Pedestrian/Base_Pedestrian.Base_Pedestrian_C",
        config_path: Optional[str] = None,
        spawn_objects: Optional[List[Dict[str, Any]]] = None,
        interact_radius: float = 30.0,
        spawn_scooter: Optional[Dict[str, Any]] = None,
        scooter_follow_cam_id: int = 0,
        scooter_cam_offset_z: float = 200.0,
        scooter_cam_pitch: float = -20.0,
    ):
        super().__init__()

        self.ip = ip
        self.port = port
        self.resolution = resolution
        self.camera_id = camera_id
        self.camera_ids = camera_ids if camera_ids is not None else [camera_id]

        self.num_agents = num_agents
        self.agent_model_path = agent_model_path
        self.spawn_position = spawn_position
        self.spawn_direction = spawn_direction

        self.ue_manager_path = ue_manager_path
        self.scooter_model_path = scooter_model_path
        self.traffic_light_model_path = traffic_light_model_path
        self.pedestrian_light_model_path = pedestrian_light_model_path
        self.pedestrian_model_path = pedestrian_model_path

        self.config_path = config_path

        self.seed: Optional[int] = None
        self.dt: float = 0.1

        self.world_json_path: Optional[str] = None
        self.ue_asset_path: Optional[str] = None
        self._world_map: Optional[dict] = None

        self.spawn_objects: List[Dict[str, Any]] = spawn_objects if spawn_objects is not None else []
        self.spawn_scooter_cfg: Optional[Dict[str, Any]] = spawn_scooter
        self.interact_radius = float(interact_radius)

        self.scooter_follow_cam_id = int(scooter_follow_cam_id)
        self.scooter_cam_offset_z = float(scooter_cam_offset_z)
        self.scooter_cam_pitch = float(scooter_cam_pitch)
        self._last_scooter_cam_yaw: float = 0.0

        self.communicator: Optional[Communicator] = None
        self.unrealcv: Optional[UnrealCV] = None

        self.agent: Optional[Humanoid] = None
        self.agent_name: Optional[str] = None

        self.agents: List[Humanoid] = []
        self.agent_names: List[str] = []

        self._traffic_ctrl: Optional[TrafficController] = None
        self._traffic_thread: Optional[threading.Thread] = None
        self._traffic_exit: threading.Event = threading.Event()

        self._has_object: bool = False
        self._picked_object_name: Optional[str] = None
        self._object_positions: Dict[str, tuple] = {
            obj["object_name"]: tuple(obj.get("position", (0, 0, 0)))
            for obj in self.spawn_objects
        }
        self._drop_positions: Dict[str, tuple] = {}
        self._dropped_pending: Dict[str, bool] = {}
        self._object_pos_trust: Dict[str, bool] = {
            obj["object_name"]: True for obj in self.spawn_objects
        }
        self._drop_requery_counter: Dict[str, int] = {}
        self._prev_collision_obj: int = 0

        self._in_vehicle: bool = False
        self._in_vehicle_name: Optional[str] = None

        self._on_scooter: bool = False
        self._scooter_obj = None
        self._scooter_id: Optional[int] = None
        self._scooter_name: Optional[str] = None

        self._npc_actor_names: Dict[str, str] = {}

        if config_path is not None:
            self._apply_config(config_path)

    def _apply_config(self, config_path: str) -> None:
        try:
            cfg = Config(config_path)
            self.ue_manager_path = cfg.get("simworld.ue_manager_path", self.ue_manager_path)
            self.seed = cfg.get("simworld.seed", self.seed)
            self.dt = cfg.get("simworld.dt", self.dt)
            self.agent_model_path = cfg.get("user.model_path", self.agent_model_path)
            self.num_agents = cfg.get("user.num_agents", self.num_agents)
            self.scooter_model_path = cfg.get("scooter.model_path", self.scooter_model_path)
            self.traffic_light_model_path = cfg.get(
                "traffic.traffic_signal.traffic_light_model_path", self.traffic_light_model_path)
            self.pedestrian_light_model_path = cfg.get(
                "traffic.traffic_signal.pedestrian_light_model_path", self.pedestrian_light_model_path)
            self.pedestrian_model_path = cfg.get(
                "traffic.pedestrian.model_path", self.pedestrian_model_path)
            self.world_json_path = cfg.get("citygen.world_json", self.world_json_path)
            self.ue_asset_path = cfg.get("citygen.ue_asset_path", self.ue_asset_path)
        except Exception as e:
            print(f"[SW] Warning: Failed to apply config {config_path}: {e}")

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str = "", device=None, **kwargs):
        config_path = kwargs.pop("config_path", pretrained_model_path)
        if config_path and Path(config_path).exists():
            return cls(config_path=config_path, **kwargs)
        return cls(**kwargs)

    def _ensure_connection(self) -> None:
        if self.communicator is not None:
            return
        print("[SW] connecting unrealcv...")
        self.unrealcv = UnrealCV(port=self.port, ip=self.ip, resolution=self.resolution)
        self.communicator = Communicator(self.unrealcv)
        if self.config_path is not None:
            self._init_full_env()
        else:
            self._init_humanoid_only()

    def _init_humanoid_only(self) -> None:
        print("[SW] light mode: spawning humanoid only...")
        self._spawn_agents(self.num_agents)
        self.communicator.spawn_ue_manager(self.ue_manager_path)
        self._spawn_extra_objects()
        if self.spawn_scooter_cfg is not None:
            self._spawn_scooter_from_cfg(self.spawn_scooter_cfg)
        print("[SW] light mode ready.")

    def _init_full_env(self) -> None:
        print("[SW] full mode: initializing environment from config...")
        cfg = Config(self.config_path)
        self._traffic_ctrl = TrafficController(
            config=cfg,
            num_vehicles=cfg.get("traffic.num_vehicles", 0),
            num_pedestrians=cfg.get("traffic.num_pedestrians", 0),
            map=cfg.get("traffic.map_path", None),
            seed=cfg.get("simworld.seed", None),
            dt=cfg.get("simworld.dt", 0.1),
        )
        self._traffic_ctrl.init_communicator(self.communicator)
        print("[SW] spawning traffic objects...")
        self._traffic_ctrl.spawn_objects_in_unreal_engine()
        self.communicator.spawn_ue_manager(self.ue_manager_path)

        self._traffic_exit.clear()
        self._traffic_thread = threading.Thread(
            target=self._run_traffic_simulation, daemon=True, name="SimWorldTrafficThread"
        )
        self._traffic_thread.start()
        print("[SW] traffic simulation thread started.")

        if self.world_json_path and Path(self.world_json_path).exists():
            with open(self.world_json_path, encoding="utf-8") as f:
                self._world_map = json.load(f)
            print(f"[SW] world map loaded: {self.world_json_path}")
        else:
            self._world_map = None

        print(f"[SW] spawning {self.num_agents} humanoid agent(s)...")
        self._spawn_agents(self.num_agents)
        self._spawn_extra_objects()
        if self.spawn_scooter_cfg is not None:
            self._spawn_scooter_from_cfg(self.spawn_scooter_cfg)
        print("[SW] full mode ready.")

    def _spawn_agents(self, count: int = 1) -> None:
        if isinstance(self.spawn_position, list):
            positions = self.spawn_position
        else:
            positions = [self.spawn_position] * count

        if isinstance(self.spawn_direction, list):
            directions = self.spawn_direction
        else:
            directions = [self.spawn_direction] * count

        for i in range(max(1, count)):
            pos = positions[i]  if i < len(positions)  else positions[-1]
            dir_ = directions[i] if i < len(directions) else directions[-1]
            agent = self._spawn_humanoid(
                Vector(float(pos[0]), float(pos[1])),
                Vector(float(dir_[0]), float(dir_[1]))
            )
            name = self.communicator.get_humanoid_name(agent.id)
            self.agents.append(agent)
            self.agent_names.append(name)

        self.agent = self.agents[0]
        self.agent_name = self.agent_names[0]

    def _spawn_humanoid(self, pos: Vector, dir_vec: Vector) -> Humanoid:
        try:
            if self.config_path:
                cfg = Config(self.config_path)
                agent = Humanoid(position=pos, direction=dir_vec,
                                 config=cfg, communicator=self.communicator)
            else:
                agent = Humanoid(position=pos, direction=dir_vec)
        except TypeError:
            agent = Humanoid(position=pos, direction=dir_vec)
        self.communicator.spawn_agent(agent, name=None,
                                      model_path=self.agent_model_path, type="humanoid")
        return agent

    def _spawn_extra_objects(self) -> None:
        for obj in self.spawn_objects:
            object_name = obj.get("object_name", "")
            model_path = obj.get("model_path", "")
            position = obj.get("position", (0, 0, 0))
            direction = obj.get("direction", (0, 0, 0))
            obj_type = obj.get("type", "object").lower()

            if not object_name or not model_path:
                print(f"[SW] Warning: skipping invalid spawn_object entry: {obj}")
                continue

            try:
                if obj_type == "dog":
                    try:
                        self.communicator.spawn_object(object_name, model_path, position, direction)
                    except Exception:
                        self.unrealcv.spawn_bp_asset(model_path, object_name)
                        self.unrealcv.set_location(
                            object_name,
                            (float(position[0]), float(position[1]),
                             float(position[2]) if len(position) > 2 else 0.0),
                        )
                    try:
                        yaw = float(direction[1]) if len(direction) > 1 else 0.0
                        self.unrealcv.set_orientation((0.0, yaw, 0.0), object_name)
                    except Exception as e:
                        print(f"[SW] dog set_orientation ignored: {e}")
                    try:
                        self.unrealcv.enable_controller(object_name, True)
                    except Exception as e:
                        print(f"[SW] dog enable_controller ignored: {e}")
                    self._npc_actor_names[object_name] = object_name
                    print(f"[SW] spawned dog: {object_name}")

                elif obj_type == "pedestrian":
                    ped_pos = Vector(float(position[0]), float(position[1]))
                    ped_dir = Vector(
                        float(direction[0]) if len(direction) > 0 else 0.0,
                        float(direction[1]) if len(direction) > 1 else 1.0,
                    )
                    ped = Pedestrian(position=ped_pos, direction=ped_dir)
                    self.communicator.spawn_agent(
                        ped, name=object_name,
                        model_path=model_path, type="pedestrian",
                    )
                    try:
                        actor_name = self.communicator.get_pedestrian_name(ped.id)
                    except Exception:
                        actor_name = object_name
                    self._npc_actor_names[object_name] = actor_name
                    self._object_positions[object_name] = (
                        float(position[0]), float(position[1]),
                        float(position[2]) if len(position) > 2 else 0.0,
                    )
                    print(f"[SW] spawned pedestrian: {object_name} → actor={actor_name}")

                else:
                    self.communicator.spawn_object(object_name, model_path, position, direction)
                    print(f"[SW] spawned object: {object_name}")

            except Exception as e:
                print(f"[SW] Warning: failed to spawn {obj_type} {object_name!r}: {e}")

    def _spawn_scooter_from_cfg(self, cfg: Dict[str, Any]) -> None:
        from .simworld.agent.scooter import Scooter
        pos = cfg.get("position", (100, 0))
        direction = cfg.get("direction", (0, 1))
        model_path = cfg.get("model_path", self.scooter_model_path)
        scooter = Scooter(Vector(pos[0], pos[1]), Vector(direction[0], direction[1]))
        self.communicator.spawn_scooter(scooter, model_path)
        self._scooter_obj = scooter
        self._scooter_id = scooter.id
        self._scooter_name = self.communicator.get_scooter_name(scooter.id)
        print(f"[SW] scooter spawned: id={self._scooter_id}, name={self._scooter_name}")

    def _run_traffic_simulation(self) -> None:
        try:
            self._traffic_ctrl.simulation(
                physical_update_function=lambda: None,
                exit_event=self._traffic_exit,
            )
        except Exception as e:
            print(f"[SW] Traffic simulation thread error: {e}")

    def _stop_traffic(self) -> None:
        if self._traffic_thread is not None and self._traffic_thread.is_alive():
            self._traffic_exit.set()
            self._traffic_thread.join(timeout=3.0)
            self._traffic_thread = None
        self._traffic_ctrl = None

    def _close(self) -> None:
        self._stop_traffic()
        if self.communicator is not None:
            try:
                self.communicator.disconnect()
            except Exception as e:
                print(f"[SW] Warning during disconnect: {e}")
            finally:
                self.communicator = None
                self.unrealcv = None
                self.agent = None
                self.agent_name = None
                self.agents.clear()
                self.agent_names.clear()
                self._scooter_obj = None
                self._scooter_id = None
                self._scooter_name = None
                self._on_scooter = False
                self._in_vehicle = False
                self._in_vehicle_name = None
                self._npc_actor_names.clear()
                self._world_map = None

    @staticmethod
    def _to_xy(obj: Any, default=(0.0, 0.0)) -> tuple:
        if hasattr(obj, "x") and hasattr(obj, "y"):
            return float(obj.x), float(obj.y)
        if isinstance(obj, (tuple, list, np.ndarray)) and len(obj) >= 2:
            return float(obj[0]), float(obj[1])
        return float(default[0]), float(default[1])

    @staticmethod
    def _yaw_to_dir_xy(yaw_deg: float) -> tuple:
        r = math.radians(float(yaw_deg))
        return math.cos(r), math.sin(r)

    @staticmethod
    def _dir_xy_to_yaw(dx: float, dy: float) -> float:
        return float(math.degrees(math.atan2(float(dy), float(dx))))

    def _get_agent_camera(self, agent_idx: int) -> int:
        # agent 0 → camera 1, agent 1 → camera 2, ...
        # camera 0 保留给全局视角
        return agent_idx + 1

    def _safe_get_location(self, name: str) -> Optional[tuple]:
        try:
            loc = self.unrealcv.get_location(name)
            if loc is not None and not isinstance(loc, str) and len(loc) >= 2:
                return (float(loc[0]), float(loc[1]),
                        float(loc[2]) if len(loc) > 2 else 0.0)
        except Exception:
            pass
        return None

    def _query_object_location_reliable(
        self, name: str, retries: int = 3, interval: float = 0.15
    ) -> Optional[tuple]:
        last_loc = None
        for attempt in range(retries):
            loc = self._safe_get_location(name)
            if loc is not None:
                if last_loc is not None:
                    if abs(loc[0]-last_loc[0]) < 5.0 and abs(loc[1]-last_loc[1]) < 5.0:
                        return loc
                last_loc = loc
            if attempt < retries - 1:
                time.sleep(interval)
        return last_loc

    @staticmethod
    def _overlay_nearby_hud(obs: Dict[str, Any]) -> Dict[str, Any]:
        frame = obs.get("rgb", None)
        if not isinstance(frame, np.ndarray) or frame.ndim < 2:
            return obs

        nearby: List[Dict[str, Any]] = obs.get("nearby_objects", [])
        nearest: Optional[str] = obs.get("nearest_object", None)

        frame = frame.copy()
        overlay = frame.copy()
        bar_h = 20 + max(len(nearby), 1) * 22
        cv2.rectangle(overlay, (0, 0), (340, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Nearby Objects:", (8, 16),
                    font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        for i, obj in enumerate(nearby):
            in_range = obj["in_range"]
            is_target = in_range and obj["object_name"] == nearest
            obj_type = obj.get("type", "object")

            if obj_type == "dog":
                color = (255, 200, 0) if in_range else (160, 160, 160)
                prefix = "[DOG]    "
            elif obj_type == "pedestrian":
                color = (100, 200, 255) if in_range else (160, 160, 160)
                prefix = "[NPC]    "
            elif obj_type == "scooter":
                color = (0, 255, 200) if in_range else (160, 160, 160)
                prefix = "[SCOOTER]"
            else:
                color = (80, 255, 80) if in_range else (160, 160, 160)
                prefix = "[TARGET] " if is_target else "         "

            text = f"{prefix} {obj['object_name']}  {obj['distance']:.0f}u"
            cv2.putText(frame, text, (8, 36 + i * 22),
                        font, 0.42, color, 1, cv2.LINE_AA)

        obs = dict(obs)
        obs["rgb"] = frame
        return obs

    def _get_observation(
        self,
        include_image: bool = True,
        use_multicam: bool = False,
    ) -> Dict[str, Any]:
        obs: Dict[str, Any] = {}

        # 1) 位置 & 朝向
        if self.agent is not None and self.communicator is not None:
            if self._on_scooter and self._scooter_name is not None:
                px, py, yaw_deg = 0.0, 0.0, 0.0
                try:
                    loc = self.unrealcv.get_location(self._scooter_name)
                    if loc is not None and not isinstance(loc, str) and len(loc) >= 2:
                        px, py = float(loc[0]), float(loc[1])
                        self._object_positions["__scooter__"] = (px, py, 0.0)
                    else:
                        _cached = self._object_positions.get("__scooter__")
                        if _cached:
                            px, py = float(_cached[0]), float(_cached[1])
                except Exception as e:
                    _cached = self._object_positions.get("__scooter__")
                    if _cached:
                        px, py = float(_cached[0]), float(_cached[1])
                    print(f"[SW] scooter get_location error: {e}")
                try:
                    ori = self.unrealcv.get_orientation(self._scooter_name)
                    if ori is not None and not isinstance(ori, str) and len(ori) >= 2:
                        yaw_deg = float(ori[1])
                except Exception as e:
                    print(f"[SW] scooter get_orientation error: {e}")

                dx, dy = self._yaw_to_dir_xy(yaw_deg)
                pos2 = Vector(px, py)
                dir2 = Vector(dx, dy)

            else:
                pos_raw = None
                dir_raw = 0.0
                try:
                    result = self.communicator.get_position_and_direction(
                        humanoid_ids=[self.agent.id])
                    pos_raw, dir_raw = result[("humanoid", self.agent.id)]
                except KeyError:
                    pass

                px, py = self._to_xy(pos_raw, default=(0.0, 0.0)) if pos_raw is not None else (0.0, 0.0)

                if (px == 0.0 and py == 0.0) and self.agent_name is not None:
                    loc = self._safe_get_location(self.agent_name)
                    if loc is not None:
                        px, py = float(loc[0]), float(loc[1])

                pos2 = Vector(px, py)
                if isinstance(dir_raw, (int, float)):
                    yaw_deg = float(dir_raw)
                    dx, dy = self._yaw_to_dir_xy(yaw_deg)
                    dir2 = Vector(dx, dy)
                else:
                    dx, dy = self._to_xy(dir_raw, default=(0.0, 1.0))
                    dir2 = Vector(dx, dy)
                    yaw_deg = self._dir_xy_to_yaw(dir2.x, dir2.y)

            obs["position"] = {"x": pos2.x, "y": pos2.y}
            obs["direction"] = {"x": dir2.x, "y": dir2.y}
            obs["yaw"] = yaw_deg
            self.agent.position = pos2
            try:
                self.agent.direction = float(yaw_deg)
            except Exception:
                pass
        else:
            obs["position"] = {"x": 0.0, "y": 0.0}
            obs["direction"] = {"x": 0.0, "y": 1.0}
            obs["yaw"] = 90.0

        if include_image:
            if use_multicam and len(self.camera_ids) > 1:
                obs["images"] = {}
                for cam_id in self.camera_ids:
                    _rgb = self.communicator.get_camera_observation(cam_id, "lit")
                    obs["images"][f"camera_{cam_id}"] = {"rgb": _rgb} if isinstance(_rgb, np.ndarray) else {}
            else:
                if self._on_scooter and self._scooter_name is not None:
                    try:
                        loc = self.unrealcv.get_location(self._scooter_name)
                        ori = self.unrealcv.get_orientation(self._scooter_name)
                        if (loc is not None and not isinstance(loc, str) and len(loc) >= 3
                                and ori is not None and not isinstance(ori, str) and len(ori) >= 2):
                            cx = float(loc[0])
                            cy = float(loc[1])
                            cz = float(loc[2]) + self.scooter_cam_offset_z
                            yaw = float(ori[1])
                            alpha = 0.3
                            yaw = alpha * yaw + (1 - alpha) * self._last_scooter_cam_yaw
                            self._last_scooter_cam_yaw = yaw
                            self.unrealcv.set_camera_location(self.scooter_follow_cam_id, (cx, cy, cz))
                            self.unrealcv.set_camera_rotation(self.scooter_follow_cam_id,
                                                              (self.scooter_cam_pitch, yaw, 0.0))
                    except Exception as e:
                        print(f"[SW] scooter follow cam error: {e}")

                _active_cam = self.scooter_follow_cam_id if self._on_scooter else self.camera_id
                _rgb = self.communicator.get_camera_observation(_active_cam, "lit")
                if isinstance(_rgb, np.ndarray):
                    obs["rgb"] = _rgb
                else:
                    print(f"[SW] get_camera_observation returned non-array: {type(_rgb)}, skipping frame")

        # 3) 碰撞
        if self.agent is not None and self.communicator is not None:
            try:
                col = self.communicator.get_collision_number(self.agent.id)
                obs["collision"] = {
                    "human": col[0] if len(col) > 0 else 0,
                    "object": col[1] if len(col) > 1 else 0,
                    "building": col[2] if len(col) > 2 else 0,
                    "vehicle": col[3] if len(col) > 3 else 0,
                }
            except Exception:
                obs["collision"] = {"human": 0, "object": 0, "building": 0, "vehicle": 0}
        else:
            obs["collision"] = {"human": 0, "object": 0, "building": 0, "vehicle": 0}

        px = obs["position"]["x"]
        py = obs["position"]["y"]

        # 4) 物体位置更新
        for obj in self.spawn_objects:
            name = obj.get("object_name", "")
            obj_type = obj.get("type", "object").lower()
            if not name:
                continue

            if obj_type in ("dog", "pedestrian"):
                actor_name = self._npc_actor_names.get(name, name)
                loc = self._safe_get_location(actor_name)
                if loc:
                    self._object_positions[name] = loc
            elif self._picked_object_name == name:
                self._object_positions[name] = (px, py, 0.0)
            else:
                locs = []
                for _ in range(2):
                    loc = self._safe_get_location(name)
                    if loc:
                        locs.append(loc)
                if len(locs) == 2:
                    diff = math.hypot(locs[1][0]-locs[0][0], locs[1][1]-locs[0][1])
                    self._object_positions[name] = locs[1] if diff > 10.0 else (
                        (locs[0][0]+locs[1][0])/2,
                        (locs[0][1]+locs[1][1])/2,
                        (locs[0][2]+locs[1][2])/2,
                    )
                elif len(locs) == 1:
                    self._object_positions[name] = locs[0]

        # 5) nearby / nearest
        nearby = []
        for obj in self.spawn_objects:
            name = obj.get("object_name", "")
            obj_type = obj.get("type", "object").lower()
            raw = self._object_positions.get(name, obj.get("position", (0, 0, 0)))
            ox, oy = float(raw[0]), float(raw[1])
            dist = math.hypot(px - ox, py - oy)
            nearby.append({
                "object_name": name,
                "distance": round(dist, 2),
                "in_range": dist <= self.interact_radius,
                "type": obj_type,
            })

        if self._scooter_name is not None and not self._on_scooter:
            loc = self._safe_get_location(self._scooter_name)
            if loc is not None:
                self._object_positions["__scooter__"] = loc
            else:
                scooter_cfg_pos = (self.spawn_scooter_cfg or {}).get("position", (0, 0))
                loc = self._object_positions.get(
                    "__scooter__",
                    (float(scooter_cfg_pos[0]), float(scooter_cfg_pos[1]), 0.0)
                )
            sox, soy = float(loc[0]), float(loc[1])
            sdist = math.hypot(px - sox, py - soy)
            nearby.append({
                "object_name": self._scooter_name,
                "distance": round(sdist, 2),
                "in_range": sdist <= self.interact_radius,
                "type": "scooter",
            })

        nearby.sort(key=lambda o: o["distance"])
        obs["nearby_objects"] = nearby
        in_range = [o for o in nearby if o["in_range"]]
        obs["nearest_object"] = in_range[0]["object_name"] if in_range else None

        # 6) 交通状态
        if self._traffic_ctrl is not None:
            try:
                obs["vehicles"] = [
                    {"id": v.id,
                     "position": {"x": float(v.position.x), "y": float(v.position.y)},
                     "speed": float(getattr(v, "speed", 0.0))}
                    for v in self._traffic_ctrl.vehicles
                ]
            except Exception:
                obs["vehicles"] = []
            try:
                obs["pedestrians"] = [
                    {"id": p.id,
                     "position": {"x": float(p.position.x), "y": float(p.position.y)},
                     "speed": float(getattr(p, "speed", 0.0))}
                    for p in self._traffic_ctrl.pedestrians
                ]
            except Exception:
                obs["pedestrians"] = []
            try:
                signals = getattr(self._traffic_ctrl, "traffic_signals", [])
                obs["traffic_lights"] = [
                    {"id": sig.id, "state": str(getattr(sig, "state", "unknown"))}
                    for sig in signals
                ]
            except Exception:
                obs["traffic_lights"] = []
        else:
            obs["vehicles"] = []
            obs["pedestrians"] = []
            obs["traffic_lights"] = []

        # 7) 地图
        obs["world_map"] = self._world_map

        # 8) 状态
        self._has_object = self._picked_object_name is not None
        obs["has_object"] = self._has_object
        obs["on_scooter"] = self._on_scooter
        obs["scooter_id"] = self._scooter_id
        obs["in_vehicle"] = self._in_vehicle
        obs["in_vehicle_name"] = self._in_vehicle_name

        if len(self.agents) >= 1:
            all_agents_obs = []
            for i, agent in enumerate(self.agents):
                cam_id = self._get_agent_camera(i)
                agent_obs = {
                    "id": agent.id,
                    "name": self.agent_names[i],
                    "camera_id": cam_id,
                }
                loc = self._safe_get_location(self.agent_names[i])
                if loc:
                    agent_obs["position"] = {"x": loc[0], "y": loc[1]}
                if include_image:
                    _rgb = self.communicator.get_camera_observation(cam_id, "lit")
                    if isinstance(_rgb, np.ndarray):
                        agent_obs["rgb"] = _rgb
                try:
                    col = self.communicator.get_collision_number(agent.id)
                    agent_obs["collision"] = {
                        "human": col[0] if len(col) > 0 else 0,
                        "object": col[1] if len(col) > 1 else 0,
                        "building": col[2] if len(col) > 2 else 0,
                        "vehicle": col[3] if len(col) > 3 else 0,
                    }
                except Exception:
                    agent_obs["collision"] = {"human": 0, "object": 0, "building": 0, "vehicle": 0}
                all_agents_obs.append(agent_obs)
            obs["all_agents"] = all_agents_obs

        obs = self._overlay_nearby_hud(obs)
        return obs

    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise TypeError(f"data must be dict, got {type(data)}")
        mode = str(data.get("mode", "observe"))
        obs_kwargs = dict(
            include_image=bool(data.get("include_image", True)),
            use_multicam= bool(data.get("use_multicam",  False)),
        )
        if mode in ("init", "observe"):
            self._ensure_connection()
            return self._get_observation(**obs_kwargs)
        if mode == "step":
            self._ensure_connection()
            action = data.get("action", None)
            if action is None:
                raise ValueError("mode='step' requires data['action']")
            if not isinstance(action, dict):
                raise TypeError(f"action must be dict, got {type(action)}")
            agent_idx = int(data.get("agent_idx", 0))
            success = self._execute_action(action, agent_idx=agent_idx)
            obs = self._get_observation(**obs_kwargs)
            obs["action_success"] = success
            obs["last_action"] = action
            return obs
        if mode == "reset":
            if self.communicator is not None:
                try:
                    self.communicator.clear_env()
                except Exception as e:
                    print(f"[SW] Warning during clear_env: {e}")
            self._close()
            self._ensure_connection()
            return self._get_observation(**obs_kwargs)
        if mode == "close":
            self._close()
            return {"closed": True}
        if mode == "query":
            self._ensure_connection()
            query = str(data.get("query", ""))
            if query == "cameras":
                return {"cameras": self.unrealcv.get_cameras()}
            if query == "objects":
                return {"objects": self.unrealcv.get_objects()}
            if query == "agent_name":
                return {"agent_name": self.agent_name, "agent_names": self.agent_names}
            if query == "traffic_state":
                if self._traffic_ctrl is None:
                    return {"error": "traffic not initialized (no config_path)"}
                return {
                    "vehicles": len(self._traffic_ctrl.vehicles),
                    "pedestrians": len(self._traffic_ctrl.pedestrians),
                }
            if query == "scooter_info":
                return {
                    "scooter_id": self._scooter_id,
                    "scooter_name": self._scooter_name,
                    "on_scooter": self._on_scooter,
                }
            if query == "npc_names":
                return {"npc_names": dict(self._npc_actor_names)}
            if query == "world_map":
                return {"world_map": self._world_map}
            if query == "config":
                return {
                    "num_agents": self.num_agents,
                    "seed": self.seed,
                    "dt": self.dt,
                }
            return {"error": f"Unknown query: {query!r}"}
        raise ValueError(f"Unknown mode: {mode!r}")

    def _execute_action(self, action: Dict[str, Any], agent_idx: int = 0) -> bool:
        agent = self.agents[agent_idx]     if agent_idx < len(self.agents)      else self.agent
        agent_name = self.agent_names[agent_idx] if agent_idx < len(self.agent_names) else self.agent_name

        action_type = str(action.get("type", "unknown"))
        try:
            if action_type == "step_forward":
                self.communicator.humanoid_step_forward(
                    agent.id,
                    float(action.get("duration", 1.0)),
                    int(action.get("direction", 0)),
                )
                return True

            if action_type == "rotate":
                self.communicator.humanoid_rotate(
                    agent.id,
                    float(action.get("angle", 90.0)),
                    str(action.get("direction", "left")),
                )
                return True

            if action_type == "stop":
                self.communicator.humanoid_stop(agent.id)
                return True

            if action_type == "set_speed":
                self.communicator.humanoid_set_speed(
                    agent.id, float(action.get("speed", 200.0)))
                return True

            if action_type == "sit_down":
                self.unrealcv.humanoid_sit_down(agent_name)
                return True

            if action_type == "stand_up":
                self.unrealcv.humanoid_stand_up(agent_name)
                return True

            if action_type == "stop_current_action":
                self.unrealcv.humanoid_stop_current_action(agent_name)
                return True

            if action_type == "scooter_control":
                if not self._on_scooter or self._scooter_id is None:
                    print("[SW] scooter_control ignored: not on scooter")
                    return False
                self.communicator.set_scooter_attributes(
                    self._scooter_id,
                    float(action.get("throttle", 0.0)),
                    float(action.get("brake",    0.0)),
                    float(action.get("steering", 0.0)),
                )
                return True

            if action_type == "vehicle_control":
                if not self._in_vehicle or self._in_vehicle_name is None:
                    print("[SW] vehicle_control ignored: not in vehicle")
                    return False
                self.unrealcv.v_set_state(
                    self._in_vehicle_name,
                    float(action.get("throttle", 0.0)),
                    float(action.get("brake",    0.0)),
                    float(action.get("steering", 0.0)),
                )
                return True

            if action_type == "pick_up":
                object_name = str(action.get("object_name", ""))
                if not object_name:
                    return False
                try:
                    loc = self.unrealcv.get_location(object_name)
                    if loc is not None and len(loc) >= 2:
                        self._object_positions[object_name] = (
                            float(loc[0]), float(loc[1]),
                            float(loc[2]) if len(loc) > 2 else 0.0)
                except Exception as e:
                    print(f"[SW] pick_up pre-check location error: {e}")
                self.communicator.humanoid_pick_up_object(agent.id, object_name)
                self._picked_object_name = object_name
                self._drop_positions.pop(object_name, None)
                time.sleep(0.8)
                return True

            if action_type == "drop_object":
                dropped_name = self._picked_object_name
                result = self.unrealcv.humanoid_drop_object(agent_name)
                self._picked_object_name = None
                if dropped_name and result is True:

                    def wait_until_settled(obj_name, max_wait=10.0, check_interval=0.3,
                                           stable_threshold=1.5, pos_tolerance=2.0):
                        deadline = time.time() + max_wait
                        last_pos = None
                        stable_since = None
                        while time.time() < deadline:
                            pos = self._safe_get_location(obj_name)
                            if pos is not None:
                                if last_pos is not None:
                                    dist = math.sqrt(
                                        (pos[0]-last_pos[0])**2 +
                                        (pos[1]-last_pos[1])**2 +
                                        (pos[2]-last_pos[2])**2
                                    )
                                    if dist < pos_tolerance:
                                        if stable_since is None:
                                            stable_since = time.time()
                                        elif time.time() - stable_since >= stable_threshold:
                                            print(f"[SW] {obj_name} settled at "
                                                  f"({pos[0]:.1f}, {pos[1]:.1f})")
                                            return pos
                                    else:
                                        stable_since = None
                                last_pos = pos
                            time.sleep(check_interval)
                        print(f"[SW] {obj_name} wait timeout, using last known pos")
                        return last_pos

                    drop_loc = wait_until_settled(dropped_name)
                    model_path = None
                    for obj in self.spawn_objects:
                        if obj.get("object_name") == dropped_name:
                            model_path = obj.get("model_path")
                            break

                    if drop_loc is not None and model_path is not None:
                        m = re.match(r'^(.*?)(\d+)$', dropped_name)
                        base, num = (m.group(1), int(m.group(2))) if m else (dropped_name+"_", 0)
                        max_num = num
                        for n in [o.get("object_name", "") for o in self.spawn_objects]:
                            mm = re.match(r'^' + re.escape(base) + r'(\d+)$', n)
                            if mm:
                                max_num = max(max_num, int(mm.group(1)))
                        new_name = f"{base}{max_num+1}"
                        try:
                            self.unrealcv.destroy(dropped_name)
                            time.sleep(1.0)
                            self.unrealcv.clean_garbage()
                            time.sleep(0.5)
                            self.communicator.spawn_object(
                                new_name, model_path,
                                (drop_loc[0], drop_loc[1], 0), (0, 0, 0))
                            time.sleep(1.0)
                            self._object_positions.pop(dropped_name, None)
                            self._object_pos_trust[new_name] = True
                            self._object_pos_trust.pop(dropped_name, None)
                            for obj in self.spawn_objects:
                                if obj.get("object_name") == dropped_name:
                                    obj["object_name"] = new_name
                                    break
                            print(f"[SW] respawn OK: {dropped_name} → {new_name} "
                                  f"at ({drop_loc[0]:.1f}, {drop_loc[1]:.1f}, 0)")
                        except Exception as e:
                            print(f"[SW] respawn error: {e}")
                            if drop_loc:
                                self._object_positions[dropped_name] = drop_loc
                    elif drop_loc:
                        self._object_positions[dropped_name] = drop_loc
                return True

            if action_type == "get_on_scooter":
                if self._on_scooter:
                    print("[SW] get_on_scooter ignored: already on scooter")
                    return False
                if self._scooter_id is None:
                    print("[SW] get_on_scooter: no scooter available")
                    return False
                self.communicator.humanoid_get_on_scooter(agent.id)
                time.sleep(0.5)
                self._on_scooter = True
                agent.scooter_id = self._scooter_id
                print(f"[SW] get_on_scooter: now riding scooter id={self._scooter_id}")
                return True

            if action_type == "get_off_scooter":
                if not self._on_scooter:
                    print("[SW] get_off_scooter ignored: not on scooter")
                    return False
                scooter_id = action.get("scooter_id", self._scooter_id)
                if scooter_id is None:
                    print("[SW] get_off_scooter: no scooter_id")
                    return False
                try:
                    self.communicator.set_scooter_attributes(scooter_id, 0.0, 1.0, 0.0)
                    time.sleep(0.3)
                except Exception:
                    pass
                self.communicator.humanoid_get_off_scooter(agent.id, scooter_id)
                time.sleep(0.5)
                self._on_scooter = False
                if self._scooter_name is not None:
                    loc = self._safe_get_location(self._scooter_name)
                    if loc is not None:
                        self._object_positions["__scooter__"] = loc
                    else:
                        self._object_positions.pop("__scooter__", None)
                print(f"[SW] get_off_scooter: dismounted scooter id={scooter_id}")
                return True

            if action_type == "enter_vehicle":
                if self._in_vehicle:
                    print("[SW] enter_vehicle ignored: already in vehicle")
                    return False
                vehicle_name = str(action.get("vehicle_name", ""))
                if not vehicle_name:
                    return False
                result = self.unrealcv.humanoid_enter_vehicle(agent_name, vehicle_name)
                if result is True:
                    self._in_vehicle = True
                    self._in_vehicle_name = vehicle_name
                    print(f"[SW] enter_vehicle SUCCESS: {vehicle_name}")
                return result is True

            if action_type == "exit_vehicle":
                if not self._in_vehicle:
                    print("[SW] exit_vehicle ignored: not in vehicle")
                    return False
                vehicle_name = str(action.get("vehicle_name", ""))
                if not vehicle_name:
                    return False
                result = self.unrealcv.humanoid_exit_vehicle(agent_name, vehicle_name)
                if result is True:
                    self._in_vehicle = False
                    self._in_vehicle_name = None
                    print(f"[SW] exit_vehicle SUCCESS: {vehicle_name}")
                return result is True

            if action_type == "argue":
                self.unrealcv.humanoid_argue(agent_name, int(action.get("argue_type", 0)))
                return True

            if action_type == "discuss":
                self.unrealcv.humanoid_discuss(agent_name, int(action.get("discuss_type", 0)))
                return True

            if action_type == "listen":
                self.unrealcv.humanoid_listen(agent_name)
                return True

            if action_type == "wave_to_dog":
                self.unrealcv.humanoid_wave_to_dog(agent_name)
                return True

            if action_type == "directing_path":
                self.unrealcv.humanoid_directing_path(agent_name)
                return True

            if action_type == "follow_path":
                self.unrealcv.humanoid_follow_path(agent_name)
                return True

            if action_type == "set_path":
                path = str(action.get("path", ""))
                if not path:
                    return False
                self.unrealcv.humanoid_set_path(agent_name, path)
                return True

            if action_type == "rescan_objects":
                return True

            print(f"[SW] Warning: Unknown action type: {action_type!r}")
            return False

        except Exception as e:
            print(f"[SW] Error executing {action_type!r}: {e}")
            return False
        