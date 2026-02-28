"""UnrealCV communication module.

This module provides a client interface for communicating with Unreal Engine,
allowing for various operations such as object spawning, movement, and image
capture.
"""
import json
import os
import struct
import time
from io import BytesIO
from threading import Lock

import cv2
import numpy as np
import PIL.Image
import unrealcv
from IPython.display import display

from ..utils.logger import Logger

class UnrealCV(object):
    """Interface class for communication with Unreal Engine.

    This class provides various functionalities for communicating with Unreal Engine,
    including basic operations and traffic system operations.
    """

    def __init__(self, port=9000, ip='127.0.0.1', resolution=(1280, 720)):
        """Initialize the UnrealCV client.

        Args:
            port: Connection port, defaults to 9000.
            ip: Connection IP address, defaults to 127.0.0.1.
            resolution: Resolution, defaults to (320, 240).
        """
        self.ip = ip
        # Build a client to connect to the environment
        self.client = unrealcv.Client((ip, port))
        self.client.connect()

        self.resolution = resolution

        self.lock = Lock()
        self.logger = Logger.get_logger('UnrealCV')
        self.ini_unrealcv(resolution)

    ###################################################
    # Basic Operations
    ###################################################
    def disconnect(self):
        """Disconnect from Unreal Engine."""
        self.client.disconnect()

    def ini_unrealcv(self, resolution=(320, 240)):
        """Initialize UnrealCV settings.

        Args:
            resolution: Resolution, defaults to (320, 240).
        """
        self.check_connection()
        [w, h] = resolution
        self.client.request(f'vrun setres {w}x{h}w', -1)  # Set resolution of display window

        self.client.request('vrun Editor.AsyncSkinnedAssetCompilation 2', -1)  # To correctly load the character
        time.sleep(1)

    def check_connection(self):
        """Check connection status, attempt to reconnect if not connected."""
        while self.client.isconnected() is False:
            self.logger.error('UnrealCV server is not running. Please try again')
            time.sleep(1)
            self.client.connect()

    # Deprecated
    def spawn(self, prefab, name):
        """Spawn an object (deprecated).

        Args:
            prefab: Prefab.
            name: Object name.
        """
        cmd = f'vset /objects/spawn {prefab} {name}'
        with self.lock:
            self.client.request(cmd)

    def spawn_bp_asset(self, prefab_path, name):
        """Spawn a blueprint asset.

        Args:
            prefab_path: Prefab path.
            name: Object name.
        """
        cmd = f'vset /objects/spawn_bp_asset {prefab_path} {name}'
        with self.lock:
            self.client.request(cmd)

    def clean_garbage(self):
        """Clean garbage objects."""
        with self.lock:
            self.client.request('vset /action/clean_garbage')

    def set_location(self, loc, name):
        """Set object location.

        Args:
            loc: Location coordinates in the form [x, y, z].
            name: Object name.
        """
        [x, y, z] = loc
        cmd = f'vset /object/{name}/location {x} {y} {z}'
        with self.lock:
            self.client.request(cmd)

    def set_orientation(self, orientation, name):
        """Set object orientation.

        Args:
            orientation: Orientation in the form [pitch, yaw, roll].
            name: Object name.
        """
        [pitch, yaw, roll] = orientation
        cmd = f'vset /object/{name}/rotation {pitch} {yaw} {roll}'
        with self.lock:
            self.client.request(cmd)

    def set_scale(self, scale, name):
        """Set object scale.

        Args:
            scale: Scale in the form [x, y, z].
            name: Object name.
        """
        [x, y, z] = scale
        cmd = f'vset /object/{name}/scale {x} {y} {z}'
        with self.lock:
            self.client.request(cmd)

    def set_color(self, actor_name, color):
        """Set object color.

        Args:
            actor_name: Object name.
            color: Color in the form [R, G, B].
        """
        [R, G, B] = color
        cmd = f'vset /object/{actor_name}/color {R} {G} {B}'
        with self.lock:
            self.client.request(cmd)

    def enable_controller(self, name, enable_controller):
        """Enable or disable controller.

        Args:
            name: Object name.
            enable_controller: Whether to enable controller.
        """
        cmd = f'vbp {name} EnableController {enable_controller}'
        with self.lock:
            self.client.request(cmd)

    def set_physics(self, actor_name, hasPhysics):
        """Set physics properties.

        Args:
            actor_name: Actor name.
            hasPhysics: Whether to enable physics.
        """
        cmd = f'vset /object/{actor_name}/physics {hasPhysics}'
        with self.lock:
            self.client.request(cmd)

    def set_collision(self, actor_name, hasCollision):
        """Set collision properties.

        Args:
            actor_name: Actor name.
            hasCollision: Whether to enable collision.
        """
        cmd = f'vset /object/{actor_name}/collision {hasCollision}'
        with self.lock:
            self.client.request(cmd)

    def set_movable(self, actor_name, isMovable):
        """Set movable properties.

        Args:
            actor_name: Actor name.
            isMovable: Whether the object is movable.
        """
        cmd = f'vset /object/{actor_name}/object_mobility {isMovable}'
        with self.lock:
            self.client.request(cmd)

    def set_fps(self, fps):
        """Set FPS.

        Args:
            fps: FPS.
        """
        cmd = f'vset /action/set_fixed_frame_rate {fps}'
        with self.lock:
            self.client.request(cmd)

    def set_mode(self, mode='async', tick_interval=0.05):
        """Set asynchronous or synchronous mode.

        Args:
            mode: Mode.
            tick_interval: Tick interval if synchronous mode.
        """
        if mode == 'sync':
            self.set_tick_interval(tick_interval)
            with self.lock:
                self.client.request('vset /action/game/pause')
        elif mode == 'async':
            with self.lock:
                self.client.request('vset /action/game/resume')
        else:
            raise ValueError(f'Invalid mode: {mode}. Please choose from "sync" or "async".')

    def set_tick_interval(self, interval):
        """Set tick interval."""
        cmd = f'vset /action/tick_intervel {interval}'
        with self.lock:
            self.client.request(cmd)

    def tick(self):
        """Tick."""
        cmd = 'vset /action/tick'
        with self.lock:
            self.client.request(cmd)

    def destroy(self, actor_name):
        """Destroy an object.

        Args:
            actor_name: Actor name.
        """
        cmd = f'vset /object/{actor_name}/destroy'
        with self.lock:
            self.client.request(cmd)

    def get_objects(self):
        """Get all objects.

        Returns:
            List of objects.
        """
        with self.lock:
            res = self.client.request('vget /objects')
        objects = np.array(res.split())
        return objects

    def set_object_name(self, name, new_name):
        """Set object name.

        Args:
            name: Object name.
            new_name: New object name.
        """
        cmd = f'vset /object/{name}/name {new_name}'
        with self.lock:
            self.client.request(cmd)

    def get_collision_num(self, actor_name):
        """Get collision number.

        Args:
            actor_name: Actor name.

        Returns:
            json: {
                "HumanCollision": 0,
                "ObjectCollision": 0,
                "BuildingCollision": 0,
                "VehicleCollision": 0
            }
        """
        with self.lock:
            res = self.client.request(f'vbp {actor_name} GetCollisionNum')
        return res

    def get_location(self, actor_name):
        """Get object location.

        Args:
            actor_name: Actor name.

        Returns:
            Location coordinates array.
        """
        cmd = f'vget /object/{actor_name}/location'
        with self.lock:
            res = self.client.request(cmd)
        location = [float(i) for i in res.split()]
        return np.array(location)

    def get_location_batch(self, actor_names):
        """Batch get object locations.

        Args:
            actor_names: List of actor names.

        Returns:
            List of location coordinate arrays.
        """
        cmd = [f'vget /object/{actor_name}/location' for actor_name in actor_names]
        with self.lock:
            res = self.client.request_batch(cmd)
        # Parse each response and convert to numpy array
        locations = [np.array([float(i) for i in r.split()]) for r in res]
        return locations

    def get_orientation(self, actor_name):
        """Get object orientation.

        Args:
            actor_name: Actor name.

        Returns:
            Orientation array.
        """
        cmd = f'vget /object/{actor_name}/rotation'
        with self.lock:
            res = self.client.request(cmd)
            orientation = [float(i) for i in res.split()]
        return np.array(orientation)

    def get_orientation_batch(self, actor_names):
        """Batch get object orientations.

        Args:
            actor_names: List of actor names.

        Returns:
            List of orientation arrays.
        """
        cmd = [f'vget /object/{actor_name}/rotation' for actor_name in actor_names]
        with self.lock:
            res = self.client.request_batch(cmd)
        # Parse each response and convert to numpy array
        orientations = [np.array([float(i) for i in r.split()]) for r in res]
        return orientations

    ##############################################################
    # Traffic System
    ##############################################################
    def v_set_state(self, object_name, throttle, brake, steering):
        """Set vehicle state.

        Args:
            object_name: Object name.
            throttle: Throttle value.
            brake: Brake value.
            steering: Steering value.
        """
        cmd = f'vbp {object_name} SetState {throttle} {brake} {steering}'
        with self.lock:
            self.client.request(cmd)

    def v_make_u_turn(self, object_name):
        """Make vehicle U-turn.

        Args:
            object_name: Object name.
        """
        cmd = f'vbp {object_name} MakeUTurn'
        with self.lock:
            self.client.request(cmd)

    def v_set_states(self, manager_object_name, states: str):
        """Batch set vehicle states.

        Args:
            manager_object_name: Manager object name.
            states: States string.
        """
        cmd = f'vbp {manager_object_name} VSetState {states}'
        with self.lock:
            self.client.request(cmd)

    def p_set_states(self, manager_object_name, states: str):
        """Batch set pedestrian states.

        Args:
            manager_object_name: Manager object name.
            states: States string.
        """
        cmd = f'vbp {manager_object_name} PSetState {states}'
        with self.lock:
            self.client.request(cmd)

    def p_stop(self, object_name):
        """Stop pedestrian movement.

        Args:
            object_name: Object name.
        """
        cmd = f'vbp {object_name} StopPedestrian'
        with self.lock:
            self.client.request(cmd)

    def p_move_forward(self, object_name):
        """Move pedestrian forward.

        Args:
            object_name: Object name.
        """
        cmd = f'vbp {object_name} MoveForward'
        with self.lock:
            self.client.request(cmd)

    def p_rotate(self, object_name, angle, direction='left'):
        """Rotate pedestrian.

        Args:
            object_name: Object name.
            angle: Angle.
            direction: Direction, defaults to 'left'.
        """
        if direction == 'right':
            clockwise = 1
        elif direction == 'left':
            angle = -angle
            clockwise = -1
        cmd = f'vbp {object_name} Rotate_Angle {1} {angle} {clockwise}'
        with self.lock:
            self.client.request(cmd)

    def p_set_speed(self, object_name, speed):
        """Set pedestrian speed.

        Args:
            object_name: Object name.
            speed: Speed.
        """
        cmd = f'vbp {object_name} SetMaxSpeed {speed}'
        with self.lock:
            self.client.request(cmd)

    def p_movement_simulation(self, object_name):
        """Start pedestrian movement simulation.

        Args:
            object_name: Object name.
        """
        cmd = f'vbp {object_name} MovementSimulation'
        with self.lock:
            self.client.request(cmd)

    def p_set_waypoints(self, object_name, waypoints):
        """Set pedestrian waypoints.

        Args:
            object_name: Object name.
            waypoints: String of waypoints. Use semicolon to separate waypoints and use comma to separate coordinates. Example: "100,100;200,200;300,300"
        """
        cmd = f'vbp {object_name} SetWaypoints {waypoints}'
        with self.lock:
            self.client.request(cmd)

    def tl_set_vehicle_green(self, object_name: str):
        """Set vehicle traffic light to green.

        Args:
            object_name: Object name.
        """
        cmd = f'vbp {object_name} SwitchVehicleFrontGreen'
        with self.lock:
            self.client.request(cmd)

    def tl_set_pedestrian_walk(self, object_name: str):
        """Set pedestrian traffic light to walk.

        Args:
            object_name: Object name.
        """
        cmd = f'vbp {object_name} SetPedestrianWalk'
        with self.lock:
            self.client.request(cmd)

    def tl_set_duration(self, object_name: str, green_duration: float, yellow_duration: float, pedestrian_green_duration: float):
        """Set traffic light duration.

        Args:
            object_name: Object name.
            green_duration: Green duration.
            yellow_duration: Yellow duration.
            pedestrian_green_duration: Pedestrian green duration.
        """
        cmd = f'vbp {object_name} SetDuration {green_duration} {yellow_duration} {pedestrian_green_duration}'
        with self.lock:
            self.client.request(cmd)

    def get_informations(self, manager_object_name):
        """Get information.

        Args:
            manager_object_name: Name of the manager object to get information from.

        Returns:
            str: Information string containing the current state of the environment.
        """
        cmd = f'vbp {manager_object_name} GetInformation'
        with self.lock:
            return self.client.request(cmd)

    def update_ue_manager(self, manager_object_name):
        """Update UE manager.

        Args:
            manager_object_name: Name of the manager object to update.
        """
        cmd = f'vbp {manager_object_name} UpdateObjects'
        with self.lock:
            self.client.request(cmd)

    def add_vehicle_signal(self, intersection_name, vehicle_signal_name):
        """Add vehicle signal.

        Args:
            intersection_name: Name of the intersection to add traffic signal to.
            vehicle_signal_name: Name of the vehicle signal to add.
        """
        cmd = f'vbp {intersection_name} AddVehicleSignal {vehicle_signal_name}'
        with self.lock:
            self.client.request(cmd)

    def add_pedestrian_signal(self, intersection_name, pedestrian_signal_name):
        """Add pedestrian signal.

        Args:
            intersection_name: Name of the intersection to add pedestrian signal to.
            pedestrian_signal_name: Name of the pedestrian signal to add.
        """
        cmd = f'vbp {intersection_name} AddPedSignal {pedestrian_signal_name}'
        with self.lock:
            self.client.request(cmd)

    def traffic_signal_start_simulation(self, intersection_name):
        """Start traffic signal simulation."""
        cmd = f'vbp {intersection_name} StartSimulation'
        with self.lock:
            self.client.request(cmd)

    ##############################################################
    # Robot System
    ##############################################################

    def dog_move(self, robot_name, action):
        """Apply transition action.

        Args:
            robot_name: Robot name.
            action: Action in the form [speed, duration, direction].
        """
        [speed, duration, direction] = action
        if speed < 0:
            # Switch direction
            if direction == 0:
                direction = 1
            elif direction == 1:
                direction = 0
            elif direction == 2:
                direction = 3
            elif direction == 3:
                direction = 2
        cmd = f'vbp {robot_name} Move_Speed {speed} {duration} {direction}'
        with self.lock:
            self.client.request(cmd)
        time.sleep(duration)

    def dog_rotate(self, robot_name, action):
        """Apply rotation action.

        Args:
            robot_name: Robot name.
            action: Action in the form [duration, angle, direction].
        """
        [duration, angle, direction] = action
        cmd = f'vbp {robot_name} Rotate_Angle {duration} {angle} {direction}'
        with self.lock:
            self.client.request(cmd)
        time.sleep(duration)

    def dog_look_up(self, robot_name):
        """Apply look up action.

        Args:
            robot_name: Robot name.
        """
        cmd = f'vbp {robot_name} lookup'
        with self.lock:
            self.client.request(cmd)

    def dog_look_down(self, robot_name):
        """Apply look down action.

        Args:
            robot_name: Robot name.
        """
        cmd = f'vbp {robot_name} lookdown'
        with self.lock:
            self.client.request(cmd)

    ##############################################################
    # Humanoid System
    # For '/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C'
    ##############################################################

    def humanoid_move_forward(self, object_name):
        """Move humanoid forward.

        Args:
            object_name: Name of the humanoid object to move forward.
        """
        cmd = f'vbp {object_name} MoveForward'
        with self.lock:
            self.client.request(cmd)

    def humanoid_rotate(self, object_name, angle, direction='left'):
        """Rotate humanoid.

        Args:
            object_name: Name of the humanoid object to rotate.
            angle: Rotation angle in degrees.
            direction: Direction of rotation, either 'left' or 'right'. Defaults to 'left'.
        """
        if direction == 'right':
            clockwise = 1
        elif direction == 'left':
            angle = -angle
            clockwise = -1
        cmd = f'vbp {object_name} TurnAround {1} {angle} {clockwise}'
        with self.lock:
            self.client.request(cmd)
        time.sleep(1)

    def humanoid_stop(self, object_name):
        """Stop humanoid.

        Args:
            object_name: Name of the humanoid object to stop.
        """
        cmd = f'vbp {object_name} StopAgent'
        with self.lock:
            self.client.request(cmd)

    def humanoid_step_forward(self, object_name, duration, direction=0):
        """Step forward.

        Args:
            object_name: Name of the humanoid object to step forward.
            duration: Duration of the step forward movement in seconds.
            direction: Direction of the step forward movement.
        """
        cmd = f'vbp {object_name} StepForward {duration} {direction}'
        with self.lock:
            self.client.request(cmd)
        time.sleep(duration)

    def humanoid_set_speed(self, object_name, speed):
        """Set humanoid speed.

        Args:
            object_name: Name of the humanoid object to set speed.
            speed: Speed to set.
        """
        cmd = f'vbp {object_name} SetMaxSpeed {speed}'
        with self.lock:
            self.client.request(cmd)

    def humanoid_sit_down(self, object_name):
        """Sit down.

        Args:
            object_name: Name of the humanoid object to sit down.
        """
        cmd = f'vbp {object_name} SitDown'
        with self.lock:
            res = self.client.request(cmd)
            success = str(json.loads(res)['Success'])
            if success == 'false':
                return False
            elif success == 'true':
                return True

    def humanoid_stand_up(self, object_name):
        """Stand up.

        Args:
            object_name: Name of the humanoid object to sit down.
        """
        cmd = f'vbp {object_name} StandUp'
        with self.lock:
            res = self.client.request(cmd)
            success = str(json.loads(res)['Success'])
            if success == 'false':
                return False
            elif success == 'true':
                return True

    def humanoid_get_on_scooter(self, object_name):
        """Get on scooter.

        Args:
            object_name: Name of the humanoid object to get on scooter.
        """
        cmd = f'vbp {object_name} GetOnScooter'
        with self.lock:
            self.client.request(cmd)
        self.clean_garbage()

    def humanoid_get_off_scooter(self, object_name):
        """Get off scooter.

        Args:
            object_name: Name of the humanoid object to get off scooter.
        """
        cmd = f'vbp {object_name} GetOffScooter'
        with self.lock:
            self.client.request(cmd)

    def humanoid_pick_up_object(self, humanoid_name, object_name):
        cmd = f'vbp {humanoid_name} PickUp {object_name}'
        with self.lock:
            res = self.client.request(cmd)
        try:
            success = str(json.loads(res)['Success'])
            return success == 'true'
        except Exception as e:
            print(f"[SW] humanoid_pick_up_object parse error: {e}, raw: {res}")
            return False

    def humanoid_drop_object(self, humanoid_name):
        """Drop object.

        Args:
            humanoid_name: Name of the humanoid to drop object.
        """
        cmd = f'vbp {humanoid_name} DropOff'
        with self.lock:
            res = self.client.request(cmd)
            success = str(json.loads(res)['Success'])
            if success == 'false':
                return False
            elif success == 'true':
                return True

    def humanoid_enter_vehicle(self, humanoid_name, vehicle_name):
        """Enter vehicle.

        Args:
            humanoid_name: Name of the humanoid to enter vehicle.
            vehicle_name: Name of the vehicle to enter.
        """
        cmd = f'vbp {humanoid_name} EnterVehicle {vehicle_name}'
        with self.lock:
            res = self.client.request(cmd)
            success = str(json.loads(res)['Success'])
            if success == 'false':
                return False
            elif success == 'true':
                return True

    def humanoid_exit_vehicle(self, humanoid_name, vehicle_name):
        """Exit vehicle.

        Args:
            humanoid_name: Name of the humanoid to enter vehicle.
            vehicle_name: Name of the vehicle to exit.
        """
        cmd = f'vbp {humanoid_name} ExitVehicle {vehicle_name}'
        with self.lock:
            res = self.client.request(cmd)
            success = str(json.loads(res)['Success'])
            if success == 'false':
                return False
            elif success == 'true':
                return True

    def humanoid_discuss(self, humanoid_name, discuss_type):
        """Discuss.

        Args:
            humanoid_name: Name of the humanoid to discuss.
            discuss_type: Type of discussion. Can be [0, 1]
        """
        cmd = f'vbp {humanoid_name} Discussion {discuss_type}'
        with self.lock:
            self.client.request(cmd)

    def humanoid_argue(self, humanoid_name, argue_type):
        """Argue.

        Args:
            humanoid_name: Name of the humanoid to argue.
            argue_type: Type of arguing. Can be [0, 1]
        """
        cmd = f'vbp {humanoid_name} Arguing {argue_type}'
        with self.lock:
            self.client.request(cmd)

    def humanoid_listen(self, humanoid_name):
        """Listen.

        Args:
            humanoid_name: Name of the humanoid to discuss.
        """
        cmd = f'vbp {humanoid_name} Listening'
        with self.lock:
            self.client.request(cmd)

    def humanoid_wave_to_dog(self, humanoid_name):
        """Wave to dog.

        Args:
            humanoid_name: Name of the humanoid to wave to dog.
        """
        cmd = f'vbp {humanoid_name} Wave2Dog'
        with self.lock:
            self.client.request(cmd)

    def humanoid_directing_path(self, humanoid_name):
        """Directing path.

        Args:
            humanoid_name: Name of the humanoid to directing path.
        """
        cmd = f'vbp {humanoid_name} Directing'
        with self.lock:
            self.client.request(cmd)

    def humanoid_stop_current_action(self, humanoid_name):
        """Stop current action.

        Args:
            humanoid_name: Name of the humanoid to stop current action.
        """
        cmd = f'vbp {humanoid_name} StopAction'
        with self.lock:
            self.client.request(cmd)

    def s_set_state(self, object_name, throttle, brake, steering):
        """Set scooter state.

        Args:
            object_name: Name of the scooter object.
            throttle: Throttle value.
            brake: Brake value.
            steering: Steering value.
        """
        cmd = f'vbp {object_name} SetState {throttle} {brake} {steering}'
        with self.lock:
            self.client.request(cmd)

    def humanoid_set_path(self, object_name, path):
        """Set humanoid path.

        Args:
            object_name: Name of the humanoid object.
            path: String of path. Use semicolon to separate waypoints and use comma to separate coordinates. Example: "100,100;200,200;300,300"
        """
        cmd = f'vbp {object_name} SetPath {path}'
        with self.lock:
            self.client.request(cmd)

    def humanoid_follow_path(self, object_name):
        """Follow path.

        Args:
            object_name: Name of the humanoid object.
        """
        cmd = f'vbp {object_name} FollowPath'
        with self.lock:
            self.client.request(cmd)

    ##############################################################
    # Camera
    ##############################################################
    def get_cameras(self):
        """Get all cameras.

        Returns:
            List of camera names.
        """
        cmd = 'vget /cameras'
        with self.lock:
            return self.client.request(cmd)

    def get_camera_location_multicam(self, camera_ids: list):
        """Get camera location batch."""
        locations = []
        for camera_id in camera_ids:
            locations.append(self.get_camera_location(camera_id))
        return locations

    def get_camera_location(self, camera_id: int):
        """Get camera location.

        Args:
            camera_id: ID of the camera to get location.

        Returns:
            Location (x, y, z) of the camera.
        """
        cmd = f'vget /camera/{camera_id}/location'
        with self.lock:
            return self.client.request(cmd)

    def get_camera_rotation(self, camera_id: int):
        """Get camera rotation.

        Args:
            camera_id: ID of the camera to get rotation.

        Returns:
            Rotation (yaw, pitch, roll) of the camera.
        """
        cmd = f'vget /camera/{camera_id}/rotation'
        with self.lock:
            return self.client.request(cmd)

    def get_camera_fov(self, camera_id: int):
        """Get camera field of view.

        Args:
            camera_id: ID of the camera to get field of view.

        Returns:
            Horizontal field of view of the camera.
        """
        cmd = f'vget /camera/{camera_id}/fov'
        with self.lock:
            return self.client.request(cmd)

    def get_camera_resolution(self, camera_id: int):
        """Get camera resolution.

        Args:
            camera_id: ID of the camera to get resolution.

        Returns:
            Resolution (width, height) of the camera.
        """
        cmd = f'vget /camera/{camera_id}/size'
        with self.lock:
            return self.client.request(cmd)

    def set_camera_location(self, camera_id: int, location: tuple):
        """Set camera location.

        Args:
            camera_id: ID of the camera to set location.
            location: Location (x, y, z) of the camera.
        """
        cmd = f'vset /camera/{camera_id}/location {location[0]} {location[1]} {location[2]}'
        with self.lock:
            self.client.request(cmd)

    def set_camera_rotation(self, camera_id: int, rotation: tuple):
        """Set camera rotation.

        Args:
            camera_id: ID of the camera to set rotation.
            rotation: Rotation (pitch, yaw, roll) of the camera.
        """
        cmd = f'vset /camera/{camera_id}/rotation {rotation[0]} {rotation[1]} {rotation[2]}'
        with self.lock:
            self.client.request(cmd)

    def set_camera_fov(self, camera_id: int, fov: float):
        """Set camera field of view.

        Args:
            camera_id: ID of the camera to set field of view.
            fov: Horizontal field of view of the camera.
        """
        cmd = f'vset /camera/{camera_id}/fov {fov}'
        with self.lock:
            self.client.request(cmd)

    def set_camera_resolution(self, camera_id: int, resolution: tuple):
        """Set camera resolution.

        Args:
            camera_id: ID of the camera to set resolution.
            resolution: Resolution (width, height) of the camera.
        """
        cmd = f'vset /camera/{camera_id}/size {resolution[0]} {resolution[1]}'
        with self.lock:
            self.client.request(cmd)

    def show_img(self, img, title='raw_img'):
        """Display an image.

        Args:
            img: Image.
            title: Title, defaults to "raw_img".
        """
        try:
            # Check if the image is a depth image (single channel)
            if len(img.shape) == 2:
                # Normalize depth image for display
                img_normalized = img / img.max()
                # Convert to 8-bit grayscale
                img_display = (img_normalized * 255).astype(np.uint8)
                # Convert to RGB for display
                img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
            else:
                # Convert OpenCV BGR image to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Ensure the image is in uint8 format before converting to PIL Image
            if img_rgb.dtype != np.uint8:
                img_rgb = (img_rgb * 255).astype(np.uint8)

            # Convert to PIL Image
            pil_img = PIL.Image.fromarray(img_rgb)
            # Display in notebook
            display(pil_img)
        except ImportError:
            # Fallback to OpenCV display if not in notebook
            if len(img.shape) == 2:
                # Normalize depth image for display
                img_normalized = img / img.max()
                # Convert to 8-bit grayscale
                img_display = (img_normalized * 255).astype(np.uint8)
                cv2.imshow(title, img_display)
            else:
                cv2.imshow(title, img)
            cv2.waitKey(3)

    def get_image(self, cam_id, viewmode, mode='direct', img_path=None):
        """Get image.

        Args:
            cam_id: Camera ID.
            viewmode: View mode. Possible values are 'lit', 'depth', 'object_mask'.
            mode: Mode.
            img_path: Image path.
        """
        image = None
        try:
            if mode == 'direct':  # get image from unrealcv in png format
                if viewmode == 'depth':
                    cmd = f'vget /camera/{cam_id}/{viewmode} npy'
                    with self.lock:
                        res = self.client.request(cmd)
                    image = self._decode_npy(res)
                else:
                    cmd = f'vget /camera/{cam_id}/{viewmode} png'
                    with self.lock:
                        res = self.client.request(cmd)
                    image = self._decode_png(res)
            elif mode == 'file':  # save image to file and read it
                img_path = os.path.join(os.getcwd(), f'{cam_id}-{viewmode}.png')
                cmd = f'vget /camera/{cam_id}/{viewmode} {img_path}'
                with self.lock:
                    img_dirs = self.client.request(cmd)
                image = cv2.imread(img_dirs)

            elif mode == 'fast':  # get image from unrealcv in bmp format
                cmd = f'vget /camera/{cam_id}/{viewmode} bmp'
                with self.lock:
                    res = self.client.request(cmd)
                # print(type(res), len(res) if hasattr(res, '__len__') else res)
                image = self._decode_bmp(res)

            elif mode == 'file_path':  # save image to file and read it
                cmd = f'vget /camera/{cam_id}/{viewmode} {img_path}'
                with self.lock:
                    img_dirs = self.client.request(cmd)
                image = cv2.imread(img_dirs)

            if image is None:
                raise ValueError(f'Failed to read image with mode={mode}, viewmode={viewmode}')
            return image

        except Exception as e:
            print(f'Error reading image: {str(e)}')
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def _decode_npy(self, res):
        """Decode NPY image.

        Args:
            res: NPY image.

        Returns:
            Decoded image.
        """
        image = np.load(BytesIO(res))
        eps = 1e-6
        depth_log = np.log(image + eps)

        depth_min = np.min(depth_log)
        depth_max = np.max(depth_log)
        normalized_depth = (depth_log - depth_min) / (depth_max - depth_min)

        gamma = 0.5
        normalized_depth = np.power(normalized_depth, gamma)

        image = (normalized_depth * 255).astype(np.uint8)

        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        return image

    def _decode_png(self, res):
        """Decode PNG image.

        Args:
            res: PNG image.

        Returns:
            Decoded image.
        """
        img = np.asarray(PIL.Image.open(BytesIO(res)))
        img = img[:, :, :-1]  # delete alpha channel
        img = img[:, :, ::-1]  # transpose channel order
        return img

    def _decode_bmp(self, res: bytes):
        """Robust BMP decoder.

        Parses header, handles row padding and top-down/bottom-up storage.
        Returns an RGB image of shape (H, W, 3).
        """
        if not isinstance(res, (bytes, bytearray)):
            raise TypeError(f'BMP decoder expects bytes, got {type(res)}')

        # --- BMP signature ---
        if res[:2] != b'BM':
            raise ValueError("Not a BMP file (missing 'BM' signature).")

        # --- Parse header (little-endian) ---
        pixel_offset = struct.unpack_from('<I', res, 10)[0]  # Pixel array start
        width = struct.unpack_from('<i', res, 18)[0]  # Signed: negative means top-down
        height_raw = struct.unpack_from('<i', res, 22)[0]  # Signed: negative => top-down
        bpp = struct.unpack_from('<H', res, 28)[0]  # Bits per pixel
        if bpp not in (24, 32):
            raise NotImplementedError(f'Unsupported BMP bpp: {bpp} (only 24/32 supported)')

        ch = bpp // 8
        w = int(width)
        h = abs(int(height_raw))

        # --- Row stride with padding to 4-byte boundary ---
        row_stride = ((bpp * w + 31) // 32) * 4  # bytes per row including padding
        needed = row_stride * h
        buf = np.frombuffer(res, dtype=np.uint8, count=needed, offset=pixel_offset)
        if buf.size < needed:
            raise ValueError(f'BMP pixel data too short: {buf.size} < expected {needed}')

        # Reshape to (H, row_stride), then slice off padding to (H, W*ch), then to (H, W, ch)
        buf = buf.reshape(h, row_stride)[:, :w * ch].reshape(h, w, ch)

        # BMP bottom-up when height_raw > 0; top-down when height_raw < 0
        if height_raw > 0:
            buf = np.flipud(buf)

        # Convert BGR(A) -> RGB and drop alpha if present
        rgb = buf[:, :, :3][:, :, ::-1]
        return rgb

    def update_objects(self, object_name):
        """Update objects.

        Args:
            object_name: UE_Manager object name.
        """
        cmd = f'vbp {object_name} UpdateObjects'
        with self.lock:
            self.client.request(cmd)

    ##############################################################
    # Weather (currently not in use for production)
    ##############################################################
    def set_sun_direction(self, weather_manager_name, pitch, yaw):
        """Set sun direction.

        Args:
            weather_manager_name: Name of the weather manager.
            pitch: Pitch of the sun. -89 - 89
            yaw: Yaw of the sun. 0 - 360
        """
        cmd = f'vbp {weather_manager_name} SetSunDirection {pitch} {yaw}'
        with self.lock:
            self.client.request(cmd)

    def get_sun_direction(self, weather_manager_name):
        """Get sun direction.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            Sun direction.
        """
        cmd = f'vbp {weather_manager_name} GetSunDirection'
        with self.lock:
            return self.client.request(cmd)

    def set_sun_intensity(self, weather_manager_name, intensity):
        """Set sun intensity.

        Args:
            weather_manager_name: Name of the weather manager.
            intensity: Intensity of the sun. 0 - 100
        """
        cmd = f'vbp {weather_manager_name} SetSunIntensity {intensity}'
        with self.lock:
            self.client.request(cmd)

    def get_sun_intensity(self, weather_manager_name):
        """Get sun intensity.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            Sun intensity.
        """
        cmd = f'vbp {weather_manager_name} GetSunIntensity'
        with self.lock:
            return self.client.request(cmd)

    def set_fog(self, weather_manager_name, density, distance, falloff):
        """Set fog.

        Args:
            weather_manager_name: Name of the weather manager.
            density: Density of the fog. 0 - 100
            distance: Distance of the fog. 0 - 5000, cm
            falloff: Falloff of the fog. 0 - 2
        """
        cmd = f'vbp {weather_manager_name} SetFog {density} {distance} {falloff}'
        with self.lock:
            self.client.request(cmd)

    def get_fog(self, weather_manager_name):
        """Get fog.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            Fog.
        """
        cmd = f'vbp {weather_manager_name} GetFog'
        with self.lock:
            return self.client.request(cmd)

    def set_atmosphere(self, weather_manager_name, rayleigh, mie):
        """Set atmosphere.

        Args:
            weather_manager_name: Name of the weather manager.
            rayleigh: RayleighScatteringScale of the atmosphere. 0 - 2.
            mie: MieScatteringScale of the atmosphere. 0 - 5.
        """
        cmd = f'vbp {weather_manager_name} SetAtmosphere {rayleigh} {mie}'
        with self.lock:
            self.client.request(cmd)

    def get_atmosphere(self, weather_manager_name):
        """Get atmosphere.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            Atmosphere.
        """
        cmd = f'vbp {weather_manager_name} GetAtmosphere'
        with self.lock:
            return self.client.request(cmd)
