"""Communicator module for interfacing with Unreal Engine.

This module provides a high-level communication interface with Unreal Engine
through the UnrealCV client, handling vehicle, pedestrian, and traffic signal management.
"""
import json
import math
import re
from threading import Lock

import numpy as np
import pandas as pd

from .unrealcv import UnrealCV
from ..utils.load_json import load_json
from ..utils.logger import Logger
from ..utils.vector import Vector

class Communicator:
    """Class for communicating with Unreal Engine through UnrealCV.

    This class is responsible for handling communication with Unreal Engine,
    including the management of vehicles, pedestrians, and traffic signals.
    """

    def __init__(self, unrealcv: UnrealCV = None):
        """Initialize the communicator.

        Args:
            unrealcv: UnrealCV instance for communication with Unreal Engine.
        """
        self.unrealcv = unrealcv
        self.ue_manager_name = None
        self.logger = Logger.get_logger('Communicator')

        self.vehicle_id_to_name = {}
        self.pedestrian_id_to_name = {}
        self.traffic_signal_id_to_name = {}
        self.humanoid_id_to_name = {}
        self.scooter_id_to_name = {}
        self.waypoint_mark_id_to_name = {}

        self.lock = Lock()

    ##############################################################
    # Humanoid Methods
    ##############################################################

    def humanoid_move_forward(self, humanoid_id):
        """Move humanoid forward.

        Args:
            humanoid_id: The unique identifier of the humanoid to move forward.
        """
        self.unrealcv.humanoid_move_forward(self.get_humanoid_name(humanoid_id))

    def humanoid_rotate(self, humanoid_id, angle, direction):
        """Rotate humanoid.

        Args:
            humanoid_id: humanoid ID.
            angle: Rotation angle.
            direction: Rotation direction.
        """
        self.unrealcv.humanoid_rotate(self.get_humanoid_name(humanoid_id), angle, direction)

    def humanoid_stop(self, humanoid_id):
        """Stop humanoid.

        Args:
            humanoid_id: humanoid ID.
        """
        self.unrealcv.humanoid_stop(self.get_humanoid_name(humanoid_id))

    def humanoid_step_forward(self, humanoid_id, duration, direction=0):
        """Step forward.

        Args:
            humanoid_id: humanoid ID.
            duration: Duration.
            direction: Direction.
        """
        self.unrealcv.humanoid_step_forward(self.get_humanoid_name(humanoid_id), duration, direction)

    def humanoid_set_speed(self, humanoid_id, speed):
        """Set humanoid speed.

        Args:
            humanoid_id: humanoid ID.
            speed: Speed.
        """
        self.unrealcv.humanoid_set_speed(self.get_humanoid_name(humanoid_id), speed)

    def humanoid_get_on_scooter(self, humanoid_id):
        """Get on scooter.

        Args:
            humanoid_id: humanoid ID.
        """
        self.unrealcv.humanoid_get_on_scooter(self.get_humanoid_name(humanoid_id))

    def humanoid_get_off_scooter(self, humanoid_id, scooter_id):
        """Get off scooter.

        Args:
            humanoid_id: humanoid ID.
            scooter_id: Scooter ID of the humanoid to get off.
        """
        with self.lock:
            self.unrealcv.humanoid_get_off_scooter(self.get_scooter_name(scooter_id))
            objects = self.unrealcv.get_objects()

        # Find the new Base_User_Agent object
        new_humanoid = None
        for obj in objects:
            if 'Base_User_Agent_C_' in obj:
                new_humanoid = obj
                break

        # Find which scooter ID no longer has a corresponding object
        old_humanoid_name = self.get_humanoid_name(humanoid_id)
        if old_humanoid_name not in objects:
            # Update the mapping to bind the new humanoid with the scooter ID
            self.unrealcv.set_object_name(new_humanoid, old_humanoid_name)

    def humanoid_sit_down(self, humanoid_id):
        """Sit down.

        Args:
            humanoid_id: humanoid ID.
        """
        self.unrealcv.humanoid_sit_down(self.get_humanoid_name(humanoid_id))

    def humanoid_stand_up(self, humanoid_id):
        """Stand up.

        Args:
            humanoid_id: humanoid ID.
        """
        self.unrealcv.humanoid_stand_up(self.get_humanoid_name(humanoid_id))

    def humanoid_pick_up_object(self, humanoid_id, object_name):
        """Pick up object.

        Args:
            humanoid_id: humanoid ID.
            object_name: Object name.
        """
        return self.unrealcv.humanoid_pick_up_object(self.get_humanoid_name(humanoid_id), object_name)

    def get_humanoid_name(self, humanoid_id):
        """Get humanoid name.

        Args:
            humanoid_id: humanoid ID.

        Returns:
            str: The formatted humanoid name.
        """
        if humanoid_id not in self.humanoid_id_to_name:
            self.humanoid_id_to_name[humanoid_id] = f'GEN_BP_Humanoid_{humanoid_id}'
        return self.humanoid_id_to_name[humanoid_id]

    ##############################################################
    # Scooter-related methods
    ##############################################################

    def get_scooter_name(self, scooter_id):
        """Get scooter name.

        Args:
            scooter_id: Scooter ID.
        """
        if scooter_id not in self.scooter_id_to_name:
            self.scooter_id_to_name[scooter_id] = f'GEN_BP_Scooter_{scooter_id}'
        return self.scooter_id_to_name[scooter_id]

    def set_scooter_attributes(self, scooter_id, throttle, brake, steering):
        """Set scooter attributes.

        Args:
            scooter_id: Scooter ID.
            throttle: Throttle.
            brake: Brake.
            steering: Steering.
        """
        name = self.get_scooter_name(scooter_id)
        self.unrealcv.s_set_state(name, throttle, brake, steering)

    ##############################################################
    # Camera-related methods
    ##############################################################

    def get_camera_observation(self, cam_id, viewmode, mode='direct'):
        """Get camera observation.

        Args:
            cam_id: Camera ID.
            viewmode: View mode. Possible values are 'lit', 'depth', 'object_mask'.
            mode: Mode, possible values are 'direct', 'file', 'fast'.

        Returns:
            Image data.
        """
        return self.unrealcv.get_image(cam_id, viewmode, mode)

    def get_camera_observation_multicam(self, cam_ids, viewmode, mode='direct'):
        """Get camera observation batch."""
        if isinstance(cam_ids, list):
            images = []
            for cam_id in cam_ids:
                images.append(self.unrealcv.get_image(cam_id, viewmode, mode))
            return images
        else:
            return self.unrealcv.get_image(cam_ids, viewmode, mode)

    def show_img(self, image):
        """Show image.

        Args:
            image: Image data.
        """
        self.unrealcv.show_img(image)

    ##############################################################
    # Vehicle-related methods
    ##############################################################

    def update_vehicle(self, vehicle_id, throttle, brake, steering):
        """Update vehicle state.

        Args:
            vehicle_id: Vehicle ID.
            throttle: Throttle value.
            brake: Brake value.
            steering: Steering value.
        """
        name = self.get_vehicle_name(vehicle_id)
        self.unrealcv.v_set_state(name, throttle, brake, steering)

    def vehicle_make_u_turn(self, vehicle_id):
        """Make vehicle perform a U-turn.

        Args:
            vehicle_id: Vehicle ID.
        """
        name = self.get_vehicle_name(vehicle_id)
        self.unrealcv.v_make_u_turn(name)

    def update_vehicles(self, states):
        """Batch update multiple vehicle states.

        Args:
            states: Dictionary containing multiple vehicle states,
                   where keys are vehicle IDs and values are state tuples.
        """
        vehicles_states_str = ''
        for vehicle_id, state in states.items():
            name = self.get_vehicle_name(vehicle_id)
            vehicle_state = f'{name},{state[0]},{state[1]},{state[2]}'

            if vehicles_states_str:
                vehicles_states_str += ';'
            vehicles_states_str += vehicle_state

        self.unrealcv.v_set_states(self.ue_manager_name, vehicles_states_str)

    def get_vehicle_name(self, vehicle_id):
        """Get vehicle name.

        Args:
            vehicle_id: Vehicle ID.

        Returns:
            Vehicle name.
        """
        if vehicle_id not in self.vehicle_id_to_name:
            self.vehicle_id_to_name[vehicle_id] = f'GEN_BP_Vehicle_{vehicle_id}'
        return self.vehicle_id_to_name[vehicle_id]

    ##############################################################
    # Pedestrian-related methods
    ##############################################################

    def pedestrian_move_forward(self, pedestrian_id):
        """Move pedestrian forward.

        Args:
            pedestrian_id: Pedestrian ID.
        """
        name = self.get_pedestrian_name(pedestrian_id)
        self.unrealcv.p_move_forward(name)

    def pedestrian_rotate(self, pedestrian_id, angle, direction):
        """Rotate pedestrian.

        Args:
            pedestrian_id: Pedestrian ID.
            angle: Rotation angle.
            direction: Rotation direction.
        """
        name = self.get_pedestrian_name(pedestrian_id)
        self.unrealcv.p_rotate(name, angle, direction)

    def pedestrian_stop(self, pedestrian_id):
        """Stop pedestrian movement.

        Args:
            pedestrian_id: Pedestrian ID.
        """
        name = self.get_pedestrian_name(pedestrian_id)
        self.unrealcv.p_stop(name)

    def set_pedestrian_speed(self, pedestrian_id, speed):
        """Set pedestrian speed.

        Args:
            pedestrian_id: Pedestrian ID.
            speed: Pedestrian speed.
        """
        name = self.get_pedestrian_name(pedestrian_id)
        self.unrealcv.p_set_speed(name, speed)

    def update_pedestrians(self, states):
        """Batch update multiple pedestrian states.

        Args:
            states: Dictionary containing multiple pedestrian states,
                   where keys are pedestrian IDs and values are states.
        """
        pedestrians_states_str = ''
        for pedestrian_id, state in states.items():
            name = self.get_pedestrian_name(pedestrian_id)
            pedestrian_state = f'{name},{state}'

            if pedestrians_states_str:
                pedestrians_states_str += ';'
            pedestrians_states_str += pedestrian_state

        self.unrealcv.p_set_states(self.ue_manager_name, pedestrians_states_str)

    def p_set_waypoints(self, pedestrian_id, waypoints):
        """Set pedestrian waypoints.

        Args:
            pedestrian_id: Pedestrian ID.
            waypoints: List of waypoints (Vector).
        """
        name = self.get_pedestrian_name(pedestrian_id)
        str_waypoints = ''
        for waypoint in waypoints:
            str_waypoints += f'{waypoint.x},{waypoint.y};'
        str_waypoints = str_waypoints[:-1]
        self.unrealcv.p_set_waypoints(name, str_waypoints)

    def get_pedestrian_name(self, pedestrian_id):
        """Get pedestrian name.

        Args:
            pedestrian_id: Pedestrian ID.

        Returns:
            Pedestrian name.
        """
        if pedestrian_id not in self.pedestrian_id_to_name:
            self.pedestrian_id_to_name[pedestrian_id] = f'GEN_BP_Pedestrian_{pedestrian_id}'
        return self.pedestrian_id_to_name[pedestrian_id]

    ##############################################################
    # Traffic signal related methods
    ##############################################################

    def traffic_signal_switch_to(self, traffic_signal_id, state='green'):
        """Switch traffic signal state.

        Args:
            traffic_signal_id: Traffic signal ID.
            state: Target state, possible values: 'green' or 'pedestrian walk'.
        """
        name = self.get_traffic_signal_name(traffic_signal_id)
        if state == 'green':
            self.unrealcv.tl_set_vehicle_green(name)
        elif state == 'pedestrian walk':
            self.unrealcv.tl_set_pedestrian_walk(name)

    def traffic_signal_set_duration(self, traffic_signal_id, green_duration, yellow_duration, pedestrian_green_duration):
        """Set traffic signal duration.

        Args:
            traffic_signal_id: Traffic signal ID.
            green_duration: Green duration.
            yellow_duration: Yellow duration.
            pedestrian_green_duration: Pedestrian green duration.
        """
        name = self.get_traffic_signal_name(traffic_signal_id)
        self.unrealcv.tl_set_duration(name, green_duration, yellow_duration, pedestrian_green_duration)

    def get_traffic_signal_name(self, traffic_signal_id):
        """Get traffic signal name.

        Args:
            traffic_signal_id: Traffic signal ID.

        Returns:
            Traffic signal name.
        """
        if traffic_signal_id not in self.traffic_signal_id_to_name:
            self.traffic_signal_id_to_name[traffic_signal_id] = f'GEN_BP_TrafficSignal_{traffic_signal_id}'
        return self.traffic_signal_id_to_name[traffic_signal_id]

    def add_traffic_signal(self, intersection_name, traffic_signal):
        """Add traffic signal.

        Args:
            intersection_name: Name of the intersection to add the traffic signal to.
            traffic_signal: Traffic signal object.
        """
        if traffic_signal.type == 'both':
            self.unrealcv.add_vehicle_signal(intersection_name, self.get_traffic_signal_name(traffic_signal.id))
        elif traffic_signal.type == 'pedestrian':
            self.unrealcv.add_pedestrian_signal(intersection_name, self.get_traffic_signal_name(traffic_signal.id))
        else:
            raise ValueError(f'Invalid traffic signal type: {traffic_signal.type}')

    def traffic_signal_start_simulation(self, intersection_name):
        """Start traffic signal simulation.

        Args:
            intersection_name: Name of the intersection to start simulation for.
        """
        self.unrealcv.traffic_signal_start_simulation(intersection_name)

    ##############################################################
    # Waypoint-related methods
    ##############################################################

    def get_waypoint_mark_name(self, waypoint_mark_id):
        """Get waypoint mark name.

        Args:
            waypoint_mark_id: Waypoint mark ID.

        Returns:
            Waypoint mark name.
        """
        if waypoint_mark_id not in self.waypoint_mark_id_to_name:
            self.waypoint_mark_id_to_name[waypoint_mark_id] = f'GEN_BP_WaypointMark_{waypoint_mark_id}'
        return self.waypoint_mark_id_to_name[waypoint_mark_id]

    ##############################################################
    # Utility methods
    ##############################################################
    def get_collision_number(self, humanoid_id):
        """Get collision number.

        Args:
            humanoid_id: Humanoid ID.

        Returns:
            Human collision number, object collision number, building collision number, vehicle collision number.
        """
        collision_json = self.unrealcv.get_collision_num(self.get_humanoid_name(humanoid_id))
        collision_data = json.loads(collision_json)
        human_collision_num = int(collision_data['HumanCollision'])
        object_collision_num = int(collision_data['ObjectCollision'])
        building_collision_num = int(collision_data['BuildingCollision'])
        vehicle_collision_num = int(collision_data['VehicleCollision'])
        return human_collision_num, object_collision_num, building_collision_num, vehicle_collision_num

    def get_position_and_direction(self, vehicle_ids=[], pedestrian_ids=[], traffic_signal_ids=[], humanoid_ids=[], scooter_ids=[]):
        """Get position and direction of vehicles, pedestrians, and traffic signals.

        Args:
            vehicle_ids: List of vehicle IDs.
            pedestrian_ids: List of pedestrian IDs.
            traffic_signal_ids: List of traffic signal IDs.
            humanoid_ids: Optional list of humanoid IDs to get their positions and directions.
            scooter_ids: Optional list of scooter IDs to get their positions and directions.

        Returns:
            Dictionary containing position and direction information for all objects.
        """
        info = json.loads(self.unrealcv.get_informations(self.ue_manager_name))
        result = {}

        # Process vehicles
        locations = info['VLocations']
        rotations = info['VRotations']
        for vehicle_id in vehicle_ids:
            name = self.get_vehicle_name(vehicle_id)

            # Parse location
            location_pattern = f'{name}X=(.*?) Y=(.*?) Z='
            match = re.search(location_pattern, locations)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                position = Vector(x, y)

                # Parse rotation
                rotation_pattern = f'{name}P=.*? Y=(.*?) R='
                match = re.search(rotation_pattern, rotations)
                if match:
                    direction = float(match.group(1))
                    result[('vehicle', vehicle_id)] = (position, direction)

        # Process pedestrians
        locations = info['PLocations']
        rotations = info['PRotations']
        for pedestrian_id in pedestrian_ids:
            name = self.get_pedestrian_name(pedestrian_id)

            location_pattern = f'{name}X=(.*?) Y=(.*?) Z='
            match = re.search(location_pattern, locations)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                position = Vector(x, y)

                rotation_pattern = f'{name}P=.*? Y=(.*?) R='
                match = re.search(rotation_pattern, rotations)
                if match:
                    direction = float(match.group(1))
                    result[('pedestrian', pedestrian_id)] = (position, direction)

        # Process traffic signals
        light_states = info['LStates']
        for traffic_signal_id in traffic_signal_ids:
            name = self.get_traffic_signal_name(traffic_signal_id)
            pattern = rf'{name}(true|false)(true|false)(\d+\.\d+)'
            match = re.search(pattern, light_states)
            if match:
                is_vehicle_green = match.group(1) == 'true'
                is_pedestrian_walk = match.group(2) == 'true'
                left_time = float(match.group(3))

                result[('traffic_signal', traffic_signal_id)] = (is_vehicle_green, is_pedestrian_walk, left_time)

        # process humanoids
        locations = info['ALocations']
        rotations = info['ARotations']
        for humanoid_id in humanoid_ids:
            name = self.get_humanoid_name(humanoid_id)
            location_pattern = f'{name}X=(.*?) Y=(.*?) Z='
            match = re.search(location_pattern, locations)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                position = Vector(x, y)

                rotation_pattern = f'{name}P=.*? Y=(.*?) R='
                match = re.search(rotation_pattern, rotations)
                if match:
                    direction = float(match.group(1))
                    result[('humanoid', humanoid_id)] = (position, direction)

        # process scooters
        locations = info['SLocations']
        rotations = info['SRotations']
        for scooter_id in scooter_ids:
            name = self.get_scooter_name(scooter_id)
            location_pattern = f'{name}X=(.*?) Y=(.*?) Z='
            match = re.search(location_pattern, locations)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                position = Vector(x, y)

                rotation_pattern = f'{name}P=.*? Y=(.*?) R='
                match = re.search(rotation_pattern, rotations)
                if match:
                    direction = float(match.group(1))
                    result[('scooter', scooter_id)] = (position, direction)

        return result

    def spawn_object(self, object_name, model_path, position, direction):
        """Spawn object.

        Args:
            object_name: Object name.
            model_path: Model path.
            position: Position. Tuple (x, y, z).
            direction: Direction. Tuple (pitch, yaw, roll).
        """
        self.unrealcv.spawn_bp_asset(model_path, object_name)
        # Convert 2D position to 3D (x,y -> x,y,z)
        location_3d = (
            position[0],  # Unreal X = 2D Y
            position[1],  # Unreal Y = 2D X
            position[2]  # Z coordinate (ground level)
        )
        # Convert 2D direction to 3D orientation (assuming rotation around Z axis)
        orientation_3d = (
            direction[0],  # Pitch
            direction[1],  # Yaw
            direction[2]  # Roll
        )
        self.unrealcv.set_location(location_3d, object_name)
        self.unrealcv.set_orientation(orientation_3d, object_name)
        self.unrealcv.set_scale((1, 1, 1), object_name)
        self.unrealcv.set_collision(object_name, True)
        self.unrealcv.set_movable(object_name, True)

    # Initialization methods
    def spawn_agent(self, agent, name, position=None, model_path='/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C', type='humanoid'):
        """Spawn agent.

        Args:
            agent: Agent object.
            name: Agent name.
            position: Position. If None, use agent's position.
            model_path: Model path.
            type: Agent type, possible values: 'humanoid', 'dog', ...
        """
        if type == 'humanoid':
            name = self.get_humanoid_name(agent.id)
        else:
            if name is None:
                raise ValueError('Agent name is required for non-humanoid agents')
            else:
                name = name
        self.unrealcv.spawn_bp_asset(model_path, name)
        # Convert 2D position to 3D (x,y -> x,y,z)
        if position is None:
            location_3d = (
                agent.position.x,  # Unreal X = 2D Y
                agent.position.y,  # Unreal Y = 2D X
                600  # Z coordinate (ground level)
            )
        else:
            location_3d = (
                position[0],
                position[1],
                position[2]
            )
        # Convert 2D direction to 3D orientation (assuming rotation around Z axis)
        orientation_3d = (
            0,  # Pitch
            math.degrees(math.atan2(agent.direction.y, agent.direction.x)),  # Yaw
            0  # Roll
        )
        print(location_3d)
        self.unrealcv.set_location(location_3d, name)
        self.unrealcv.set_orientation(orientation_3d, name)
        self.unrealcv.set_scale((1, 1, 1), name)  # Default scale
        self.unrealcv.set_collision(name, True)
        self.unrealcv.set_movable(name, True)

    def spawn_scooter(self, scooter, model_path):
        """Spawn scooter.

        Args:
            scooter: Scooter object.
            model_path: Model path.
        """
        name = self.get_scooter_name(scooter.id)
        self.unrealcv.spawn_bp_asset(model_path, name)
        # Convert 2D position to 3D (x,y -> x,y,z)
        location_3d = (
            scooter.position.x,  # Unreal X = 2D Y
            scooter.position.y,  # Unreal Y = 2D X
            0  # Z coordinate (ground level)
        )
        # Convert 2D direction to 3D orientation (assuming rotation around Z axis)
        orientation_3d = (
            0,  # Pitch
            math.degrees(math.atan2(scooter.direction.y, scooter.direction.x)),  # Yaw
            0  # Roll
        )
        self.unrealcv.set_location(location_3d, name)
        self.unrealcv.set_orientation(orientation_3d, name)
        self.unrealcv.set_scale((1, 1, 1), name)  # Default scale
        self.unrealcv.set_collision(name, True)
        self.unrealcv.set_movable(name, True)

    def spawn_vehicles(self, vehicles):
        """Spawn vehicles.

        Args:
            vehicles: List of vehicle objects.
        """
        for vehicle in vehicles:
            name = self.get_vehicle_name(vehicle.id)
            self.unrealcv.spawn_bp_asset(vehicle.vehicle_reference, name)
            # Convert 2D position to 3D (x,y -> x,y,z)
            location_3d = (
                vehicle.position.x,  # Unreal X = 2D Y
                vehicle.position.y,  # Unreal Y = 2D X
                0  # Z coordinate (ground level)
            )
            # Convert 2D direction to 3D orientation (assuming rotation around Z axis)
            orientation_3d = (
                0,  # Pitch
                math.degrees(math.atan2(vehicle.direction.y, vehicle.direction.x)),  # Yaw
                0  # Roll
            )
            self.unrealcv.set_location(location_3d, name)
            self.unrealcv.set_orientation(orientation_3d, name)
            self.unrealcv.set_scale((1, 1, 1), name)  # Default scale
            self.unrealcv.set_collision(name, True)
            self.unrealcv.set_movable(name, True)

    def spawn_pedestrians(self, pedestrians, model_path='/Game/TrafficSystem/Pedestrian/Base_Pedestrian.Base_Pedestrian_C'):
        """Spawn pedestrians.

        Args:
            pedestrians: List of pedestrian objects.
            model_path: Pedestrian model path.
        """
        for pedestrian in pedestrians:
            name = self.get_pedestrian_name(pedestrian.id)
            self.unrealcv.spawn_bp_asset(model_path, name)
            # Convert 2D position to 3D (x,y -> x,y,z)
            location_3d = (
                pedestrian.position.x,  # Unreal X = 2D Y
                pedestrian.position.y,  # Unreal Y = 2D X
                110  # Z coordinate (ground level)
            )
            # Convert 2D direction to 3D orientation (assuming rotation around Z axis)
            orientation_3d = (
                0,  # Pitch
                math.degrees(math.atan2(pedestrian.direction.y, pedestrian.direction.x)),  # Yaw
                0  # Roll
            )
            self.unrealcv.set_location(location_3d, name)
            self.unrealcv.set_orientation(orientation_3d, name)
            self.unrealcv.set_scale((1, 1, 1), name)  # Default scale
            self.unrealcv.set_collision(name, True)
            self.unrealcv.set_movable(name, True)

    def spawn_traffic_signals(self, traffic_signals, traffic_light_model_path='/Game/city_props/BP/props/street_light/BP_street_light.BP_street_light_C', pedestrian_light_model_path='/Game/city_props/BP/props/street_light/BP_street_light_ped.BP_street_light_ped_C'):
        """Spawn traffic signals.

        Args:
            traffic_signals: List of traffic signal objects to spawn.
            traffic_light_model_path: Path to the traffic light model asset.
            pedestrian_light_model_path: Path to the pedestrian signal light model asset.
        """
        for traffic_signal in traffic_signals:
            name = self.get_traffic_signal_name(traffic_signal.id)
            if traffic_signal.type == 'pedestrian':
                model_name = pedestrian_light_model_path
            elif traffic_signal.type == 'both':
                model_name = traffic_light_model_path
            self.unrealcv.spawn_bp_asset(model_name, name)
            # Convert 2D position to 3D (x,y -> x,y,z)
            location_3d = (
                traffic_signal.position.x,
                traffic_signal.position.y,
                0  # Z coordinate (ground level)
            )
            # Convert 2D direction to 3D orientation (assuming rotation around Z axis)
            orientation_3d = (
                0,  # Pitch
                math.degrees(math.atan2(traffic_signal.direction.y, traffic_signal.direction.x)),  # Yaw
                0  # Roll
            )
            self.unrealcv.set_location(location_3d, name)
            self.unrealcv.set_orientation(orientation_3d, name)
            self.unrealcv.set_scale((1, 1, 1), name)  # Default scale
            self.unrealcv.set_collision(name, True)
            self.unrealcv.set_movable(name, False)

    def spawn_intersection(self, intersection_name, model_path):
        """Spawn intersection.

        Args:
            intersection_name: Name of the intersection to spawn.
            model_path: Model path.
        """
        self.unrealcv.spawn_bp_asset(model_path, intersection_name)

    def spawn_waypoint_mark(self, waypoints, model_path):
        """Spawn waypoint marks.

        Args:
            waypoints: List of waypoint objects.
            model_path: Waypoint mark model path.
        """
        id_counter = 0
        for waypoint in waypoints:
            name = self.get_waypoint_mark_name(id_counter)
            id_counter += 1
            self.unrealcv.spawn_bp_asset(model_path, name)
            location_3d = (
                waypoint.position.x,
                waypoint.position.y,
                30  # Z coordinate (ground level)
            )
            self.unrealcv.set_location(location_3d, name)
            orientation_3d = (
                0,  # Pitch
                math.degrees(math.atan2(waypoint.direction.y, waypoint.direction.x)),  # Yaw
                0  # Roll
            )
            self.unrealcv.set_orientation(orientation_3d, name)
            self.unrealcv.set_scale((1, 1, 1), name)
            self.unrealcv.set_collision(name, False)
            self.unrealcv.set_movable(name, False)

    def spawn_ue_manager(self, ue_manager_path):
        """Spawn UE manager.

        Args:
            ue_manager_path: Path to the UE manager asset in the content browser.
        """
        self.ue_manager_name = 'GEN_BP_UEManager'
        self.unrealcv.spawn_bp_asset(ue_manager_path, self.ue_manager_name)

    def update_objects(self):
        """Update objects."""
        self.unrealcv.update_objects(self.ue_manager_name)

    def generate_world(self, world_json, ue_asset_path, run_time=True):
        """Generate world.

        Args:
            world_json: World configuration JSON file path.
            ue_asset_path: Unreal Engine asset path.
            run_time: Whether to run the world generation in real time.

        Returns:
            set: A set of generated object IDs.
        """
        generated_ids = set()
        # Load world from JSON
        world_setting = load_json(world_json)
        # Use pandas data structure, convert JSON data to pandas dataframe
        nodes = world_setting['nodes']
        node_df = pd.json_normalize(nodes, sep='_')
        node_df.set_index('id', inplace=True)

        # Load asset library
        asset_library = load_json(ue_asset_path)

        def _parse_rgb(color_str):
            """Parse RGB values from color string like '(R=255,G=255,B=0)'.

            Args:
                color_str: Color string.
            """
            pattern = r'R=(\d+),G=(\d+),B=(\d+)'
            match = re.search(pattern, color_str)
            if match:
                return [int(match.group(1)), int(match.group(2)), int(match.group(3))]
            return [0, 0, 0]  # Default to black if parsing fails

        def _process_node(row):
            """Process a single node.

            Args:
                row: Node row.
            """
            # Spawn each node on the map
            id = row.name  # name is the index of the row
            try:
                instance_ref = asset_library[node_df.loc[id, 'instance_name']]['asset_path']
                color = asset_library['colors'][asset_library[node_df.loc[id, 'instance_name']]['color']]
                rgb_values = _parse_rgb(color)
            except KeyError:
                self.logger.error(f"Can't find node {node_df.loc[id, 'instance_name']} in asset library")
                return
            else:
                self.unrealcv.spawn_bp_asset(instance_ref, id)
                if run_time:
                    self.unrealcv.set_color(id, rgb_values)
                location = node_df.loc[id, ['properties_location_x', 'properties_location_y', 'properties_location_z']].to_list()
                self.unrealcv.set_location(location, id)
                orientation = node_df.loc[id, ['properties_orientation_pitch', 'properties_orientation_yaw', 'properties_orientation_roll']].to_list()
                self.unrealcv.set_orientation(orientation, id)
                scale = node_df.loc[id, ['properties_scale_x', 'properties_scale_y', 'properties_scale_z']].to_list()
                self.unrealcv.set_scale(scale, id)
                self.unrealcv.set_collision(id, True)
                self.unrealcv.set_movable(id, False)
                generated_ids.add(id)

        node_df.apply(_process_node, axis=1)

        return generated_ids

    # Utility methods
    def clear_env(self, keep_roads=False):
        """Clear all objects in the environment."""
        # Lazy import to avoid circular dependency
        from ..agent.humanoid import Humanoid
        from ..agent.pedestrian import Pedestrian
        from ..agent.scooter import Scooter
        from ..agent.vehicle import Vehicle

        # Get all objects in the environment
        objects = [obj.lower() for obj in self.unrealcv.get_objects()]  # Convert objects to lowercase
        # Define unwanted objects
        if keep_roads:
            unwanted_terms = ['GEN_BP_']
        else:
            unwanted_terms = ['GEN_BP_', 'GEN_Road_']
        unwanted_terms = [term.lower() for term in unwanted_terms]  # Convert unwanted terms to lowercase

        # Get all objects starting with the unwanted terms
        indexes = np.concatenate([np.flatnonzero(np.char.startswith(objects, term)) for term in unwanted_terms])
        # Destroy them
        if indexes is not None:
            for index in indexes:
                self.unrealcv.destroy(objects[index])

        self.unrealcv.clean_garbage()
        Humanoid._id_counter = 0
        Scooter._id_counter = 0
        Pedestrian._id_counter = 0
        Vehicle._id_counter = 0

    def clean_traffic_only(self, vehicles, pedestrians, traffic_signals):
        """Clean traffic objects only.

        Args:
            vehicles: List of vehicles.
            pedestrians: List of pedestrians.
            traffic_signals: List of traffic signals.
        """
        for vehicle in vehicles:
            self.unrealcv.destroy(self.get_vehicle_name(vehicle.id))
        for traffic_signal in traffic_signals:
            self.unrealcv.destroy(self.get_traffic_signal_name(traffic_signal.id))
        for pedestrian in pedestrians:
            self.unrealcv.destroy(self.get_pedestrian_name(pedestrian.id))

        self.unrealcv.destroy(self.ue_manager_name)
        self.unrealcv.clean_garbage()

    def disconnect(self):
        """Disconnect from Unreal Engine."""
        self.unrealcv.disconnect()

    ##############################################################
    # Weather-related methods (currently not in use for production)
    ##############################################################

    def set_sun_direction(self, weather_manager_name, pitch, yaw):
        """Set sun direction.

        Args:
            weather_manager_name: Name of the weather manager.
            pitch: Pitch of the sun. Range: -89 to 89 degrees.
            yaw: Yaw of the sun. Range: 0 to 360 degrees.
        """
        self.unrealcv.set_sun_direction(weather_manager_name, pitch, yaw)

    def get_sun_direction(self, weather_manager_name):
        """Get sun direction.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            tuple: (pitch, yaw) in degrees, or (None, None) if parsing fails.
        """
        try:
            sun_direction_str = self.unrealcv.get_sun_direction(weather_manager_name)

            # Clean up the string by removing \r\n\t and extra spaces
            cleaned_str = re.sub(r'[\r\n\t]', '', sun_direction_str)
            cleaned_str = re.sub(r'\s+', ' ', cleaned_str).strip()

            # Parse JSON
            sun_data = json.loads(cleaned_str)

            # Extract rotation string and parse P, Y, R values
            rotation_str = sun_data.get('rotation', '')
            # Parse format: "P=-26.000000 Y=-20.000000 R=0.000000"
            pitch_match = re.search(r'P=([-\d.]+)', rotation_str)
            yaw_match = re.search(r'Y=([-\d.]+)', rotation_str)

            if pitch_match and yaw_match:
                pitch = float(pitch_match.group(1))
                yaw = float(yaw_match.group(1))
                return pitch, yaw
            else:
                self.logger.error(f'Failed to parse rotation values from: {rotation_str}')
                return None, None

        except Exception as e:
            self.logger.error(f'Failed to parse sun direction response: {e}')
            return None, None

    def get_sun_intensity(self, weather_manager_name):
        """Get sun intensity.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            float: Sun intensity value, or 0.0 if parsing fails.
        """
        try:
            sun_intensity_str = self.unrealcv.get_sun_intensity(weather_manager_name)

            # Clean up the string by removing \r\n\t and extra spaces
            cleaned_str = re.sub(r'[\r\n\t]', '', sun_intensity_str)
            cleaned_str = re.sub(r'\s+', ' ', cleaned_str).strip()

            # Parse JSON
            sun_data = json.loads(cleaned_str)

            # Extract SunIntensity value
            intensity = float(sun_data.get('SunIntensity', 0))

            return intensity
        except Exception as e:
            self.logger.error(f'Failed to parse sun intensity response: {e}')
            return 0.0

    def set_sun_intensity(self, weather_manager_name, intensity):
        """Set sun intensity.

        Args:
            weather_manager_name: Name of the weather manager.
            intensity: Intensity of the sun. 0 - 100
        """
        self.unrealcv.set_sun_intensity(weather_manager_name, intensity)

    def set_fog(self, weather_manager_name, density, distance, falloff):
        """Set fog parameters.

        Args:
            weather_manager_name: Name of the weather manager.
            density: Fog density. Range: 0-100.
            distance: Fog distance in cm. Range: 0-5000.
            falloff: Fog falloff. Range: 0-2.
        """
        self.unrealcv.set_fog(weather_manager_name, density, distance, falloff)

    def get_fog(self, weather_manager_name):
        """Get fog parameters.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            tuple: (density, distance, falloff), or (None, None, None) if parsing fails.
        """
        try:
            fog_str = self.unrealcv.get_fog(weather_manager_name)

            # Clean up the string by removing \r\n\t and extra spaces
            cleaned_str = re.sub(r'[\r\n\t]', '', fog_str)
            cleaned_str = re.sub(r'\s+', ' ', cleaned_str).strip()

            # Parse JSON
            fog_data = json.loads(cleaned_str)

            density = float(fog_data.get('FogDensity', 0))
            distance = float(fog_data.get('FogDistance', 0))
            falloff = float(fog_data.get('FogFalloff', 0))

            return density, distance, falloff
        except Exception as e:
            self.logger.error(f'Failed to parse fog response: {e}')
            return None, None, None

    def set_atmosphere(self, weather_manager_name, rayleigh, mie):
        """Set atmosphere parameters.

        Args:
            weather_manager_name: Name of the weather manager.
            rayleigh: Rayleigh scattering scale. Range: 0-2.
            mie: Mie scattering scale. Range: 0-5.
        """
        self.unrealcv.set_atmosphere(weather_manager_name, rayleigh, mie)

    def get_atmosphere(self, weather_manager_name):
        """Get atmosphere parameters.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            tuple: (rayleigh_scattering_scale, mie_scattering_scale), or (None, None) if parsing fails.
        """
        try:
            atmosphere_str = self.unrealcv.get_atmosphere(weather_manager_name)

            # Clean up the string by removing \r\n\t and extra spaces
            cleaned_str = re.sub(r'[\r\n\t]', '', atmosphere_str)
            cleaned_str = re.sub(r'\s+', ' ', cleaned_str).strip()

            # Parse JSON
            atmosphere_data = json.loads(cleaned_str)

            rayleigh = float(atmosphere_data.get('Rayleigh Scattering Scale', 0))
            mie = float(atmosphere_data.get('Mie Scattering Scale', 0))

            return rayleigh, mie
        except Exception as e:
            self.logger.error(f'Failed to parse atmosphere response: {e}')
            return None, None

    def get_weather_info(self, weather_manager_name):
        """Get all weather information.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            dict: Dictionary containing all weather parameters, or None if any parsing fails.
        """
        try:
            weather_info = {}

            # Get sun direction
            pitch, yaw = self.get_sun_direction(weather_manager_name)
            if pitch is not None and yaw is not None:
                weather_info['sun_direction'] = {'pitch': pitch, 'yaw': yaw}

            # Get fog parameters
            density, distance, falloff = self.get_fog(weather_manager_name)
            if density is not None and distance is not None and falloff is not None:
                weather_info['fog'] = {'density': density, 'distance': distance, 'falloff': falloff}

            # Get atmosphere parameters
            rayleigh, mie = self.get_atmosphere(weather_manager_name)
            if rayleigh is not None and mie is not None:
                weather_info['atmosphere'] = {'rayleigh': rayleigh, 'mie': mie}

            return weather_info
        except Exception as e:
            self.logger.error(f'Failed to get weather info: {e}')
            return None

    def spawn_weather_manager(self, weather_manager_name, weather_manager_model_path='/Game/Weather/BP_WeatherManager.BP_WeatherManager_C'):
        """Spawn weather manager.

        Args:
            weather_manager_name: Name of the weather manager.
            weather_manager_model_path: Path to the weather manager model.
        """
        self.unrealcv.spawn_bp_asset(weather_manager_model_path, weather_manager_name)
