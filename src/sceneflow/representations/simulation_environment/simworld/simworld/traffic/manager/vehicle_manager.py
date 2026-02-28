"""Vehicle management module for traffic simulation.

This module handles the creation, spawning, and updating of vehicles in the simulation.
It manages vehicle lifecycle, movement logic, and interaction with traffic signals.
"""
import random

from ...agent.vehicle import Vehicle, VehicleState
from ..base.traffic_signal import TrafficSignalState
from ...utils.load_json import load_json
from ...utils.logger import Logger
from ...utils.traffic_utils import cal_waypoints


class VehicleManager:
    """Manages vehicles in the traffic simulation.

    This class handles spawning vehicles on roads, updating their positions and states,
    and managing their interactions with other elements in the simulation.
    """
    def __init__(self, roads, num_vehicles, config):
        """Initialize the vehicle manager with configuration and initial vehicles.

        Args:
            roads: List of road segments where vehicles can be placed.
            num_vehicles: Number of vehicles to create initially.
            config: Configuration dictionary with simulation parameters.
        """
        self.config = config
        self.vehicles = []
        self.roads = roads
        self.num_vehicles = num_vehicles

        self.vehicle_types = load_json(self.config['traffic.vehicle.model_file_path'])

        # logger
        self.logger = Logger.get_logger('VehicleController')
        self.logger.info(f'VehicleController initialized with {num_vehicles} vehicles')

        self.last_states = {}   # {vehicle_id: (throttle, brake, steering)}

        self.init_vehicles()

    def init_vehicles(self):
        """Initialize the vehicles and place them on the roads.

        Vehicles are randomly placed on lanes, ensuring they maintain safe distances
        from other vehicles and intersections.
        """
        while len(self.vehicles) < self.num_vehicles:
            target_road = random.choice(self.roads)
            target_lane = random.choice(list(target_road.lanes.values()))
            target_position = random.uniform(target_lane.start, target_lane.end)

            # check if the vehicle is too close to intersection
            if target_position.distance(target_lane.end) < 3 * self.config['traffic.distance_between_objects']:
                continue

            possible_vehicles = target_lane.vehicles
            for vehicle in possible_vehicles:
                # check if the vehicle is too close to another vehicle
                if vehicle.position.distance(target_position) < 2 * self.config['traffic.distance_between_objects'] + vehicle.length:
                    break
            else:
                target_direction = target_lane.direction
                # Randomly select a vehicle type
                vehicle_type = random.choice(list(self.vehicle_types.values()))
                new_vehicle = Vehicle(position=target_position, direction=target_direction, current_lane=target_lane,
                                      vehicle_reference=vehicle_type['reference'], config=self.config,
                                      length=vehicle_type['length'], width=vehicle_type['width'])

                waypoints = cal_waypoints(target_position, target_lane.end, self.config['traffic.gap_between_waypoints'])

                new_vehicle.add_waypoint(waypoints)
                target_lane.add_vehicle(new_vehicle)
                self.vehicles.append(new_vehicle)

                self.logger.info(f"Spawned Vehicle: {new_vehicle.id} of type {vehicle_type['name']} on Lane {target_lane.id} at {target_position}")

    def spawn_vehicles(self, communicator):
        """Spawn vehicles in the simulation environment.

        Args:
            communicator: Communication interface to the simulation environment.
        """
        communicator.spawn_vehicles(self.vehicles)

    def update_vehicles(self, communicator, intersection_controller, pedestrians):
        """Update vehicle states and movements based on environment conditions.

        This method handles vehicle movement logic, including waypoint following,
        intersection crossing, obstacle avoidance, and traffic signal compliance.

        Args:
            communicator: Interface for sending updates to the simulation.
            intersection_controller: Controller for managing intersection logic.
            pedestrians: List of pedestrians to check for collision avoidance.
        """
        for vehicle in self.vehicles:
            # update vehicle waypoints
            if vehicle.waypoints and len(vehicle.waypoints) > 0:
                # Calculate vector from current position to waypoint
                to_waypoint = vehicle.waypoints[0] - vehicle.position
                # Calculate dot product between vehicle direction and to_waypoint vector
                dot_product = vehicle.direction.dot(to_waypoint.normalize())
                # If angle is greater than 90 degrees (dot product < 0), remove the waypoint
                if dot_product < 0:
                    vehicle.waypoints.pop(0)

            if vehicle.state == VehicleState.WAITING:
                communicator.vehicle_make_u_turn(vehicle.id)
                vehicle.set_attributes(0.05, 0, -1)  # throttle = 0, brake = 0, steering = -1
                vehicle.state = VehicleState.MAKING_U_TURN

            if vehicle.state == VehicleState.MAKING_U_TURN:
                if vehicle.completed_u_turn():
                    vehicle.state = VehicleState.MOVING
                    vehicle.steering_pid.reset()
                else:
                    continue

            if vehicle.is_close_to_object(self.vehicles, pedestrians):
                self.logger.debug(f'Vehicle {vehicle.id} is close to another vehicle, stop it')
                if not vehicle.state == VehicleState.STOPPED:
                    vehicle.set_attributes(0, 1, 0)  # throttle = 0, brake = 1, steering = 0
                    vehicle.state = VehicleState.STOPPED
                continue

            # Check if vehicle has reached current waypoint
            if vehicle.is_close_to_end():
                self.logger.debug(f'Vehicle {vehicle.id} has reached current waypoint, get next waypoints')
                next_lane, waypoints, current_intersection, is_u_turn = intersection_controller.get_waypoints_for_vehicle(vehicle.current_lane)

                if is_u_turn:
                    vehicle.state = VehicleState.WAITING
                    vehicle.add_waypoint(waypoints)
                    vehicle.change_to_next_lane(next_lane)
                    # communicator.set_state(vehicle.vehicle_id, 0, 1, 0)
                    vehicle.set_attributes(0, 1, 0)  # throttle = 0, brake = 1, steering = 0
                    continue
                else:
                    vehicle_light_state, _ = current_intersection.get_traffic_light_state(vehicle.current_lane)
                    if vehicle_light_state == TrafficSignalState.VEHICLE_GREEN:
                        self.logger.debug(f'Vehicle {vehicle.id} has green light on lane {vehicle.current_lane.id}, add waypoints')
                        vehicle.add_waypoint(waypoints)
                        vehicle.change_to_next_lane(next_lane)
                    else:
                        if not vehicle.state == VehicleState.STOPPED:
                            self.logger.debug(f'Vehicle {vehicle.id} has red light on lane {vehicle.current_lane.id}, stop it')
                            # communicator.set_state(vehicle.vehicle_id, 0, 1, 0)
                            vehicle.set_attributes(0, 1, 0)  # throttle = 0, brake = 1, steering = 0
                            vehicle.state = VehicleState.STOPPED
                        continue

            if vehicle.waypoints and len(vehicle.waypoints) > 0:
                throttle, brake, steering, changed = vehicle.compute_control(vehicle.waypoints[0], 0.1)
                if not vehicle.state == VehicleState.MOVING:
                    vehicle.state = VehicleState.MOVING

        changed_states = {}
        for vehicle in self.vehicles:
            vehicle_id = vehicle.id
            current_state = vehicle.get_attributes()

            if vehicle_id not in self.last_states or current_state != self.last_states[vehicle_id]:
                changed_states[vehicle_id] = current_state
                self.last_states[vehicle_id] = current_state

        if changed_states:
            communicator.update_vehicles(changed_states)

    def stop_vehicles(self, communicator):
        """Stop all vehicles in the simulation."""
        for vehicle in self.vehicles:
            vehicle.state = VehicleState.STOPPED
            communicator.update_vehicle(vehicle.id, 0, 1, 0)
