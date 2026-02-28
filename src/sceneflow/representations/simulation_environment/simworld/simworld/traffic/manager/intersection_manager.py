"""Traffic intersection management module for traffic simulation.

This module handles the management of intersections, traffic signals, and vehicle/pedestrian
routing through intersections.
"""
import random
import threading
from math import acos, degrees
from typing import List

import numpy as np

from ...communicator.communicator import Communicator
from ..base.intersection import Intersection
from ..base.sidewalk import Sidewalk
from ..base.traffic_lane import TrafficLane
from ..base.traffic_signal import (TrafficSignal, TrafficSignalState)
from ...utils.logger import Logger
from ...utils.traffic_utils import (cal_waypoints, extend_control_point, get_bezier_points)
from ...utils.vector import Vector


class IntersectionManager:
    """Manages intersections and traffic signals in the simulation.

    This class handles the initialization of intersections, traffic signal state management,
    and routing of vehicles and pedestrians through intersections.
    """
    def __init__(self, intersections: List[Intersection], config):
        """Initialize the intersection manager with a list of intersections.

        Args:
            intersections: List of intersection objects to manage.
            config: Configuration dictionary with simulation parameters.
        """
        self.config = config
        self.intersections = intersections
        self.python_states = {}  # {light_id: python_state}

        # logger
        self.logger = Logger.get_logger('IntersectionManager')
        self.logger.info(f'IntersectionManager initialized with {len(intersections)} intersections')

        self.lock = threading.Lock()

        self.init_intersections()

    def init_intersections(self):
        """Initialize all intersections with their configuration.

        Sets up lanes, sidewalks, and traffic lights for each intersection.
        """
        for intersection in self.intersections:
            intersection.init_intersection(self.config)

            self.logger.info(f'Intersection {intersection.id}')
            self.logger.info(f'Add Lanes: {intersection.lanes}')
            self.logger.info(f'Add Sidewalks: {intersection.sidewalks}')
            self.logger.info(f'Add Traffic Lights: {intersection.traffic_lights}')

    def spawn_traffic_signals(self, communicator: Communicator):
        """Spawn all traffic signals in the simulation environment.

        Args:
            communicator: Interface for communicating with the simulation environment.
        """
        all_traffic_signals: List[TrafficSignal] = []
        for intersection in self.intersections:
            all_traffic_signals.extend(intersection.traffic_lights)
            all_traffic_signals.extend(intersection.pedestrian_lights)
        communicator.spawn_traffic_signals(all_traffic_signals, self.config['traffic.traffic_signal.traffic_light_model_path'], self.config['traffic.traffic_signal.pedestrian_light_model_path'])

    def set_traffic_signal_duration(self, communicator: Communicator):
        """Set the duration of all traffic signals.

        Args:
            communicator: Interface for communicating with the simulation environment.
        """
        for intersection in self.intersections:
            for light in intersection.traffic_lights:
                communicator.traffic_signal_set_duration(light.id, self.config['traffic.traffic_signal.green_light_duration'], self.config['traffic.traffic_signal.yellow_light_duration'], self.config['traffic.traffic_signal.pedestrian_green_light_duration'])
            for light in intersection.pedestrian_lights:
                communicator.traffic_signal_set_duration(light.id, self.config['traffic.traffic_signal.green_light_duration'], self.config['traffic.traffic_signal.yellow_light_duration'], self.config['traffic.traffic_signal.pedestrian_green_light_duration'])

    def get_intersection_by_lane(self, lane: TrafficLane):
        """Find the intersection that contains a specific lane.

        Args:
            lane: The traffic lane to find the intersection for.

        Returns:
            The intersection containing the lane, or None if not found.
        """
        for intersection in self.intersections:
            if lane in intersection.lanes.keys():
                return intersection
        return None

    def get_intersection_by_sidewalk(self, sidewalk: Sidewalk, current_waypoint: Vector):
        """Find the closest intersection that contains a specific sidewalk.

        Args:
            sidewalk: The sidewalk to find the intersection for.
            current_waypoint: The current position to determine closest intersection.

        Returns:
            The closest intersection containing the sidewalk, or None if not found.
        """
        closest_intersection = None
        min_distance = float('inf')

        for intersection in self.intersections:
            if sidewalk in intersection.sidewalks.keys():
                distance = current_waypoint.distance(intersection.center)
                if distance < min_distance:
                    min_distance = distance
                    closest_intersection = intersection

        return closest_intersection

    def get_next_lane_for_vehicle(self, lane: TrafficLane):
        """Determine the next lane for a vehicle to follow at an intersection.

        This method chooses the next lane based on a weighted random selection,
        with straight paths given higher probability.

        Args:
            lane: The current lane the vehicle is on.

        Returns:
            A tuple of (intersection, next_lane) or (None, None) if no valid next lane.
        """
        target_intersection = self.get_intersection_by_lane(lane)
        if target_intersection is None:
            return None, None

        possible_lanes = target_intersection.lanes[lane]
        weighted_lanes = []

        for next_lane in possible_lanes:
            dot_product = lane.direction.dot(next_lane.direction)
            angle = degrees(acos(np.clip(dot_product, -1.0, 1.0)))

            # more possible to choose straight lane
            if angle <= 5:
                weighted_lanes.extend([next_lane] * 3)
            else:
                weighted_lanes.append(next_lane)

        if not weighted_lanes:
            return None, None

        return target_intersection, random.choice(weighted_lanes)

    def get_waypoints_for_vehicle(self, lane: TrafficLane):
        """Generate waypoints for a vehicle to navigate through an intersection.

        Args:
            lane: The current lane the vehicle is on.

        Returns:
            A tuple of (next_lane, waypoints, intersection, is_u_turn) or (None, None, None, None).
        """
        current_intersection, next_lane = self.get_next_lane_for_vehicle(lane)
        if next_lane is None:
            return None, None, None, None

        dot_product = lane.direction.dot(next_lane.direction)
        angle = degrees(acos(np.clip(dot_product, -1.0, 1.0)))

        if angle <= 5:
            waypoints = cal_waypoints(lane.end, next_lane.start, self.config['traffic.gap_between_waypoints'])
            return next_lane, waypoints, current_intersection, 0
        elif angle == 180:
            return next_lane, [next_lane.start], current_intersection, 1
        else:
            return next_lane, get_bezier_points(lane.end, next_lane.start, extend_control_point(lane.end, next_lane.start, current_intersection.center, 0.15), self.config['traffic.steering_point_num']), current_intersection, 0

    def get_waypoints_for_pedestrian(self, sidewalk: Sidewalk, current_waypoint: Vector):
        """Get the next sidewalk, crosswalk and waypoints for a pedestrian.

        Args:
            sidewalk: The current sidewalk the pedestrian is on.
            current_waypoint: The current position of the pedestrian.

        Returns:
            A tuple of (next_sidewalk, crosswalk, waypoints, intersection) or (None, None, None, None).
        """
        current_intersection = self.get_intersection_by_sidewalk(sidewalk, current_waypoint)
        if current_intersection is None:
            self.logger.debug('Pedestrian has no intersection')
            return None, None, None, None

        possible_sidewalks = current_intersection.sidewalks[sidewalk]

        next_sidewalk, crosswalk = random.choice(possible_sidewalks)

        # If there is a crosswalk, add the two points of the next_sidewalk as waypoints
        if crosswalk is not None:
            next_waypoints = [next_sidewalk.start, next_sidewalk.end] if current_waypoint.distance(next_sidewalk.start) < current_waypoint.distance(next_sidewalk.end) else [next_sidewalk.end, next_sidewalk.start]
        else:
            # If there is no crosswalk, add the farther point of the next_sidewalk as waypoints
            next_waypoints = [next_sidewalk.start] if current_waypoint.distance(next_sidewalk.start) > current_waypoint.distance(next_sidewalk.end) else [next_sidewalk.end]

        return next_sidewalk, crosswalk, next_waypoints, current_intersection

    def update_intersections(self, communicator: Communicator):
        """Control the traffic lights at intersections.

        Manages the cycle of traffic signals, switching between vehicle and pedestrian phases.

        Args:
            communicator: Interface for communicating with the simulation environment.
        """
        for intersection in self.intersections:
            if len(intersection.traffic_lights) == 0:
                continue

            # If all lights are red, turn the first light green
            if intersection.all_traffic_lights_red() and intersection.cycle_count == 0:
                self.logger.debug(f'Intersection {intersection.id} will turn green on lane {intersection.traffic_lights[0].lane_id}')
                communicator.traffic_signal_switch_to(intersection.traffic_lights[0].id, 'green')
                self.python_states[intersection.traffic_lights[0].id] = (TrafficSignalState.VEHICLE_GREEN, TrafficSignalState.PEDESTRIAN_RED)
                intersection.increment_cycle_count()
                continue

            # Check if we need to switch to pedestrian crossing
            if intersection.has_completed_cycle():
                if all(light.get_state()[0] == TrafficSignalState.VEHICLE_RED for light in intersection.traffic_lights):
                    self.logger.debug(f'Intersection {intersection.id} will turn green on all pedestrian lanes')
                    for light in intersection.pedestrian_lights:
                        communicator.traffic_signal_switch_to(light.id, 'pedestrian walk')
                    for light in intersection.traffic_lights:
                        communicator.traffic_signal_switch_to(light.id, 'pedestrian walk')
                        self.python_states[light.id] = (TrafficSignalState.VEHICLE_RED, TrafficSignalState.PEDESTRIAN_GREEN)
                    intersection.reset_cycle_count()
                continue

            # Find green light and check if it needs to change
            for i, light in enumerate(intersection.traffic_lights):
                if light.get_state()[0] == TrafficSignalState.VEHICLE_RED:  # UE state is red
                    # Check if UE state is different from Python state (indicating need to change)
                    if self.python_states.get(light.id, (None, None))[0] == TrafficSignalState.VEHICLE_GREEN:
                        self.logger.debug(f'Intersection {intersection.id} will turn red on lane {light.lane_id}')
                        # Set current light to red
                        self.python_states[light.id] = (TrafficSignalState.VEHICLE_RED, TrafficSignalState.PEDESTRIAN_RED)

                        # Set next light to green
                        next_light = intersection.traffic_lights[(i + 1) % len(intersection.traffic_lights)]
                        self.logger.debug(f'Intersection {intersection.id} will turn green on lane {next_light.lane_id}')
                        self.python_states[next_light.id] = (TrafficSignalState.VEHICLE_GREEN, TrafficSignalState.PEDESTRIAN_RED)
                        communicator.traffic_signal_switch_to(next_light.id, 'green')

                        intersection.increment_cycle_count()
                        break
