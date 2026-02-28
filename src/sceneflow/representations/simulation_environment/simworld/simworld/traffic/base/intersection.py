"""Intersection module for representing and managing intersections in the simulation.

This module defines the intersection class which is responsible for managing the connections
between roads, lanes, sidewalks, and controlling traffic signals.
"""
import math
from typing import List

from .crosswalk import Crosswalk
from .road import Road
from .traffic_lane import TrafficLane
from .traffic_signal import (TrafficSignal, TrafficSignalState)
from ...utils.vector import Vector


class Intersection:
    """Represents a road intersection in the simulation.

    Manages the connection of multiple roads, including lanes, sidewalks, and traffic signals.
    Handles the traffic flow logic at the intersection.
    """
    _id_counter = 0

    def __init__(self, center: Vector, roads: List[Road]):
        """Initialize an intersection.

        Args:
            center: The center point of the intersection.
            roads: List of roads connected to this intersection.
        """
        self.id = Intersection._id_counter
        Intersection._id_counter += 1
        self.center = center
        self.roads = roads  # roads connected to the intersection

        self.traffic_lights: List[TrafficSignal] = []  # [traffic_light]
        self.pedestrian_lights: List[TrafficSignal] = []  # [pedestrian_light]
        self.crosswalks = []  # [crosswalk]

        # number of sidewalks should be twice of the number of lanes
        self.lane_mapping = {}  # {incoming_lane: [outgoing_lane]}
        self.sidewalk_mapping = {}  # {sidewalk: [(sidewalk, crosswalk) or (sidewalk, None)]}

        # for traffic light cycle
        self.cycle_count = 0    # 0 for the pedestrian green light
        self.total_lights = 0

        self.is_light_changing = False

    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter for intersections.

        Used when resetting the simulation to ensure IDs start from 0.
        """
        cls._id_counter = 0

    def __repr__(self):
        """Return a string representation of the intersection.

        Returns:
            A string containing the intersection's attributes.
        """
        return f'Intersection(id={self.id}, center={self.center}, lanes={self.lane_mapping})'

    def init_intersection(self, config):
        """Initialize the intersection with configuration.

        Sets up lane connections, sidewalk connections, and traffic signals.

        Args:
            config: Configuration dictionary with simulation parameters.
        """
        self.connect_lanes(config)
        self.connect_sidewalks(config)
        self.add_traffic_signals(config)

    def add_traffic_signals(self, config):
        """Add traffic signals to the intersection.

        One signal for each road, controlling both lanes and crosswalks on that road.

        Args:
            config: Configuration dictionary with traffic signal parameters.

        Raises:
            ValueError: If there are no lanes to add traffic signals to.
        """
        if len(self.lane_mapping) == 0:
            raise ValueError('No lanes to add traffic light')

        if len(self.lane_mapping) == 1:  # u-turn intersection
            return

        # Sort incoming lanes clockwise based on their angle from the center
        sorted_lanes = sorted(
            self.lane_mapping.keys(),
            key=lambda lane: (
                -math.atan2(
                    lane.start.y - self.center.y,
                    lane.start.x - self.center.x
                ) % (2 * math.pi)
            )
        )

        # Add traffic lights in clockwise order
        for incoming_lane in sorted_lanes:
            # In UE, Y axis is inverted. The normal direction is on the left side of the direction vector
            incoming_lane_normal_direction = Vector(incoming_lane.direction.y, -incoming_lane.direction.x)
            traffic_light_center = self.center + (incoming_lane_normal_direction * config['traffic.traffic_signal.light_normal_offset']) + (incoming_lane.direction * config['traffic.traffic_signal.light_radial_offset'])

            dir_traffic_light = Vector(-incoming_lane.direction.x, -incoming_lane.direction.y)

            for crosswalk in self.crosswalks:
                if crosswalk.road_id == incoming_lane.road_id:
                    self.traffic_lights.append(TrafficSignal(traffic_light_center, dir_traffic_light, incoming_lane.id, crosswalk.id, 'both'))
                    break

        # Simplified check for pedestrian lights (for 3-way intersection)
        for light in self.traffic_lights:
            extended_position = light.position + ((self.center - light.position) * 2)
            if not any(existing_light.position == extended_position for existing_light in self.traffic_lights):
                self.pedestrian_lights.append(TrafficSignal(extended_position, light.direction * -1, None, None, 'pedestrian'))

        # for 2-way intersection
        if len(self.traffic_lights) + len(self.pedestrian_lights) < 4:
            for light in self.traffic_lights:
                extended_position = light.position + light.direction * config['traffic.traffic_signal.light_radial_offset'] * 2
                if not any(existing_light.position == extended_position for existing_light in self.traffic_lights):
                    negative_normal_direction = Vector(-light.direction.y, light.direction.x)
                    self.pedestrian_lights.append(TrafficSignal(extended_position, negative_normal_direction, None, None, 'pedestrian'))

        # Count only 'both' type traffic lights
        self.total_lights = len(self.traffic_lights)

    def connect_lanes(self, config):
        """Connect lanes at the intersection.

        Creates mappings from incoming lanes to outgoing lanes.

        Args:
            config: Configuration dictionary with lane parameters.
        """
        # Make a U-turn at the intersection
        if len(self.roads) == 1:
            for i in range(config['traffic.num_lanes']):
                incoming_lane = self.roads[0].lanes['forward' + str(i+1)]
                outgoing_lane = self.roads[0].lanes['backward' + str(i+1)]
                self.lane_mapping[incoming_lane] = [outgoing_lane]
        else:
            # Forbid U-turn at the intersection
            # Handle multiple roads intersection
            for lane_level in range(config['traffic.num_lanes']):
                lane_num = lane_level + 1
                # Collect all incoming and outgoing lanes at this level
                incoming_lanes = []
                outgoing_lanes = []

                for road in self.roads:
                    # Add forward lane (incoming) if end is closer to intersection
                    forward_lane = road.lanes[f'forward{lane_num}']
                    if forward_lane.end.distance(self.center) < forward_lane.start.distance(self.center):
                        incoming_lanes.append((road.id, forward_lane))
                    else:
                        outgoing_lanes.append((road.id, forward_lane))

                    # Add backward lane (incoming) if end is closer to intersection
                    backward_lane = road.lanes[f'backward{lane_num}']
                    if backward_lane.end.distance(self.center) < backward_lane.start.distance(self.center):
                        incoming_lanes.append((road.id, backward_lane))
                    else:
                        outgoing_lanes.append((road.id, backward_lane))

                # Connect each incoming lane to all possible outgoing lanes from different roads
                for incoming_road_id, incoming_lane in incoming_lanes:
                    valid_outgoing_lanes = [
                        out_lane for out_road_id, out_lane in outgoing_lanes
                        if out_road_id != incoming_road_id  # Exclude lanes from the same road
                    ]
                    self.lane_mapping[incoming_lane] = valid_outgoing_lanes

    def connect_sidewalks(self, config):
        """Connect sidewalks at the intersection.

        Creates mappings between sidewalks and determines crosswalk connections.

        Args:
            config: Configuration dictionary with sidewalk parameters.

        Raises:
            ValueError: If a road has an unexpected number of crosswalks.
        """
        # Collect all sidewalks
        all_sidewalks = []
        for road in self.roads:
            all_sidewalks.extend([road.sidewalks['forward'], road.sidewalks['backward']])
            # Only add the crosswalk that is closer to this intersection
            if len(road.crosswalks) == 2:
                crosswalk1, crosswalk2 = road.crosswalks
                # Calculate the minimum distance of each crosswalk to the intersection center
                dist1 = min(crosswalk1.start.distance(self.center), crosswalk1.end.distance(self.center))
                dist2 = min(crosswalk2.start.distance(self.center), crosswalk2.end.distance(self.center))
                # Add the closer crosswalk
                self.crosswalks.append(crosswalk1 if dist1 < dist2 else crosswalk2)
            else:
                raise ValueError(f'Road {road.id} has {len(road.crosswalks)} crosswalks')

        # Calculate maximum connection distance
        max_distance = math.sqrt((2 * config['traffic.sidewalk_offset'])**2 + (2 * config['traffic.crosswalk_offset'])**2)

        # For each sidewalk, find its connections to other sidewalks and crosswalks
        for sidewalk in all_sidewalks:
            # Get the point closer to intersection center
            start_dist = sidewalk.start.distance(self.center)
            end_dist = sidewalk.end.distance(self.center)
            incoming_point = sidewalk.end if end_dist < start_dist else sidewalk.start

            # Find valid connections to other sidewalks and crosswalks
            valid_connections = []
            for other_sidewalk in all_sidewalks:
                if other_sidewalk == sidewalk:
                    continue

                # Get the point of other sidewalk closer to intersection center
                other_start_dist = other_sidewalk.start.distance(self.center)
                other_end_dist = other_sidewalk.end.distance(self.center)
                other_point = other_sidewalk.end if other_end_dist < other_start_dist else other_sidewalk.start

                if incoming_point == other_point:
                    valid_connections.append((other_sidewalk, None))
                    continue

                # If points are within range, add to valid connections with no crosswalk
                # 200 is a magic number to avoid error when calculating float
                if incoming_point.distance(other_point) < max_distance - 200:
                    for crosswalk in self.crosswalks:
                        if (crosswalk.start == incoming_point or crosswalk.end == incoming_point) and \
                           (crosswalk.start == other_point or crosswalk.end == other_point):
                            valid_connections.append((other_sidewalk, crosswalk))
                            break

            self.sidewalk_mapping[sidewalk] = valid_connections

    def all_traffic_lights_red(self):
        """Check if all traffic lights at the intersection are red.

        Returns:
            True if all traffic lights are red, False otherwise.
        """
        if len(self.traffic_lights) == 0:
            return False

        for traffic_light in self.traffic_lights:
            if not traffic_light.get_state() == (TrafficSignalState.VEHICLE_RED, TrafficSignalState.PEDESTRIAN_RED):
                return False
        return True

    def get_traffic_light_state(self, incoming_lane: TrafficLane):
        """Get the state of the traffic light for a specific lane.

        Args:
            incoming_lane: The lane to get the traffic light state for.

        Returns:
            The state of the traffic light for the lane.
        """
        for traffic_light in self.traffic_lights:
            if traffic_light.lane_id == incoming_lane.id:
                return traffic_light.get_state()
        return None

    def get_crosswalk_light_state(self, crosswalk: Crosswalk):
        """Get the state of the traffic light for a specific crosswalk.

        Args:
            crosswalk: The crosswalk to get the traffic light state for.

        Returns:
            A tuple of (state, left_time) for the crosswalk.
        """
        for traffic_light in self.traffic_lights:
            if traffic_light.crosswalk_id == crosswalk.id:
                state = traffic_light.get_state()[1]
                left_time = traffic_light.get_left_time()
                return state, left_time

        # If no traffic light is found, return the pedestrian green light
        return TrafficSignalState.PEDESTRIAN_GREEN, 20

    def has_completed_cycle(self):
        """Check if all traffic lights have cycled through green once.

        Returns:
            True if a complete cycle has been completed, False otherwise.
        """
        if self.cycle_count >= self.total_lights:
            return True
        return False

    def reset_cycle_count(self):
        """Reset the cycle count when a light cycle is completed."""
        self.cycle_count = 0

    def increment_cycle_count(self):
        """Increment the cycle count when a light turns green."""
        self.cycle_count += 1

    @property
    def lanes(self):
        """Get the lane mapping for this intersection.

        Returns:
            Dictionary mapping incoming lanes to outgoing lanes.
        """
        return self.lane_mapping

    @property
    def sidewalks(self):
        """Get the sidewalk mapping for this intersection.

        Returns:
            Dictionary mapping sidewalks to their connections.
        """
        return self.sidewalk_mapping
