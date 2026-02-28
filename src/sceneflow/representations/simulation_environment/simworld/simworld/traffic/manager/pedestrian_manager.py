"""Pedestrian management module for traffic simulation.

This module handles the creation, spawning, and updating of pedestrians in the simulation.
It manages pedestrian lifecycle, movement logic, and interaction with traffic signals.
"""
import random

from ...agent.pedestrian import Pedestrian, PedestrianState
from ..base.traffic_signal import TrafficSignalState
from ...utils.logger import Logger


class PedestrianManager:
    """Manages pedestrians in the traffic simulation.

    This class handles spawning pedestrians on sidewalks, updating their positions and states,
    and managing their interactions with intersections and traffic signals.
    """
    def __init__(self, roads, num_pedestrians, config):
        """Initialize the pedestrian manager with configuration and initial pedestrians.

        Args:
            roads: List of road segments where pedestrians can be placed.
            num_pedestrians: Number of pedestrians to create initially.
            config: Configuration dictionary with simulation parameters.
        """
        self.config = config
        self.pedestrians = []
        self.num_pedestrians = num_pedestrians
        self.roads = roads

        self.logger = Logger.get_logger('PedestrianController')
        self.logger.info(f'PedestrianController initialized with {num_pedestrians} pedestrians')

        self.model_path = self.config['traffic.pedestrian.model_path']

        self.init_pedestrians()

    def init_pedestrians(self):
        """Initialize the pedestrians and place them on the sidewalks.

        Pedestrians are randomly placed on sidewalks, ensuring they maintain safe distances
        from other pedestrians and intersections.
        """
        while len(self.pedestrians) < self.num_pedestrians:
            target_road = random.choice(self.roads)
            target_sidewalk = random.choice(list(target_road.sidewalks.values()))
            target_position = random.uniform(target_sidewalk.start, target_sidewalk.end)

            # check if the pedestrian is too close to the intersection
            if target_position.distance(target_sidewalk.end) < self.config['traffic.crosswalk_offset']:
                continue

            possible_pedestrians = target_sidewalk.pedestrians
            for pedestrian in possible_pedestrians:
                if pedestrian.position.distance(target_position) < 0.5 * self.config['traffic.distance_between_objects']:
                    break
            else:
                target_direction = target_sidewalk.direction * random.choice([1, -1])
                new_pedestrian = Pedestrian(position=target_position, direction=target_direction, current_sidewalk=target_sidewalk,
                                            speed=random.choice([self.config['traffic.pedestrian.min_speed'], self.config['traffic.pedestrian.max_speed']]))

                if target_direction.dot(target_sidewalk.direction) < 0:  # if the target direction is opposite to the sidewalk direction, add the start point as a waypoint
                    new_pedestrian.add_waypoint([target_sidewalk.start])
                else:   # if the target direction is the same as the sidewalk direction, add the end point as a waypoint
                    new_pedestrian.add_waypoint([target_sidewalk.end])
                target_sidewalk.add_pedestrian(new_pedestrian)
                self.pedestrians.append(new_pedestrian)

                self.logger.info(f'Spawned Pedestrian: {new_pedestrian.id} on Sidewalk {target_sidewalk.id} at {target_position}')

    def spawn_pedestrians(self, communicator):
        """Spawn pedestrians in the simulation environment.

        Args:
            communicator: Communication interface to the simulation environment.
        """
        communicator.spawn_pedestrians(self.pedestrians, self.model_path)

    def set_pedestrians_max_speed(self, communicator):
        """Set the maximum speed for each pedestrian in the simulation.

        Args:
            communicator: Communication interface to the simulation environment.
        """
        for pedestrian in self.pedestrians:
            communicator.set_pedestrian_speed(pedestrian.id, pedestrian.speed)

    def update_pedestrians(self, communicator, intersection_controller):
        """Update pedestrian states and movements based on environment conditions.

        This method handles pedestrian movement logic, including waypoint following,
        intersection crossing, and traffic signal compliance.

        Args:
            communicator: Interface for sending updates to the simulation.
            intersection_controller: Controller for managing intersection logic.
        """
        for pedestrian in self.pedestrians:
            if pedestrian.state == PedestrianState.TURN_AROUND:
                if pedestrian.complete_turn():
                    pedestrian.state = PedestrianState.MOVE_FORWARD
                else:
                    continue

            # get the next sidewalk and waypoints for the pedestrian at the intersection
            if pedestrian.is_close_to_end(self.config['traffic.pedestrian.waypoint_distance_threshold']):
                next_sidewalk, crosswalk, waypoints, current_intersection = intersection_controller.get_waypoints_for_pedestrian(pedestrian.current_sidewalk, pedestrian.waypoints[0])
                if next_sidewalk is None or waypoints is None:
                    self.logger.debug(f'Pedestrian {pedestrian.id} has no waypoints to move to')
                    continue

                if crosswalk is not None:
                    pedestrian_light_state, left_time = current_intersection.get_crosswalk_light_state(crosswalk)
                    if pedestrian_light_state == TrafficSignalState.PEDESTRIAN_GREEN and left_time > min(15, self.config['traffic.traffic_signal.pedestrian_green_light_duration']):
                        pedestrian.add_waypoint(waypoints)
                        pedestrian.change_to_next_sidewalk(next_sidewalk)
                        self.logger.debug(f'Pedestrian {pedestrian.id} is moving to Sidewalk {next_sidewalk.id} with waypoints {waypoints}')
                    else:
                        if not pedestrian.state == PedestrianState.STOP:
                            self.logger.debug(f'Pedestrian {pedestrian.id} is waiting at crosswalk {crosswalk.id}')
                            pedestrian.state = PedestrianState.STOP
                            communicator.pedestrian_stop(pedestrian.id)
                        continue
                else:
                    pedestrian.add_waypoint(waypoints)
                    pedestrian.change_to_next_sidewalk(next_sidewalk)
                    self.logger.debug(f'Pedestrian {pedestrian.id} is moving to Sidewalk {next_sidewalk.id} with waypoints {waypoints}')

            # pop waypoint if the pedestrian has reached the waypoint
            if pedestrian.waypoints and len(pedestrian.waypoints) > 0:
                to_waypoint = pedestrian.waypoints[0] - pedestrian.position
                dot_product = pedestrian.direction.dot(to_waypoint.normalize())
                if dot_product < 0:
                    self.logger.debug(f'Pedestrian {pedestrian.id} passed waypoint {pedestrian.waypoints[0]}')
                    pedestrian.pop_waypoint()

            # compute the control input for the pedestrian
            if pedestrian.waypoints:
                if pedestrian.state == PedestrianState.STOP:
                    pedestrian.state = PedestrianState.MOVE_FORWARD
                    communicator.pedestrian_move_forward(pedestrian.id)

                angle, turn_direction = pedestrian.compute_control(pedestrian.waypoints[0])
                if angle != 0:
                    pedestrian.state = PedestrianState.TURN_AROUND
                    communicator.pedestrian_rotate(pedestrian.id, angle, turn_direction)

    def stop_pedestrians(self, communicator):
        """Stop all pedestrians in the simulation."""
        for pedestrian in self.pedestrians:
            pedestrian.state = PedestrianState.STOP
            communicator.pedestrian_stop(pedestrian.id)
