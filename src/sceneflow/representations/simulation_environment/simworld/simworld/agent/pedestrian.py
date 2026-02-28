"""Pedestrian agent module for simulating pedestrians in traffic."""
import math
from enum import Enum, auto

import numpy as np

from .base_agent import BaseAgent
from ..traffic.base.sidewalk import Sidewalk
from ..utils.vector import Vector


class PedestrianState(Enum):
    """Enumeration of possible pedestrian states."""
    STOP = auto()
    MOVE_FORWARD = auto()
    TURN_AROUND = auto()


class Pedestrian(BaseAgent):
    """Pedestrian agent for traffic simulation."""

    _id_counter = 0

    def __init__(self, position: Vector, direction: Vector, current_sidewalk: Sidewalk = None, speed: float = 100):
        """Initialize a pedestrian agent.

        Args:
            position: Initial position vector.
            direction: Initial direction vector.
            current_sidewalk: Sidewalk where the pedestrian starts.
            speed: Pedestrian movement speed in units per second.
        """
        super().__init__(position, direction)
        self.id = Pedestrian._id_counter
        Pedestrian._id_counter += 1

        self.current_sidewalk = current_sidewalk
        self.waypoints = []
        self.state = PedestrianState.STOP

        self.speed = speed

    @classmethod
    def reset_id_counter(cls):
        """Reset the pedestrian ID counter to zero."""
        cls._id_counter = 0

    def __str__(self):
        """Return a string representation of the pedestrian."""
        return f'Pedestrian(id={self.id}, position={self.position}, direction={self.direction})'

    def __repr__(self):
        """Return a detailed string representation of the pedestrian."""
        return f'Pedestrian(id={self.id}, current_sidewalk={self.current_sidewalk.id}, position={self.position}, direction={self.direction}, waypoints={self.waypoints})'

    def change_to_next_sidewalk(self, next_sidewalk):
        """Change the pedestrian's current sidewalk to the next sidewalk.

        Args:
            next_sidewalk: The sidewalk to change to.
        """
        self.current_sidewalk.pedestrians.remove(self)
        self.current_sidewalk = next_sidewalk
        self.current_sidewalk.pedestrians.append(self)

    def add_waypoint(self, waypoints: list[Vector]):
        """Add waypoints to the pedestrian's path.

        Args:
            waypoints: List of waypoint vectors to add.
        """
        self.waypoints.extend(waypoints)

    def pop_waypoint(self):
        """Remove and return the first waypoint.

        Returns:
            Vector: The first waypoint.
        """
        return self.waypoints.pop(0)

    def is_close_to_end(self, waypoint_distance_threshold: float):
        """Check if the pedestrian is close to the end of the sidewalk.

        Args:
            waypoint_distance_threshold: Distance threshold to consider close to waypoint.

        Returns:
            bool: True if pedestrian is close to end of the sidewalk.
        """
        if len(self.waypoints) == 0:
            return False
        return self.position.distance(self.waypoints[-1]) < waypoint_distance_threshold

    def complete_turn(self):
        """Check if the pedestrian has completed turning around.

        Returns:
            bool: True if the turn is completed.
        """
        if self.state == PedestrianState.TURN_AROUND and len(self.waypoints) > 0:
            to_waypoint = self.waypoints[0] - self.position
            angle = math.degrees(math.acos(np.clip(self.direction.dot(to_waypoint.normalize()), -1, 1)))
            # complete turn if agent is facing the waypoint or the angle has passed the point
            return angle < 2 or angle >= 90
        return False

    def compute_control(self, waypoint: Vector):
        """Compute control values for pedestrian movement.

        Args:
            waypoint: Target waypoint to move towards.

        Returns:
            tuple: Angle to turn and direction to turn ('left', 'right', or None).
        """
        to_waypoint = waypoint - self.position

        angle = math.degrees(math.acos(np.clip(self.direction.dot(to_waypoint.normalize()), -1, 1)))

        cross_product = self.direction.cross(to_waypoint)
        turn_direction = 'left' if cross_product < 0 else 'right'

        if angle < 2:
            return 0, None
        else:
            return angle, turn_direction
