"""Scooter agent module."""

from .base_agent import BaseAgent
from ..utils.vector import Vector


class Scooter(BaseAgent):
    """Scooter agent for traffic simulation."""

    _id_counter = 0

    def __init__(self, position: Vector, direction: Vector):
        """Initialize scooter agent.

        Args:
            position: Position.
            direction: Direction.
        """
        super().__init__(position, direction)
        self.id = Scooter._id_counter
        Scooter._id_counter += 1

        self.throttle = 0
        self.brake = 0
        self.steering = 0

    def __str__(self):
        """Return a string representation of the scooter."""
        return f'Scooter(id={self.id}, position={self.position}, direction={self.direction})'

    def __repr__(self):
        """Return a detailed string representation of the scooter."""
        return self.__str__()

    def get_attributes(self):
        """Get the attributes of the scooter."""
        return self.throttle, self.brake, self.steering

    def set_attributes(self, throttle: float, brake: float, steering: float):
        """Set the attributes of the scooter."""
        self.throttle = throttle
        self.brake = brake
        self.steering = steering
