"""Base agent class for all agents in the simulation."""
import math

from ..utils.vector import Vector


class BaseAgent:
    """Base class for all agents in the simulation."""

    def __init__(self, position: Vector, direction: Vector):
        """Initialize the base agent.

        Args:
            position: Initial position vector.
            direction: Initial direction vector.
        """
        self._position = position if isinstance(position, Vector) else Vector(0, 0)
        self._direction = direction if isinstance(direction, Vector) else Vector(0, 0)  
        self._yaw = 0

    @property
    def position(self):
        """Get the position of the agent.

        Returns:
            Vector: The position of the agent.
        """
        return self._position

    @property
    def direction(self):
        """Get the direction of the agent.

        Returns:
            Vector: The direction of the agent.
        """
        return self._direction

    @property
    def yaw(self):
        """Get the yaw of the agent.

        Returns:
            float: The yaw of the agent.
        """
        return self._yaw

    @position.setter
    def position(self, position: Vector):
        """Set the position of the agent.

        Args:
            position: The new position vector.
        """
        self._position = position

    @direction.setter
    def direction(self, yaw: float):
        """Set the direction of the agent.

        Args:
            yaw: The new yaw of the agent.
        """
        self._yaw = yaw
        self._direction = Vector(math.cos(math.radians(yaw)), math.sin(math.radians(yaw))).normalize()
