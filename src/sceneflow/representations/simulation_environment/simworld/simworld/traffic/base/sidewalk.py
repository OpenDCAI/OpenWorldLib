"""Sidewalk module for representing sidewalks in the simulation.

This module defines the sidewalk class used to represent pedestrian paths
along the sides of roads in the simulation.
"""
from ...utils.vector import Vector


class Sidewalk:
    """Represents a sidewalk in the simulation.

    A sidewalk is a path along the side of a road where pedestrians can walk.
    It maintains its start and end points, direction, and a list of pedestrians on it.
    """
    _id_counter = 0

    def __init__(self, road_id: int, start: Vector, end: Vector):
        """Initialize a sidewalk.

        Args:
            road_id: The ID of the road this sidewalk belongs to.
            start: The starting point of the sidewalk.
            end: The ending point of the sidewalk.
        """
        self.id = Sidewalk._id_counter
        Sidewalk._id_counter += 1
        self.road_id = road_id
        self.start = start
        self.end = end
        self.pedestrians = []
        self.direction = self.get_direction()

    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter for sidewalks.

        Used when resetting the simulation to ensure IDs start from 0.
        """
        cls._id_counter = 0

    def get_direction(self):
        """Calculate and return the normalized direction vector of the sidewalk.

        Returns:
            A normalized Vector representing the sidewalk's direction.
        """
        return Vector(self.end.x - self.start.x, self.end.y - self.start.y).normalize()

    def __repr__(self):
        """Return a string representation of the sidewalk.

        Returns:
            A string containing the sidewalk's attributes.
        """
        return f'Sidewalk(id={self.id}, road_id={self.road_id}, start={self.start}, end={self.end}, direction={self.direction})'

    def add_pedestrian(self, pedestrian):
        """Add a pedestrian to this sidewalk.

        Args:
            pedestrian: The pedestrian object to add to the sidewalk.
        """
        self.pedestrians.append(pedestrian)

    def remove_pedestrian(self, pedestrian):
        """Remove a pedestrian from this sidewalk.

        Args:
            pedestrian: The pedestrian object to remove from the sidewalk.
        """
        self.pedestrians.remove(pedestrian)
