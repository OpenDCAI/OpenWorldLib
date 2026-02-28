"""Crosswalk module for representing crosswalks in the simulation.

This module defines the crosswalk class used to represent pedestrian crossing paths
that connect sidewalks across roads in the simulation.
"""
from ...utils.vector import Vector

class Crosswalk:
    """Represents a crosswalk in the simulation.

    A crosswalk connects two sidewalks and allows pedestrians to cross roads safely.
    It defines the start and end points of the crossing path.
    """
    _id_counter = 0

    def __init__(self, start: Vector, end: Vector, road_id: int):
        """Initialize a crosswalk.

        Args:
            start: The starting point of the crosswalk.
            end: The ending point of the crosswalk.
            road_id: The ID of the road this crosswalk belongs to.
        """
        self.id = Crosswalk._id_counter
        Crosswalk._id_counter += 1
        self.start = start
        self.end = end

        self.road_id = road_id
        self.sidewalks = []

    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter for crosswalks.

        Used when resetting the simulation to ensure IDs start from 0.
        """
        cls._id_counter = 0

    def __repr__(self):
        """Return a string representation of the crosswalk.

        Returns:
            A string containing the crosswalk's attributes.
        """
        return f'Crosswalk(id={self.id}, start={self.start}, end={self.end}, road_id={self.road_id})'
