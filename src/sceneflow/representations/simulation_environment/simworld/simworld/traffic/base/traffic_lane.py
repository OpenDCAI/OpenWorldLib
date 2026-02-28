"""Traffic lane module for representing traffic lanes in the simulation.

This module defines the traffic lane class used to represent road lanes
where vehicles can travel in the simulation.
"""
from ...utils.vector import Vector


class TrafficLane:
    """Represents a traffic lane in the simulation.

    A traffic lane is a part of a road where vehicles can travel in one direction.
    It maintains its start and end points, direction, and a list of vehicles on it.
    """
    _id_counter = 0

    def __init__(self, road_id: int, start: Vector, end: Vector):
        """Initialize a traffic lane.

        Args:
            road_id: The ID of the road this lane belongs to.
            start: The starting point of the lane.
            end: The ending point of the lane.
        """
        self.id = TrafficLane._id_counter
        TrafficLane._id_counter += 1
        self.road_id = road_id
        self.start = start
        self.end = end
        self.vehicles = []
        self.direction = self.get_direction()

    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter for traffic lanes.

        Used when resetting the simulation to ensure IDs start from 0.
        """
        cls._id_counter = 0

    def get_direction(self):
        """Calculate and return the normalized direction vector of the lane.

        Returns:
            A normalized Vector representing the lane's direction.
        """
        return Vector(self.end.x - self.start.x, self.end.y - self.start.y).normalize()

    def __repr__(self):
        """Return a string representation of the traffic lane.

        Returns:
            A string containing the lane's attributes.
        """
        return f'Lane(id={self.id}, road_id={self.road_id}, start={self.start}, end={self.end}, direction={self.direction})'

    def add_vehicle(self, vehicle):
        """Add a vehicle to this lane.

        Args:
            vehicle: The vehicle object to add to the lane.
        """
        self.vehicles.append(vehicle)

    def remove_vehicle(self, vehicle):
        """Remove a vehicle from this lane.

        Args:
            vehicle: The vehicle object to remove from the lane.
        """
        self.vehicles.remove(vehicle)
