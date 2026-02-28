"""Road module for representing roads in the simulation.

This module defines the road class which is responsible for generating and managing
lanes, sidewalks, and crosswalks that make up the road infrastructure.
"""
from .crosswalk import Crosswalk
from .sidewalk import Sidewalk
from .traffic_lane import TrafficLane
from ...utils.vector import Vector


class Road:
    """Represents a road in the simulation.

    A road consists of multiple lanes, sidewalks, and crosswalks. It connects
    two points in the simulation and provides the infrastructure for vehicles and pedestrians.
    """
    _id_counter = 0

    def __init__(self, start: Vector, end: Vector, num_lanes: int, lane_offset: float, intersection_offset: float, sidewalk_offset: float, crosswalk_offset: float):
        """Initialize a road.

        Args:
            start: The starting point of the road.
            end: The ending point of the road.
            num_lanes: Number of lanes in each direction.
            lane_offset: Offset distance between lanes.
            intersection_offset: Distance to offset lanes from intersection.
            sidewalk_offset: Distance to offset sidewalks from the road edge.
            crosswalk_offset: Distance to offset crosswalks from the road edge.
        """
        self.id = Road._id_counter
        Road._id_counter += 1
        self.start = start
        self.end = end
        self.num_lanes = num_lanes
        self.lane_offset = lane_offset
        self.sidewalk_offset = sidewalk_offset
        self.intersection_offset = intersection_offset
        self.crosswalk_offset = crosswalk_offset
        self.direction = self.get_direction()
        self.center = self.get_center()

        # {'forward1': Lane, 'backward1': Lane, 'forward2': Lane, 'backward2': Lane, ...}
        # forward and backward is relative to the road direction
        self.lanes = {}

        # {'forward': Sidewalk, 'backward': Sidewalk}
        # forward and backward is relative to the road direction
        self.sidewalks = {}

        # There should be two crosswalks for each road
        self.crosswalks = []

    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter for roads.

        Used when resetting the simulation to ensure IDs start from 0.
        """
        cls._id_counter = 0

    def __repr__(self):
        """Return a string representation of the road.

        Returns:
            A string containing the road's attributes.
        """
        return f'Road(id={self.id}, start={self.start}, end={self.end}, center={self.center}, direction={self.direction})'

    def get_direction(self):
        """Calculate and return the normalized direction vector of the road.

        Returns:
            A normalized Vector representing the road's direction.
        """
        return Vector(self.end.x - self.start.x, self.end.y - self.start.y).normalize()

    def get_center(self):
        """Calculate and return the center point of the road.

        Returns:
            A Vector representing the center point of the road.
        """
        return Vector((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)

    def init_road(self):
        """Initialize the road by creating lanes, sidewalks, and crosswalks.

        This method should be called after creating a road to set up all its components.
        """
        self.add_lanes()
        self.add_sidewalks()
        self.add_crosswalks()

    def add_lanes(self):
        """Create and add traffic lanes to the road.

        Creates forward and backward lanes for each lane number, offset appropriately
        from the road center line.
        """
        normal_vector = Vector(self.direction.y, -self.direction.x)

        for i in range(self.num_lanes):
            lane_start = self.start - normal_vector * (self.lane_offset * (i + 1)) + self.direction * self.intersection_offset
            lane_end = self.end - normal_vector * (self.lane_offset * (i + 1)) - self.direction * self.intersection_offset
            self.lanes['forward' + str(i+1)] = TrafficLane(self.id, lane_start, lane_end)
            lane_start = self.end + normal_vector * (self.lane_offset * (i + 1)) - self.direction * self.intersection_offset
            lane_end = self.start + normal_vector * (self.lane_offset * (i + 1)) + self.direction * self.intersection_offset
            self.lanes['backward' + str(i+1)] = TrafficLane(self.id, lane_start, lane_end)

    def add_sidewalks(self):
        """Create and add sidewalks to the road.

        Creates forward and backward sidewalks on each side of the road,
        offset appropriately from the road edge.
        """
        normal_vector = Vector(self.direction.y, -self.direction.x)
        sidewalk_start = self.start - normal_vector * (self.sidewalk_offset) + self.direction * self.crosswalk_offset
        sidewalk_end = self.end - normal_vector * (self.sidewalk_offset) - self.direction * self.crosswalk_offset
        self.sidewalks['forward'] = Sidewalk(self.id, sidewalk_start, sidewalk_end)

        sidewalk_start = self.end + normal_vector * (self.sidewalk_offset) - self.direction * self.crosswalk_offset
        sidewalk_end = self.start + normal_vector * (self.sidewalk_offset) + self.direction * self.crosswalk_offset
        self.sidewalks['backward'] = Sidewalk(self.id, sidewalk_start, sidewalk_end)

    def add_crosswalks(self):
        """Create and add crosswalks to the road.

        Creates crosswalks connecting the sidewalks at each end of the road.
        """
        crosswalk_start = self.sidewalks['forward'].end
        crosswalk_end = self.sidewalks['backward'].start
        self.crosswalks.append(Crosswalk(crosswalk_start, crosswalk_end, self.id))

        crosswalk_start = self.sidewalks['backward'].end
        crosswalk_end = self.sidewalks['forward'].start
        self.crosswalks.append(Crosswalk(crosswalk_start, crosswalk_end, self.id))
