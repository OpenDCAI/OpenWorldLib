"""Road network generation module for the city simulation.

This module handles the procedural generation of road networks, including the creation
of road segments, intersections, and the overall structure of the city's transportation system.
"""
import math
import random
from typing import List

from ...citygen.dataclass.dataclass import (Intersection, MetaInfo,
                                                  Point, Segment)
from ...citygen.road.road_manager import RoadManager
from ...utils.load_json import load_json
from ...utils.math_utils import MathUtils
from ...utils.priority_queue import PriorityQueue
from ...utils.road_utils import RoadUtils


class RoadGenerator:
    """Handles procedural road network generation."""

    def __init__(self, config, num_segments: int = None):
        """Initialize the road generator."""
        # Store references
        self.config = config
        self.num_segments = num_segments
        self.road_manager = RoadManager(config)
        self.queue = PriorityQueue()

    def generate_roads_from_file(self, input_path: str) -> None:
        """Generate roads from a JSON file.

        Args:
            input_path: Path to the JSON file containing road data.
        """
        # Read and parse the JSON file
        data = load_json(input_path)
        road_data = data['roads']

        # Process each road segment
        for road in road_data:
            start = Point(
                road['start']['x'],
                road['start']['y']
            )
            end = Point(
                road['end']['x'],
                road['end']['y']
            )
            meta = MetaInfo(
                highway=road['is_highway'],
                t=0.0
            )
            segment = Segment(start, end, meta)
            self.road_manager.add_segment(segment)

        # Find and add intersections
        self.find_intersections()

    def generate_initial_segments(self) -> None:
        """Generate the initial road segments at the city center."""
        start = Point(0, 0)

        if self.config['citygen.road.two_segment_init']:
            # Create two opposing highway segments
            self.queue.enqueue(
                self.create_segment(start, 0, self.config['citygen.road.segment_length'], False)
            )
            self.queue.enqueue(
                self.create_segment(start, 180, self.config['citygen.road.segment_length'], False)
            )
        else:
            # Create single highway segment
            self.queue.enqueue(
                self.create_segment(start, 0, self.config['citygen.road.segment_length'], False)
            )

    def generate_step(self) -> bool:
        """Process one step of road network generation.

        Returns:
            True when generation should stop, False otherwise.
        """
        if len(self.road_manager.roads) >= self.num_segments:
            return True

        if self.queue.empty():
            return True

        segment = self.queue.dequeue()
        if not segment:
            return True

        # Try to merge segment endpoints with existing points
        RoadUtils.merge_point_if_close(
            segment, 'start', self.road_manager.roads, self.road_manager.merge_distance
        )
        RoadUtils.merge_point_if_close(
            segment, 'end', self.road_manager.roads, self.road_manager.merge_distance
        )

        # Skip if segment became too short after merging
        if MathUtils.length(segment.start, segment.end) < self.road_manager.merge_distance:
            return False

        # Add segment if valid
        if self.road_manager.can_place_segment(segment):
            self.road_manager.add_segment(segment)
            self.generate_next_segments(segment)

        return False

    def create_segment(
        self, start: Point, angle: float, length: float, is_highway: bool, t: float = 0
    ) -> Segment:
        """Create a new road segment with specified parameters.

        Args:
            start: Starting point of the segment.
            angle: Direction angle in degrees.
            length: Length of the segment.
            is_highway: Whether this is a highway segment.
            t: Time parameter for generation sequencing.

        Returns:
            A new road segment.
        """
        # Calculate end point using angle and length
        angle_rad = math.radians(angle)
        end = Point(
            start.x + length * round(math.cos(angle_rad), 10),
            start.y + length * round(math.sin(angle_rad), 10),
        )

        return Segment(start, end, MetaInfo(highway=is_highway, t=t))

    def generate_next_segments(self, segment: Segment):
        """Generate potential next segments from the current segment.

        Args:
            segment: The current segment to generate branches from.
        """
        is_highway = segment.q.highway
        current_t = segment.q.t

        # Calculate current segment angle
        angle = segment.get_angle()

        # Prevent road generation outside bounds
        if abs(segment.end.x) > self.config['citygen.quadtree.bounds.width'] / 2 - 200 or abs(segment.end.y) > self.config['citygen.quadtree.bounds.height'] / 2 - 200:
            return

        potential_segments = []

        if is_highway:
            self._generate_highway_branches(
                segment, angle, current_t, potential_segments
            )
        else:
            self._generate_normal_branches(
                segment, angle, current_t, potential_segments
            )
        self._select_and_queue_segments(potential_segments)

    def _generate_highway_branches(
        self,
        segment: Segment,
        angle: float,
        current_t: float,
        potential_segments: List[Segment],
    ):
        """Generate potential highway branches including straight continuation and side roads.

        Args:
            segment: The current segment.
            angle: Current segment angle.
            current_t: Current time parameter.
            potential_segments: List to store generated segments.
        """
        # Try straight continuation
        # TODO: can add different angle for different road
        new_angle = angle

        if self._is_angle_valid(segment, new_angle):
            new_segment = self.create_segment(
                segment.end,
                new_angle,
                self.config['citygen.road.segment_length'],
                True,
                current_t + self.config['citygen.road.time_delay_between_segments'],
            )
            if self.road_manager.can_place_segment(new_segment):
                potential_segments.append(new_segment)

        # Try side branches
        self._generate_side_branches(
            segment, angle, current_t, potential_segments, True
        )

    def _generate_normal_branches(
        self,
        segment: Segment,
        angle: float,
        current_t: float,
        potential_segments: List[Segment],
    ):
        """Generate potential normal road branches.

        Args:
            segment: The current segment.
            angle: Current segment angle.
            current_t: Current time parameter.
            potential_segments: List to store generated segments.
        """
        # Try straight continuation
        # TODO: can add different angle for different road
        new_angle = angle

        if self._is_angle_valid(segment, new_angle):
            new_segment = self.create_segment(
                segment.end,
                new_angle,
                self.config['citygen.road.segment_length'],
                False,
                current_t + self.config['citygen.road.time_delay_between_segments'],
            )
            if self.road_manager.can_place_segment(new_segment):
                potential_segments.append(new_segment)
        # Try side branches

        self._generate_side_branches(
            segment, angle, current_t, potential_segments, False
        )

    def _generate_side_branches(
        self,
        segment: Segment,
        angle: float,
        current_t: float,
        potential_segments: List[Segment],
        is_highway: bool,
    ):
        """Generate side branches for both highway and normal roads.

        Args:
            segment: The current segment.
            angle: Current segment angle.
            current_t: Current time parameter.
            potential_segments: List to store generated segments.
            is_highway: Whether the parent segment is a highway.
        """
        possible_directions = [-90, 90]
        random.shuffle(possible_directions)

        for base_angle in possible_directions:
            branch_angle = angle + base_angle

            if self._is_angle_valid(segment, branch_angle):
                new_segment = self.create_segment(
                    segment.end,
                    branch_angle,
                    self.config['citygen.road.segment_length'],
                    False,
                    current_t + self.config['citygen.road.time_delay_between_segments']
                )
                if self.road_manager.can_place_segment(new_segment):
                    potential_segments.append(new_segment)

    def _is_angle_valid(self, segment: Segment, new_angle: float) -> bool:
        """Check if the new angle is valid considering existing road angles.

        Args:
            segment: The current segment.
            new_angle: The angle to validate.

        Returns:
            True if the angle is valid, False otherwise.
        """
        current_angles = RoadUtils.get_current_angles(
            segment, self.road_manager.roads, self.queue.elements, self.road_manager.merge_distance
        )

        return all(
            abs(MathUtils.angle_difference(new_angle, ea))
            >= self.config['citygen.road.minimum_intersection_deviation']
            for ea in current_angles
        )

    def _select_and_queue_segments(self, potential_segments: List[Segment]):
        """Select which potential segments to add to the generation queue.

        Args:
            potential_segments: List of potential segments to select from.
        """
        if not potential_segments:
            return
        # Prioritize highways (10% chance to keep at least one)

        highways = [s for s in potential_segments if s.q.highway]
        if highways and random.random() < 0.1:
            self.queue.enqueue(highways[0])
            potential_segments.remove(highways[0])

        # Add remaining segments with 50% probability each
        for segment in potential_segments:
            if random.random() < 0.5:
                self.queue.enqueue(segment)

    def find_intersections(self) -> None:
        """Find all road intersections in the network."""
        self.road_manager.intersections = []
        processed_points = set()

        for segment in self.road_manager.roads:
            for point in [segment.start, segment.end]:
                point_key = (point.x, point.y)
                if point_key in processed_points:
                    continue

                # Find all segments connected to this point
                connected_segments = [
                    s for s in self.road_manager.roads
                    if MathUtils.length(s.start, point) < 0.1
                    or MathUtils.length(s.end, point) < 0.1
                ]

                if len(connected_segments) > 1:
                    self.road_manager.intersections.append(Intersection(point, connected_segments))
                processed_points.add(point_key)
