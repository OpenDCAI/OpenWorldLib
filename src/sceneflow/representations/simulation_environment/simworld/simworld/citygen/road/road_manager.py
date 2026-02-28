"""Road network management module for the city simulation.

This module provides functionality for managing road segments, including spatial indexing,
conflict detection, and intersection identification.
"""
from typing import List

from ...citygen.dataclass.dataclass import Bounds, Intersection, Segment
from ...utils.math_utils import MathUtils
from ...utils.quadtree import QuadTree


class RoadManager:
    """Manages road segments and their spatial relationships."""

    def __init__(self, config):
        """Initialize the road manager."""
        # Core data structures
        self.roads: List[Segment] = []
        self.intersections: List[Intersection] = []

        # Configuration
        self.config = config
        self.merge_distance = self.config['citygen.road.road_snap_distance']

        # Initialize spatial index
        bounds = Bounds(self.config['citygen.quadtree.bounds.x'], self.config['citygen.quadtree.bounds.y'], self.config['citygen.quadtree.bounds.width'], self.config['citygen.quadtree.bounds.height'])
        self.road_quadtree = QuadTree[Segment](
            bounds,
            self.config['citygen.quadtree.max_objects'],
            self.config['citygen.quadtree.max_levels']
        )

    def add_segment(self, segment: Segment) -> None:
        """Add a road segment to the network."""
        self.roads.append(segment)
        bounds = self._create_bounds_for_segment(segment)
        self.road_quadtree.insert(bounds, segment)

    def can_place_segment(self, segment: Segment) -> bool:
        """Check if a segment can be placed without conflicts."""
        if self.config['citygen.road.ignore_conflicts']:
            return True

        # Check minimum length
        if MathUtils.length(segment.start, segment.end) < self.merge_distance:
            return False

        # Check for intersections with existing segments
        bounds = self._create_bounds_for_segment(segment)
        nearby_segments = self.road_quadtree.retrieve(bounds)

        for other in nearby_segments:
            if other.start == segment.start and other.end == segment.end:
                continue
            if MathUtils.do_line_segments_intersect(
                segment.start, segment.end, other.start, other.end
            ):
                return False
        return True

    def get_nearby_segments(self, bounds: Bounds) -> List[Segment]:
        """Get segments within the specified bounds."""
        return self.road_quadtree.retrieve(bounds)

    def _create_bounds_for_segment(self, segment: Segment) -> Bounds:
        """Create a bounding box for a road segment."""
        min_x = min(segment.start.x, segment.end.x)
        min_y = min(segment.start.y, segment.end.y)
        width = abs(segment.end.x - segment.start.x)
        height = abs(segment.end.y - segment.start.y)

        # Add some padding to the bounds
        padding = self.merge_distance

        return Bounds(
            min_x - padding,
            min_y - padding,
            width + 2 * padding,
            height + 2 * padding,
            segment.get_angle(),    # RoadUtils.create_bounds_for_segment initializes angle as well
        )

    def remove_segment(self, segment: Segment) -> None:
        """Remove a road segment from the network."""
        self.roads.remove(segment)
        bounds = self._create_bounds_for_segment(segment)
        self.road_quadtree.remove(bounds, segment)

    def get_segment_by_id(self, id: int) -> Segment:
        """Get a road segment by its ID."""
        return self.roads[id]

    def update_segment(self, old_segment: Segment, new_segment: Segment) -> None:
        """Update a road segment's position in the spatial index."""
        old_bounds = self._create_bounds_for_segment(old_segment)
        self.road_quadtree.remove(old_bounds, old_segment)

        new_bounds = self._create_bounds_for_segment(new_segment)
        self.road_quadtree.insert(new_bounds, new_segment)

    def rebuild_quadtree(self):
        """Rebuild the quadtree."""
        self.road_quadtree.clear()
        for segment in self.roads:
            bounds = self._create_bounds_for_segment(segment)
            self.road_quadtree.insert(bounds, segment)
