"""Building generator module for generating buildings along road segments."""
import math
import random

from ...citygen.building.building_manager import BuildingManager
from ...citygen.dataclass.dataclass import (Bounds, Building, Point,
                                                  Segment)
from ...utils.logger import Logger
from ...utils.math_utils import MathUtils
from ...utils.quadtree import QuadTree

class BuildingGenerator:
    """Building generator class for placing buildings along road segments."""
    def __init__(self, config, building_types):
        """Initialize the building generator.

        Args:
            config: Configuration dictionary with parameters for building generation.
            building_types: List of available building types.
        """
        self.config = config
        self.building_manager = BuildingManager(self.config)
        self.sorted_buildings = sorted(building_types, key=lambda b: b.width, reverse=True)

        self.building_counts = {b: 0 for b in building_types}
        self.building_to_segment = {}

        self.logger = Logger.get_logger('BuildingGenerator')

    def get_next_building_type(self):
        """Choose the next building type to generate.

        Returns:
            The next building type to be placed.
        """
        limited_buildings = [b for b in self.sorted_buildings if b.num_limit != -1]
        random.shuffle(limited_buildings)
        for building_type in limited_buildings:
            current_count = self.building_counts[building_type]
            if current_count < building_type.num_limit:
                return building_type

        unlimited_buildings = [b for b in self.sorted_buildings if b.num_limit == -1]
        if unlimited_buildings:
            return random.choice(unlimited_buildings)

        return self.sorted_buildings[-1]

    def get_smallest_available_building_type(self):
        """From the smallest building to the largest, return the first building type that has not reached the limit.

        Returns:
            The smallest available building type or None if no types are available.
        """
        for building_type in reversed(self.sorted_buildings):
            if building_type.num_limit == -1 or self.building_counts[building_type] < building_type.num_limit:
                return building_type
        return None

    def generate_buildings_along_segment(self, segment: Segment, road_quadtree: QuadTree[Segment]):
        """Generate buildings along both sides of a road segment.

        Args:
            segment: Road segment to place buildings along.
            road_quadtree: Quadtree containing all road segments.
        """
        # Calculate road direction vector
        dx = segment.end.x - segment.start.x
        dy = segment.end.y - segment.start.y

        length = math.sqrt(dx * dx + dy * dy)

        if length < 1:
            return

        # Unit vector
        dx, dy = dx / length, dy / length

        # Perpendicular vector (left side)
        perpendicular_dx = -dy
        perpendicular_dy = dx

        # Building distance from road
        INTERSECTION_BUFFER = self.config['citygen.building.building_intersection_distance']

        # Generate buildings on both sides of road
        for side in [-1, 1]:  # -1: left, 1: right
            # current_pos is the distance from the start of the segment to the center of the building that is being placed
            current_pos = INTERSECTION_BUFFER
            building_type = self.get_next_building_type()
            # current_pos += building_type.width / 2 + Config.BUILDING_SIDE_OFFSET
            current_pos += building_type.width / 2
            overlap_building_flag = False
            overlap_road_flag = False
            while current_pos < length - INTERSECTION_BUFFER:
                # while True:
                offset = self.config['citygen.building.building_side_distance'] + building_type.height / 2
                # (x, y) is the center of the building
                x = (
                    segment.start.x
                    + dx * current_pos
                    + side * perpendicular_dx * offset
                )
                y = (
                    segment.start.y
                    + dy * current_pos
                    + side * perpendicular_dy * offset
                )

                rotation = (math.degrees(math.atan2(dy, dx)) + (180 if side == 1 else 0)) % 360
                building_bounds = Bounds(x - building_type.width / 2, y - building_type.height / 2, building_type.width, building_type.height, rotation)

                if self.building_manager.can_place_building(building_bounds) and not self.check_building_road_overlap(building_bounds, road_quadtree):
                    building = Building(
                        building_type=building_type,
                        bounds=building_bounds,
                        rotation=rotation,
                    )
                    self.building_manager.add_building(building)
                    self.building_to_segment[building] = segment

                    self.building_counts[building_type] += 1

                    next_building_type = self.get_next_building_type()
                    while True:
                        next_pos = current_pos + building_type.width / 2 + next_building_type.width / 2 + self.config['citygen.building.building_building_distance']
                        next_offset = self.config['citygen.building.building_side_distance'] + next_building_type.height / 2
                        next_x = (
                            segment.start.x
                            + dx * next_pos
                            + side * perpendicular_dx * next_offset
                        )
                        next_y = (
                            segment.start.y
                            + dy * next_pos
                            + side * perpendicular_dy * next_offset
                        )
                        next_building_bounds = Bounds(next_x - next_building_type.width / 2, next_y - next_building_type.height / 2, next_building_type.width, next_building_type.height, rotation)
                        if self.building_manager.can_place_building(next_building_bounds) and not self.check_building_road_overlap(next_building_bounds, road_quadtree):
                            building_type = next_building_type
                            current_pos = next_pos
                            break
                        else:
                            idx = self.sorted_buildings.index(next_building_type) + 1
                            while idx < len(self.sorted_buildings):
                                if self.building_counts.get(self.sorted_buildings[idx], 0) == 0:
                                    next_building_type = self.sorted_buildings[idx]
                                    break
                                idx += 1
                            if idx == len(self.sorted_buildings):
                                next_building_type = self.get_next_building_type()
                                break

                elif not self.building_manager.can_place_building(building_bounds):  # overlap with other buildings
                    if not overlap_building_flag:
                        building_type = self.get_smallest_available_building_type()
                        if building_type is None:
                            break
                        current_pos = INTERSECTION_BUFFER + self.config['citygen.building.building_side_distance'] + building_type.width / 2
                        overlap_building_flag = True
                        overlap_road_flag = True
                    else:
                        building_type = self.get_smallest_available_building_type()
                        if building_type is None:
                            break
                        current_pos += random.uniform(0, 1)
                elif self.check_building_road_overlap(building_bounds, road_quadtree):  # overlap with roads
                    if not overlap_road_flag:
                        current_pos += self.config['citygen.building.building_side_distance']
                        overlap_road_flag = True
                    else:
                        building_type = self.get_smallest_available_building_type()
                        if building_type is None:
                            break
                        current_pos += random.uniform(0, 1)

    def filter_overlapping_buildings(self, segments_quadtree: QuadTree[Segment]):
        """Filter out buildings that overlap with roads after building generation.

        Args:
            segments_quadtree: Quadtree containing all road segments.
        """
        buildings_to_remove = []
        self.logger.info(f'Generated buildings: {len(self.building_manager.buildings)}')
        for building in self.building_manager.buildings:
            if self.check_building_road_overlap(building.bounds, segments_quadtree):
                buildings_to_remove.append(building)
        self.logger.info(f'Buildings to keep: {len(self.building_manager.buildings) - len(buildings_to_remove)}')
        for building in set(buildings_to_remove):
            self.building_manager.remove_building(building)

    def check_building_road_overlap(
        self, building_bounds: Bounds, segments_quadtree: QuadTree[Segment]
    ) -> bool:
        """Check if building overlaps with roads.

        Args:
            building_bounds: Bounds of the building to check.
            segments_quadtree: Quadtree containing all road segments.

        Returns:
            True if the building overlaps with any road, False otherwise.
        """
        ROAD_CLEARANCE = self.config['citygen.building.building_road_distance']

        # Create a search bounds that includes margin around the building
        search_margin = max(ROAD_CLEARANCE, self.config['citygen.building.building_intersection_distance'])
        search_bounds = Bounds(
            building_bounds.x - search_margin,
            building_bounds.y - search_margin,
            building_bounds.width + 2 * search_margin,
            building_bounds.height + 2 * search_margin,
            building_bounds.rotation,
        )

        # Get only nearby segments using quadtree
        nearby_segments = segments_quadtree.retrieve(search_bounds)

        # building center
        center = Point(
            building_bounds.x + building_bounds.width / 2,
            building_bounds.y + building_bounds.height / 2,
        )

        # building corners
        corners = [
            MathUtils.rotate_point(center, Point(building_bounds.x, building_bounds.y), building_bounds.rotation),
            MathUtils.rotate_point(center, Point(building_bounds.x + building_bounds.width, building_bounds.y), building_bounds.rotation),
            MathUtils.rotate_point(center, Point(building_bounds.x, building_bounds.y + building_bounds.height), building_bounds.rotation),
            MathUtils.rotate_point(center, Point(building_bounds.x + building_bounds.width, building_bounds.y + building_bounds.height), building_bounds.rotation),
        ]

        for segment in nearby_segments:
            for corner in corners:
                if MathUtils.point_segment_distance(corner, segment) <= ROAD_CLEARANCE:
                    return True

            building_edges = [
                (corners[0], corners[1]),
                (corners[1], corners[3]),
                (corners[2], corners[3]),
                (corners[0], corners[2]),
            ]

            # Check if building intersects with road segment
            for edge_start, edge_end in building_edges:
                if MathUtils.do_line_segments_intersect(
                    segment.start, segment.end, edge_start, edge_end
                ):
                    return True

            for point in [segment.start, segment.end]:
                dx = point.x - (building_bounds.x + building_bounds.width / 2)
                dy = point.y - (building_bounds.y + building_bounds.height / 2)
                if math.sqrt(dx * dx + dy * dy) < self.config['citygen.building.building_intersection_distance']:
                    return True

        return False
