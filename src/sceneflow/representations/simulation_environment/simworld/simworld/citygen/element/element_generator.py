"""Element generation module for the city simulation.

This module provides functionality for generating various elements in the city,
such as trees, street furniture, and other objects around buildings and roads.
"""
import math
import random
from concurrent.futures import ProcessPoolExecutor
from typing import List

from ...citygen.dataclass.dataclass import (Bounds, Building, Element,
                                                  Segment)
from ...citygen.element.element_manager import ElementManager
from ...utils.bbox_utils import BboxUtils
from ...utils.quadtree import QuadTree


class ElementGenerator:
    """Generator for placing elements in the city environment.

    This class is responsible for creating and placing elements such as trees,
    street furniture, and other decorative objects around buildings and along roads.
    """
    def __init__(self, config, element_types, map_element_offsets):
        """Initialize the element generator.

        Args:
            config: Configuration dictionary with simulation parameters.
            element_types: List of available element types to generate.
            map_element_offsets: Mapping of offset distances to element types.
        """
        self.config = config
        self.element_manager = ElementManager(config)
        self.element_types = element_types
        self.map_element_offsets = map_element_offsets
        self.element_to_owner = {}

    def _add_elements_around_building(self, building: Building) -> List[Element]:
        """Generate elements around a building.

        Creates elements at various positions around a building, considering
        the building's shape and orientation.

        Args:
            building: The building to place elements around.

        Returns:
            A list of generated elements.
        """
        elements = []
        DISTANCE = self.config['citygen.element.element_building_distance']
        NUM_ATTEMPTS = self.config['citygen.element.add_building_attempts']

        building_center = building.center
        building_rotation = math.radians(building.rotation)

        # face road and back road
        # 1 for back road, -1 for face road
        # for side in [1]:
        #     for _ in range(NUM_ATTEMPTS):
        #         # Random angle and distance from building
        #         distance = DISTANCE

        #         x_offset = random.uniform(-building.width/2 + distance/2, building.width/2 - distance/2)
        #         y_offset = random.uniform(-building.height/2 + distance/2, building.height/2 - distance/2)

        #         x = building_center.x +\
        #             (building.width / 2 * abs(math.cos(building_rotation)) + building.height / 2 * abs(math.sin(building_rotation)) + distance) * -math.sin(building_rotation) * side +\
        #             (x_offset * abs(math.cos(building_rotation)) + y_offset * abs(math.sin(building_rotation))) * math.cos(building_rotation)
        #         y = building_center.y +\
        #             (building.height / 2 * abs(math.cos(building_rotation)) + building.width / 2 * abs(math.sin(building_rotation)) + distance) * math.cos(building_rotation) * side +\
        #             (y_offset * abs(math.cos(building_rotation)) + x_offset * abs(math.sin(building_rotation))) * math.sin(building_rotation)

        #         element_type = random.choice(self.element_types)

        #         # Align element rotation with building rotation plus some randomness
        #         element_rotation = random.uniform(-180, 180)

        #         element_bounds = Bounds(
        #             x - element_type.width / 2,
        #             y - element_type.height / 2,
        #             element_type.width,
        #             element_type.height,
        #             element_rotation
        #         )

        #         elements.append(Element(
        #             element_type=element_type,
        #             bounds=element_bounds,
        #             rotation=element_rotation,
        #             building=building
        #         ))

        # left side and right side (building as reference)
        # 1 for right side, -1 for left side
        for side in [1, -1]:
            for _ in range(NUM_ATTEMPTS):
                # Random angle and distance from building
                distance = DISTANCE

                x_offset = random.uniform(-building.width/2 + distance/2, building.width/2 - distance/2)
                y_offset = random.uniform(-building.height/2 + distance/2, building.height/2 - distance/2)

                x = building_center.x +\
                    (building.width / 2 * abs(math.cos(building_rotation)) + building.height / 2 * abs(math.sin(building_rotation)) + distance) * -math.cos(building_rotation) * side +\
                    (x_offset * abs(math.cos(building_rotation)) + y_offset * abs(math.sin(building_rotation))) * math.sin(building_rotation)
                y = building_center.y +\
                    (building.height / 2 * abs(math.cos(building_rotation)) + building.width / 2 * abs(math.sin(building_rotation)) + distance) * -math.sin(building_rotation) * side +\
                    (y_offset * abs(math.cos(building_rotation)) + x_offset * abs(math.sin(building_rotation))) * math.cos(building_rotation)

                element_type = random.choice(self.element_types)

                # Align element rotation with building rotation plus some randomness
                element_rotation = random.uniform(-180, 180)

                element_bounds = Bounds(
                    x - element_type.width / 2,
                    y - element_type.height / 2,
                    element_type.width,
                    element_type.height,
                    element_rotation
                )

                elements.append(Element(
                    element_type=element_type,
                    bounds=element_bounds,
                    rotation=element_rotation,
                    building=building
                ))

        return elements

    def _add_elements_spline_road(self, segment: Segment):
        """Generate elements on a spline road.

        Places elements like trees and street furniture along a road segment,
        considering both sides of the road.

        Args:
            segment: The road segment to place elements along.

        Returns:
            A list of generated elements.
        """
        # Calculate road properties
        dx = segment.end.x - segment.start.x
        dy = segment.end.y - segment.start.y
        length = math.sqrt(dx**2 + dy**2)

        if length < 1:
            return []

        # Sort points to ensure consistent direction
        start_point, end_point = sorted(
            [segment.start, segment.end],
            key=lambda p: (p.y, p.x)
        )

        # Calculate angles based on sorted points
        road_angle = math.atan2(end_point.y - start_point.y, end_point.x - start_point.x)
        element_angle = road_angle + math.pi/2

        def generate_elements_for_offset(offset_config, density_config, road_angle, element_angle):
            if offset_config not in self.map_element_offsets or not self.map_element_offsets[offset_config]:
                return []

            number_of_elements = int(length * density_config)
            if number_of_elements == 0:
                return []

            step_length = length / number_of_elements

            # Generate all element positions at once
            # position: element position projected to the road
            positions = [
                (
                    start_point.x + math.cos(road_angle) * step_length * (i + 1),
                    start_point.y + math.sin(road_angle) * step_length * (i + 1)
                )
                for i in range(2, number_of_elements - 2)
            ]

            elements = []
            for px, py in positions:
                element_type = next(
                    (element for element in self.element_types
                     if element.name == random.choice(self.map_element_offsets[offset_config])),
                    None
                )
                if not element_type:
                    continue

                rotation = random.uniform(-180, 180)

                # Generate elements for both sides of the road
                elements.extend([
                    Element(
                        element_type=element_type,
                        bounds=Bounds(
                            px + side * offset_config * math.cos(element_angle) - element_type.width/2,
                            py + side * offset_config * math.sin(element_angle) - element_type.height/2,
                            element_type.width,
                            element_type.height,
                            rotation
                        ),
                        rotation=rotation
                    )
                    for side in [-1, 1]
                ])

            return elements

        # Generate all types of elements
        return sum(
            (
                generate_elements_for_offset(offset, density, road_angle, element_angle)
                for offset, density in [
                    (self.config['citygen.element.parking_offset'], self.config['citygen.element.parking_density']),
                    (self.config['citygen.element.furniture_offset'], self.config['citygen.element.furniture_density']),
                    (self.config['citygen.element.tree_offset'], self.config['citygen.element.tree_density']),
                ]
            ),
            []
        )

    def generate_elements_around_buildings_multithread(self, buildings: List[Building]):
        """Generate elements around buildings using multiprocessing.

        Args:
            buildings: List of buildings to place elements around.
        """
        with ProcessPoolExecutor(max_workers=self.config['citygen.element.generation_thread_number']) as executor:
            # generate elements around buildings
            future_elements = list(executor.map(self._add_elements_around_building, buildings))

            # add generated elements to element manager
            for idx, elements in enumerate(future_elements):
                for element in elements:
                    if self.element_manager.can_place_element(element.bounds):
                        self.element_manager.add_element(element)
                        self.element_to_owner[element] = buildings[idx]

    def generate_elements_on_road_multithread(self, segments: List[Segment]):
        """Generate elements on roads using multiprocessing.

        Args:
            segments: List of road segments to place elements along.
        """
        with ProcessPoolExecutor(max_workers=self.config['citygen.element.generation_thread_number']) as executor:
            future_elements = list(executor.map(self._add_elements_spline_road, segments))
            for idx, elements in enumerate(future_elements):
                for element in elements:
                    self.element_manager.add_element(element)
                    self.element_to_owner[element] = segments[idx]

    def generate_elements_around_building(self, building: Building):
        """Generate and add elements around a single building.

        Args:
            building: The building to place elements around.
        """
        elements = self._add_elements_around_building(building)
        for element in elements:
            if self.element_manager.can_place_element(element.bounds):
                self.element_manager.add_element(element)

    def filter_elements_by_buildings(self, building_quadtree: QuadTree[Building]):
        """Filter out elements that overlap with buildings.

        Args:
            building_quadtree: Quadtree containing all buildings for efficient spatial querying.
        """
        elements_to_remove = []

        for element in self.element_manager.elements:
            buffer = 100    # magic number. temporary solution for errors in bounding box
            building = element.building
            check_bounds = Bounds(
                building.bounds.x - buffer,
                building.bounds.y - buffer,
                building.bounds.width + 2 * buffer,
                building.bounds.height + 2 * buffer
            )

            # Use quadtree to efficiently find nearby buildings
            nearby_buildings = building_quadtree.retrieve(check_bounds)
            for building in nearby_buildings:
                if BboxUtils.bbox_overlap(element.bounds, building.bounds):
                    elements_to_remove.append(element)
                    break  # No need to check other buildings once overlap found

        for element in set(elements_to_remove):
            self.element_manager.remove_element(element)
