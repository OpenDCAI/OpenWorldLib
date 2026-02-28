"""Route generation module for creating travel routes in the simulation.

This module handles the creation of travel routes based on road segments and points of interest
in the generated city, providing navigation paths for the simulation.
"""
import random
from typing import List

from ...citygen.dataclass.dataclass import (Bounds, Building, Element,
                                                  Point, Segment)
from ...citygen.route.route_manager import RouteManager
from ...utils.logger import Logger
from ...utils.math_utils import MathUtils
from ...utils.quadtree import QuadTree


class RouteGenerator:
    """Generates routes between points of interest in the city.

    This class creates travel routes along roads and between elements in the city,
    providing navigation paths for agents in the simulation.
    """
    def __init__(self, config):
        """Initialize the route generator.

        Args:
            config: Configuration dictionary with simulation parameters.
        """
        self.config = config
        self.route_manager = RouteManager()
        self.logger = Logger.get_logger('RouteGenerator')

    def generate_route_along_road(self, road: Segment):
        """Generate a route along a road segment.

        Creates a route with random points along a given road segment.

        Args:
            road: The road segment to generate the route along.
        """
        # Generate random number of points along road
        num_points = random.randint(self.config['citygen.route.min_points_per_route'], self.config['citygen.route.max_points_per_route'])
        points = []
        self.logger.info(f'Generating route with {num_points} points')
        for _ in range(num_points):
            # Generate random point along road segment
            point = MathUtils.interpolate_point(road.start, road.end, random.random())
            points.append(point)
        self.route_manager.add_route_points(points)

    def generate_route_based_on_elements(self, elements: List[Element]):
        """Generate routes based on elements' positions.

        Creates routes connecting specific elements in the city.

        Args:
            elements: List of elements to use as route endpoints.
        """
        num_points = 2
        element = random.sample(elements, 1)[0]
        points = []
        for _ in range(num_points):
            point = element.center
            points.append(point)
        self.route_manager.add_route_points(points)

    def generate_target_point_randomly(self):
        """Generate a random target point based on available elements.

        Returns:
            A randomly selected point based on an element's position.
        """
        elements = self.element_manager.elements
        element = random.sample(elements, 1)[0]
        target_point = element.center
        return target_point

    def get_point_around_label(self, point: Point, quadtrees: List[QuadTree], distance: float = 50, k: int = 5):
        """Get the label of a point based on its K nearest neighbors.

        Analyzes the surroundings of a point to determine its contextual information.

        Args:
            point: The point to analyze.
            quadtrees: List of quadtrees containing spatial data.
            distance: Search radius around the point.
            k: Number of nearest neighbors to consider.

        Returns:
            Dictionary containing statistics about nearby elements and buildings.
        """
        bounds = Bounds(point.x - distance, point.y - distance, 2 * distance, 2 * distance)

        neighbors = []
        for quadtree in quadtrees:
            nodes = quadtree.retrieve(bounds)
            neighbors += nodes

        # Count different types of neighbors
        elements = [neighbor for neighbor in neighbors if isinstance(neighbor, Element)][:k]
        buildings = [neighbor for neighbor in neighbors if isinstance(neighbor, Building)][:k]
        # Not using segments in the current implementation
        # segments = [neighbor for neighbor in neighbors if isinstance(neighbor, Segment)][:k]

        element_types = [element.element_type for element in elements]
        # Not using building_types in the current implementation
        # building_types = [building.building_type for building in buildings]

        element_stats = {element_type.name: element_types.count(element_type) for element_type in element_types}

        building_stats = {building.building_type.name: MathUtils.get_direction_description_for_points(point, building.center) for building in buildings}

        stats = {
            'element_stats': element_stats,
            'building_stats': building_stats,
        }

        return stats
