"""Route manager module for generating and managing routes.

This module provides functionality to create and manage routes between points in the simulation.
"""
from typing import List

from ...citygen.dataclass.dataclass import Point, Route


class RouteManager:
    """Manager for generating and managing routes.

    This class provides functionality to create and manage routes between points in the simulation.
    """
    def __init__(self):
        """Initialize the RouteManager.

        Initializes an empty list to store routes.
        """
        self.routes = []

    def add_route_points(self, points: List[Point]):
        """Add a route with points.

        Args:
            points: List of points defining the route.
        """
        start = points[0]
        end = points[-1]
        route = Route(points, start, end)

        self.routes.append(route)
