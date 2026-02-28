"""Building manager module for managing buildings in the city.

This module provides functionality to add, remove, and check for collisions with existing buildings
in the city's quadtree structure.
"""
from typing import List

from ...citygen.dataclass.dataclass import Bounds, Building
from ...utils.bbox_utils import BboxUtils
from ...utils.quadtree import QuadTree


class BuildingManager:
    """Building manager class for managing buildings in the city.

    This class provides functionality to add, remove, and check for collisions with existing buildings
    in the city's quadtree structure.
    """
    def __init__(self, config):
        """Initialize the building manager.

        Args:
            config: Configuration dictionary with simulation parameters.
        """
        self.config = config
        self.building_quadtree = QuadTree[Building](
            Bounds(self.config['citygen.quadtree.bounds.x'], self.config['citygen.quadtree.bounds.y'], self.config['citygen.quadtree.bounds.width'], self.config['citygen.quadtree.bounds.height']),
            self.config['citygen.quadtree.max_objects'],
            self.config['citygen.quadtree.max_levels'])
        self.buildings: List[Building] = []

    def can_place_building(self, bounds: Bounds, buffer: float = None) -> bool:
        """Check if a building can be placed at the specified location.

        Args:
            bounds: Bounds of the building.
            buffer: Buffer distance.

        Returns:
            True if the building can be placed, False otherwise.
        """
        if buffer is None:
            buffer = self.config['citygen.building.building_building_distance']

        # Add buffer spacing to extend check range
        check_bounds = Bounds(
            bounds.x - buffer,
            bounds.y - buffer,
            bounds.width + 2 * buffer,
            bounds.height + 2 * buffer,
            bounds.rotation,
        )

        # Check for collisions with existing buildings
        candidates = self.building_quadtree.retrieve(check_bounds)

        for building in candidates:
            if BboxUtils.bbox_overlap(building.bounds, check_bounds):
                return False
        return True

    def add_building(self, building: Building):
        """Add new building to manager.

        Args:
            building: Building to add.
        """
        self.buildings.append(building)
        # To be added to quadtree later
        self.building_quadtree.insert(building.bounds, building)

    def remove_building(self, building: Building):
        """Remove a building from the quadtree and list.

        Args:
            building: Building to remove.
        """
        self.buildings.remove(building)
        self.building_quadtree.remove(building.bounds, building)

    def rebuild_quadtree(self):
        """Rebuild the quadtree."""
        self.building_quadtree.clear()
        for building in self.buildings:
            self.building_quadtree.insert(building.bounds, building)
