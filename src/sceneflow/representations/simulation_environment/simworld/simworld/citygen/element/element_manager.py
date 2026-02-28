"""Element manager module for managing elements in the city.

This module provides functionality to add, remove, and check for collisions with existing elements
in the city's quadtree structure.
"""
from ...citygen.dataclass.dataclass import Bounds, Element
from ...utils.bbox_utils import BboxUtils
from ...utils.logger import Logger
from ...utils.quadtree import QuadTree


class ElementManager:
    """Manage elements for the city."""

    def __init__(self, config):
        """Initialize the element manager.

        Args:
            config: Configuration dictionary with simulation parameters.
        """
        self.config = config
        self.elements = []
        self.element_quadtree = QuadTree[Element](
            Bounds(self.config['citygen.quadtree.bounds.x'], self.config['citygen.quadtree.bounds.y'], self.config['citygen.quadtree.bounds.width'], self.config['citygen.quadtree.bounds.height']),
            self.config['citygen.quadtree.max_objects'],
            self.config['citygen.quadtree.max_levels']
        )
        self.logger = Logger.get_logger('ElementManager')

    def add_element(self, element: Element):
        """Add a element to the quadtree and list.

        Args:
            element: Element to add.
        """
        self.elements.append(element)
        self.element_quadtree.insert(element.bounds, element)

    def can_place_element(self, bounds: Bounds, buffer: float = None) -> bool:
        """Check if a element can be placed at the specified location.

        Args:
            bounds: Bounds of the element.
            buffer: Buffer distance.

        Returns:
            True if the element can be placed, False otherwise.
        """
        if buffer is None:
            buffer = self.config['citygen.element.element_element_distance']
        check_bounds = Bounds(
            bounds.x - buffer,
            bounds.y - buffer,
            bounds.width + 2 * buffer,
            bounds.height + 2 * buffer,
            bounds.rotation,
        )

        # Check for collisions with existing items
        candidates = self.element_quadtree.retrieve(check_bounds)

        for item in candidates:
            if BboxUtils.bbox_overlap(item.bounds, check_bounds):
                return False
        return True

    def remove_element(self, element: Element):
        """Remove a element from the quadtree and list."""
        try:
            self.elements.remove(element)
            self.element_quadtree.remove(element.bounds, element)
        except ValueError:
            self.logger.error(f'Element {element.element_type.name} not found in elements list')

    def rebuild_quadtree(self):
        """Rebuild the quadtree."""
        self.element_quadtree.clear()
        for element in self.elements:
            self.element_quadtree.insert(element.bounds, element)
