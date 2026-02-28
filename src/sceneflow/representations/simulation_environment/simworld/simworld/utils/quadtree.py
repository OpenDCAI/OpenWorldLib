"""Quadtree implementation for efficient spatial partitioning and querying."""
from typing import Generic, List, Optional, TypeVar

from ..citygen.dataclass.dataclass import Bounds

T = TypeVar('T')


class QuadTree(Generic[T]):
    """Quadtree data structure for efficient spatial partitioning and querying.

    A quadtree recursively divides space into four quadrants to efficiently store and
    query spatial data.

    Attributes:
        bounds: The spatial bounds of this quadtree node.
        max_objects: Maximum number of objects before splitting.
        max_levels: Maximum depth of the quadtree.
        level: Current depth level of this node.
        objects: List of object bounds in this node.
        items: List of items corresponding to the bounds.
        nodes: Child nodes of this quadtree.
    """

    def __init__(self, bounds: Bounds, max_objects=10, max_levels=4, level=0):
        """Initialize a new quadtree node.

        Args:
            bounds: The spatial bounds of this quadtree node.
            max_objects: Maximum number of objects before splitting.
            max_levels: Maximum depth of the quadtree.
            level: Current depth level of this node.
        """
        self.bounds = bounds
        self.max_objects = max_objects
        self.max_levels = max_levels
        self.level = level
        self.objects: List[Bounds] = []
        self.items: List[T] = []
        self.nodes: List[Optional[QuadTree]] = [None] * 4

    def split(self):
        """Split this node into four child nodes.

        Divides the current node into four equal quadrants and redistributes
        the contained objects among them.
        """
        width = self.bounds.width / 2
        height = self.bounds.height / 2
        x = self.bounds.x
        y = self.bounds.y

        self.nodes[0] = QuadTree(
            Bounds(x + width, y, width, height),
            self.max_objects,
            self.max_levels,
            self.level + 1,
        )
        self.nodes[1] = QuadTree(
            Bounds(x, y, width, height),
            self.max_objects,
            self.max_levels,
            self.level + 1,
        )
        self.nodes[2] = QuadTree(
            Bounds(x, y + height, width, height),
            self.max_objects,
            self.max_levels,
            self.level + 1,
        )
        self.nodes[3] = QuadTree(
            Bounds(x + width, y + height, width, height),
            self.max_objects,
            self.max_levels,
            self.level + 1,
        )

        for i, rect in enumerate(self.objects):
            for node in self.get_relevant_nodes(rect):
                node.insert(rect, self.items[i])

    def get_relevant_nodes(self, rect: Bounds) -> List['QuadTree[T]']:
        """Get the child nodes that intersect with the given rectangle.

        Args:
            rect: The bounding rectangle to test intersection with.

        Returns:
            List of child nodes that intersect with the rectangle.
        """
        nodes = []
        mid_x = self.bounds.x + self.bounds.width / 2
        mid_y = self.bounds.y + self.bounds.height / 2

        top = rect.y <= mid_y
        bottom = rect.y + rect.height > mid_y

        if rect.x <= mid_x:
            if top:
                nodes.append(self.nodes[1])
            if bottom:
                nodes.append(self.nodes[2])
        if rect.x + rect.width > mid_x:
            if top:
                nodes.append(self.nodes[0])
            if bottom:
                nodes.append(self.nodes[3])
        return [n for n in nodes if n is not None]

    def insert(self, rect: Bounds, item: T):
        """Insert an item with its bounds into the quadtree.

        Args:
            rect: The bounding rectangle of the item.
            item: The item to insert.
        """
        if any(self.nodes):
            for node in self.get_relevant_nodes(rect):
                node.insert(rect, item)
            return
        self.objects.append(rect)
        self.items.append(item)

        if len(self.objects) > self.max_objects and self.level < self.max_levels:
            if not any(self.nodes):
                self.split()

    def retrieve(self, rect: Bounds) -> List[T]:
        """Retrieve all items that might intersect with the given rectangle.

        Args:
            rect: The bounding rectangle to query.

        Returns:
            List of items that might intersect with the rectangle.
        """
        result = []
        if any(self.nodes):
            for node in self.get_relevant_nodes(rect):
                result.extend(node.retrieve(rect))
        else:
            result = self.items
        return result

    def retrieve_exact(self, query_rect: Bounds) -> List[T]:
        """Retrieve all items that have an exact intersection with the given rectangle.

        Args:
            query_rect: The bounding rectangle to query.

        Returns:
            List of items that intersect with the rectangle.
        """
        candidates = self.retrieve(query_rect)
        candidate_rects = [candidate.bounds for candidate in candidates]
        result = []
        for item, item_rect in zip(candidates, candidate_rects):
            if query_rect.intersects(item_rect):
                result.append(item)
        return result

    def clear(self):
        """Clear the quadtree, removing all items and resetting to initial state."""
        self.objects = []
        self.items = []
        self.nodes = [None] * 4

    def remove(self, bounds: Bounds, item: T) -> bool:
        """Remove an item and its bounds from the quadtree.

        Args:
            bounds: The bounding rectangle of the item.
            item: The item to remove.

        Returns:
            True if item was found and removed, False otherwise.
        """
        # Check if item exists in current node
        if item in self.items:
            index = self.items.index(item)
            self.items.pop(index)
            self.objects.pop(index)

            # Recursively remove from child nodes if they exist
            if any(self.nodes):
                for node in self.get_relevant_nodes(bounds):
                    node.remove(bounds, item)
            return True

        # If not in current node but has children, try removing from relevant child nodes
        elif any(self.nodes):
            for node in self.get_relevant_nodes(bounds):
                if node.remove(bounds, item):
                    return True

        return False
