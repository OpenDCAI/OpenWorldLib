"""Priority queue implementation for managing segments.

This module provides a priority queue implementation specifically designed for
managing segments in the simulation. It supports basic queue operations with
priority-based dequeuing.
"""
from typing import List, Optional

from ..citygen.dataclass.dataclass import Segment


class PriorityQueue:
    """A priority queue implementation for managing segments.

    This class implements a priority queue where segments are ordered based on their
    t value. It provides standard queue operations with priority-based dequeuing.
    """

    def __init__(self):
        """Initialize an empty priority queue."""
        self.elements: List[Segment] = []

    def enqueue(self, segment: Segment):
        """Add a segment to the queue.

        Args:
            segment: The segment to add to the queue.
        """
        self.elements.append(segment)

    def dequeue(self) -> Optional[Segment]:
        """Get the segment with the minimum t value.

        Returns:
            The segment with the minimum t value, or None if the queue is empty.
        """
        if not self.elements:
            return None
        min_t = float('inf')
        min_idx = 0

        for i, segment in enumerate(self.elements):
            if segment.q.t < min_t:
                min_t = segment.q.t
                min_idx = i
        return self.elements.pop(min_idx)

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            True if the queue is empty, False otherwise.
        """
        return len(self.elements) == 0

    def __iter__(self):
        """Make the queue iterable.

        Returns:
            An iterator over the queue elements.
        """
        return iter(self.elements)

    def __len__(self):
        """Get the number of elements in the queue.

        Returns:
            The number of elements in the queue.
        """
        return len(self.elements)

    @property
    def size(self):
        """Get the number of elements in the queue.

        Returns:
            The number of elements in the queue.
        """
        return len(self.elements)
