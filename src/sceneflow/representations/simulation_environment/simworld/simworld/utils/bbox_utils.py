"""Utility module for bounding box calculations and operations.

This module provides functionality for checking intersections between rotated bounding boxes
and other geometric operations without external dependencies.
"""
import math

from ..citygen.dataclass.dataclass import Bounds


class BboxUtils:
    """Utility class for bounding box operations.

    Provides static methods for geometric operations on bounding boxes,
    particularly for detecting collisions between rotated rectangles.
    """
    @staticmethod
    def bbox_overlap(a: Bounds, b: Bounds) -> bool:
        """Check if two rotated bounding boxes overlap without using shapely.

        Args:
            a: First bounding box.
            b: Second bounding box.

        Returns:
            True if the bounding boxes overlap, False otherwise.
        """
        def rotate_point(cx, cy, x, y, angle):
            """Rotate a point around a center by a given angle (in radians).

            Args:
                cx: Center x coordinate.
                cy: Center y coordinate.
                x: Point x coordinate.
                y: Point y coordinate.
                angle: Rotation angle in radians.

            Returns:
                Tuple of rotated (x, y) coordinates.
            """
            dx, dy = x - cx, y - cy
            rotated_x = cx + dx * math.cos(angle) - dy * math.sin(angle)
            rotated_y = cy + dx * math.sin(angle) + dy * math.cos(angle)
            return rotated_x, rotated_y

        def get_corners(bounds):
            """Get the rotated corners of a bounding box.

            Args:
                bounds: The bounding box.

            Returns:
                List of corner coordinates as (x, y) tuples.
            """
            cx = bounds.x + bounds.width / 2
            cy = bounds.y + bounds.height / 2
            w2, h2 = bounds.width / 2, bounds.height / 2
            corners = [
                (-w2, -h2), (w2, -h2),
                (w2, h2), (-w2, h2)
            ]
            angle = math.radians(bounds.rotation)
            return [rotate_point(cx, cy, cx + x, cy + y, angle) for x, y in corners]

        def is_point_in_polygon(px, py, polygon):
            """Check if a point is inside a polygon using the ray-casting method.

            Args:
                px: Point x coordinate.
                py: Point y coordinate.
                polygon: List of vertex coordinates forming the polygon.

            Returns:
                True if the point is inside the polygon, False otherwise.
            """
            n = len(polygon)
            inside = False
            x1, y1 = polygon[0]
            for i in range(1, n + 1):
                x2, y2 = polygon[i % n]
                if min(y1, y2) < py <= max(y1, y2) and px <= max(x1, x2):
                    if y1 != y2:
                        xinters = (py - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x1 == x2 or px <= xinters:
                        inside = not inside
                x1, y1 = x2, y2
            return inside

        def do_segments_intersect(p1, q1, p2, q2):
            """Check if two line segments intersect.

            Args:
                p1: First point of first segment.
                q1: Second point of first segment.
                p2: First point of second segment.
                q2: Second point of second segment.

            Returns:
                True if the segments intersect, False otherwise.
            """
            def orientation(p, q, r):
                val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                if val == 0:
                    return 0
                return 1 if val > 0 else 2

            def on_segment(p, q, r):
                return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            if o1 != o2 and o3 != o4:
                return True
            if o1 == 0 and on_segment(p1, p2, q1):
                return True
            if o2 == 0 and on_segment(p1, q2, q1):
                return True
            if o3 == 0 and on_segment(p2, p1, q2):
                return True
            if o4 == 0 and on_segment(p2, q1, q2):
                return True
            return False

        # Get rotated corners
        a_corners = get_corners(a)
        b_corners = get_corners(b)

        # Check if any corner of A is inside B
        for corner in a_corners:
            if is_point_in_polygon(corner[0], corner[1], b_corners):
                return True

        # Check if any corner of B is inside A
        for corner in b_corners:
            if is_point_in_polygon(corner[0], corner[1], a_corners):
                return True

        # Check if any edge of A intersects with any edge of B
        a_edges = [(a_corners[i], a_corners[(i + 1) % 4]) for i in range(4)]
        b_edges = [(b_corners[i], b_corners[(i + 1) % 4]) for i in range(4)]

        for edge_a in a_edges:
            for edge_b in b_edges:
                if do_segments_intersect(edge_a[0], edge_a[1], edge_b[0], edge_b[1]):
                    return True

        return False
