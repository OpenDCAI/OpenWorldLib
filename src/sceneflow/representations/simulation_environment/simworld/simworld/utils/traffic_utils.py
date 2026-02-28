"""Utility functions for traffic simulation and route generation."""

import numpy as np

from ..utils.vector import Vector


def bezier(start: Vector, end: Vector, control_point: Vector, t: float) -> Vector:
    """Calculate a point on a quadratic Bezier curve.

    Args:
        start: Starting point of the curve.
        end: Ending point of the curve.
        control_point: Control point that influences the curve shape.
        t: Parameter value between 0 and 1 representing position along the curve.

    Returns:
        A Vector representing the point on the Bezier curve at parameter t.
    """
    # Calculate the x and y coordinates
    return start * (1 - t) ** 2 + control_point * (1 - t) * t * 2 + end * t ** 2


def get_bezier_points(start: Vector, end: Vector, control_point: Vector, num_points: int) -> list[Vector]:
    """Generate waypoints on a quadratic Bezier curve.

    Args:
        start: Starting point of the curve.
        end: Ending point of the curve.
        control_point: Control point that influences the curve shape.
        num_points: Number of points to sample along the curve.

    Returns:
        List of waypoints on the Bezier curve (excluding the start point).
    """
    t = np.linspace(0, 1, num_points)
    return [bezier(start, end, control_point, t_i) for t_i in t[1:]]


def extend_control_point(start: Vector, end: Vector, intersection: Vector, extension_factor: float = 0.1) -> Vector:
    """Extend the control point from the intersection for smoother curves.

    Args:
        start: Starting point of the curve.
        end: Ending point of the curve.
        intersection: Intersection point (typically junction center).
        extension_factor: Extension factor, larger values produce smoother curves.

    Returns:
        The extended control point as a Vector.
    """
    # Calculate the vector from the intersection to the start and end
    vec_start = start - intersection
    vec_end = end - intersection

    # Calculate the dot product to determine the angle
    dot_product = vec_start.dot(vec_end)

    # Determine the direction of extension based on the dot product
    direction = 1 if dot_product > 0 else -1
    # Add the two vectors and normalize, then extend in the appropriate direction
    control_vec = intersection + (vec_start + vec_end) * extension_factor * direction

    # Extend the control point from the intersection
    return control_vec


def cal_waypoints(start: Vector, end: Vector, gap_between_waypoints: float) -> list[Vector]:
    """Add waypoints between start and end at regular intervals.

    Args:
        start: Starting point.
        end: Ending point.
        gap_between_waypoints: Distance between consecutive waypoints.

    Returns:
        List of waypoints between start and end (including end point).
    """
    distance = start.distance(end)
    num_waypoints = int(distance // gap_between_waypoints)
    waypoints = []
    for i in range(1, num_waypoints + 1):
        fraction = i * gap_between_waypoints / distance
        waypoints.append(start + (end - start) * fraction)
    waypoints.append(end)
    return waypoints
