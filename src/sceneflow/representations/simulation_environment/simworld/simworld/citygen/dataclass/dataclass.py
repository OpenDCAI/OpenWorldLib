"""Module for data classes defining various data structures for city generation."""
import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class Point:
    """A point in a 2D plane."""
    x: float
    y: float

    def __hash__(self):
        """Return the hash value of the point."""
        return hash((self.x, self.y))

    def to_dict(self):
        """Convert the point to dictionary representation."""
        return {
            'x': self.x,
            'y': self.y
        }


@dataclass
class MetaInfo:
    """Metadata for road segments."""
    highway: bool = False
    t: float = 0.0


@dataclass(frozen=True, eq=True)
class Bounds:
    """A bounding box with x, y, width, height, and rotation.

    (x, y) is the bottom-left corner of the bounding box
    width: width of the bounding box
    height: height of the bounding box
    """
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0

    def __hash__(self):
        """Return the hash value of the bounds."""
        return hash((self.x, self.y, self.width, self.height))

    def to_dict(self):
        """Convert the bounds to dictionary representation."""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'rotation': self.rotation
        }

    def intersects(self, other: 'Bounds') -> bool:
        """Checks if two Bounds objects' bounding boxes intersect."""
        return not (self.x + self.width < other.x or
                    self.x > other.x + other.width or
                    self.y + self.height < other.y or
                    self.y > other.y + other.height)


@dataclass
class Segment:
    """A road segment connecting two points."""
    start: Point
    end: Point
    q: MetaInfo = field(default_factory=MetaInfo)
    bounds: Bounds = Bounds(0, 0, 0, 0)

    def get_angle(self) -> float:
        """Calculate the angle of the segment in degrees."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return math.degrees(math.atan2(dy, dx))

    def to_dict(self):
        """Convert the segment to dictionary representation."""
        return {
            'start': self.start.to_dict(),
            'end': self.end.to_dict()
        }

    def __post_init__(self):
        """Calculates segment length, sets width, creates Bounds object after Segment initialization."""
        width = 50
        length = ((self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2) ** 0.5
        object.__setattr__(self, 'bounds', Bounds(
            self.start.x - width / 2,
            self.start.y - length / 2,
            width,
            length,
            self.get_angle()
        ))


@dataclass
class Intersection:
    """A road intersection."""
    point: Point
    segments: List[Segment]


@dataclass
class Route:
    """A route consisting of multiple points."""
    points: List[Point]
    start: Point
    end: Point
    bounds: Bounds = Bounds(0, 0, 0, 0)

    def __hash__(self):
        """Return the hash value of the route."""
        return hash((self.start, self.end))

    def __post_init__(self):
        """Sets the bounds attribute of centered around the start point."""
        object.__setattr__(self, 'bounds', Bounds(
            self.start.x - 1,
            self.start.y - 1,
            2,
            2,
        ))


# Building types
@dataclass(frozen=True, eq=True)
class BuildingType:
    """A building type."""
    name: str
    width: float
    height: float
    num_limit: int = -1

    def __hash__(self):
        """Return the hash value of the building type."""
        return hash(
            (self.name, self.width, self.height, self.num_limit)
        )

    def to_dict(self):
        """Convert the building type to dictionary representation."""
        return {
            'name': self.name,
            'width': self.width,
            'height': self.height,
            'num_limit': self.num_limit
        }


@dataclass(frozen=True, eq=True)
class Building:
    """A building."""
    building_type: BuildingType
    bounds: Bounds
    rotation: float = 0.0  # Building rotation angle (aligned with road)
    center: Point = field(init=False)  # Center point of the building
    width: float = field(init=False)
    height: float = field(init=False)

    def __post_init__(self):
        """Calculate center point after initialization."""
        # Use object.__setattr__ to bypass the frozen restriction
        object.__setattr__(
            self,
            'center',
            Point(
                self.bounds.x + self.bounds.width / 2,
                self.bounds.y + self.bounds.height / 2
            )
        )
        object.__setattr__(self, 'width', self.bounds.width)
        object.__setattr__(self, 'height', self.bounds.height)
        object.__setattr__(self, 'rotation', self.bounds.rotation)

    def __hash__(self):
        """Make Building hashable."""
        return hash(
            (
                self.building_type.name,
                self.bounds.x,
                self.bounds.y,
                self.bounds.width,
                self.bounds.height,
                self.rotation,
            )
        )

    def to_dict(self):
        """Convert the building to dictionary representation."""
        return {
            'building_type': self.building_type.to_dict(),
            'bounds': self.bounds.to_dict(),
            'rotation': self.rotation,
            'center': self.center.to_dict()
        }


@dataclass(frozen=True, eq=True)
class ElementType:
    """An element type."""
    name: str
    width: float
    height: float

    def __hash__(self):
        """Return the hash value of the element type."""
        return hash((self.name, self.width, self.height))

    def to_dict(self):
        """Convert the element type to dictionary representation."""
        return {
            'name': self.name,
            'width': self.width,
            'height': self.height
        }


@dataclass(frozen=True, eq=True)
class Element:
    """An element is a small object that can be placed in the city."""
    element_type: ElementType
    bounds: Bounds
    rotation: float = 0.0
    center: Point = field(init=False)
    building: Building = None

    def __post_init__(self):
        """Calculate center point after initialization."""
        # Use object.__setattr__ to bypass the frozen restriction
        object.__setattr__(
            self,
            'center',
            Point(
                self.bounds.x + self.bounds.width / 2,
                self.bounds.y + self.bounds.height / 2
            )
        )

    def __hash__(self):
        """Return the hash value of the element."""
        return hash((self.element_type.name, self.bounds, self.rotation, self.center))

    def to_dict(self):
        """Convert the element to dictionary representation."""
        return {
            'element_type': self.element_type.to_dict(),
            'bounds': self.bounds.to_dict(),
            'rotation': self.rotation,
            'center': self.center.to_dict()
        }
