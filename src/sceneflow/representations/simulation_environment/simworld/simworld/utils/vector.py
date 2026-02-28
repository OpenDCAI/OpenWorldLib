"""Two-dimensional vector utilities module, providing Vector class and related operations."""
from dataclasses import dataclass


@dataclass
class Vector:
    """Two-dimensional vector class.

    Used for representing and manipulating 2D vectors, providing basic vector operations.

    Attributes:
        x: X coordinate.
        y: Y coordinate.
    """

    x: float
    y: float

    def __init__(self, x, y=None):
        """Initialize the vector.

        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        if y is None and isinstance(x, (list, tuple)):
            # Handle list/tuple input like [1, 1] or (1, 1)
            self.x = float(x[0])
            self.y = float(x[1])
        elif y is None and isinstance(x, str):
            # Handle string input
            # Remove all whitespace and unnecessary characters
            clean_str = x.replace(' ', '').strip('()[]{}')
            # Split by comma or other common separators
            coords = clean_str.split(',')
            if len(coords) == 2:
                self.x = float(coords[0])
                self.y = float(coords[1])
            else:
                raise ValueError(f'Invalid vector string format: {x}')
        elif y is None and isinstance(x, dict):
            # Handle dictionary input
            self.x = float(x.get('x', x.get(0, 0)))
            self.y = float(x.get('y', x.get(1, 0)))
        else:
            # Handle traditional x, y input
            self.x = float(x)
            self.y = float(y) if y is not None else 0.0

        # Round values as per original implementation
        self.x = round(self.x, 4)
        self.y = round(self.y, 4)

    def __post_init__(self):
        """Post-initialization processing, rounds coordinates."""
        self.x = round(self.x, 4)
        self.y = round(self.y, 4)

    def normalize(self) -> 'Vector':
        """Normalize the vector.

        Returns:
            Normalized vector.
        """
        magnitude = (self.x ** 2 + self.y ** 2) ** 0.5
        if magnitude == 0:
            return Vector(0, 0)
        return Vector(round(self.x / magnitude, 4), round(self.y / magnitude, 4))

    def __add__(self, other: 'Vector') -> 'Vector':
        """Vector addition.

        Args:
            other: Another vector.

        Returns:
            Sum of the two vectors.
        """
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector') -> 'Vector':
        """Vector subtraction.

        Args:
            other: Another vector.

        Returns:
            Difference of the two vectors.
        """
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float) -> 'Vector':
        """Vector multiplication by a scalar.

        Args:
            other: Scalar value.

        Returns:
            Product of vector and scalar.
        """
        return Vector(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> 'Vector':
        """Vector division by a scalar.

        Args:
            other: Scalar value.

        Returns:
            Quotient of vector and scalar.
        """
        return Vector(self.x / other, self.y / other)

    def distance(self, other: 'Vector') -> float:
        """Calculate distance to another vector.

        Args:
            other: Another vector.

        Returns:
            Euclidean distance between the two vectors.
        """
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def __eq__(self, other: 'Vector') -> bool:
        """Check if two vectors are equal.

        Uses tolerance comparison, allowing small errors.

        Args:
            other: Another vector.

        Returns:
            True if vectors are approximately equal, False otherwise.
        """
        # Compare two vectors with a tolerance
        return abs(self.x - other.x) < 1e-3 and abs(self.y - other.y) < 1e-3

    def __hash__(self) -> int:
        """Calculate hash value of the vector.

        Returns:
            Hash value of the vector.
        """
        return hash((self.x, self.y))

    def dot(self, other: 'Vector') -> float:
        """Calculate dot product with another vector.

        Args:
            other: Another vector.

        Returns:
            Dot product of the two vectors.
        """
        return round(self.x * other.x + self.y * other.y, 4)

    def cross(self, other: 'Vector') -> float:
        """Calculate cross product with another vector.

        Args:
            other: Another vector.

        Returns:
            Cross product of the two vectors.
        """
        return round(self.x * other.y - self.y * other.x, 4)

    def length(self) -> float:
        """Calculate length of the vector.

        Returns:
            Length of the vector.
        """
        return round(((self.x ** 2 + self.y ** 2) ** 0.5), 4)
