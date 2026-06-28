import logging
import math
from numbers import Number
from typing import Union

import numpy as np

from ._base import AxiomError, VectorData
from ._config import AxiomConfig

logger = logging.getLogger(__name__)


class Vector:
    """An ``n``-dimensional vector backed by a NumPy array.

    Supports basic arithmetic, dot/cross products, norms, and projections.
    """

    def __init__(self, data: VectorData, dtype=None):
        """Initialize a Vector from a list of numbers.

        Args:
            data: List of numeric coordinates.
            dtype: NumPy dtype (default ``float``).
        """
        self._data = np.array(data, dtype=dtype or float)

    def __len__(self) -> int:
        """Return the number of elements.

        Returns:
            int: Length of the vector.
        """
        return len(self._data)

    def __repr__(self) -> str:
        """Return a pretty-printed string representation.

        Returns:
            str: e.g. ``Vector([1.0, 2.0, 3.0])``.
        """
        prec = AxiomConfig.load().precision
        rounded = [round(float(x), prec) for x in self._data]
        return f"Vector({rounded})"

    def to_latex(self) -> str:
        """Return a LaTeX vector string.

        Returns:
            str: ``\\begin{pmatrix} a & b & ... \\end{pmatrix}``.
        """
        prec = AxiomConfig.load().precision
        entries = " & ".join(str(round(float(x), prec)) for x in self._data)
        return f"\\begin{{pmatrix}}{entries}\\end{{pmatrix}}"

    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter notebooks."""
        return f"$${self.to_latex()}$$"

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        prec = AxiomConfig.load().precision
        cells = "".join(
            f"<td style='text-align:center;padding:2px 8px'>{round(float(x), prec)}</td>"
            for x in self._data
        )
        return (
            "<table style='display:inline-block;border:none;'>"
            f"<tr>{cells}</tr>"
            "</table>"
        )

    def __getitem__(self, key):
        """Access elements by index or slice.

        Args:
            key: Integer index or slice.

        Returns:
            float or np.ndarray: The element(s) at the given key.
        """
        return self._data[key]

    @property
    def dtype(self):
        """NumPy dtype of the underlying array."""
        return self._data.dtype

    def astype(self, dtype):
        """Return a copy cast to a different dtype.

        Args:
            dtype: NumPy dtype or equivalent string (e.g. ``'float32'``).

        Returns:
            Vector: New vector with the requested dtype.
        """
        return Vector(self._data.astype(dtype), dtype=dtype)

    def __neg__(self) -> 'Vector':
        """Negate all elements.

        Returns:
            Vector: The negated vector.
        """
        return Vector(-self._data)

    def __abs__(self) -> float:
        """Compute the Euclidean norm (magnitude).

        Returns:
            float: The magnitude.
        """
        return float(np.linalg.norm(self._data))

    def to_list(self) -> VectorData:
        """Convert the vector to a plain Python list.

        Returns:
            VectorData: List of floats.
        """
        return self._data.tolist()

    def __add__(self, other):
        """Add two vectors element-wise.

        Args:
            other (Vector): The vector to add.

        Returns:
            Vector: Element-wise sum.
        """
        return Vector(self._data + other._data)

    def __sub__(self, other):
        """Subtract two vectors element-wise.

        Args:
            other (Vector): The vector to subtract.

        Returns:
            Vector: Element-wise difference.
        """
        return Vector(self._data - other._data)

    def __mul__(self, other):
        """Scalar multiplication or dot product.

        When ``other`` is a number, returns a scaled vector.
        When ``other`` is a :class:`Vector`, returns the dot product.

        Args:
            other: A number or :class:`Vector`.

        Returns:
            Vector or float: Scaled vector or dot product.
        """
        if isinstance(other, Number):
            return Vector(self._data * other)
        if isinstance(other, Vector):
            return np.dot(self._data, other._data)
        return NotImplemented

    def __matmul__(self, other):
        """Matrix multiplication is not defined for vectors.

        Returns:
            NotImplemented: Always.
        """
        return NotImplemented

    def __rmatmul__(self, other):
        """Reverse matrix multiplication is not defined for vectors.

        Returns:
            NotImplemented: Always.
        """
        return NotImplemented

    def __rmul__(self, other: Number) -> 'Vector':
        """Right-side scalar multiplication.

        Args:
            other (Number): Scalar value.

        Returns:
            Vector: Scaled vector.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Number) -> 'Vector':
        """Divide all elements by a scalar.

        Args:
            other (Number): Divisor.

        Returns:
            Vector: Quotient vector.
        """
        return Vector(self._data / other)

    def __eq__(self, other) -> bool:
        """Check element-wise equality with another Vector.

        Args:
            other (Vector): Vector to compare.

        Returns:
            bool: ``True`` if all elements match.
        """
        if not isinstance(other, Vector):
            return NotImplemented
        return np.array_equal(self._data, other._data)

    def magnitude(self) -> float:
        """Compute the Euclidean norm (same as :meth:`__abs__`).

        Returns:
            float: The magnitude.
        """
        return np.linalg.norm(self._data)

    def norm(self, ord: Union[int, float, str] = 2) -> float:
        """Compute the vector norm of a given order.

        Args:
            ord: Order of the norm (default 2 for Euclidean).

        Returns:
            float: The computed norm.
        """
        return float(np.linalg.norm(self._data, ord=ord))

    def normalize(self) -> 'Vector':
        """Return a unit vector in the same direction.

        Returns:
            Vector: Unit vector (zero vector if magnitude is zero).
        """
        mag = self.magnitude()
        if mag == 0:
            return Vector(np.zeros_like(self._data))
        return Vector(self._data / mag)

    def angle_between(self, other: 'Vector', in_degrees: bool = False) -> float:
        """Compute the angle between this vector and another.

        Args:
            other (Vector): The other vector.
            in_degrees (bool): Return angle in degrees (default radians).

        Returns:
            float: The angle.
        """
        dot_product = self * other
        magnitudes = self.magnitude() * other.magnitude()
        if magnitudes == 0:
            return 0.0
        cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad) if in_degrees else angle_rad

    def cross(self, other: 'Vector') -> 'Vector':
        """Compute the cross product (3D vectors only).

        Args:
            other (Vector): The other 3D vector.

        Returns:
            Vector: The cross product.

        Raises:
            AxiomError: If either vector is not 3-dimensional.
        """
        if len(self) != 3 or len(other) != 3:
            raise AxiomError("Cross product is only defined for 3D vectors.")
        return Vector(np.cross(self._data, other._data))

    def dot(self, other: 'Vector') -> float:
        """Compute the dot product with another vector.

        Args:
            other (Vector): The other vector.

        Returns:
            float: The dot product.
        """
        return float(np.dot(self._data, other._data))

    def project_onto(self, other: 'Vector') -> 'Vector':
        """Project this vector onto another.

        Args:
            other (Vector): The vector to project onto.

        Returns:
            Vector: The projection (zero vector if ``other`` is zero).
        """
        denom = other.dot(other)
        if denom == 0:
            return Vector(np.zeros_like(self._data))
        return other * (self.dot(other) / denom)

    @staticmethod
    def dot_static(a: 'Vector', b: 'Vector') -> float:
        """Compute the dot product of two vectors (static version).

        Args:
            a (Vector): First vector.
            b (Vector): Second vector.

        Returns:
            float: The dot product.
        """
        return float(np.dot(a._data, b._data))

    @staticmethod
    def cross_static(a: 'Vector', b: 'Vector') -> 'Vector':
        """Compute the cross product of two 3D vectors (static version).

        Args:
            a (Vector): First 3D vector.
            b (Vector): Second 3D vector.

        Returns:
            Vector: The cross product.

        Raises:
            AxiomError: If either vector is not 3-dimensional.
        """
        if len(a) != 3 or len(b) != 3:
            raise AxiomError("Cross product is only defined for 3D vectors.")
        return Vector(np.cross(a._data, b._data))
