import numpy as np
import math
from numbers import Number
from ._base import AxiomError, VectorData

class Vector:
    def __init__(self, data: VectorData):
        self._data = np.array(data, dtype=float)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Vector({self._data.tolist()})"

    def __getitem__(self, key):
        return self._data[key]

    def to_list(self) -> VectorData:
        return self._data.tolist()

    def __add__(self, other):
        return Vector(self._data + other._data)

    def __sub__(self, other):
        return Vector(self._data - other._data)

    def __mul__(self, other):
        if isinstance(other, Number):
            return Vector(self._data * other)
        if isinstance(other, Vector):
            return np.dot(self._data, other._data)
        return NotImplemented

    def __matmul__(self, other):
        return NotImplemented

    def __rmul__(self, other: Number) -> 'Vector':
        return self.__mul__(other)

    def __truediv__(self, other: Number) -> 'Vector':
        return Vector(self._data / other)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return np.array_equal(self._data, other._data)

    def magnitude(self) -> float:
        return np.linalg.norm(self._data)

    def normalize(self) -> 'Vector':
        mag = self.magnitude()
        if mag == 0:
            return Vector(np.zeros_like(self._data))
        return Vector(self._data / mag)

    def angle_between(self, other: 'Vector', in_degrees: bool = False) -> float:
        dot_product = self * other
        magnitudes = self.magnitude() * other.magnitude()
        if magnitudes == 0:
            return 0.0
        cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad) if in_degrees else angle_rad

    def cross(self, other: 'Vector') -> 'Vector':
        if len(self) != 3 or len(other) != 3:
            raise AxiomError("Cross product is only defined for 3D vectors.")
        return Vector(np.cross(self._data, other._data))
