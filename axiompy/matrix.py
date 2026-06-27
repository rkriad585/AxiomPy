import numpy as np
from typing import Any, Tuple
from numbers import Number
from ._base import MatrixData
from .vector import Vector

class Matrix:
    def __init__(self, data: MatrixData):
        self._data = np.array(data, dtype=float)

    def __repr__(self) -> str:
        s = [[str(round(e, 4)) for e in row] for row in self._data]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = ' '.join('{{:>{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        return "Matrix(\n  " + "\n  ".join(table) + "\n)"

    def to_list(self) -> MatrixData:
        return self._data.tolist()

    @property
    def shape(self) -> Tuple[int, int]:
        return self._data.shape

    @property
    def T(self) -> 'Matrix':
        return Matrix(self._data.T)

    @property
    def determinant(self) -> float:
        return np.linalg.det(self._data)

    @property
    def inverse(self) -> 'Matrix':
        return Matrix(np.linalg.inv(self._data))

    @property
    def trace(self) -> float:
        return np.trace(self._data)

    @property
    def rank(self) -> int:
        return np.linalg.matrix_rank(self._data)

    def __add__(self, other):
        return Matrix(self._data + other._data)

    def __sub__(self, other):
        return Matrix(self._data - other._data)

    def __mul__(self, other: Number):
        return Matrix(self._data * other)

    def __matmul__(self, other: Any):
        if isinstance(other, Matrix):
            return Matrix(self._data @ other._data)
        if isinstance(other, Vector):
            return Vector(self._data @ other._data)
        return NotImplemented

    def __pow__(self, power: int) -> 'Matrix':
        return Matrix(np.linalg.matrix_power(self._data, power))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return np.array_equal(self._data, other._data)
