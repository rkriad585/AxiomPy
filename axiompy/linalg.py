import numpy as np
from typing import Tuple
from .matrix import Matrix
from .vector import Vector


class LinearAlgebra:
    @staticmethod
    def identity(n: int) -> Matrix:
        return Matrix(np.identity(n))

    @staticmethod
    def zeros(shape: Tuple[int, int]) -> Matrix:
        return Matrix(np.zeros(shape))

    @staticmethod
    def ones(shape: Tuple[int, int]) -> Matrix:
        return Matrix(np.ones(shape))

    @staticmethod
    def solve_linear(A: Matrix, b: Vector) -> Vector:
        return Vector(np.linalg.solve(A._data, b._data))
