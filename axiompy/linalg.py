import numpy as np
from typing import Tuple
from .matrix import Matrix

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
