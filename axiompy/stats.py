import numpy as np
from .matrix import Matrix

class Statistics:
    @staticmethod
    def covariance_matrix(data: Matrix) -> Matrix:
        return Matrix(np.cov(data._data, rowvar=False))
