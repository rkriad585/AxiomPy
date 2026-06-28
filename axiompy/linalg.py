import logging

import numpy as np

from .matrix import Matrix
from .vector import Vector

logger = logging.getLogger(__name__)


class LinearAlgebra:
    """Linear algebra operations including matrix creation and system solving."""

    @staticmethod
    def identity(n: int) -> Matrix:
        """Create an identity matrix of size n x n.

        Args:
            n (int): The size of the identity matrix.

        Returns:
            Matrix: An n x n identity matrix.
        """
        return Matrix(np.identity(n))

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        """Create a zero matrix of the given shape.

        Args:
            shape (Tuple[int, int]): The (rows, cols) shape of the matrix.

        Returns:
            Matrix: A zero matrix of the specified shape.
        """
        return Matrix(np.zeros(shape))

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        """Create a matrix of ones with the given shape.

        Args:
            shape (Tuple[int, int]): The (rows, cols) shape of the matrix.

        Returns:
            Matrix: A matrix of ones of the specified shape.
        """
        return Matrix(np.ones(shape))

    @staticmethod
    def solve_linear(A: Matrix, b: Vector) -> Vector:
        """Solve the linear system Ax = b.

        Args:
            A (Matrix): The coefficient matrix.
            b (Vector): The right-hand side vector.

        Returns:
            Vector: The solution vector x.
        """
        return Vector(np.linalg.solve(A._data, b._data))

    @staticmethod
    def least_squares(A: Matrix, b: Vector) -> Vector:
        """Solve the least-squares problem min ||Ax - b||₂.

        Args:
            A (Matrix): The coefficient matrix.
            b (Vector): The right-hand side vector.

        Returns:
            Vector: The least-squares solution vector x.
        """
        logger.debug("Least squares %s", A.shape)
        x, *_ = np.linalg.lstsq(A._data, b._data, rcond=None)
        return Vector(x.tolist())

    @staticmethod
    def cross_product_matrix(v: Vector) -> Matrix:
        """Create the skew-symmetric cross-product matrix of a 3-vector.

        For a vector ``v = [x, y, z]``, returns the matrix::

            [[  0, -z,  y],
             [  z,  0, -x],
             [ -y,  x,  0]]

        such that ``cross_product_matrix(v) @ w == cross(v, w)``.

        Args:
            v (Vector): A 3-element vector.

        Returns:
            Matrix: The 3x3 skew-symmetric matrix.
        """
        x, y, z = v.to_list()
        return Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    @staticmethod
    def householder(v: Vector) -> Matrix:
        """Compute the Householder reflector matrix for a given vector.

        Returns the matrix ``H = I - 2 v vᵀ / (vᵀ v)``.

        Args:
            v (Vector): The vector defining the reflection.

        Returns:
            Matrix: The Householder reflection matrix.
        """
        arr = np.array(v.to_list(), dtype=float)
        I = np.eye(arr.shape[0])
        H = I - 2 * np.outer(arr, arr) / (arr @ arr)
        return Matrix(H)
