from __future__ import annotations

import logging
from numbers import Number
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._mmap import MmapArray
    from ._sparse import SparseMatrix

import numpy as np

from ._base import MatrixData
from ._config import AxiomConfig
from .vector import Vector

logger = logging.getLogger(__name__)


class Matrix:
    """An ``m x n`` matrix backed by a NumPy array.

    Supports basic arithmetic, matrix multiplication (``@``),
    decompositions (LU, QR, Cholesky), and properties such as
    :attr:`determinant`, :attr:`inverse`, :attr:`trace`, and :attr:`rank`.
    """

    def __init__(self, data: MatrixData, dtype=None):
        """Initialize a Matrix from a 2-D list of numbers.

        Args:
            data: List of rows, each a list of numeric values.
            dtype: NumPy dtype (default ``float``).
        """
        self._data = np.array(data, dtype=dtype or float)

    def __repr__(self) -> str:
        """Return a pretty-printed string representation.

        Returns:
            str: Formatted matrix with aligned columns.
        """
        prec = AxiomConfig.load().precision
        s = [[str(round(e, prec)) for e in row] for row in self._data]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = ' '.join(f'{{:>{x}}}' for x in lens)
        table = [fmt.format(*row) for row in s]
        return "Matrix(\n  " + "\n  ".join(table) + "\n)"

    def to_list(self) -> MatrixData:
        """Convert the matrix to a plain Python list of lists.

        Returns:
            MatrixData: Nested list of floats.
        """
        return self._data.tolist()

    @property
    def dtype(self):
        """NumPy dtype of the underlying array."""
        return self._data.dtype

    def astype(self, dtype):
        """Return a copy cast to a different dtype.

        Args:
            dtype: NumPy dtype or equivalent string (e.g. ``'float32'``).

        Returns:
            Matrix: New matrix with the requested dtype.
        """
        return Matrix(self._data.astype(dtype), dtype=dtype)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the matrix dimensions.

        Returns:
            Tuple[int, int]: ``(rows, columns)``.
        """
        return self._data.shape

    @property
    def T(self) -> Matrix:
        """Return the transpose of the matrix.

        Returns:
            Matrix: The transposed matrix.
        """
        return Matrix(self._data.T)

    @property
    def determinant(self) -> float:
        """Compute the determinant of the matrix.

        Returns:
            float: The determinant.
        """
        return np.linalg.det(self._data)

    @property
    def inverse(self) -> Matrix:
        """Compute the multiplicative inverse of the matrix.

        Returns:
            Matrix: The inverse matrix.
        """
        return Matrix(np.linalg.inv(self._data))

    @property
    def trace(self) -> float:
        """Compute the trace (sum of diagonal elements).

        Returns:
            float: The trace.
        """
        return np.trace(self._data)

    @property
    def rank(self) -> int:
        """Compute the rank of the matrix.

        Returns:
            int: The rank.
        """
        return np.linalg.matrix_rank(self._data)

    def norm(self, ord: Any = 'fro') -> float:
        """Compute the matrix norm.

        Args:
            ord: Order of the norm (default ``'fro'`` for Frobenius).

        Returns:
            float: The computed norm.
        """
        return float(np.linalg.norm(self._data, ord=ord))

    def __add__(self, other):
        """Add two matrices element-wise.

        Args:
            other (Matrix): The matrix to add.

        Returns:
            Matrix: Element-wise sum.
        """
        return Matrix(self._data + other._data)

    def __sub__(self, other):
        """Subtract two matrices element-wise.

        Args:
            other (Matrix): The matrix to subtract.

        Returns:
            Matrix: Element-wise difference.
        """
        return Matrix(self._data - other._data)

    def __mul__(self, other: Number):
        """Scalar multiplication.

        Args:
            other (Number): Scalar value.

        Returns:
            Matrix: Scaled matrix.
        """
        return Matrix(self._data * other)

    def __matmul__(self, other: Any):
        """Matrix-matrix or matrix-vector multiplication.

        Args:
            other: :class:`Matrix` or :class:`Vector`.

        Returns:
            Matrix or Vector: The product.
        """
        if isinstance(other, Matrix):
            return Matrix(self._data @ other._data)
        if isinstance(other, Vector):
            return Vector(self._data @ other._data)
        return NotImplemented

    def __rmatmul__(self, other):
        """Reverse matrix multiplication is not defined for scalars.

        Returns:
            NotImplemented: Always.
        """
        return NotImplemented

    def __pow__(self, power: int) -> Matrix:
        """Raise the matrix to an integer power.

        Args:
            power: Integer exponent.

        Returns:
            Matrix: ``self`` raised to ``power``.
        """
        return Matrix(np.linalg.matrix_power(self._data, power))

    def __eq__(self, other) -> bool:
        """Check element-wise equality with another Matrix.

        Args:
            other (Matrix): Matrix to compare.

        Returns:
            bool: ``True`` if all elements match.
        """
        if not isinstance(other, Matrix):
            return NotImplemented
        return np.array_equal(self._data, other._data)

    def lu_decompose(self) -> tuple[Matrix, Matrix, Matrix]:
        """Compute the LU decomposition with partial pivoting.

        Returns:
            Tuple[Matrix, Matrix, Matrix]: ``(P, L, U)`` where
            ``P @ A = L @ U``.
        """
        logger.debug("LU decomposition of %s", self.shape)
        P, L, U = scipy_lu(self._data)
        return Matrix(P), Matrix(L), Matrix(U)

    def qr_decompose(self) -> tuple[Matrix, Matrix]:
        """Compute the QR decomposition.

        Returns:
            Tuple[Matrix, Matrix]: ``(Q, R)`` where ``A = Q @ R``.
        """
        logger.debug("QR decomposition of %s", self.shape)
        Q, R = np.linalg.qr(self._data)
        return Matrix(Q), Matrix(R)

    def cholesky_decompose(self) -> Matrix:
        """Compute the Cholesky decomposition.

        The matrix must be symmetric positive-definite.

        Returns:
            Matrix: Lower triangular factor ``L`` such that ``A = L @ L.T``.
        """
        logger.debug("Cholesky decomposition of %s", self.shape)
        return Matrix(np.linalg.cholesky(self._data))

    def svd_decompose(self) -> tuple[Matrix, Vector, Matrix]:
        """Compute the singular value decomposition.

        Returns:
            Tuple[Matrix, Vector, Matrix]: ``(U, S, Vt)`` where
            ``A = U @ diag(S) @ Vt``.
        """
        logger.debug("SVD of %s", self.shape)
        U, S, Vt = np.linalg.svd(self._data)
        return Matrix(U), Vector(S.tolist()), Matrix(Vt)

    def eigenvalues(self) -> Vector:
        """Compute the eigenvalues of a symmetric matrix.

        Returns:
            Vector: Eigenvalues in ascending order.
        """
        logger.debug("Eigenvalues of %s", self.shape)
        return Vector(np.linalg.eigvalsh(self._data).tolist())

    def eigenvectors(self) -> Matrix:
        """Compute the eigenvectors of a symmetric matrix.

        Returns:
            Matrix: Column eigenvectors corresponding to :meth:`eigenvalues`,
            arranged in ascending order.
        """
        logger.debug("Eigenvectors of %s", self.shape)
        _, V = np.linalg.eigh(self._data)
        return Matrix(V)

    def condition_number(self, p: Any = 2) -> float:
        """Compute the condition number of the matrix.

        Args:
            p: Norm order for the condition number (default ``2`` uses SVD).

        Returns:
            float: The condition number.
        """
        return float(np.linalg.cond(self._data, p=p))

    def pinv(self) -> Matrix:
        """Compute the Moore-Penrose pseudoinverse.

        Returns:
            Matrix: The pseudoinverse.
        """
        logger.debug("Pseudoinverse of %s", self.shape)
        return Matrix(np.linalg.pinv(self._data))

    def to_sparse(self, tol: float = 1e-15) -> SparseMatrix:
        """Convert to a sparse ``SparseMatrix``.

        Args:
            tol: Zero threshold (entries with absolute value ≤ tol are omitted).

        Returns:
            ``SparseMatrix`` in COO format.
        """
        from ._sparse import SparseMatrix

        return SparseMatrix.from_dense(self, tol=tol)

    @classmethod
    def from_mmap(cls, mmap: MmapArray) -> Matrix:
        """Load a :class:`Matrix` from a memory-mapped array.

        Args:
            mmap: An open ``MmapArray`` instance.

        Returns:
            Matrix: Dense matrix loaded into memory.
        """
        mmap.open()
        return cls(mmap._mmap.copy())


def scipy_lu(A):
    """Perform LU decomposition with partial pivoting (no SciPy dependency).

    Args:
        A: A square NumPy array.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(P, L, U)`` where
        ``P @ A = L @ U``.
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(float)
    P = np.eye(n)
    for k in range(n - 1):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    return P, L, U
