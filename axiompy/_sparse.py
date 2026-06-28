"""Sparse matrix - COO/CSR format with arithmetic and matrix-vector product."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Number

import numpy as np

from ._base import AxiomError
from .matrix import Matrix
from .vector import Vector

__all__ = ["SparseMatrix"]


def _ensure_vector(v):
    if isinstance(v, Vector):
        return v
    return Vector(v)


class SparseMatrix:
    """A sparse matrix stored in COO (coordinate) format.

    Supports CSR-converted matrix-vector products, arithmetic with scalars,
    and conversion to/from :class:`Matrix`.

    Args:
        rows: Sequence of row indices.
        cols: Sequence of column indices.
        data: Sequence of non-zero values.
        shape: ``(nrows, ncols)`` of the full matrix.
    """

    def __init__(
        self,
        rows: Sequence[int],
        cols: Sequence[int],
        data: Sequence[Number],
        shape: tuple[int, int],
    ):
        if len(rows) != len(cols) or len(rows) != len(data):
            raise AxiomError("rows, cols, and data must have the same length")
        nrows, ncols = shape
        for r, c in zip(rows, cols):
            if not (0 <= r < nrows and 0 <= c < ncols):
                raise AxiomError(f"Index ({r},{c}) out of bounds for shape {shape}")

        self._rows = np.asarray(rows, dtype=np.int64)
        self._cols = np.asarray(cols, dtype=np.int64)
        self._data = np.asarray(data, dtype=np.float64)
        self._shape = shape
        self._nnz = len(data)
        # CSR cache
        self._csr_rowptr: np.ndarray | None = None
        self._csr_cols: np.ndarray | None = None
        self._csr_data: np.ndarray | None = None

    # ---- properties ----

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(nrows, ncols)``."""
        return self._shape

    @property
    def nnz(self) -> int:
        """Number of stored non-zero entries."""
        return self._nnz

    @property
    def density(self) -> float:
        """Fraction of non-zero entries (``nnz / (rows * cols)``)."""
        return self._nnz / (self._shape[0] * self._shape[1])

    # ---- factories ----

    @classmethod
    def from_dense(cls, mat: Matrix, tol: float = 1e-15) -> SparseMatrix:
        """Convert a dense :class:`Matrix` to a sparse ``SparseMatrix``.

        Entries with absolute value ≤ *tol* are treated as zero.

        Args:
            mat: Dense matrix.
            tol: Zero threshold.

        Returns:
            A new ``SparseMatrix``.
        """
        nrows, ncols = mat._data.shape
        rows, cols, data = [], [], []
        for i in range(nrows):
            for j in range(ncols):
                val = mat._data[i, j]
                if abs(val) > tol:
                    rows.append(i)
                    cols.append(j)
                    data.append(float(val))
        return cls(rows, cols, data, (nrows, ncols))

    @classmethod
    def identity(cls, n: int) -> SparseMatrix:
        """Return an ``n x n`` identity matrix in sparse format.

        Args:
            n: Size of the identity.

        Returns:
            ``SparseMatrix`` identity.
        """
        return cls(range(n), range(n), [1.0] * n, (n, n))

    # ---- conversion ----

    def to_coo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(row_indices, col_indices, data)`` arrays."""
        return self._rows.copy(), self._cols.copy(), self._data.copy()

    def to_dense(self) -> Matrix:
        """Materialise to a dense :class:`Matrix`."""
        out = np.zeros(self._shape, dtype=np.float64)
        out[self._rows, self._cols] = self._data
        return Matrix(out)

    def to_csr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build CSR representation (row_ptr, col_indices, data).

        Returns:
            Tuple ``(row_ptr, col_indices, data)``.
        """
        if self._csr_rowptr is not None:
            return self._csr_rowptr, self._csr_cols, self._csr_data

        nrows = self._shape[0]
        order = np.lexsort((self._cols, self._rows))
        sorted_rows = self._rows[order]
        sorted_cols = self._cols[order]
        sorted_data = self._data[order]

        row_ptr = np.zeros(nrows + 1, dtype=np.int64)
        np.add.at(row_ptr, sorted_rows + 1, 1)
        row_ptr = np.cumsum(row_ptr, dtype=np.int64)

        self._csr_rowptr = row_ptr
        self._csr_cols = sorted_cols
        self._csr_data = sorted_data
        return row_ptr, sorted_cols, sorted_data

    # ---- arithmetic ----

    def __add__(self, other):
        if isinstance(other, SparseMatrix):
            return self._add_sparse(other)
        if isinstance(other, (int, float)):
            return self._add_scalar(other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, SparseMatrix):
            return self._add_sparse(other * (-1))
        if isinstance(other, (int, float)):
            return self._add_scalar(-other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return (self * (-1))._add_scalar(other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            return SparseMatrix(
                self._rows.copy(),
                self._cols.copy(),
                self._data * other,
                self._shape,
            )
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return self._matmul_vector(other)
        if isinstance(other, SparseMatrix):
            return self._matmul_sparse(other)
        if isinstance(other, Matrix):
            return self.to_dense().__matmul__(other)
        return NotImplemented

    def __neg__(self):
        return self * (-1)

    def __repr__(self) -> str:
        return (
            f"SparseMatrix(shape={self._shape}, nnz={self._nnz}, "
            f"density={self.density:.4g})"
        )

    def to_latex(self) -> str:
        """Return a LaTeX matrix string of the dense form."""
        return self.to_dense().to_latex()

    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter notebooks."""
        return f"$${self.to_latex()}$$"

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        return self.to_dense()._repr_html_()

    # ---- internal helpers ----

    def _add_scalar(self, scalar: Number) -> SparseMatrix:
        """Add a scalar to every element (sparsity pattern changes)."""
        dense = self.to_dense()
        dense_data = dense._data + scalar
        return SparseMatrix.from_dense(Matrix(dense_data))

    def _add_sparse(self, other: SparseMatrix) -> SparseMatrix:
        if self._shape != other._shape:
            raise AxiomError("Cannot add sparse matrices with different shapes")
        # Merge COO lists
        rows = np.concatenate([self._rows, other._rows])
        cols = np.concatenate([self._cols, other._cols])
        data = np.concatenate([self._data, other._data])
        # Sum duplicates via bincount-like logic
        lin_idx = rows * self._shape[1] + cols
        order = np.argsort(lin_idx)
        rows, cols, data = rows[order], cols[order], data[order]
        # Sum equal indices
        unique_lin, inv, _counts = np.unique(lin_idx[order], return_inverse=True, return_counts=True)
        out_data = np.zeros(len(unique_lin), dtype=np.float64)
        np.add.at(out_data, inv, data)

        out_rows = unique_lin // self._shape[1]
        out_cols = unique_lin % self._shape[1]
        # Strip explicit zeros
        mask = np.abs(out_data) > 1e-15
        return SparseMatrix(out_rows[mask], out_cols[mask], out_data[mask], self._shape)

    def _matmul_vector(self, vec: Vector) -> Vector:
        _vec = _ensure_vector(vec)
        if _vec._data.shape[0] != self._shape[1]:
            raise AxiomError(
                f"Shape mismatch: sparse {self._shape} @ vector ({_vec._data.shape[0]},)"
            )
        # Use CSR for O(nnz) matvec
        row_ptr, cols, data = self.to_csr()
        nrows = self._shape[0]
        vdata = _vec._data
        result = np.zeros(nrows, dtype=np.float64)
        for i in range(nrows):
            s = 0.0
            for jj in range(row_ptr[i], row_ptr[i + 1]):
                s += data[jj] * vdata[cols[jj]]
            result[i] = s
        return Vector(result)

    def _matmul_sparse(self, other: SparseMatrix) -> SparseMatrix:
        if self._shape[1] != other._shape[0]:
            raise AxiomError(
                f"Shape mismatch: {self._shape} @ {other._shape}"
            )
        a_row_ptr, a_cols, a_data = self.to_csr()
        b_row_ptr, b_cols, b_data = other.to_csr()
        nrows = self._shape[0]
        ncols = other._shape[1]

        out_rows, out_cols, out_data = [], [], []
        for i in range(nrows):
            row_start = a_row_ptr[i]
            row_end = a_row_ptr[i + 1]
            if row_start == row_end:
                continue
            # Accumulate row i * B columns via per-column dot
            accum = {}
            for jj in range(row_start, row_end):
                a_val = a_data[jj]
                k = a_cols[jj]
                # B row k -> sparse * scalar
                bk_start = b_row_ptr[k]
                bk_end = b_row_ptr[k + 1]
                for kk in range(bk_start, bk_end):
                    j = b_cols[kk]
                    b_val = b_data[kk]
                    accum[j] = accum.get(j, 0.0) + a_val * b_val
            for j, val in accum.items():
                if abs(val) > 1e-15:
                    out_rows.append(i)
                    out_cols.append(j)
                    out_data.append(val)
        return SparseMatrix(out_rows, out_cols, out_data, (nrows, ncols))
