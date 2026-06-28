"""Memory-mapped out-of-core array operations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from .matrix import Matrix
from .vector import Vector

__all__ = ["MmapArray", "open_mmap"]


class MmapArray:
    """Memory-mapped array with chunked operations for out-of-core workloads.

    Wraps ``numpy.memmap`` and provides chunked arithmetic and matmul
    so that large arrays never need to reside fully in RAM.

    Args:
        path: File path for the memory-mapped array.
        shape: Shape of the array.
        dtype: NumPy dtype (default ``'float64'``).
        mode: File mode — ``'r'``, ``'r+'``, ``'w+'``, or ``'c'``.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        shape: tuple[int, int],
        dtype: str = "float64",
        mode: str = "r+",
    ):
        self._path = Path(path)
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._mode = mode
        self._mmap: np.memmap | None = None
        self._chunk_size = 1024  # rows per chunk

    # ---- context manager ----

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ---- open / close ----

    def open(self):
        """Open (or create) the memory-mapped file."""
        if self._mmap is not None:
            return
        exists = self._path.exists()
        if not exists and self._mode in ("r",):
            raise FileNotFoundError(f"{self._path} not found for read mode")
        if not exists:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._mmap = np.memmap(
            self._path,
            dtype=self._dtype,
            mode=self._mode,
            shape=self._shape,
        )

    def close(self):
        """Flush and close the memory-mapped file."""
        if self._mmap is not None:
            self._mmap.flush()
            del self._mmap
            self._mmap = None

    def flush(self):
        """Force a write to disk."""
        if self._mmap is not None:
            self._mmap.flush()

    # ---- properties ----

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(nrows, ncols)``."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the array."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions (always 2)."""
        return 2

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._shape[0] * self._shape[1]

    @property
    def path(self) -> Path:
        """Path to the backing file."""
        return self._path

    # ---- factories ----

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        path: str | os.PathLike | None = None,
    ) -> MmapArray:
        """Create a memory-mapped array from an existing NumPy array.

        Args:
            arr: Source array.
            path: Destination path (auto-generated temp file if omitted).

        Returns:
            ``MmapArray`` backed by a file on disk.
        """
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".mmap", prefix="axiom_")
            os.close(fd)
            os.unlink(path)
        mm = cls(path, arr.shape, arr.dtype, mode="w+")
        mm.open()
        mm._mmap[:] = arr[:]
        mm.flush()
        return mm

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int],
        dtype: str = "float64",
        path: str | os.PathLike | None = None,
    ) -> MmapArray:
        """Create a zero-initialized memory-mapped array.

        Args:
            shape: ``(nrows, ncols)``.
            dtype: NumPy dtype.
            path: Destination path (auto-generated if omitted).

        Returns:
            ``MmapArray`` filled with zeros.
        """
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".mmap", prefix="axiom_")
            os.close(fd)
            os.unlink(path)
        mm = cls(path, shape, dtype, mode="w+")
        mm.open()
        return mm

    # ---- chunked read ----

    def read_rows(self, start: int, end: int) -> np.ndarray:
        """Read a slice of rows into memory.

        Args:
            start: Start row index.
            end: End row index (exclusive).

        Returns:
            NumPy array of shape ``(end - start, ncols)``.
        """
        self._ensure_open()
        if end > self._shape[0]:
            end = self._shape[0]
        return self._mmap[start:end].copy()

    def __getitem__(self, key):
        self._ensure_open()
        return self._mmap[key]

    def __setitem__(self, key, value):
        self._ensure_open()
        self._mmap[key] = value

    # ---- chunked operations ----

    def matmul(self, other: np.ndarray | Vector | Matrix) -> Vector | Matrix:
        """Chunked matrix-vector or matrix-matrix product.

        Args:
            other: Vector, Matrix, or ndarray.

        Returns:
            ``Vector`` for 1-D inputs, ``Matrix`` for 2-D.
        """
        self._ensure_open()
        if isinstance(other, (Vector, Matrix)):
            other_arr = other._data
        else:
            other_arr = np.asarray(other, dtype=self._dtype)

        if other_arr.ndim == 1:
            ncols = other_arr.shape[0]
            out = np.zeros(self._shape[0], dtype=self._dtype)
            for start in range(0, self._shape[0], self._chunk_size):
                end = min(start + self._chunk_size, self._shape[0])
                chunk = self._mmap[start:end]
                out[start:end] = chunk @ other_arr
            return Vector(out)
        else:
            ncols = other_arr.shape[1]
            out = np.zeros((self._shape[0], ncols), dtype=self._dtype)
            for start in range(0, self._shape[0], self._chunk_size):
                end = min(start + self._chunk_size, self._shape[0])
                chunk = self._mmap[start:end]
                out[start:end] = chunk @ other_arr
            return Matrix(out)

    def add(self, other: float | np.ndarray) -> MmapArray:
        """Element-wise addition (in-place on the mmap file).

        Args:
            other: Scalar or array of matching shape.

        Returns:
            Self (modified in-place).
        """
        self._ensure_open()
        if np.isscalar(other):
            for start in range(0, self._shape[0], self._chunk_size):
                end = min(start + self._chunk_size, self._shape[0])
                self._mmap[start:end] += other
        else:
            other_arr = np.asarray(other, dtype=self._dtype)
            for start in range(0, self._shape[0], self._chunk_size):
                end = min(start + self._chunk_size, self._shape[0])
                self._mmap[start:end] += other_arr[start:end]
        self.flush()
        return self

    def mul(self, other: float) -> MmapArray:
        """Element-wise scalar multiplication (in-place).

        Args:
            other: Scalar multiplier.

        Returns:
            Self (modified in-place).
        """
        self._ensure_open()
        for start in range(0, self._shape[0], self._chunk_size):
            end = min(start + self._chunk_size, self._shape[0])
            self._mmap[start:end] *= other
        self.flush()
        return self

    def to_dense(self) -> Matrix:
        """Load the entire array into memory as a :class:`Matrix`.

        Warning: this defeats the purpose of memory-mapping for large arrays.
        """
        self._ensure_open()
        return Matrix(self._mmap.copy())

    def __repr__(self) -> str:
        return (
            f"MmapArray(shape={self._shape}, dtype={self._dtype.name}, "
            f"path={self._path.name})"
        )

    # ---- helpers ----

    def _ensure_open(self):
        if self._mmap is None:
            raise RuntimeError("MmapArray is not open. Use the context manager or call open().")


def open_mmap(
    path: str | os.PathLike,
    shape: tuple[int, int] | None = None,
    dtype: str = "float64",
    mode: str = "r",
) -> MmapArray:
    """Convenience function to open an existing memory-mapped array.

    Args:
        path: File path.
        shape: Shape (required for ``'w+'`` / ``'r+'`` creation).
        dtype: NumPy dtype.
        mode: File mode (``'r'``, ``'r+'``, ``'w+'``, ``'c'``).

    Returns:
        ``MmapArray`` instance.
    """
    if shape is None and mode in ("w+",):
        raise ValueError("shape is required when creating a new file (mode 'w+')")
    actual_shape = shape or (0, 0)
    return MmapArray(path, actual_shape, dtype, mode)
