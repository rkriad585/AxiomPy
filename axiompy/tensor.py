import logging
import operator
from typing import Callable, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

Shape = tuple[int, ...]
ArrayLike = Union[list, np.ndarray]


class Tensor:
    """``n``-dimensional array wrapper backed by NumPy.

    Supports indexing, slicing, broadcasting arithmetic, tensor contraction,
    outer product, and Kronecker product.
    """

    def __init__(self, data: ArrayLike, dtype: type = float):
        if isinstance(data, np.ndarray):
            self._data = data.astype(dtype)
        else:
            self._data = np.array(data, dtype=dtype)

    # ---- shape / dim helpers -----------------------------------------------

    @property
    def shape(self) -> Shape:
        """Return the tensor shape."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self._data.ndim

    @property
    def size(self) -> int:
        """Return the total number of elements."""
        return self._data.size

    def reshape(self, *shape: int) -> 'Tensor':
        """Return a reshaped view (data is shared).

        Args:
            *shape: New shape compatible with current size.

        Returns:
            Tensor: Reshaped tensor.
        """
        return Tensor(self._data.reshape(*shape))

    def flatten(self) -> 'Tensor':
        """Return a 1-D copy of the data.

        Returns:
            Tensor: 1-dimensional tensor.
        """
        return Tensor(self._data.flatten())

    # ---- indexing / iteration ----------------------------------------------

    def __getitem__(self, key) -> 'Tensor':
        return Tensor(self._data[key])

    def __setitem__(self, key, value) -> None:
        if isinstance(value, Tensor):
            self._data[key] = value._data
        else:
            self._data[key] = value

    def __len__(self) -> int:
        if self.ndim == 0:
            return 1
        return self.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield Tensor(self._data[i])

    def to_list(self) -> list:
        """Convert to a plain Python list."""
        return self._data.tolist()

    # ---- arithmetic --------------------------------------------------------

    def __neg__(self) -> 'Tensor':
        return Tensor(-self._data)

    def __pos__(self) -> 'Tensor':
        return Tensor(+self._data)

    def __abs__(self) -> 'Tensor':
        return Tensor(np.abs(self._data))

    def _apply(self, other, fn: Callable) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(fn(self._data, other._data))
        if isinstance(other, (int, float, complex, np.number)):
            return Tensor(fn(self._data, other))
        return NotImplemented

    def __add__(self, other) -> 'Tensor':
        return self._apply(other, operator.add)

    def __radd__(self, other) -> 'Tensor':
        return Tensor(operator.add(other, self._data))

    def __sub__(self, other) -> 'Tensor':
        return self._apply(other, operator.sub)

    def __rsub__(self, other) -> 'Tensor':
        return Tensor(operator.sub(other, self._data))

    def __mul__(self, other) -> 'Tensor':
        return self._apply(other, operator.mul)

    def __rmul__(self, other) -> 'Tensor':
        return Tensor(operator.mul(other, self._data))

    def __truediv__(self, other) -> 'Tensor':
        return self._apply(other, operator.truediv)

    def __rtruediv__(self, other) -> 'Tensor':
        return Tensor(operator.truediv(other, self._data))

    def __pow__(self, other) -> 'Tensor':
        return self._apply(other, operator.pow)

    # ---- comparison --------------------------------------------------------

    def __eq__(self, other) -> bool:
        if isinstance(other, Tensor):
            return bool(np.array_equal(self._data, other._data))
        if isinstance(other, (int, float, complex, np.number)):
            return bool(np.all(self._data == other))
        return NotImplemented

    def __ne__(self, other) -> bool:
        if isinstance(other, Tensor):
            return not bool(np.array_equal(self._data, other._data))
        if isinstance(other, (int, float, complex, np.number)):
            return bool(np.any(self._data != other))
        return NotImplemented

    def allclose(self, other: 'Tensor', rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check element-wise approximate equality.

        Args:
            other: Tensor to compare against.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            True if all elements are within tolerance.
        """
        if isinstance(other, Tensor):
            return bool(np.allclose(self._data, other._data, rtol=rtol, atol=atol))
        return False

    # ---- reductions --------------------------------------------------------

    def sum(self, axis: Optional[int] = None) -> 'Tensor':
        return Tensor(np.sum(self._data, axis=axis))

    def mean(self, axis: Optional[int] = None) -> 'Tensor':
        return Tensor(np.mean(self._data, axis=axis))

    def max(self, axis: Optional[int] = None) -> 'Tensor':
        return Tensor(np.max(self._data, axis=axis))

    def min(self, axis: Optional[int] = None) -> 'Tensor':
        return Tensor(np.min(self._data, axis=axis))

    def std(self, axis: Optional[int] = None) -> 'Tensor':
        return Tensor(np.std(self._data, axis=axis))

    # ---- linear algebra helpers --------------------------------------------

    def dot(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication (last axis vs second-last)."""
        return Tensor(np.dot(self._data, other._data))

    def transpose(self, *axes: int) -> 'Tensor':
        if axes:
            return Tensor(np.transpose(self._data, axes))
        return Tensor(np.transpose(self._data))

    @property
    def T(self) -> 'Tensor':
        return Tensor(np.transpose(self._data))

    # ---- tensor-specific operations ----------------------------------------

    @staticmethod
    def contract(a: 'Tensor', b: 'Tensor', axes: Union[int, tuple, list]) -> 'Tensor':
        """Tensor contraction — generalized dot product over specified axes.

        Args:
            a: First tensor.
            b: Second tensor.
            axes: If int, sum over last ``axes`` axes of ``a`` and first
                  ``axes`` of ``b``.  If tuple of two sequences, sum over
                  ``axes[0]`` of ``a`` and ``axes[1]`` of ``b``.

        Returns:
            Tensor: Contraction result.
        """
        return Tensor(np.tensordot(a._data, b._data, axes=axes))

    @staticmethod
    def outer(a: 'Tensor', b: 'Tensor') -> 'Tensor':
        """Outer (tensor) product of two tensors.

        Every element of ``a`` is multiplied by every element of ``b``.
        The result has shape ``a.shape + b.shape``.

        Args:
            a: First tensor.
            b: Second tensor.

        Returns:
            Tensor: Outer product.
        """
        return Tensor(np.multiply.outer(a._data, b._data))

    @staticmethod
    def kronecker(a: 'Tensor', b: 'Tensor') -> 'Tensor':
        """Kronecker product — block-wise multiplication.

        Args:
            a: First tensor.
            b: Second tensor.

        Returns:
            Tensor: Kronecker product.
        """
        return Tensor(np.kron(a._data, b._data))

    @staticmethod
    def einsum(subscript: str, *operands: 'Tensor') -> 'Tensor':
        """Einstein summation convention.

        Delegates to ``np.einsum``.

        Args:
            subscript: Einstein summation subscript string.
            *operands: Tensors to contract.

        Returns:
            Tensor: Result of the einsum.
        """
        return Tensor(np.einsum(subscript, *(o._data for o in operands)))

    # ---- factory helpers ---------------------------------------------------

    @staticmethod
    def zeros(*shape: int, dtype: type = float) -> 'Tensor':
        return Tensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def ones(*shape: int, dtype: type = float) -> 'Tensor':
        return Tensor(np.ones(shape, dtype=dtype))

    @staticmethod
    def eye(n: int, dtype: type = float) -> 'Tensor':
        return Tensor(np.eye(n, dtype=dtype))

    @staticmethod
    def linspace(start: float, stop: float, num: int) -> 'Tensor':
        return Tensor(np.linspace(start, stop, num))

    @staticmethod
    def arange(*args, dtype: type = float) -> 'Tensor':
        return Tensor(np.arange(*args, dtype=dtype))

    # ---- repr / str --------------------------------------------------------

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data=\n{self._data})"

    def __str__(self) -> str:
        if self.ndim <= 2:
            return str(self._data)
        return f"Tensor(shape={self.shape})"
