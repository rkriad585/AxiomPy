import abc

import numpy as np


class Backend(abc.ABC):
    """Abstract interface for numerical linear algebra backends.

    All backend implementations must subclass this and implement every
    abstract method.
    """

    @abc.abstractmethod
    def array(self, data, dtype=None) -> np.ndarray:
        """Create a NumPy array from the given data.

        Args:
            data: Array-like input.
            dtype: Optional dtype for the array.

        Returns:
            np.ndarray: The constructed array.
        """
        ...

    @abc.abstractmethod
    def dot(self, a, b) -> np.ndarray:
        """Compute the dot product of two arrays.

        Args:
            a: First array.
            b: Second array.

        Returns:
            np.ndarray: The dot product.
        """
        ...

    @abc.abstractmethod
    def inv(self, a) -> np.ndarray:
        """Compute the multiplicative inverse of a matrix.

        Args:
            a: Square matrix.

        Returns:
            np.ndarray: The inverse matrix.
        """
        ...

    @abc.abstractmethod
    def solve(self, A, b) -> np.ndarray:
        """Solve the linear system ``Ax = b``.

        Args:
            A: Coefficient matrix.
            b: Right-hand side vector or matrix.

        Returns:
            np.ndarray: Solution ``x``.
        """
        ...

    @abc.abstractmethod
    def det(self, a) -> float:
        """Compute the determinant of a matrix.

        Args:
            a: Square matrix.

        Returns:
            float: The determinant.
        """
        ...

    @abc.abstractmethod
    def matrix_power(self, a, power) -> np.ndarray:
        """Raise a square matrix to an integer power.

        Args:
            a: Square matrix.
            power: Integer exponent.

        Returns:
            np.ndarray: ``a`` raised to ``power``.
        """
        ...

    @abc.abstractmethod
    def norm(self, a, ord=None) -> float:
        """Compute the norm of a vector or matrix.

        Args:
            a: Input array.
            ord: Order of the norm (default: 2-norm for vectors, Frobenius for matrices).

        Returns:
            float: The computed norm.
        """
        ...

    @abc.abstractmethod
    def qr(self, a) -> tuple[np.ndarray, np.ndarray]:
        """Compute the QR decomposition of a matrix.

        Args:
            a: Matrix to decompose.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(Q, R)`` matrices.
        """
        ...

    @abc.abstractmethod
    def cholesky(self, a) -> np.ndarray:
        """Compute the Cholesky decomposition of a positive-definite matrix.

        Args:
            a: Positive-definite square matrix.

        Returns:
            np.ndarray: Lower triangular Cholesky factor ``L``.
        """
        ...

    @abc.abstractmethod
    def eigvals(self, a) -> np.ndarray:
        """Compute the eigenvalues of a square matrix.

        Args:
            a: Square matrix.

        Returns:
            np.ndarray: Eigenvalues.
        """
        ...

    @abc.abstractmethod
    def matrix_rank(self, a) -> int:
        """Compute the rank of a matrix.

        Args:
            a: Input matrix.

        Returns:
            int: The matrix rank.
        """
        ...


class NumpyBackend(Backend):
    """Concrete backend that delegates all operations to NumPy."""

    def array(self, data, dtype=None):
        """Create a NumPy array from the given data.

        Args:
            data: Array-like input.
            dtype: Optional dtype for the array.

        Returns:
            np.ndarray: The constructed array.
        """
        return np.array(data, dtype=dtype)

    def dot(self, a, b):
        """Compute the dot product of two arrays.

        Args:
            a: First array.
            b: Second array.

        Returns:
            np.ndarray: The dot product.
        """
        return np.dot(a, b)

    def inv(self, a):
        """Compute the multiplicative inverse of a matrix.

        Args:
            a: Square matrix.

        Returns:
            np.ndarray: The inverse matrix.
        """
        return np.linalg.inv(a)

    def solve(self, A, b):
        """Solve the linear system ``Ax = b``.

        Args:
            A: Coefficient matrix.
            b: Right-hand side vector or matrix.

        Returns:
            np.ndarray: Solution ``x``.
        """
        return np.linalg.solve(A, b)

    def det(self, a):
        """Compute the determinant of a matrix.

        Args:
            a: Square matrix.

        Returns:
            float: The determinant.
        """
        return np.linalg.det(a)

    def matrix_power(self, a, power):
        """Raise a square matrix to an integer power.

        Args:
            a: Square matrix.
            power: Integer exponent.

        Returns:
            np.ndarray: ``a`` raised to ``power``.
        """
        return np.linalg.matrix_power(a, power)

    def norm(self, a, ord=None):
        """Compute the norm of a vector or matrix.

        Args:
            a: Input array.
            ord: Order of the norm (default: 2-norm for vectors, Frobenius for matrices).

        Returns:
            float: The computed norm.
        """
        return np.linalg.norm(a, ord=ord)

    def qr(self, a):
        """Compute the QR decomposition of a matrix.

        Args:
            a: Matrix to decompose.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(Q, R)`` matrices.
        """
        return np.linalg.qr(a)

    def cholesky(self, a):
        """Compute the Cholesky decomposition of a positive-definite matrix.

        Args:
            a: Positive-definite square matrix.

        Returns:
            np.ndarray: Lower triangular Cholesky factor ``L``.
        """
        return np.linalg.cholesky(a)

    def eigvals(self, a):
        """Compute the eigenvalues of a square matrix.

        Args:
            a: Square matrix.

        Returns:
            np.ndarray: Eigenvalues.
        """
        return np.linalg.eigvals(a)

    def matrix_rank(self, a):
        """Compute the rank of a matrix.

        Args:
            a: Input matrix.

        Returns:
            int: The matrix rank.
        """
        return np.linalg.matrix_rank(a)


# Import after class definitions to avoid circular imports
from ._cupy_backend import CuPyBackend  # noqa: E402
from ._jax_backend import JaxBackend  # noqa: E402
from ._pure_backend import PurePythonBackend  # noqa: E402

_backends: dict[str, type[Backend]] = {
    "numpy": NumpyBackend,
    "pure": PurePythonBackend,
    "jax": JaxBackend,
    "cupy": CuPyBackend,
}

_current_backend: Backend = NumpyBackend()


def get_backend() -> Backend:
    """Return the currently active backend instance.

    Returns:
        Backend: The active backend.
    """
    return _current_backend


def register_backend(name: str, backend_cls: type[Backend]):
    """Register a new backend implementation by name.

    Args:
        name: Identifier for the backend.
        backend_cls: Class implementing the :class:`Backend` interface.
    """
    _backends[name] = backend_cls


def set_backend(name: str = "numpy"):
    """Switch the active backend.

    Args:
        name: Backend identifier (must have been registered).

    Raises:
        ValueError: If the backend name is unknown.
    """
    global _current_backend
    if name not in _backends:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(_backends.keys())}")
    _current_backend = _backends[name]()
