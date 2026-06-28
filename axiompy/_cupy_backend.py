"""CuPy backend — CUDA-accelerated via cupy.

Requires ``cupy`` (optional dependency) and a CUDA-capable GPU.
"""
import logging

from ._backend import Backend

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
    logger.info("cupy not available — CuPyBackend will raise on instantiation")


class CuPyBackend(Backend):
    """Backend that delegates operations to ``cupy`` for CUDA acceleration.

    All arrays are stored on the GPU (device memory).
    """

    def __init__(self):
        if not _HAS_CUPY:
            raise ImportError(
                "CuPyBackend requires cupy. "
                "Install with: uv pip install cupy"
            )

    def array(self, data, dtype=None):
        return cp.array(data, dtype=dtype)

    def dot(self, a, b):
        return cp.dot(a, b)

    def inv(self, a):
        return cp.linalg.inv(a)

    def solve(self, A, b):
        return cp.linalg.solve(A, b)

    def det(self, a):
        return float(cp.linalg.det(a))

    def matrix_power(self, a, power):
        return cp.linalg.matrix_power(a, power)

    def norm(self, a, ord=None):
        return float(cp.linalg.norm(a, ord=ord))

    def qr(self, a):
        Q, R = cp.linalg.qr(a)
        return Q, R

    def cholesky(self, a):
        return cp.linalg.cholesky(a)

    def eigvals(self, a):
        return cp.linalg.eigvals(a)

    def matrix_rank(self, a):
        return int(cp.linalg.matrix_rank(a))
