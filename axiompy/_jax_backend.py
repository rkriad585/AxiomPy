"""JAX backend — JIT-compiled via jax.numpy.

Requires ``jax`` and ``jaxlib`` (optional dependency).
"""
import logging

from ._backend import Backend

logger = logging.getLogger(__name__)

try:
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    logger.info("jax not available — JaxBackend will raise on instantiation")


class JaxBackend(Backend):
    """Backend that delegates operations to ``jax.numpy`` for JIT compilation.

    GPU acceleration is automatic when JAX is installed with CUDA support.
    """

    def __init__(self):
        if not _HAS_JAX:
            raise ImportError(
                "JaxBackend requires jax and jaxlib. "
                "Install with: uv pip install jax jaxlib"
            )

    def array(self, data, dtype=None):
        return jnp.array(data, dtype=dtype)

    def dot(self, a, b):
        return jnp.dot(a, b)

    def inv(self, a):
        return jnp.linalg.inv(a)

    def solve(self, A, b):
        return jnp.linalg.solve(A, b)

    def det(self, a):
        return float(jnp.linalg.det(a))

    def matrix_power(self, a, power):
        return jnp.linalg.matrix_power(a, power)

    def norm(self, a, ord=None):
        return float(jnp.linalg.norm(a, ord=ord))

    def qr(self, a):
        Q, R = jnp.linalg.qr(a)
        return Q, R

    def cholesky(self, a):
        return jnp.linalg.cholesky(a)

    def eigvals(self, a):
        return jnp.linalg.eigvals(a)

    def matrix_rank(self, a):
        return int(jnp.linalg.matrix_rank(a))
