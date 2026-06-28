"""Tests for JaxBackend and CuPyBackend.

These backends require optional dependencies (jax / cupy) so tests
are skipped if the library is not installed.
"""

import pytest

from axiompy._backend import get_backend, set_backend
from axiompy._cupy_backend import _HAS_CUPY, CuPyBackend
from axiompy._jax_backend import _HAS_JAX, JaxBackend

# ---- JAX backend tests -----------------------------------------------------

@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
class TestJaxBackend:
    def test_switch_to_jax(self):
        set_backend("jax")
        assert isinstance(get_backend(), JaxBackend)
        set_backend("numpy")

    def test_jax_array(self):
        b = JaxBackend()
        a = b.array([1, 2, 3])
        assert a.shape == (3,)

    def test_jax_dot(self):
        b = JaxBackend()
        r = b.dot([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert r.shape == (2, 2)
        assert abs(r[0, 0] - 19) < 1e-10

    def test_jax_det(self):
        b = JaxBackend()
        assert abs(b.det([[1, 2], [3, 4]]) - (-2.0)) < 1e-10

    def test_jax_solve(self):
        b = JaxBackend()
        x = b.solve([[2, 1], [1, 3]], [5, 6])
        assert abs(x[0] - 1.8) < 1e-10
        assert abs(x[1] - 1.4) < 1e-10

    def test_jax_norm(self):
        b = JaxBackend()
        assert abs(b.norm([3, 4]) - 5.0) < 1e-10

    def test_jax_qr(self):
        b = JaxBackend()
        Q, R = b.qr([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
        import jax.numpy as jnp
        prod = Q @ R
        A = jnp.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
        assert jnp.allclose(prod, A, atol=1e-10)

    def test_jax_eigvals(self):
        b = JaxBackend()
        ev = b.eigvals([[4, 1], [1, 3]])
        ev_sorted = sorted(ev.tolist(), reverse=True)
        assert any(abs(v - 4.618034) < 1e-4 for v in ev_sorted)

    def test_jax_matrix_rank(self):
        b = JaxBackend()
        assert b.matrix_rank([[1, 2], [3, 4]]) == 2
        assert b.matrix_rank([[1, 2], [2, 4]]) == 1


# ---- CuPy backend tests ----------------------------------------------------

@pytest.mark.skipif(not _HAS_CUPY, reason="cupy not installed")
class TestCuPyBackend:
    def test_switch_to_cupy(self):
        set_backend("cupy")
        assert isinstance(get_backend(), CuPyBackend)
        set_backend("numpy")

    def test_cupy_array(self):
        b = CuPyBackend()
        a = b.array([1, 2, 3])
        assert a.shape == (3,)

    def test_cupy_dot(self):
        b = CuPyBackend()
        r = b.dot([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert r.shape == (2, 2)

    def test_cupy_det(self):
        b = CuPyBackend()
        assert abs(b.det([[1, 2], [3, 4]]) - (-2.0)) < 1e-10

    def test_cupy_solve(self):
        b = CuPyBackend()
        x = b.solve([[2, 1], [1, 3]], [5, 6])
        assert abs(float(x[0]) - 1.8) < 1e-10

    def test_cupy_matrix_rank(self):
        b = CuPyBackend()
        assert b.matrix_rank([[1, 2], [3, 4]]) == 2

    def test_raise_when_not_available(self):
        # Verify the ImportError path
        import axiompy._cupy_backend as cb
        old = cb._HAS_CUPY
        try:
            cb._HAS_CUPY = False
            with pytest.raises(ImportError, match="CuPyBackend requires cupy"):
                CuPyBackend()
        finally:
            cb._HAS_CUPY = old


# ---- JaxBacket raise when not available ------------------------------------

@pytest.mark.skipif(_HAS_JAX, reason="jax IS installed — can't test missing case")
def test_jax_raise_when_not_available():
    with pytest.raises(ImportError, match="JaxBackend requires jax"):
        JaxBackend()
