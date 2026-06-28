import pytest
from axiompy import Axiom, set_backend, register_backend, get_backend, Backend


def test_default_backend():
    assert get_backend() is not None


def test_set_backend_unknown():
    with pytest.raises(ValueError, match="Unknown backend"):
        set_backend("nonexistent")


def test_facade_backend():
    assert Axiom.backend is not None


def test_register_backend():
    from axiompy._backend import NumpyBackend

    class FakeBackend(Backend):
        def array(self, data, dtype=None):
            return data
        def dot(self, a, b):
            return 0
        def inv(self, a):
            return a
        def solve(self, A, b):
            return b
        def det(self, a):
            return 0.0
        def matrix_power(self, a, power):
            return a
        def norm(self, a, ord=None):
            return 0.0
        def qr(self, a):
            return a, a
        def cholesky(self, a):
            return a
        def eigvals(self, a):
            return []
        def matrix_rank(self, a):
            return 0

    register_backend("fake", FakeBackend)
    set_backend("fake")
    try:
        assert isinstance(get_backend(), FakeBackend)
    finally:
        set_backend("numpy")
