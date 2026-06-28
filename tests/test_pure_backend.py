"""Tests for PurePythonBackend."""

import pytest

from axiompy._backend import get_backend, set_backend
from axiompy._pure_array import PureArray
from axiompy._pure_backend import PurePythonBackend


@pytest.fixture(params=["numpy", "pure"])
def backend(request):
    """Parametrized fixture: runs each test against both numpy and pure backends."""
    saved = get_backend()
    set_backend(request.param)
    yield request.param
    set_backend("numpy" if request.param == "pure" else "pure")
    # Restore original
    set_backend("numpy" if saved != "numpy" else "numpy")


class TestPurePythonBackend:
    def test_array_1d(self):
        b = PurePythonBackend()
        a = b.array([1, 2, 3])
        assert a.shape == (3,)
        assert a.tolist() == [1, 2, 3]

    def test_array_2d(self):
        b = PurePythonBackend()
        a = b.array([[1, 2], [3, 4]])
        assert a.shape == (2, 2)

    def test_dot_1d(self):
        b = PurePythonBackend()
        result = b.dot([1, 2, 3], [4, 5, 6])
        assert result == 32

    def test_dot_2d(self):
        b = PurePythonBackend()
        r = b.dot([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert r.shape == (2, 2)
        assert r.tolist() == [[19, 22], [43, 50]]

    def test_inv(self):
        b = PurePythonBackend()
        r = b.inv([[1, 2], [3, 4]])
        assert r.shape == (2, 2)
        data = r.tolist()
        # Check A @ A^{-1} ≈ I
        prod = [[sum(data[i][k] * ([[1, 2], [3, 4]][k][j]) for k in range(2)) for j in range(2)]
                for i in range(2)]
        assert all(abs(prod[i][j] - (1.0 if i == j else 0.0)) < 1e-10
                   for i in range(2) for j in range(2))

    def test_solve(self):
        b = PurePythonBackend()
        x = b.solve([[2, 1], [1, 3]], [5, 6])
        assert x.shape == (2,)
        data = x.tolist()
        assert abs(data[0] - 1.8) < 1e-10
        assert abs(data[1] - 1.4) < 1e-10

    def test_det(self):
        b = PurePythonBackend()
        assert abs(b.det([[1, 2], [3, 4]]) - (-2.0)) < 1e-10
        assert b.det([[1, 2], [2, 4]]) == 0.0

    def test_matrix_power(self):
        b = PurePythonBackend()
        r = b.matrix_power([[1, 1], [0, 1]], 3)
        assert r.tolist() == [[1, 3], [0, 1]]

    def test_norm(self):
        b = PurePythonBackend()
        assert abs(b.norm([3, 4]) - 5.0) < 1e-10
        assert abs(b.norm([3, 4], ord=1) - 7.0) < 1e-10

    def test_cholesky(self):
        b = PurePythonBackend()
        L = b.cholesky([[4, 1], [1, 3]])
        L_data = L.tolist()
        # L @ L^T should give original
        A_recovered = [[L_data[0][0]**2, L_data[0][0]*L_data[1][0]],
                       [L_data[0][0]*L_data[1][0], L_data[1][0]**2 + L_data[1][1]**2]]
        assert all(abs(A_recovered[i][j] - [[4, 1], [1, 3]][i][j]) < 1e-10
                   for i in range(2) for j in range(2))

    def test_matrix_rank(self):
        b = PurePythonBackend()
        assert b.matrix_rank([[1, 2], [3, 4]]) == 2
        assert b.matrix_rank([[1, 2], [2, 4]]) == 1

    def test_qr(self):
        b = PurePythonBackend()
        A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]
        Q, R = b.qr(A)
        Q_data = Q.tolist()
        R_data = R.tolist()
        # Q @ R ≈ A
        m, n = 3, 3
        prod = [[sum(Q_data[i][k] * R_data[k][j] for k in range(n)) for j in range(n)]
                for i in range(m)]
        assert all(abs(prod[i][j] - A[i][j]) < 1e-10
                   for i in range(m) for j in range(n))
        # Q is orthogonal: Q^T @ Q ≈ I
        qtq = [[sum(Q_data[k][i] * Q_data[k][j] for k in range(m)) for j in range(n)]
               for i in range(n)]
        assert all(abs(qtq[i][j] - (1.0 if i == j else 0.0)) < 1e-10
                   for i in range(n) for j in range(n))

    def test_eigvals(self):
        b = PurePythonBackend()
        ev = b.eigvals([[4, 1], [1, 3]])
        ev_data = sorted(ev.tolist())
        assert any(abs(v - 2.381966) < 1e-4 for v in ev_data)
        assert any(abs(v - 4.618034) < 1e-4 for v in ev_data)

    def test_fft(self):
        b = PurePythonBackend()
        x = [1.0, 0.0, -1.0, 0.0]
        r = b.fft(x)
        data = r.tolist()
        assert len(data) == 4
        # Expected FFT of [1, 0, -1, 0]: [0, 2, 0, 2]
        assert abs(data[0]) < 1e-10
        assert abs(data[1] - 2.0) < 1e-10
        assert abs(data[2]) < 1e-10
        assert abs(data[3] - 2.0) < 1e-10

    def test_array_from_purearray(self):
        b = PurePythonBackend()
        pa = PureArray([1, 2, 3])
        a = b.array(pa)
        assert a.tolist() == [1, 2, 3]
