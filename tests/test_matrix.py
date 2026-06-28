import pytest
import numpy as np
from axiompy import Axiom


class TestMatrix:
    def test_creation(self, m1):
        assert m1.shape == (2, 2)
        assert m1.to_list() == [[1.0, 2.0], [3.0, 4.0]]

    def test_add(self, m1, m2):
        m3 = m1 + m2
        assert m3.to_list() == [[6.0, 8.0], [10.0, 12.0]]

    def test_sub(self, m1, m2):
        m3 = m1 - m2
        assert m3.to_list() == [[-4.0, -4.0], [-4.0, -4.0]]

    def test_scalar_mul(self, m1):
        m3 = m1 * 2.0
        assert m3.to_list() == [[2.0, 4.0], [6.0, 8.0]]

    def test_matmul(self, m1, m2):
        m3 = m1 @ m2
        assert m3.to_list() == [[19.0, 22.0], [43.0, 50.0]]

    def test_matmul_vector(self, m1):
        v = Axiom.Vector([1, 2])
        result = m1 @ v
        assert result.to_list() == [5.0, 11.0]

    def test_pow(self, m1):
        m3 = m1 ** 2
        assert m3.to_list() == [[7.0, 10.0], [15.0, 22.0]]

    def test_eq(self, m1):
        m3 = Axiom.Matrix([[1, 2], [3, 4]])
        assert m1 == m3

    def test_ne(self, m1, m2):
        assert m1 != m2

    def test_transpose(self, m1):
        assert m1.T.to_list() == [[1.0, 3.0], [2.0, 4.0]]

    def test_determinant(self, m1):
        assert m1.determinant == pytest.approx(-2.0)

    def test_inverse(self, m1):
        inv = m1.inverse
        ident = m1 @ inv
        for row in ident.to_list():
            for val in row:
                assert val == pytest.approx(0.0 if abs(val) < 1e-10 else 1.0, abs=1e-10)

    def test_trace(self, m1):
        assert m1.trace == pytest.approx(5.0)

    def test_rank(self, m1):
        assert m1.rank == 2

    def test_norm(self, m1):
        assert m1.norm() == pytest.approx(5.477225, rel=1e-5)

    def test_lu_decompose(self, m1):
        P, L, U = m1.lu_decompose()
        reconstructed = P @ L @ U
        assert (m1 - reconstructed).norm() < 1e-10

    def test_qr_decompose(self, m1):
        Q, R = m1.qr_decompose()
        reconstructed = Q @ R
        assert (m1 - reconstructed).norm() < 1e-10

    def test_cholesky_decompose(self):
        A = Axiom.Matrix([[4, 2], [2, 3]])
        L = A.cholesky_decompose()
        reconstructed = L @ L.T
        assert (A - reconstructed).norm() < 1e-10

    def test_matmul_notimplemented(self, m1):
        result = m1.__matmul__("string")
        assert result is NotImplemented

    def test_svd_decompose(self, m1):
        U, S, Vt = m1.svd_decompose()
        reconstructed = U @ Axiom.Matrix(np.diag(S.to_list())) @ Vt
        assert (m1 - reconstructed).norm() < 1e-10

    def test_eigenvalues(self):
        A = Axiom.Matrix([[4, 2], [2, 3]])
        evals = A.eigenvalues()
        assert len(evals.to_list()) == 2
        assert evals.to_list()[0] == pytest.approx(1.438, abs=1e-3)

    def test_eigenvectors(self):
        A = Axiom.Matrix([[4, 2], [2, 3]])
        evecs = A.eigenvectors()
        assert evecs.shape == (2, 2)

    def test_condition_number(self, m1):
        c = m1.condition_number()
        assert c == pytest.approx(14.933, rel=1e-3)

    def test_pinv(self, m1):
        pinv = m1.pinv()
        ident = m1 @ pinv
        np.testing.assert_allclose(ident.to_list(), np.eye(2), atol=1e-10)
