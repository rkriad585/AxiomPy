import pytest
from axiompy import Axiom


class TestLinearAlgebra:
    def test_identity(self):
        I = Axiom.linalg.identity(3)
        assert I.shape == (3, 3)
        assert I.to_list() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_zeros(self):
        Z = Axiom.linalg.zeros((2, 3))
        assert Z.shape == (2, 3)
        assert Z.to_list() == [[0, 0, 0], [0, 0, 0]]

    def test_ones(self):
        O = Axiom.linalg.ones((2, 2))
        assert O.shape == (2, 2)
        assert O.to_list() == [[1, 1], [1, 1]]

    def test_solve_linear(self):
        A = Axiom.Matrix([[3, 1], [1, 2]])
        b = Axiom.Vector([5, 5])
        x = Axiom.linalg.solve_linear(A, b)
        assert x.to_list() == [1.0, 2.0]

    def test_least_squares(self):
        A = Axiom.Matrix([[1, 0], [1, 0], [1, 0]])
        b = Axiom.Vector([2, 3, 4])
        x = Axiom.linalg.least_squares(A, b)
        assert x.to_list()[0] == pytest.approx(3.0, abs=1e-10)
        assert x.to_list()[1] == pytest.approx(0.0, abs=1e-10)

    def test_cross_product_matrix(self):
        v = Axiom.Vector([1, 2, 3])
        M = Axiom.linalg.cross_product_matrix(v)
        w = Axiom.Vector([4, 5, 6])
        result = M @ w
        cross = v.cross(w)
        assert result.to_list() == pytest.approx(cross.to_list(), abs=1e-10)

    def test_householder(self):
        v = Axiom.Vector([1, 0, 0])
        H = Axiom.linalg.householder(v)
        neg = H @ v
        assert neg.to_list() == pytest.approx([-1.0, 0.0, 0.0], abs=1e-10)
