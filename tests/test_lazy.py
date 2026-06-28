"""Tests for axiompy._lazy — deferred computation graphs."""


from axiompy import Axiom
from axiompy._lazy import LazyExpr


class TestLazyExpr:
    def test_constant(self):
        v = Axiom.Vector([1, 2, 3])
        expr = LazyExpr.constant(v)
        result = expr.compute()
        assert result == v

    def test_add_vectors(self):
        v = Axiom.Vector([1, 2])
        w = Axiom.Vector([3, 4])
        expr = Axiom.lazy.lazy(v) + Axiom.lazy.lazy(w)
        result = expr.compute()
        assert result == v + w

    def test_sub_vectors(self):
        v = Axiom.Vector([5, 6])
        w = Axiom.Vector([3, 4])
        expr = Axiom.lazy.lazy(v) - Axiom.lazy.lazy(w)
        result = expr.compute()
        assert result == v - w

    def test_mul_scalar(self):
        v = Axiom.Vector([1, 2, 3])
        expr = Axiom.lazy.lazy(v) * 3.0
        result = expr.compute()
        assert result == v * 3.0

    def test_rmul(self):
        v = Axiom.Vector([1, 2])
        expr = 2.0 * Axiom.lazy.lazy(v)
        result = expr.compute()
        assert result == 2.0 * v

    def test_neg(self):
        v = Axiom.Vector([1, -2])
        expr = -Axiom.lazy.lazy(v)
        result = expr.compute()
        assert result == -v

    def test_matmul(self):
        m = Axiom.Matrix([[1, 2], [3, 4]])
        v = Axiom.Vector([5, 6])
        expr = Axiom.lazy.lazy(m) @ Axiom.lazy.lazy(v)
        result = expr.compute()
        assert result == m @ v

    def test_matrix_add(self):
        A = Axiom.Matrix([[1, 2], [3, 4]])
        B = Axiom.Matrix([[5, 6], [7, 8]])
        expr = Axiom.lazy.lazy(A) + Axiom.lazy.lazy(B)
        result = expr.compute()
        assert result == A + B

    def test_transpose(self):
        A = Axiom.Matrix([[1, 2], [3, 4]])
        expr = Axiom.lazy.lazy(A).T
        result = expr.compute()
        assert result == A.T

    def test_pow(self):
        A = Axiom.Matrix([[1, 1], [0, 1]])
        expr = Axiom.lazy.lazy(A) ** 3
        result = expr.compute()
        assert result == A ** 3

    def test_chained(self):
        v = Axiom.Vector([1, 2, 3])
        w = Axiom.Vector([4, 5, 6])
        # (v + w) * 2.0 - v
        expr = (Axiom.lazy.lazy(v) + Axiom.lazy.lazy(w)) * 2.0 - Axiom.lazy.lazy(v)
        result = expr.compute()
        expected = (v + w) * 2.0 - v
        assert result == expected

    def test_nested_matmul(self):
        A = Axiom.Matrix([[1, 2], [3, 4]])
        B = Axiom.Matrix([[0, 1], [1, 0]])
        C = Axiom.Matrix([[2, 0], [0, 2]])
        expr = (Axiom.lazy.lazy(A) @ Axiom.lazy.lazy(B)) @ Axiom.lazy.lazy(C)
        result = expr.compute()
        expected = (A @ B) @ C
        assert result == expected

    def test_repr(self):
        v = Axiom.Vector([1, 2])
        expr = Axiom.lazy.lazy(v)
        assert "Lazy" in repr(expr)
        assert "Vector" in repr(expr)

    def test_lazy_scope_compute(self):
        v = Axiom.Vector([1, 2])
        expr = Axiom.lazy.lazy(v) * 3.0
        result = Axiom.lazy.compute(expr)
        assert result == v * 3.0

    def test_add_scalar_matrix(self):
        A = Axiom.Matrix([[1, 2], [3, 4]])
        expr = Axiom.lazy.lazy(A) + 5.0
        result = expr.compute()
        assert result.to_list() == [[6.0, 7.0], [8.0, 9.0]]

    def test_sub_scalar_matrix(self):
        A = Axiom.Matrix([[1, 2], [3, 4]])
        expr = Axiom.lazy.lazy(A) - 1.0
        result = expr.compute()
        assert result.to_list() == [[0.0, 1.0], [2.0, 3.0]]

    def test_rsub_matrix(self):
        A = Axiom.Matrix([[1, 2], [3, 4]])
        expr = 10.0 - Axiom.lazy.lazy(A)
        result = expr.compute()
        assert result.to_list() == [[9.0, 8.0], [7.0, 6.0]]
