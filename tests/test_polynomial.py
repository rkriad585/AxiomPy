import pytest

from axiompy import Axiom


class TestPolynomial:
    def test_creation(self):
        p = Axiom.Polynomial([1, -3, 2])
        assert p.degree == 2

    def test_trailing_zeros(self):
        p = Axiom.Polynomial([1, 0, 0])
        assert p.degree == 0
        assert p.coeffs == [1.0]

    def test_call(self):
        p = Axiom.Polynomial([1, -3, 2])
        assert p(0) == 1.0
        assert p(1) == 0.0
        assert p(2) == 3.0

    def test_add(self):
        p1 = Axiom.Polynomial([1, 2])
        p2 = Axiom.Polynomial([3, 4, 5])
        p3 = p1 + p2
        assert p3.coeffs == [4.0, 6.0, 5.0]

    def test_sub(self):
        p1 = Axiom.Polynomial([5, 4, 3])
        p2 = Axiom.Polynomial([1, 2, 1])
        p3 = p1 - p2
        assert p3.coeffs == [4.0, 2.0, 2.0]

    def test_scalar_mul(self):
        p = Axiom.Polynomial([1, 2, 3])
        p2 = p * 2.0
        assert p2.coeffs == [2.0, 4.0, 6.0]

    def test_rmul(self):
        p = Axiom.Polynomial([1, 2, 3])
        p2 = 2.0 * p
        assert p2.coeffs == [2.0, 4.0, 6.0]

    def test_poly_mul(self):
        p1 = Axiom.Polynomial([1, 1])
        p2 = Axiom.Polynomial([1, 1])
        p3 = p1 * p2
        assert p3.coeffs == [1.0, 2.0, 1.0]

    def test_eq(self):
        p1 = Axiom.Polynomial([1, 2, 3])
        p2 = Axiom.Polynomial([1, 2, 3])
        assert p1 == p2

    def test_derivative(self):
        p = Axiom.Polynomial([1, -3, 2])
        dp = p.derivative()
        assert dp.coeffs == [-3.0, 4.0]

    def test_derivative_constant(self):
        p = Axiom.Polynomial([5])
        dp = p.derivative()
        assert dp.coeffs == [0.0]

    def test_integral(self):
        p = Axiom.Polynomial([1, 2])
        ip = p.integral(constant=1.0)
        assert ip(0) == 1.0

    def test_roots_quadratic(self):
        p = Axiom.Polynomial([1, -3, 2])
        roots = p.roots()
        assert sorted(r.real for r in roots) == [0.5, 1.0]

    def test_roots_linear(self):
        p = Axiom.Polynomial([3, 1])
        roots = p.roots()
        assert roots == [-3.0]

    def test_roots_constant(self):
        p = Axiom.Polynomial([5])
        assert p.roots() == []

    def test_lagrange_interpolate(self):
        p = Axiom.Polynomial.lagrange_interpolate([0, 1, 2], [0, 1, 4])
        assert p(0) == pytest.approx(0.0)
        assert p(1) == pytest.approx(1.0)
        assert p(2) == pytest.approx(4.0)

    def test_floordiv(self):
        p = Axiom.Polynomial([1, 2, 1])
        q = Axiom.Polynomial([1, 1])
        quotient = p // q
        assert quotient.coeffs == [1.0, 1.0]

    def test_mod(self):
        p = Axiom.Polynomial([1, 2, 1])
        q = Axiom.Polynomial([1, 1])
        remainder = p % q
        assert remainder.coeffs == [0.0]

    def test_gcd(self):
        p1 = Axiom.Polynomial([1, 2, 1])
        p2 = Axiom.Polynomial([1, 1])
        g = p1.gcd(p2)
        assert g.coeffs == [1.0, 1.0]

    def test_compose(self):
        p = Axiom.Polynomial([0, 1])
        q = Axiom.Polynomial([1, 1])
        composed = p.compose(q)
        assert composed(0) == 1.0
        assert composed(1) == 2.0

    def test_chebyshev_roots(self):
        roots = Axiom.Polynomial.chebyshev_roots(5)
        assert len(roots) == 5
        for r in roots:
            assert -1 <= r <= 1

    def test_fit(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 4, 9, 16]
        p = Axiom.Polynomial.fit(xs, ys, degree=2)
        assert p(0) == pytest.approx(0.0, abs=1e-4)
        assert p(1) == pytest.approx(1.0, abs=1e-4)
        assert p(2) == pytest.approx(4.0, abs=1e-4)
