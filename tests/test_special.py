import math

import pytest

from axiompy import Axiom


class TestGamma:
    def test_gamma_integer(self):
        assert Axiom.gamma(1) == pytest.approx(1.0)
        assert Axiom.gamma(2) == pytest.approx(1.0)
        assert Axiom.gamma(3) == pytest.approx(2.0)
        assert Axiom.gamma(4) == pytest.approx(6.0)
        assert Axiom.gamma(5) == pytest.approx(24.0)

    def test_gamma_half(self):
        assert Axiom.gamma(0.5) == pytest.approx(math.sqrt(math.pi), rel=1e-5)

    def test_gamma_reflection(self):
        assert Axiom.gamma(-0.5) == pytest.approx(-2 * math.sqrt(math.pi), rel=1e-5)


class TestBeta:
    def test_beta_symmetric(self):
        assert Axiom.beta(1, 1) == pytest.approx(1.0)
        assert Axiom.beta(2, 2) == pytest.approx(1.0 / 6, rel=1e-5)

    def test_beta_identity(self):
        assert Axiom.beta(1, 2) == pytest.approx(0.5, rel=1e-5)


class TestErf:
    def test_erf_zero(self):
        assert Axiom.erf(0) == pytest.approx(0.0)

    def test_erf_inf(self):
        assert Axiom.erf(10) == pytest.approx(1.0, abs=1e-6)
        assert Axiom.erf(-10) == pytest.approx(-1.0, abs=1e-6)

    def test_erfc(self):
        assert Axiom.erfc(0) == pytest.approx(1.0)
        assert Axiom.erfc(10) == pytest.approx(0.0, abs=1e-6)


class TestBessel:
    def test_j0(self):
        assert Axiom.bessel_j(0, 0) == pytest.approx(1.0)
        assert Axiom.bessel_j(0, 1) == pytest.approx(0.7651976866, rel=1e-5)

    def test_j1(self):
        assert Axiom.bessel_j(1, 0) == pytest.approx(0.0)

    def test_y0(self):
        assert Axiom.bessel_y(0, 1) == pytest.approx(0.08825692, rel=1e-3)


class TestLegendre:
    def test_p0(self):
        assert Axiom.legendre_p(0, 0.5) == pytest.approx(1.0)

    def test_p1(self):
        assert Axiom.legendre_p(1, 0.5) == pytest.approx(0.5)

    def test_p2(self):
        assert Axiom.legendre_p(2, 0.5) == pytest.approx(-0.125)


class TestCombinatorial:
    def test_factorial(self):
        assert Axiom.factorial(0) == 1
        assert Axiom.factorial(5) == 120

    def test_binomial(self):
        assert Axiom.binomial(5, 2) == 10
        assert Axiom.binomial(5, 0) == 1
        assert Axiom.binomial(5, 5) == 1

    def test_double_factorial(self):
        assert Axiom.double_factorial(0) == 1
        assert Axiom.double_factorial(1) == 1
        assert Axiom.double_factorial(5) == 15  # 5*3*1
        assert Axiom.double_factorial(6) == 48  # 6*4*2
