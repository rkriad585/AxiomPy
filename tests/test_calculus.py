import math
import pytest
from axiompy import Axiom


class TestCalculus:
    def test_numerical_derivative(self):
        f = lambda x: x ** 2
        assert Axiom.calc.numerical_derivative(f, 3.0) == pytest.approx(6.0, rel=1e-4)

    def test_integrate_trapezoid(self):
        f = lambda x: x ** 2
        result = Axiom.calc.integrate_trapezoid(f, 0, 1, n=1000)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-2)

    def test_integrate_simpson(self):
        f = lambda x: x ** 2
        result = Axiom.calc.integrate_simpson(f, 0, 1, n=100)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-4)

    def test_integrate_monte_carlo(self):
        f = lambda x: x ** 2
        result = Axiom.calc.integrate_monte_carlo(f, 0, 1, n=50000)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-1)

    def test_gradient(self):
        f = lambda p: p[0] ** 2 + p[1] ** 2
        grad = Axiom.calc.gradient(f, [1.0, 2.0])
        assert grad[0] == pytest.approx(2.0, rel=1e-4)
        assert grad[1] == pytest.approx(4.0, rel=1e-4)

    def test_richardson_extrapolation(self):
        f = lambda x: x ** 3
        assert Axiom.calc.richardson_extrapolation(f, 2.0) == pytest.approx(12.0, rel=1e-4)

    def test_integrate_gauss_legendre(self):
        f = lambda x: x ** 2
        result = Axiom.calc.integrate_gauss_legendre(f, 0, 1, n=5)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-4)

    def test_integrate_romberg(self):
        f = lambda x: x ** 2
        result = Axiom.calc.integrate_romberg(f, 0, 1, tol=1e-8)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-6)

    def test_jacobian(self):
        f = lambda p: [p[0] ** 2 + p[1], p[0] - p[1] ** 2]
        J = Axiom.calc.jacobian(f, [1.0, 2.0])
        assert J[0][0] == pytest.approx(2.0, rel=1e-4)
        assert J[0][1] == pytest.approx(1.0, rel=1e-4)
        assert J[1][0] == pytest.approx(1.0, rel=1e-4)
        assert J[1][1] == pytest.approx(-4.0, rel=1e-4)

    def test_hessian(self):
        f = lambda p: p[0] ** 2 + 3 * p[0] * p[1] + p[1] ** 2
        H = Axiom.calc.hessian(f, [1.0, 1.0])
        assert H[0][0] == pytest.approx(2.0, rel=1e-3)
        assert H[0][1] == pytest.approx(3.0, rel=1e-3)
        assert H[1][0] == pytest.approx(3.0, rel=1e-3)
        assert H[1][1] == pytest.approx(2.0, rel=1e-3)

    def test_ode_euler(self):
        f = lambda t, y: [y[0]]  # dy/dt = y -> y = exp(t)
        ts, ys = Axiom.calc.ode_euler(f, [1.0], (0.0, 1.0), dt=0.001)
        assert ys[-1][0] == pytest.approx(math.e, rel=1e-2)

    def test_ode_rk4(self):
        f = lambda t, y: [y[0]]
        ts, ys = Axiom.calc.ode_rk4(f, [1.0], (0.0, 1.0), dt=0.1)
        assert ys[-1][0] == pytest.approx(math.e, rel=1e-4)
