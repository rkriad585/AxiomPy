import math

import pytest

from axiompy import Axiom
from axiompy.odes import lotka_volterra_odes, pendulum_odes


class TestSolveIVP:
    def test_euler(self):
        f = lambda t, y: [y[0]]
        ts, ys = Axiom.solve_ivp(f, [1.0], (0, 1), method='euler', dt=0.01)
        assert abs(ys[-1][0] - math.e) < 0.015

    def test_rk4(self):
        f = lambda t, y: [y[0]]
        ts, ys = Axiom.solve_ivp(f, [1.0], (0, 1), method='rk4', dt=0.1)
        assert abs(ys[-1][0] - math.exp(1)) < 0.01

    def test_rk45_adaptive(self):
        f = lambda t, y: [y[0]]
        ts, ys = Axiom.solve_ivp(f, [1.0], (0, 1), method='rk45', dt=0.1)
        assert abs(ys[-1][0] - math.exp(1)) < 0.01

    def test_adams_bashforth(self):
        f = lambda t, y: [y[0]]
        ts, ys = Axiom.solve_ivp(f, [1.0], (0, 1), method='adams_bashforth', dt=0.01)
        assert abs(ys[-1][0] - math.exp(1)) < 0.05

    def test_two_dim(self):
        def f(t, y):
            return [-y[1], y[0]]
        ts, ys = Axiom.solve_ivp(f, [1.0, 0.0], (0, math.pi / 2), method='rk4', dt=0.01)
        assert abs(ys[-1][0]) < 0.01
        assert abs(ys[-1][1] - 1.0) < 0.01

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            Axiom.solve_ivp(lambda t, y: [0], [0], (0, 1), method='foo')


class TestSolveBVP:
    def test_bvp_simple(self):
        # y'' = -y, y(0)=0, y(pi/2)=1  =>  y(x) = sin(x)
        def f(t, y):
            return [y[1], -y[0]]
        def bc(y_end):
            return [y_end[0] - 1.0]
        ts, ys = Axiom.solve_bvp(f, bc, (0, math.pi / 2), guess=[0.0, 1.0], dt=0.05)
        assert abs(ys[-1][0] - 1.0) < 0.05


class TestSystemFactories:
    def test_pendulum_odes(self):
        rhs = Axiom.pendulum_odes(g=9.81, L=1.0, b=0.0)
        result = rhs(0, [0.0, 1.0])
        assert len(result) == 2
        assert result[0] == 1.0  # theta_dot = omega
        assert result[1] == pytest.approx(-0.0, abs=1e-10)

    def test_lotka_volterra_odes(self):
        rhs = Axiom.lotka_volterra_odes(alpha=1.0, beta=0.1, gamma=1.5, delta=0.075)
        result = rhs(0, [10.0, 5.0])
        assert len(result) == 2
        assert result[0] == pytest.approx(10.0 - 0.1 * 10 * 5)

    def test_pendulum_module_level(self):
        rhs = pendulum_odes()
        assert len(rhs(0, [1.0, 0.0])) == 2

    def test_lotka_module_level(self):
        rhs = lotka_volterra_odes()
        assert len(rhs(0, [10, 2])) == 2

    def test_pendulum_integration(self):
        rhs = Axiom.pendulum_odes(L=1.0, b=0.1)
        ts, ys = Axiom.solve_ivp(rhs, [0.1, 0.0], (0, 10), method='rk4', dt=0.01)
        assert len(ts) > 10
        assert len(ys) == len(ts)
