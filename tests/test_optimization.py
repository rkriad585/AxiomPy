import pytest
from axiompy import Axiom


class TestOptimization:
    def test_gradient_descent(self):
        f = lambda x: x ** 2 + 2 * x + 1
        df = lambda x: 2 * x + 2
        result = Axiom.optimization.gradient_descent(f, df, start=5.0, lr=0.1, steps=100)
        assert result == pytest.approx(-1.0, rel=1e-4)

    def test_newton_method(self):
        f = lambda x: x ** 2 + 2 * x + 1
        df = lambda x: 2 * x + 2
        d2f = lambda x: 2.0
        result = Axiom.optimization.newton_method(f, df, d2f, start=5.0, steps=10)
        assert result == pytest.approx(-1.0, rel=1e-4)

    def test_bisection(self):
        result = Axiom.optimization.bisection(lambda x: x ** 2 - 4, 0, 5)
        assert result == pytest.approx(2.0, rel=1e-4)

    def test_bisection_no_root(self):
        result = Axiom.optimization.bisection(lambda x: x ** 2 + 1, 0, 5)
        assert result is None

    def test_newton_zero_denominator(self):
        df = lambda x: 2 * x
        d2f = lambda x: 0.0
        result = Axiom.optimization.newton_method(
            lambda x: x ** 2, df, d2f, start=1.0, steps=10
        )
        assert result == pytest.approx(1.0)

    def test_golden_section(self):
        result = Axiom.optimization.golden_section(lambda x: x ** 2, -2, 2, tol=1e-6)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_conjugate_gradient(self):
        A = Axiom.Matrix([[4, 1], [1, 3]])
        b = Axiom.Vector([1, 2])
        x = Axiom.optimization.conjugate_gradient(A, b)
        residual = (A @ x - b).magnitude()
        assert residual == pytest.approx(0.0, abs=1e-6)

    def test_nelder_mead(self):
        result = Axiom.optimization.nelder_mead(lambda p: p[0] ** 2 + p[1] ** 2,
                                                  [10.0, 10.0], max_iter=2000)
        assert result[0] == pytest.approx(0.0, abs=5e-4)
        assert result[1] == pytest.approx(0.0, abs=5e-4)

    def test_lbfgs(self):
        f = lambda p: p[0] ** 2 + p[1] ** 2
        grad = lambda p: [2 * p[0], 2 * p[1]]
        result = Axiom.optimization.lbfgs(f, grad, [5.0, 5.0], max_iter=50)
        assert result[0] == pytest.approx(0.0, abs=1e-4)
        assert result[1] == pytest.approx(0.0, abs=1e-4)

    def test_simulated_annealing(self):
        f = lambda p: p[0] ** 2 + p[1] ** 2
        result = Axiom.optimization.simulated_annealing(f, [5.0, 5.0],
                                                          temp=5.0, cooling=0.95, max_iter=2000)
        assert result[0] == pytest.approx(0.0, abs=1.0)
        assert result[1] == pytest.approx(0.0, abs=1.0)
