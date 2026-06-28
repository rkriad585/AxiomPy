import pytest
from axiompy import Axiom


class TestMandelbrot:
    def test_basic_shape(self):
        grid = Axiom.mandelbrot(10, 8, max_iter=20)
        assert len(grid) == 8
        assert len(grid[0]) == 10

    def test_origin_in_set(self):
        grid = Axiom.mandelbrot(5, 5, (-2, 1), (-1.5, 1.5), max_iter=100)
        # c = 0 is in the set
        cx = 2  # approx center
        cy = 2
        assert grid[cy][cx] == 100


class TestJulia:
    def test_basic_shape(self):
        grid = Axiom.julia(-0.7 + 0.27j, 10, 8, max_iter=20)
        assert len(grid) == 8
        assert len(grid[0]) == 10


class TestLogisticMap:
    def test_length(self):
        orbit = Axiom.logistic_map(3.5, 0.5, 10)
        assert len(orbit) == 11

    def test_fixed_ranges(self):
        orbit = Axiom.logistic_map(2.0, 0.5, 10)
        for x in orbit:
            assert 0 <= x <= 1

    def test_chaotic(self):
        orbit = Axiom.logistic_map(4.0, 0.2, 100)
        assert orbit[-1] != pytest.approx(orbit[-2], abs=1e-6)


class TestBifurcationDiagram:
    def test_output_format(self):
        points = Axiom.bifurcation_diagram(2.5, 4.0, r_steps=10, n_transient=5, n_plot=5)
        assert len(points) > 0
        r_vals, x_vals = zip(*points)
        assert all(2.5 <= r <= 4.0 for r in r_vals)


class TestLyapunov:
    def test_stable_fixed_point(self):
        """For r=2.0, logistic map has a stable fixed point, λ < 0."""
        f = lambda x: 2.0 * x * (1 - x)
        lam = Axiom.lyapunov_exponent(f, 0.5, n=500)
        assert lam < 0

    def test_chaotic(self):
        """For r=4.0, logistic map is chaotic, λ > 0."""
        f = lambda x: 4.0 * x * (1 - x)
        lam = Axiom.lyapunov_exponent(f, 0.5, n=1000)
        assert lam > 0
