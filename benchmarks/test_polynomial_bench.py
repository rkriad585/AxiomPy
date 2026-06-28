"""Benchmarks for Polynomial operations vs numpy."""

import numpy as np
import pytest

from axiompy import Axiom

A = Axiom


@pytest.mark.benchmark(group="poly-eval")
def test_poly_eval(benchmark):
    p = A.Polynomial([float(i) for i in range(50)])
    result = benchmark(p, 1.5)
    assert result is not None


@pytest.mark.benchmark(group="poly-eval")
def test_numpy_poly_eval(benchmark):
    coeffs = [float(i) for i in range(50)]
    result = benchmark(np.polyval, coeffs[::-1], 1.5)
    assert result is not None


@pytest.mark.benchmark(group="poly-roots")
def test_poly_roots(benchmark):
    p = A.Polynomial([1.0, -3.0, 2.0])
    result = benchmark(p.roots)
    assert len(result) == 2


@pytest.mark.benchmark(group="poly-roots")
def test_numpy_poly_roots(benchmark):
    coeffs = [1.0, -3.0, 2.0]
    result = benchmark(np.roots, coeffs)
    assert len(result) == 2
