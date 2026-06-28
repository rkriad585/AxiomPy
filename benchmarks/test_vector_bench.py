"""Benchmarks for Vector operations vs numpy."""

import numpy as np
import pytest

from axiompy import Axiom

A = Axiom


@pytest.mark.benchmark(group="vector-creation")
def test_vector_creation(benchmark):
    data = [float(i) for i in range(1000)]
    result = benchmark(A.Vector, data)
    assert len(result) == 1000


@pytest.mark.benchmark(group="vector-creation")
def test_numpy_creation(benchmark):
    data = [float(i) for i in range(1000)]
    result = benchmark(np.array, data)
    assert len(result) == 1000


@pytest.mark.benchmark(group="vector-dot")
def test_vector_dot(benchmark):
    v = A.Vector([float(i) for i in range(1000)])
    w = A.Vector([float(i * 2) for i in range(1000)])
    result = benchmark(v.dot, w)
    assert result > 0


@pytest.mark.benchmark(group="vector-dot")
def test_numpy_dot(benchmark):
    v = np.array([float(i) for i in range(1000)])
    w = np.array([float(i * 2) for i in range(1000)])
    result = benchmark(np.dot, v, w)
    assert result > 0


@pytest.mark.benchmark(group="vector-norm")
def test_vector_norm(benchmark):
    v = A.Vector([float(i) for i in range(1000)])
    result = benchmark(v.magnitude)
    assert result > 0


@pytest.mark.benchmark(group="vector-norm")
def test_numpy_norm(benchmark):
    v = np.array([float(i) for i in range(1000)])
    result = benchmark(np.linalg.norm, v)
    assert result > 0


@pytest.mark.benchmark(group="vector-add")
def test_vector_add(benchmark):
    v = A.Vector([float(i) for i in range(1000)])
    w = A.Vector([float(i * 2) for i in range(1000)])
    result = benchmark(v.__add__, w)
    assert len(result) == 1000


@pytest.mark.benchmark(group="vector-add")
def test_numpy_add(benchmark):
    v = np.array([float(i) for i in range(1000)])
    w = np.array([float(i * 2) for i in range(1000)])
    result = benchmark(v.__add__, w)
    assert len(result) == 1000
