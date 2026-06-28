"""Benchmarks for Matrix operations vs numpy."""

import numpy as np
import pytest

from axiompy import Axiom

A = Axiom
N = 50


@pytest.mark.benchmark(group="matrix-matmul")
def test_matrix_matmul(benchmark):
    data1 = [[float(i * N + j) for j in range(N)] for i in range(N)]
    data2 = [[float(j * N + i) for j in range(N)] for i in range(N)]
    m1 = A.Matrix(data1)
    m2 = A.Matrix(data2)
    result = benchmark(m1.__matmul__, m2)
    assert result.shape == (N, N)


@pytest.mark.benchmark(group="matrix-matmul")
def test_numpy_matmul(benchmark):
    data1 = np.array([[float(i * N + j) for j in range(N)] for i in range(N)])
    data2 = np.array([[float(j * N + i) for j in range(N)] for i in range(N)])
    result = benchmark(np.matmul, data1, data2)
    assert result.shape == (N, N)


@pytest.mark.benchmark(group="matrix-determinant")
def test_matrix_determinant(benchmark):
    np.random.seed(42)
    data = np.random.randn(N, N) + np.eye(N) * N
    m = A.Matrix(data.tolist())
    result = benchmark(lambda: m.determinant)
    assert result is not None


@pytest.mark.benchmark(group="matrix-determinant")
def test_numpy_determinant(benchmark):
    np.random.seed(42)
    data = np.random.randn(N, N) + np.eye(N) * N
    result = benchmark(np.linalg.det, data)
    assert result is not None


@pytest.mark.benchmark(group="matrix-inverse")
def test_matrix_inverse(benchmark):
    np.random.seed(42)
    data = np.random.randn(N, N) + np.eye(N) * N
    m = A.Matrix(data.tolist())
    result = benchmark(lambda: m.inverse)
    assert result is not None


@pytest.mark.benchmark(group="matrix-inverse")
def test_numpy_inverse(benchmark):
    np.random.seed(42)
    data = np.random.randn(N, N) + np.eye(N) * N
    result = benchmark(np.linalg.inv, data)
    assert result is not None
