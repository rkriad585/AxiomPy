"""Tests for the sparse matrix module."""

import pytest
import numpy as np

from axiompy import Axiom
from axiompy._base import AxiomError
from axiompy.matrix import Matrix
from axiompy.vector import Vector
from axiompy._sparse import SparseMatrix


class TestSparseMatrixConstruction:
    def test_from_dense(self):
        M = Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        sp = SparseMatrix.from_dense(M)
        assert sp.shape == (3, 3)
        assert sp.nnz == 3
        assert sp.density == pytest.approx(3 / 9)

    def test_from_dense_with_tol(self):
        M = Matrix([[1, 1e-10], [1e-10, 2]])
        sp = SparseMatrix.from_dense(M, tol=1e-6)
        assert sp.nnz == 2

    def test_identity(self):
        sp = SparseMatrix.identity(5)
        assert sp.shape == (5, 5)
        assert sp.nnz == 5

    def test_to_dense(self):
        sp = SparseMatrix([0, 1], [0, 1], [3.0, 4.0], (3, 3))
        dense = sp.to_dense()
        assert dense.to_list() == [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
        ]

    def test_to_coo(self):
        rows, cols, data = SparseMatrix([0], [2], [9.0], (2, 3)).to_coo()
        assert rows.tolist() == [0]
        assert cols.tolist() == [2]
        assert data.tolist() == [9.0]

    def test_matrix_to_sparse(self):
        M = Matrix([[1, 0], [0, 2]])
        sp = M.to_sparse()
        assert sp.nnz == 2
        assert sp.shape == (2, 2)

    def test_invalid_shape_mismatch(self):
        with pytest.raises(AxiomError):
            SparseMatrix([0, 1], [0, 1], [1.0], (2, 2))

    def test_index_out_of_bounds(self):
        with pytest.raises(AxiomError):
            SparseMatrix([5], [0], [1.0], (3, 3))


class TestSparseMatrixArithmetic:
    def test_add_sparse(self):
        a = SparseMatrix([0, 1], [0, 1], [1.0, 2.0], (2, 2))
        b = SparseMatrix([0, 1], [1, 0], [3.0, 4.0], (2, 2))
        c = a + b
        assert c.nnz == 4
        assert c.to_dense().to_list() == [[1.0, 3.0], [4.0, 2.0]]

    def test_add_scalar(self):
        sp = SparseMatrix([0, 1], [0, 1], [1.0, 2.0], (2, 2))
        result = sp + 1.0
        assert np.isclose(result.to_dense().to_list(), [[2.0, 1.0], [1.0, 3.0]]).all()

    def test_radd_scalar(self):
        sp = SparseMatrix([0, 1], [0, 1], [1.0, 2.0], (2, 2))
        result = 1.0 + sp
        assert np.isclose(result.to_dense().to_list(), [[2.0, 1.0], [1.0, 3.0]]).all()

    def test_sub_sparse(self):
        a = SparseMatrix([0, 1], [0, 1], [5.0, 5.0], (2, 2))
        b = SparseMatrix([0, 1], [0, 1], [1.0, 2.0], (2, 2))
        c = a - b
        assert c.to_dense().to_list() == [[4.0, 0.0], [0.0, 3.0]]

    def test_sub_scalar(self):
        sp = SparseMatrix([0, 1], [0, 1], [5.0, 5.0], (2, 2))
        result = sp - 2.0
        assert np.isclose(result.to_dense().to_list(), [[3.0, -2.0], [-2.0, 3.0]]).all()

    def test_rsub_scalar(self):
        sp = SparseMatrix([0, 1], [0, 1], [1.0, 2.0], (2, 2))
        result = 10.0 - sp
        assert np.isclose(result.to_dense().to_list(), [[9.0, 10.0], [10.0, 8.0]]).all()

    def test_mul_scalar(self):
        sp = SparseMatrix([0], [0], [3.0], (2, 2))
        result = sp * 2.0
        assert result.to_dense().to_list() == [[6.0, 0.0], [0.0, 0.0]]

    def test_rmul_scalar(self):
        sp = SparseMatrix([0], [0], [3.0], (2, 2))
        result = 2.0 * sp
        assert result.to_dense().to_list() == [[6.0, 0.0], [0.0, 0.0]]

    def test_neg(self):
        sp = SparseMatrix([0, 1], [0, 1], [1.0, 2.0], (2, 2))
        result = -sp
        assert result.to_dense().to_list() == [[-1.0, 0.0], [0.0, -2.0]]


class TestSparseMatrixMatmul:
    def test_matmul_vector(self):
        sp = SparseMatrix([0, 1], [0, 1], [2.0, 3.0], (2, 2))
        v = Vector([1.0, 4.0])
        result = sp @ v
        assert result.to_list() == [2.0, 12.0]

    def test_matmul_vector_non_square(self):
        sp = SparseMatrix([0], [0], [5.0], (2, 3))
        v = Vector([1.0, 2.0, 3.0])
        result = sp @ v
        assert result.to_list() == [5.0, 0.0]

    def test_matmul_sparse(self):
        a = SparseMatrix([0, 1], [0, 1], [1.0, 1.0], (2, 2))
        b = SparseMatrix([0, 1], [0, 1], [2.0, 3.0], (2, 2))
        c = a @ b
        assert c.to_dense().to_list() == [[2.0, 0.0], [0.0, 3.0]]

    def test_matmul_shape_mismatch(self):
        a = SparseMatrix([0], [0], [1.0], (2, 2))
        v = Vector([1.0])
        with pytest.raises(AxiomError):
            a @ v

    def test_matmul_dense_matrix(self):
        sp = SparseMatrix([0, 1], [0, 1], [1.0, 1.0], (2, 2))
        M = Matrix([[2.0, 0.0], [0.0, 3.0]])
        result = sp @ M
        assert result.to_list() == [[2.0, 0.0], [0.0, 3.0]]


class TestSparseMatrixProperties:
    def test_repr(self):
        sp = SparseMatrix([0], [0], [1.0], (3, 3))
        r = repr(sp)
        assert "SparseMatrix" in r
        assert "nnz=1" in r
        assert "density" in r
