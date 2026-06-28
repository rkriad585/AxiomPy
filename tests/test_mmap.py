"""Tests for memory-mapped array and dtype support."""

import numpy as np

from axiompy import Axiom
from axiompy._mmap import MmapArray
from axiompy.matrix import Matrix


class TestDtype:
    def test_vector_dtype(self):
        v = Axiom.Vector([1, 2, 3])
        assert v.dtype == np.float64

    def test_vector_astype(self):
        v = Axiom.Vector([1, 2, 3])
        v32 = v.astype("float32")
        assert v32.dtype == np.float32
        assert v32.to_list() == [1.0, 2.0, 3.0]

    def test_matrix_dtype(self):
        M = Axiom.Matrix([[1, 2], [3, 4]])
        assert M.dtype == np.float64

    def test_matrix_astype(self):
        M = Axiom.Matrix([[1, 2], [3, 4]])
        m32 = M.astype("float32")
        assert m32.dtype == np.float32
        assert m32.to_list() == [[1.0, 2.0], [3.0, 4.0]]

    def test_astype_int32(self):
        v = Axiom.Vector([1.5, 2.7])
        vi = v.astype("int32")
        assert vi.dtype == np.int32
        assert vi.to_list() == [1.0, 2.0]


class TestMmapArray:
    def test_create_zeros(self, tmp_path):
        path = tmp_path / "test.mmap"
        mm = MmapArray.zeros((4, 3), path=path)
        mm.open()
        arr = mm[:]
        assert arr.shape == (4, 3)
        assert arr.sum() == 0.0
        mm.close()

    def test_from_array(self, tmp_path):
        path = tmp_path / "test_from.mmap"
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        mm = MmapArray.from_array(arr, path=path)
        mm.open()
        assert np.allclose(mm[:], arr)
        mm.close()

    def test_shape_and_dtype(self, tmp_path):
        path = tmp_path / "shape.mmap"
        mm = MmapArray.zeros((10, 5), dtype="float32", path=path)
        mm.open()
        assert mm.shape == (10, 5)
        assert mm.dtype == np.float32
        assert mm.ndim == 2
        assert mm.size == 50
        mm.close()

    def test_read_rows(self, tmp_path):
        path = tmp_path / "rows.mmap"
        arr = np.arange(20.0).reshape(5, 4)
        mm = MmapArray.from_array(arr, path=path)
        mm.open()
        chunk = mm.read_rows(1, 4)
        assert chunk.shape == (3, 4)
        assert np.allclose(chunk, arr[1:4])
        mm.close()

    def test_context_manager(self, tmp_path):
        path = tmp_path / "ctx.mmap"
        with MmapArray.zeros((2, 3), path=path) as mm:
            mm[0, 0] = 42.0
            mm.flush()
        with MmapArray(path, (2, 3), mode="r") as mm:
            assert mm[0, 0] == 42.0

    def test_chunked_matmul_vector(self, tmp_path):
        path = tmp_path / "matvec.mmap"
        arr = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
        mm = MmapArray.from_array(arr, path=path)
        mm._chunk_size = 2  # force chunking
        v = Axiom.Vector([2.0, 1.0])
        result = mm.matmul(v)
        assert result.to_list() == [2.0, 2.0, 6.0]

    def test_chunked_matmul_matrix(self, tmp_path):
        path = tmp_path / "matmat.mmap"
        arr = np.eye(3)
        mm = MmapArray.from_array(arr, path=path)
        mm._chunk_size = 2
        b = Matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = mm.matmul(b)
        assert result.to_list() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    def test_chunked_add_scalar(self, tmp_path):
        path = tmp_path / "add.mmap"
        arr = np.ones((4, 3))
        mm = MmapArray.from_array(arr, path=path)
        mm._chunk_size = 2
        mm.add(5.0)
        assert np.allclose(mm[:], 6.0)

    def test_chunked_mul_scalar(self, tmp_path):
        path = tmp_path / "mul.mmap"
        arr = np.ones((4, 3)) * 2.0
        mm = MmapArray.from_array(arr, path=path)
        mm._chunk_size = 2
        mm.mul(3.0)
        assert np.allclose(mm[:], 6.0)

    def test_to_dense(self, tmp_path):
        path = tmp_path / "dense.mmap"
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        mm = MmapArray.from_array(arr, path=path)
        dense = mm.to_dense()
        assert dense.to_list() == [[1.0, 2.0], [3.0, 4.0]]

    def test_matrix_from_mmap(self, tmp_path):
        path = tmp_path / "from_mmap.mmap"
        arr = np.array([[5.0, 6.0], [7.0, 8.0]])
        mm = MmapArray.from_array(arr, path=path)
        M = Axiom.Matrix.from_mmap(mm)
        assert M.to_list() == [[5.0, 6.0], [7.0, 8.0]]

    def test_repr(self, tmp_path):
        path = tmp_path / "repr.mmap"
        mm = MmapArray.zeros((2, 2), path=path)
        r = repr(mm)
        assert "MmapArray" in r
        assert "(2, 2)" in r

    def test_close_and_reopen(self, tmp_path):
        path = tmp_path / "reopen.mmap"
        mm = MmapArray.zeros((2, 2), path=path)
        mm.open()
        mm[0, 0] = 99.0
        mm.close()
        mm2 = MmapArray(path, (2, 2), mode="r")
        mm2.open()
        assert mm2[0, 0] == 99.0
        mm2.close()

    def test_open_twice_noop(self, tmp_path):
        path = tmp_path / "double.mmap"
        mm = MmapArray.zeros((2, 2), path=path)
        mm.open()
        mm.open()  # should be no-op
        assert mm._mmap is not None
        mm.close()
