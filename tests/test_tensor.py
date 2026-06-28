import pytest
from axiompy import Axiom
from axiompy.tensor import Tensor


class TestTensorCreation:
    def test_from_list(self):
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)
        assert t.ndim == 2
        assert t.size == 4

    def test_from_ndarray(self):
        import numpy as np
        t = Tensor(np.array([1, 2, 3]))
        assert t.shape == (3,)

    def test_factory_zeros(self):
        t = Tensor.zeros(2, 3)
        assert t.shape == (2, 3)
        assert t.sum().to_list() == 0.0

    def test_factory_ones(self):
        t = Tensor.ones(4)
        assert t.shape == (4,)
        assert t.to_list() == [1.0, 1.0, 1.0, 1.0]

    def test_factory_eye(self):
        t = Tensor.eye(3)
        assert t.shape == (3, 3)
        assert t.to_list() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def test_factory_linspace(self):
        t = Tensor.linspace(0, 1, 5)
        assert t.shape == (5,)
        assert t.to_list() == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_factory_arange(self):
        t = Tensor.arange(0, 5, 2)
        assert t.to_list() == [0.0, 2.0, 4.0]

    def test_facade_access(self):
        t = Axiom.Tensor([1, 2, 3])
        assert isinstance(t, Tensor)


class TestTensorArithmetic:
    def test_add(self):
        a = Tensor([1, 2])
        b = Tensor([3, 4])
        c = a + b
        assert c.to_list() == [4.0, 6.0]

    def test_sub(self):
        a = Tensor([5, 6])
        b = Tensor([3, 4])
        c = a - b
        assert c.to_list() == [2.0, 2.0]

    def test_mul(self):
        a = Tensor([1, 2])
        b = Tensor([3, 4])
        c = a * b
        assert c.to_list() == [3.0, 8.0]

    def test_div(self):
        a = Tensor([2, 4])
        b = Tensor([1, 2])
        c = a / b
        assert c.to_list() == [2.0, 2.0]

    def test_scalar_mult(self):
        t = Tensor([1, 2, 3])
        r = t * 2
        assert r.to_list() == [2.0, 4.0, 6.0]

    def test_neg(self):
        t = -Tensor([1, -2])
        assert t.to_list() == [-1.0, 2.0]

    def test_pow(self):
        t = Tensor([1, 2, 3]) ** 2
        assert t.to_list() == [1.0, 4.0, 9.0]

    def test_broadcasting(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = Tensor([1, 2, 3])
        c = a + b
        assert c.to_list() == [[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]]


class TestTensorIndexing:
    def test_getitem(self):
        t = Tensor([[1, 2], [3, 4]])
        assert t[0, 1].to_list() == 2.0
        row = t[0]
        assert row.shape == (2,)

    def test_setitem(self):
        t = Tensor.zeros(2, 2)
        t[0, 0] = 42
        assert t[0, 0].to_list() == 42.0

    def test_len(self):
        t = Tensor([[1, 2], [3, 4], [5, 6]])
        assert len(t) == 3

    def test_iter(self):
        t = Tensor([10, 20, 30])
        vals = [x.to_list() for x in t]
        assert vals == [10.0, 20.0, 30.0]


class TestTensorReshape:
    def test_reshape(self):
        t = Tensor.arange(0, 6).reshape(2, 3)
        assert t.shape == (2, 3)

    def test_flatten(self):
        t = Tensor([[1, 2], [3, 4]])
        f = t.flatten()
        assert f.shape == (4,)
        assert f.to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_transpose(self):
        t = Tensor([[1, 2], [3, 4]])
        tt = t.T
        assert tt.to_list() == [[1.0, 3.0], [2.0, 4.0]]


class TestTensorReductions:
    def test_sum(self):
        t = Tensor([1, 2, 3])
        assert t.sum().to_list() == 6.0

    def test_mean(self):
        t = Tensor([1, 2, 3, 4])
        assert t.mean().to_list() == 2.5

    def test_max_min(self):
        t = Tensor([3, 1, 4, 1, 5])
        assert t.max().to_list() == 5.0
        assert t.min().to_list() == 1.0

    def test_std(self):
        t = Tensor([1, 1, 1])
        assert t.std().to_list() == 0.0


class TestTensorOps:
    def test_contract(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = Tensor.contract(a, b, axes=1)
        assert c.shape == (2, 2)
        assert c.to_list() == [[19.0, 22.0], [43.0, 50.0]]

    def test_outer(self):
        a = Tensor([1, 2])
        b = Tensor([3, 4, 5])
        o = Tensor.outer(a, b)
        assert o.shape == (2, 3)
        assert o.to_list() == [[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]

    def test_kronecker(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[0, 5], [6, 7]])
        k = Tensor.kronecker(a, b)
        assert k.shape == (4, 4)

    def test_einsum(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = Tensor.einsum("ij,jk->ik", a, b)
        assert c.to_list() == [[19.0, 22.0], [43.0, 50.0]]

    def test_dot(self):
        a = Tensor([1, 2])
        b = Tensor([[3], [4]])
        c = a.dot(b)
        assert c.shape == (1,)

    def test_eq(self):
        a = Tensor([1, 2])
        b = Tensor([1, 2])
        c = Tensor([3, 4])
        assert a == b
        assert a != c

    def test_allclose(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([1.0001, 2.0001])
        assert a.allclose(b, rtol=1e-3)
        assert not a.allclose(b, rtol=1e-6)
