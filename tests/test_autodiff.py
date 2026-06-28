import pytest
from axiompy import Axiom


class TestAutoDiff:
    def test_add(self):
        x = Axiom.autodiff.Variable(2.0)
        y = Axiom.autodiff.Variable(3.0)
        z = x + y
        z.backward()
        assert z.value == pytest.approx(5.0)
        assert x.grad == pytest.approx(1.0)
        assert y.grad == pytest.approx(1.0)

    def test_mul(self):
        x = Axiom.autodiff.Variable(2.0)
        y = Axiom.autodiff.Variable(3.0)
        z = x * y
        z.backward()
        assert z.value == pytest.approx(6.0)
        assert x.grad == pytest.approx(3.0)
        assert y.grad == pytest.approx(2.0)

    def test_pow(self):
        x = Axiom.autodiff.Variable(3.0)
        z = x ** 2
        z.backward()
        assert z.value == pytest.approx(9.0)
        assert x.grad == pytest.approx(6.0)

    def test_neg(self):
        x = Axiom.autodiff.Variable(2.0)
        z = -x
        z.backward()
        assert z.value == pytest.approx(-2.0)
        assert x.grad == pytest.approx(-1.0)

    def test_sub(self):
        x = Axiom.autodiff.Variable(5.0)
        y = Axiom.autodiff.Variable(3.0)
        z = x - y
        z.backward()
        assert z.value == pytest.approx(2.0)
        assert x.grad == pytest.approx(1.0)
        assert y.grad == pytest.approx(-1.0)

    def test_div(self):
        x = Axiom.autodiff.Variable(6.0)
        y = Axiom.autodiff.Variable(2.0)
        z = x / y
        z.backward()
        assert z.value == pytest.approx(3.0)
        assert x.grad == pytest.approx(0.5)
        assert y.grad == pytest.approx(-1.5)

    def test_radd(self):
        x = Axiom.autodiff.Variable(2.0)
        z = 5.0 + x
        z.backward()
        assert z.value == pytest.approx(7.0)
        assert x.grad == pytest.approx(1.0)

    def test_rmul(self):
        x = Axiom.autodiff.Variable(2.0)
        z = 3.0 * x
        z.backward()
        assert z.value == pytest.approx(6.0)
        assert x.grad == pytest.approx(3.0)

    def test_rsub(self):
        x = Axiom.autodiff.Variable(2.0)
        z = 5.0 - x
        z.backward()
        assert z.value == pytest.approx(3.0)
        assert x.grad == pytest.approx(-1.0)

    def test_rtruediv(self):
        x = Axiom.autodiff.Variable(2.0)
        z = 6.0 / x
        z.backward()
        assert z.value == pytest.approx(3.0)
        assert x.grad == pytest.approx(-1.5)

    def test_sin(self):
        import math
        x = Axiom.autodiff.Variable(math.pi / 2)
        z = Axiom.autodiff.sin(x)
        z.backward()
        assert z.value == pytest.approx(1.0, rel=1e-5)
        assert x.grad == pytest.approx(0.0, abs=1e-5)

    def test_exp(self):
        x = Axiom.autodiff.Variable(0.0)
        z = Axiom.autodiff.exp(x)
        z.backward()
        assert z.value == pytest.approx(1.0)
        assert x.grad == pytest.approx(1.0)

    def test_log(self):
        x = Axiom.autodiff.Variable(2.0)
        z = Axiom.autodiff.log(x)
        z.backward()
        assert z.value == pytest.approx(0.693147, rel=1e-5)
        assert x.grad == pytest.approx(0.5)

    def test_tanh(self):
        x = Axiom.autodiff.Variable(0.0)
        z = Axiom.autodiff.tanh(x)
        z.backward()
        assert z.value == pytest.approx(0.0)
        assert x.grad == pytest.approx(1.0)

    def test_sigmoid(self):
        x = Axiom.autodiff.Variable(0.0)
        z = Axiom.autodiff.sigmoid(x)
        z.backward()
        assert z.value == pytest.approx(0.5)
        assert x.grad == pytest.approx(0.25)

    def test_sqrt(self):
        x = Axiom.autodiff.Variable(4.0)
        z = Axiom.autodiff.sqrt(x)
        z.backward()
        assert z.value == pytest.approx(2.0)
        assert x.grad == pytest.approx(0.25)

    def test_gradient_descent(self):
        f = lambda x: x ** 2 + 2 * x + 1
        minimum = Axiom.autodiff.gradient_descent(f, start=5.0, lr=0.1, steps=100)
        assert minimum == pytest.approx(-1.0, rel=1e-4)

    def test_adam(self):
        f = lambda x: x ** 2 + 2 * x + 1
        minimum = Axiom.autodiff.adam(f, start=5.0, lr=0.5, steps=200)
        assert minimum == pytest.approx(-1.0, rel=1e-2)

    def test_hessian(self):
        f = lambda x: x ** 3
        h = Axiom.autodiff.hessian(f, 2.0)
        assert h == pytest.approx(12.0, rel=1e-3)

    def test_tanh_method(self):
        x = Axiom.autodiff.Variable(0.0)
        z = x.tanh()
        z.backward()
        assert z.value == pytest.approx(0.0)
        assert x.grad == pytest.approx(1.0)

    def test_sigmoid_method(self):
        x = Axiom.autodiff.Variable(0.0)
        z = x.sigmoid()
        z.backward()
        assert z.value == pytest.approx(0.5)
        assert x.grad == pytest.approx(0.25)

    def test_to_ascii(self):
        x = Axiom.autodiff.Variable(2.0)
        y = Axiom.autodiff.Variable(3.0)
        z = x * y
        z.backward()
        ascii_repr = z.to_ascii()
        assert "Variable" in ascii_repr
        assert "value=" in ascii_repr
        assert "grad=" in ascii_repr
