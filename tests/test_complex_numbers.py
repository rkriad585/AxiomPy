import math

import pytest

from axiompy import Axiom


class TestComplexNumber:
    def test_creation(self):
        z = Axiom.ComplexNumber(3, 4)
        assert z.real == 3.0
        assert z.imag == 4.0

    def test_from_polar(self):
        z = Axiom.ComplexNumber.from_polar(1, math.pi / 2)
        assert z.real == pytest.approx(0.0, abs=1e-10)
        assert z.imag == pytest.approx(1.0)

    def test_roots_of_unity(self):
        roots = Axiom.ComplexNumber.roots_of_unity(4)
        assert len(roots) == 4
        for r in roots:
            assert r.modulus() == pytest.approx(1.0)

    def test_conjugate(self):
        z = Axiom.ComplexNumber(1, 2)
        c = z.conjugate
        assert c.real == 1.0
        assert c.imag == -2.0

    def test_modulus(self):
        z = Axiom.ComplexNumber(3, 4)
        assert z.modulus() == pytest.approx(5.0)

    def test_argument(self):
        z = Axiom.ComplexNumber(0, 1)
        assert z.argument() == pytest.approx(math.pi / 2)

    def test_power(self):
        z = Axiom.ComplexNumber(0, 1)
        z2 = z.power(2)
        assert z2.real == pytest.approx(-1.0)
        assert z2.imag == pytest.approx(0.0)

    def test_arithmetic(self):
        a = Axiom.ComplexNumber(1, 2)
        b = Axiom.ComplexNumber(3, 4)
        s = a + b
        assert s.real == 4.0 and s.imag == 6.0
        d = a - b
        assert d.real == -2.0 and d.imag == -2.0
        p = a * b
        assert p.real == -5.0 and p.imag == 10.0
        q = a / b
        assert q.modulus() == pytest.approx(0.447213, rel=1e-5)

    def test_cfft(self):
        x = [1, 0, -1, 0]
        X = Axiom.ComplexNumber.cfft(x)
        assert len(X) == 4

    def test_complex_vector(self):
        v = Axiom.ComplexVector([1+2j, 3+4j])
        assert len(v) == 2
        assert v[0].real == 1.0

    def test_complex_matrix(self):
        m = Axiom.ComplexMatrix([[1+1j, 2+2j], [3+3j, 4+4j]])
        assert m.shape == (2, 2)
