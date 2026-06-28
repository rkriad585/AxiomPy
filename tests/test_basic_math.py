import pytest

from axiompy import Axiom


def test_add():
    assert Axiom.math.add(5, 3) == 8


def test_sub():
    assert Axiom.math.sub(10, 4) == 6


def test_mul():
    assert Axiom.math.mul(6, 7) == 42


def test_div():
    assert Axiom.math.div(10, 3) == pytest.approx(3.33333, rel=1e-4)


def test_percentage():
    assert Axiom.math.percentage(25, 100) == 25.0


def test_factorial():
    assert Axiom.math.factorial(5) == 120


def test_is_prime():
    assert Axiom.math.is_prime(7)
    assert not Axiom.math.is_prime(4)


def test_sqrt():
    assert Axiom.math.sqrt(9) == 3.0


def test_power():
    assert Axiom.math.power(2, 10) == 1024


def test_fraction():
    f = Axiom.Fraction(1, 3)
    g = Axiom.Fraction(2, 3)
    assert f + g == Axiom.Fraction(1, 1)
