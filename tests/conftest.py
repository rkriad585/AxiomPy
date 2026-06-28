import pytest

from axiompy import Axiom


@pytest.fixture
def v1():
    return Axiom.Vector([1.0, 2.0, 3.0])


@pytest.fixture
def v2():
    return Axiom.Vector([4.0, 5.0, 6.0])


@pytest.fixture
def m1():
    return Axiom.Matrix([[1, 2], [3, 4]])


@pytest.fixture
def m2():
    return Axiom.Matrix([[5, 6], [7, 8]])
