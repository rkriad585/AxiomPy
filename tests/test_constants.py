import math

from axiompy import Axiom


def test_pi():
    assert Axiom.constants.PI == math.pi


def test_e():
    assert Axiom.constants.E == math.e


def test_physical():
    assert Axiom.constants.C == 299792458.0
    assert Axiom.constants.G == 6.67430e-11


def test_list_all():
    consts = Axiom.constants.list_all()
    assert "PI" in consts
    assert "C" in consts
