import math

from axiompy import Axiom


def test_pi():
    assert Axiom.constants.PI == math.pi


def test_e():
    assert Axiom.constants.E == math.e


def test_physical():
    assert Axiom.constants.C == 299792458.0
    assert Axiom.constants.G == 6.67430e-11


def test_new_physics_constants():
    assert Axiom.constants.R == 8.314462618
    assert Axiom.constants.E_CHARGE == 1.602176634e-19
    assert Axiom.constants.HBAR == 1.0545718176469264e-34
    assert Axiom.constants.ALPHA == 0.0072973525693
    assert Axiom.constants.RYDBERG == 10973731.56816
    assert Axiom.constants.STEFAN_BOLTZMANN == 5.670374419e-08
    assert Axiom.constants.WIEN == 0.002897771955


def test_mass_constants():
    assert Axiom.constants.M_E == 9.1093837015e-31
    assert Axiom.constants.M_P == 1.67262192369e-27
    assert Axiom.constants.M_N == 1.67492749804e-27
    assert Axiom.constants.M_MU == 1.883531627e-28


def test_astronomy_constants():
    assert Axiom.constants.AU == 149597870700.0
    assert Axiom.constants.PARSEC == 3.085677581491367e+16
    assert Axiom.constants.LIGHT_YEAR == 9460730472580800.0
    assert Axiom.constants.SOLAR_MASS == 1.98847e+30
    assert Axiom.constants.EARTH_MASS == 5.9722e+24


def test_time_constants():
    assert Axiom.constants.MINUTE == 60.0
    assert Axiom.constants.HOUR == 3600.0
    assert Axiom.constants.DAY == 86400.0
    assert Axiom.constants.YEAR == 31557600.0


def test_len():
    assert len(Axiom.constants) == 35


def test_contains():
    assert "PI" in Axiom.constants
    assert "C" in Axiom.constants
    assert "SOLAR_MASS" in Axiom.constants


def test_list_by_category():
    cats = Axiom.constants.list_by_category()
    assert "Math" in cats
    assert "Physics" in cats
    assert "Astronomy" in cats
    assert "PI" in cats["Math"]
    assert "C" in cats["Physics"]
    assert "SOLAR_MASS" in cats["Astronomy"]


def test_find_exact():
    results = Axiom.constants.find("PI")
    assert "PI" in results


def test_find_partial():
    results = Axiom.constants.find("mass")
    assert "SOLAR_MASS" in results
    assert "EARTH_MASS" in results


def test_find_case_insensitive():
    # "mass" matches M_E, M_P, M_N, SOLAR_MASS, EARTH_MASS (case-insensitive)
    results = Axiom.constants.find("MASS")
    assert "SOLAR_MASS" in results


def test_find_empty():
    results = Axiom.constants.find("xyznonexistent")
    assert results == {}


def test_tau():
    assert Axiom.constants.TAU == 2 * math.pi
