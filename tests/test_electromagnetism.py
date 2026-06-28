import pytest

from axiompy import Axiom
from axiompy.electromagnetism import Electromagnetism


class TestElectromagnetism:
    def test_electric_potential_single(self):
        charge = Electromagnetism.Charge(1e-9, (0, 0, 0))
        V = Electromagnetism.electric_potential([charge], (1, 0, 0))
        expected = Electromagnetism.K_E * 1e-9 / 1.0
        assert pytest.approx(expected, rel=1e-6) == V

    def test_electric_potential_dipole(self):
        charges = [
            Electromagnetism.Charge(1e-9, (0, 0, 0)),
            Electromagnetism.Charge(-1e-9, (1, 0, 0)),
        ]
        V = Electromagnetism.electric_potential(charges, (0.5, 0.5, 0))
        assert isinstance(V, float)

    def test_magnetic_field(self):
        charge = Electromagnetism.Charge(1e-9, (0, 0, 0))
        B = Electromagnetism.calculate_magnetic_field(
            [charge], (0, 0, 1), [(1, 0, 0)]
        )
        assert B.magnitude() > 0

    def test_dipole_moment(self):
        charges = [
            Electromagnetism.Charge(1e-9, (0, 0, 0)),
            Electromagnetism.Charge(-1e-9, (1, 0, 0)),
        ]
        p = Electromagnetism.dipole_moment(charges)
        assert isinstance(p, Axiom.Vector)
        assert p.to_list() == [-1e-9, 0.0, 0.0]

    def test_combine_fields(self):
        from axiompy import Axiom
        fields = [
            Axiom.Vector([1, 0, 0]),
            Axiom.Vector([0, 2, 0]),
            Axiom.Vector([0, 0, 3]),
        ]
        net = Electromagnetism.combine_fields(fields)
        assert net.to_list() == [1.0, 2.0, 3.0]
