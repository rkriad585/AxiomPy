import pytest
from axiompy import Axiom
from axiompy._base import AxiomError


class TestVector:
    def test_creation(self):
        v = Axiom.Vector([1.0, 2.0, 3.0])
        assert len(v) == 3
        assert v.to_list() == [1.0, 2.0, 3.0]

    def test_neg(self, v1):
        nv = -v1
        assert nv.to_list() == [-1.0, -2.0, -3.0]

    def test_abs(self, v1):
        assert abs(v1) == pytest.approx(3.741657, rel=1e-5)

    def test_add(self, v1, v2):
        v3 = v1 + v2
        assert v3.to_list() == [5.0, 7.0, 9.0]

    def test_sub(self, v1, v2):
        v3 = v1 - v2
        assert v3.to_list() == [-3.0, -3.0, -3.0]

    def test_scalar_mul(self, v1):
        v3 = v1 * 2.0
        assert v3.to_list() == [2.0, 4.0, 6.0]

    def test_rmul(self, v1):
        v3 = 2.0 * v1
        assert v3.to_list() == [2.0, 4.0, 6.0]

    def test_dot_product(self, v1, v2):
        d = v1 * v2
        assert d == pytest.approx(32.0)

    def test_truediv(self, v1):
        v3 = v1 / 2.0
        assert v3.to_list() == [0.5, 1.0, 1.5]

    def test_eq(self, v1):
        v3 = Axiom.Vector([1.0, 2.0, 3.0])
        assert v1 == v3

    def test_ne(self, v1, v2):
        assert v1 != v2

    def test_magnitude(self, v1):
        assert v1.magnitude() == pytest.approx(3.741657, rel=1e-5)

    def test_norm_l2(self, v1):
        assert v1.norm() == pytest.approx(3.741657, rel=1e-5)

    def test_norm_l1(self, v1):
        assert v1.norm(1) == pytest.approx(6.0)

    def test_normalize(self, v1):
        n = v1.normalize()
        assert n.magnitude() == pytest.approx(1.0)

    def test_angle_between(self, v1, v2):
        angle = v1.angle_between(v2)
        assert angle == pytest.approx(0.225726, rel=1e-4)

    def test_angle_between_degrees(self, v1, v2):
        angle = v1.angle_between(v2, in_degrees=True)
        assert angle == pytest.approx(12.933, rel=1e-2)

    def test_cross(self):
        a = Axiom.Vector([1, 0, 0])
        b = Axiom.Vector([0, 1, 0])
        c = a.cross(b)
        assert c.to_list() == [0.0, 0.0, 1.0]

    def test_cross_invalid(self, v1):
        with pytest.raises(AxiomError):
            v1.cross(Axiom.Vector([1, 2]))

    def test_dot_method(self, v1, v2):
        assert v1.dot(v2) == pytest.approx(32.0)

    def test_project_onto(self):
        a = Axiom.Vector([3, 4, 0])
        b = Axiom.Vector([1, 0, 0])
        proj = a.project_onto(b)
        assert proj.to_list() == [3.0, 0.0, 0.0]

    def test_static_dot(self, v1, v2):
        assert Axiom.Vector.dot_static(v1, v2) == pytest.approx(32.0)

    def test_static_cross(self):
        a = Axiom.Vector([1, 0, 0])
        b = Axiom.Vector([0, 1, 0])
        c = Axiom.Vector.cross_static(a, b)
        assert c.to_list() == [0.0, 0.0, 1.0]

    def test_getitem(self, v1):
        assert v1[0] == 1.0
        assert v1[2] == 3.0

    def test_zero_vector(self):
        z = Axiom.Vector([0, 0, 0])
        assert z.magnitude() == 0.0
        n = z.normalize()
        assert n.to_list() == [0.0, 0.0, 0.0]

    def test_vector_not_implemented(self, v1):
        result = v1.__matmul__(5)
        assert result is NotImplemented
