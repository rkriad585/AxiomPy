import math

import pytest

from axiompy import Axiom
from axiompy.geometry import Line, Plane, Point, Sphere


class TestPoint:
    def test_creation_2d(self):
        p = Point(1, 2)
        assert p.x == 1 and p.y == 2

    def test_creation_3d(self):
        p = Point(1, 2, 3)
        assert p.z == 3

    def test_distance(self):
        a = Point(0, 0)
        b = Point(3, 4)
        assert a.distance_to(b) == 5.0

    def test_add_sub(self):
        a = Point(1, 2)
        b = Point(3, 4)
        assert (a + b) == Point(4, 6)
        assert (a - b) == Point(-2, -2)

    def test_scalar_mul(self):
        p = Point(1, 2)
        assert p * 3 == Point(3, 6)
        assert 3 * p == Point(3, 6)

    def test_dot(self):
        a = Point(1, 2)
        b = Point(3, 4)
        assert a.dot(b) == 11

    def test_cross_3d(self):
        a = Point(1, 0, 0)
        b = Point(0, 1, 0)
        c = a.cross(b)
        assert c == Point(0, 0, 1)

    def test_norm(self):
        assert Point(3, 4).norm() == 5.0

    def test_normalize(self):
        u = Point(3, 4).normalize()
        assert u.norm() == pytest.approx(1.0)

    def test_facade(self):
        p = Axiom.Point(1, 2)
        assert isinstance(p, Point)


class TestLine:
    def test_creation(self):
        l = Line(Point(0, 0), Point(1, 1))

    def test_distance_to_point(self):
        l = Line(Point(0, 0), Point(1, 0))
        d = l.distance_to_point(Point(0, 1))
        assert d == 1.0

    def test_intersection(self):
        l1 = Line(Point(0, 0), Point(1, 1))
        l2 = Line(Point(0, 1), Point(1, 0))
        p = l1.intersection(l2)
        assert p == Point(0.5, 0.5)

    def test_no_intersection(self):
        l1 = Line(Point(0, 0), Point(1, 1))
        l2 = Line(Point(0, 1), Point(1, 2))
        assert l1.intersection(l2) is None

    def test_contains(self):
        l = Line(Point(0, 0), Point(2, 0))
        assert l.contains_point(Point(1, 0))
        assert not l.contains_point(Point(1, 1))


class TestPlane:
    def test_creation(self):
        pl = Plane(Point(0, 0, 0), Point(0, 0, 1))

    def test_distance_to_point(self):
        pl = Plane(Point(0, 0, 0), Point(0, 0, 1))
        assert pl.distance_to_point(Point(0, 0, 5)) == 5.0

    def test_contains(self):
        pl = Plane(Point(0, 0, 0), Point(0, 0, 1))
        assert pl.contains_point(Point(1, 2, 0))
        assert not pl.contains_point(Point(0, 0, 1))


class TestSphere:
    def test_volume(self):
        s = Sphere(Point(0, 0, 0), 1)
        assert s.volume() == pytest.approx(4 / 3 * math.pi)

    def test_surface_area(self):
        s = Sphere(Point(0, 0, 0), 1)
        assert s.surface_area() == pytest.approx(4 * math.pi)

    def test_contains(self):
        s = Sphere(Point(0, 0, 0), 1)
        assert s.contains_point(Point(0.5, 0, 0))
        assert not s.contains_point(Point(2, 0, 0))

    def test_intersect_line(self):
        s = Sphere(Point(0, 0, 0), 1)
        l = Line(Point(-2, 0, 0), Point(2, 0, 0))
        pts = s.intersect_line(l)
        assert len(pts) == 2
        assert pts[0] == Point(-1, 0, 0)
        assert pts[1] == Point(1, 0, 0)


class TestHelpers:
    def test_distance_func(self):
        d = Axiom.distance(Point(0, 0), Point(3, 4))
        assert d == 5.0

    def test_closest_point_on_line(self):
        l = Line(Point(0, 0), Point(1, 0))
        p = Axiom.closest_point_on_line(l, Point(0.5, 5))
        assert p == Point(0.5, 0)

    def test_project_on_plane(self):
        pl = Plane(Point(0, 0, 0), Point(0, 0, 1))
        p = Axiom.project_point_on_plane(pl, Point(1, 2, 3))
        assert p == Point(1, 2, 0)


class TestConvexHull:
    def test_triangle(self):
        pts = [Point(0, 0), Point(1, 0), Point(0, 1)]
        hull = Axiom.convex_hull(pts)
        assert len(hull) == 3

    def test_square(self):
        pts = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
        hull = Axiom.convex_hull(pts)
        assert len(hull) == 4

    def test_collinear(self):
        pts = [Point(0, 0), Point(1, 0), Point(2, 0)]
        hull = Axiom.convex_hull(pts)
        assert len(hull) == 2


class TestClosestPair:
    def test_two_points(self):
        a, b = Point(0, 0), Point(3, 4)
        p1, p2 = Axiom.closest_pair([a, b])
        assert {p1, p2} == {a, b}

    def test_many_points(self):
        pts = [Point(0, 0), Point(10, 10), Point(1, 1), Point(5, 5)]
        p1, p2 = Axiom.closest_pair(pts)
        assert p1.distance_to(p2) == pytest.approx(math.sqrt(2))
