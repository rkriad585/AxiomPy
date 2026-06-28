import math
from typing import Optional

# ---- helper ---------------------------------------------------------------

def _cross_2d(ax, ay, bx, by):
    return ax * by - ay * bx


# ---- Primitives -----------------------------------------------------------


class Point:
    """A point in 2D or 3D space."""

    def __init__(self, *coords: float):
        if len(coords) not in (2, 3):
            raise ValueError("Point takes 2 or 3 coordinates")
        self.coords = tuple(coords)

    @property
    def x(self) -> float:
        return self.coords[0]

    @property
    def y(self) -> float:
        return self.coords[1]

    @property
    def z(self) -> float:
        if len(self.coords) < 3:
            return 0.0
        return self.coords[2]

    def __repr__(self) -> str:
        return f"Point{self.coords}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.coords == other.coords

    def __hash__(self) -> int:
        return hash(self.coords)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(*(a - b for a, b in zip(self.coords, other.coords)))

    def __add__(self, other: 'Point') -> 'Point':
        return Point(*(a + b for a, b in zip(self.coords, other.coords)))

    def __mul__(self, scalar: float) -> 'Point':
        return Point(*(c * scalar for c in self.coords))

    def __rmul__(self, scalar: float) -> 'Point':
        return self.__mul__(scalar)

    def dim(self) -> int:
        return len(self.coords)

    def distance_to(self, other: 'Point') -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.coords, other.coords)))

    def dot(self, other: 'Point') -> float:
        return sum(a * b for a, b in zip(self.coords, other.coords))

    def cross(self, other: 'Point') -> 'Point':
        """3D cross product (raises ValueError for 2D points)."""
        if self.dim() != 3 or other.dim() != 3:
            raise ValueError("Cross product is only defined for 3D points")
        return Point(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        return math.sqrt(sum(c * c for c in self.coords))

    def normalize(self) -> 'Point':
        n = self.norm()
        if n == 0:
            return Point(*([0.0] * self.dim()))
        return Point(*(c / n for c in self.coords))


class Line:
    """A line defined by two points."""

    def __init__(self, p1: Point, p2: Point):
        if p1.dim() != p2.dim():
            raise ValueError("Points must have the same dimension")
        self.p1 = p1
        self.p2 = p2

    def __repr__(self) -> str:
        return f"Line({self.p1}, {self.p2})"

    def direction(self) -> Point:
        return self.p2 - self.p1

    def point_at(self, t: float) -> Point:
        return self.p1 + self.direction() * t

    def distance_to_point(self, p: Point) -> float:
        """Shortest distance from point to line."""
        d = self.direction()
        if d.norm() == 0:
            return p.distance_to(self.p1)
        if p.dim() == 2:
            # 2D cross product (scalar)
            ap = (p - self.p1)
            area = abs(ap.x * d.y - ap.y * d.x)
            return area / d.norm()
        return abs((p - self.p1).cross(d).norm()) / d.norm()

    def contains_point(self, p: Point, tol: float = 1e-10) -> bool:
        return self.distance_to_point(p) < tol

    def intersection(self, other: 'Line') -> Optional[Point]:
        """Intersection of two lines (2D only)."""
        if self.p1.dim() != 2 or other.p1.dim() != 2:
            raise ValueError("Line intersection only supported in 2D")
        x1, y1 = self.p1.coords
        x2, y2 = self.p2.coords
        x3, y3 = other.p1.coords
        x4, y4 = other.p2.coords
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return Point(x1 + t * (x2 - x1), y1 + t * (y2 - y1))


class Plane:
    """A plane in 3D defined by a point and a normal vector."""

    def __init__(self, point: Point, normal: Point):
        if point.dim() != 3 or normal.dim() != 3:
            raise ValueError("Plane requires 3D points")
        self.point = point
        self.normal = normal.normalize()

    def __repr__(self) -> str:
        return f"Plane(point={self.point}, normal={self.normal})"

    def distance_to_point(self, p: Point) -> float:
        return abs((p - self.point).dot(self.normal))

    def contains_point(self, p: Point, tol: float = 1e-10) -> bool:
        return self.distance_to_point(p) < tol


class Sphere:
    """A sphere in 3D space."""

    def __init__(self, center: Point, radius: float):
        if center.dim() != 3:
            raise ValueError("Sphere requires a 3D center point")
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        self.center = center
        self.radius = radius

    def __repr__(self) -> str:
        return f"Sphere(center={self.center}, radius={self.radius})"

    def surface_area(self) -> float:
        return 4.0 * math.pi * self.radius ** 2

    def volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius ** 3

    def contains_point(self, p: Point, tol: float = 1e-10) -> bool:
        return self.center.distance_to(p) <= self.radius + tol

    def distance_to_point(self, p: Point) -> float:
        return max(0.0, self.center.distance_to(p) - self.radius)

    def intersect_line(self, line: Line) -> list[Point]:
        """Ray-sphere intersection. Returns 0, 1, or 2 points."""
        oc = line.p1 - self.center
        d = line.direction()
        a = d.dot(d)
        b = 2.0 * oc.dot(d)
        c = oc.dot(oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return []
        if disc == 0:
            t = -b / (2 * a)
            return [line.point_at(t)]
        t1 = (-b - math.sqrt(disc)) / (2 * a)
        t2 = (-b + math.sqrt(disc)) / (2 * a)
        return [line.point_at(t1), line.point_at(t2)]


# ---- Distance helpers ----------------------------------------------------


def distance(p1: Point, p2: Point) -> float:
    """Euclidean distance between two points."""
    return p1.distance_to(p2)


def closest_point_on_line(line: Line, p: Point) -> Point:
    """Project point onto line (orthogonal projection)."""
    d = line.direction()
    if d.norm() == 0:
        return line.p1
    t = (p - line.p1).dot(d) / d.dot(d)
    return line.point_at(t)


def project_point_on_plane(plane: Plane, p: Point) -> Point:
    """Orthogonal projection of point onto a plane."""
    n = plane.normal
    t = (p - plane.point).dot(n)
    return p - (n * t)


# ---- Convex hull (Andrew's monotone chain) --------------------------------


def convex_hull(points: list[Point]) -> list[Point]:
    """Compute the convex hull of 2D points (Andrew's monotone chain).

    Args:
        points: List of 2D points.

    Returns:
        List of vertices on the hull in counter-clockwise order.
    """
    pts = sorted(set(points), key=lambda p: (p.x, p.y))
    if len(pts) <= 1:
        return pts

    def _ccw(o, a, b):
        return _cross_2d(a.x - o.x, a.y - o.y, b.x - o.x, b.y - o.y) > 0

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and not _ccw(lower[-2], lower[-1], p):
            lower.pop()
        lower.append(p)

    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and not _ccw(upper[-2], upper[-1], p):
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


# ---- Closest pair (divide & conquer) ------------------------------------


def closest_pair(points: list[Point]) -> tuple[Point, Point]:
    """Find the closest pair of 2D points (divide & conquer).

    Args:
        points: List of 2D points (>= 2).

    Returns:
        Tuple of the two closest points.

    Raises:
        ValueError: If fewer than 2 points.
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points")
    pts_x = sorted(points, key=lambda p: p.x)
    pts_y = sorted(points, key=lambda p: p.y)

    def _sqdist(a, b):
        return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

    def _closest(px, py):
        n = len(px)
        if n <= 3:
            best_d = float("inf")
            best_pair = (px[0], px[1])
            for i in range(n):
                for j in range(i + 1, n):
                    d = _sqdist(px[i], px[j])
                    if d < best_d:
                        best_d = d
                        best_pair = (px[i], px[j])
            return best_pair, best_d

        mid = n // 2
        mid_x = px[mid].x
        left_x = px[:mid]
        right_x = px[mid:]
        left_y = [p for p in py if p.x <= mid_x]
        right_y = [p for p in py if p.x > mid_x]

        (left_pair, d_left) = _closest(left_x, left_y)
        (right_pair, d_right) = _closest(right_x, right_y)
        if d_left < d_right:
            best_d = d_left
            best_pair = left_pair
        else:
            best_d = d_right
            best_pair = right_pair

        strip = [p for p in py if abs(p.x - mid_x) < math.sqrt(best_d)]
        m = len(strip)
        for i in range(m):
            for j in range(i + 1, min(i + 7, m)):
                d = _sqdist(strip[i], strip[j])
                if d < best_d:
                    best_d = d
                    best_pair = (strip[i], strip[j])
        return best_pair, best_d

    pair, _ = _closest(pts_x, pts_y)
    return pair
