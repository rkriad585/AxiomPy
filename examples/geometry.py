from axiompy import Axiom

# ---- Points ----
p1 = Axiom.Point(0, 0)
p2 = Axiom.Point(3, 4)
print(f"Distance between {p1} and {p2}: {p1.distance_to(p2)}")
print(f"Dot product: {p1.dot(p2)}")
print(f"3 * {p1} = {3 * p1}")

p3 = Axiom.Point(1, 2, 3)
p4 = Axiom.Point(4, 5, 6)
print(f"\n3D cross product of {p3} and {p4}: {p3.cross(p4)}")

# ---- Line ----
line = Axiom.Line(Axiom.Point(0, 0), Axiom.Point(1, 0))
print(f"\nLine: {line}")
print(f"Distance to (0, 1): {line.distance_to_point(Axiom.Point(0, 1))}")
proj = Axiom.closest_point_on_line(line, Axiom.Point(0.3, 5))
print(f"Projection of (0.3, 5): {proj}")

l1 = Axiom.Line(Axiom.Point(0, 0), Axiom.Point(1, 1))
l2 = Axiom.Line(Axiom.Point(0, 1), Axiom.Point(1, 0))
print(f"Line intersection: {l1.intersection(l2)}")

# ---- Plane ----
plane = Axiom.Plane(Axiom.Point(0, 0, 0), Axiom.Point(0, 0, 1))
print(f"\nPlane: {plane}")
print(f"Distance to (0, 0, 5): {plane.distance_to_point(Axiom.Point(0, 0, 5))}")
print(f"Project (1, 2, 3): {Axiom.project_point_on_plane(plane, Axiom.Point(1, 2, 3))}")

# ---- Sphere ----
sphere = Axiom.Sphere(Axiom.Point(0, 0, 0), 1.0)
print(f"\nSphere volume: {sphere.volume():.4f}")
print(f"Surface area: {sphere.surface_area():.4f}")
print(f"Contains (0.5, 0, 0): {sphere.contains_point(Axiom.Point(0.5, 0, 0))}")

ray = Axiom.Line(Axiom.Point(-2, 0, 0), Axiom.Point(2, 0, 0))
hits = sphere.intersect_line(ray)
print(f"Ray hits: {hits}")

# ---- Convex hull ----
pts = [Axiom.Point(0, 0), Axiom.Point(1, 0), Axiom.Point(1, 1),
       Axiom.Point(0, 1), Axiom.Point(0.5, 0.5)]
hull = Axiom.convex_hull(pts)
print(f"\nConvex hull of 5 points: {len(hull)} vertices: {hull}")

# ---- Closest pair ----
pts2 = [Axiom.Point(0, 0), Axiom.Point(10, 10), Axiom.Point(1, 1), Axiom.Point(5, 5)]
a, b = Axiom.closest_pair(pts2)
print(f"\nClosest pair: {a} -- {b}  (dist={a.distance_to(b):.4f})")
