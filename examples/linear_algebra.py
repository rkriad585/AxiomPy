from axiompy import Axiom

M = Axiom.Matrix([[1, 2], [3, 4]])
v = Axiom.Vector([5, 6])
I = Axiom.linalg.identity(2)

print("M:\n", M)
print("v:", v)
print("M @ v =", M @ v)
print("M @ M:\n", M @ M)
print("M ** 2:\n", M ** 2)
print("M + I:\n", M + I)

print("det(M) =", M.determinant)
print("trace(M) =", M.trace)
print("M^-1:\n", M.inverse)

v1 = Axiom.Vector([1, 0, 0])
v2 = Axiom.Vector([0, 1, 0])
print("v1 × v2 =", v1.cross(v2))
print("angle =", v1.angle_between(v2, in_degrees=True))
