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
print("rank(M) =", M.rank)
print("M^-1:\n", M.inverse)
print("norm(M) =", M.norm())
print("norm(M, ord=1) =", M.norm(ord=1))

v1 = Axiom.Vector([1, 0, 0])
v2 = Axiom.Vector([0, 1, 0])
print("v1 × v2 =", v1.cross(v2))
print("angle =", v1.angle_between(v2, in_degrees=True))
print("v1 · v2 =", v1.dot(v2))
print("v1 norm(L1) =", v1.norm(ord=1))
print("v1 norm(L2) =", v1.norm(ord=2))
print("v1 norm(L∞) =", v1.norm(ord=float('inf')))
print("v1 project onto v2 =", v1.project_onto(v2))
print("-v1 =", -v1)
print("abs(v1) =", abs(v1))

# Solve Ax = b
A = Axiom.Matrix([[3, 1], [1, 2]])
b = Axiom.Vector([9, 8])
x = Axiom.linalg.solve_linear(A, b)
print("Solve 3x + y = 9, x + 2y = 8  =>", x)

# LU decomposition
P, L, U = M.lu_decompose()
print("\nLU decomposition:")
print("P:\n", P)
print("L:\n", L)
print("U:\n", U)
print("P @ L @ U:\n", P @ L @ U)

# QR decomposition
Q, R = M.qr_decompose()
print("\nQR decomposition:")
print("Q:\n", Q)
print("R:\n", R)
