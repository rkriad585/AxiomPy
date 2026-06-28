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
print("v1 x v2 =", v1.cross(v2))
print("angle =", v1.angle_between(v2, in_degrees=True))
print("v1 . v2 =", v1.dot(v2))
print("v1 norm(L1) =", v1.norm(ord=1))
print("v1 norm(L2) =", v1.norm(ord=2))
print("v1 norm(Loo) =", v1.norm(ord=float('inf')))
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

# --- Phase 5.1 features ---

# SVD
A_mat = Axiom.Matrix([[1, 2], [3, 4], [5, 6]])
U, S, Vt = A_mat.svd_decompose()
print("\nSVD of 3x2 matrix:")
print("U:\n", U)
print("Singular values:", S)
print("Vt:\n", Vt)

# Eigen decomposition (symmetric matrix)
S_mat = Axiom.Matrix([[4, 1], [1, 3]])
print("\nSymmetric matrix:\n", S_mat)
print("Eigenvalues:", S_mat.eigenvalues())
print("Eigenvectors:\n", S_mat.eigenvectors())

# Condition number
print("Condition number:", S_mat.condition_number())

# Pseudoinverse
A2 = Axiom.Matrix([[1, 2], [3, 4], [5, 6]])
pinv = A2.pinv()
print("\nPseudoinverse of 3x2:\n", pinv)

# Least squares
b_vec = Axiom.Vector([1, 0, 1])
x_ls = Axiom.linalg.least_squares(A2, b_vec)
print("Least squares solution:", x_ls)

# Cross product matrix
v_vec = Axiom.Vector([1, 2, 3])
cp = Axiom.linalg.cross_product_matrix(v_vec)
print("\nCross product matrix of", v_vec, ":\n", cp)

# Householder reflection
H = Axiom.linalg.householder(Axiom.Vector([3, 4]))
print("\nHouseholder reflector for [3, 4]:\n", H)
print("H @ [3, 4] =", H @ Axiom.Vector([3, 4]))
