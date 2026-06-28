"""Pure Python backend example — no numpy dependency required."""
from axiompy import Axiom, set_backend
from axiompy._pure_backend import PurePythonBackend

# Switch to the pure-Python backend
set_backend("pure")
print(f"Active backend: {Axiom.backend.__class__.__name__}")

# PureArray creation via the backend
b = Axiom.backend
a = b.array([[1, 2], [3, 4]])
print(f"\nArray:  {a.tolist()}")
print(f"Shape:  {a.shape}")

# Dot product
c = b.dot(a, a)
print(f"\nA @ A:  {c.tolist()}")

# Determinant
d = b.det([[1, 2], [3, 4]])
print(f"\ndet(A): {d:.4f}  (expected: -2.0)")

# Matrix inverse
inv = b.inv([[1, 2], [3, 4]])
print(f"\nA^-1:   {inv.tolist()}")

# Solve linear system
x = b.solve([[2, 1], [1, 3]], [5, 6])
print(f"\nSolve 2x+y=5, x+3y=6:  x = {x.tolist()}  (expected: [1.8, 1.4])")

# Matrix powers
p3 = b.matrix_power([[1, 1], [0, 1]], 3)
print(f"\n[[1,1],[0,1]]^3: {p3.tolist()}  (expected: [[1,3],[0,1]])")

# Norms
print(f"\nnorm([3,4]):   {b.norm([3, 4]):.4f}  (expected: 5.0)")
print(f"norm([3,4],1):  {b.norm([3, 4], ord=1):.4f}  (expected: 7.0)")

# QR decomposition
Q, R = b.qr([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
print("\nQR decomposition:")
print(f"  Q: {Q.tolist()}")
print(f"  R: {R.tolist()}")

# Cholesky decomposition
L = b.cholesky([[4, 1], [1, 3]])
print(f"\nCholesky L: {L.tolist()}")

# Eigenvalues
ev = b.eigvals([[4, 1], [1, 3]])
print(f"\nEigenvalues of [[4,1],[1,3]]: {sorted(ev.tolist(), reverse=True)}")

# Matrix rank
print(f"\nrank([[1,2],[3,4]]): {b.matrix_rank([[1,2],[3,4]])}  (expected: 2)")
print(f"rank([[1,2],[2,4]]): {b.matrix_rank([[1,2],[2,4]])}  (expected: 1)")

# FFT (PurePythonBackend-specific)
if isinstance(b, PurePythonBackend):
    fft_out = b.fft([1.0, 0.0, -1.0, 0.0])
    print(f"\nFFT of [1, 0, -1, 0]: {[f'{v.real:.4f}+{v.imag:.4f}j' for v in fft_out.tolist()]}")

# Switch back to numpy
set_backend("numpy")
print(f"\nBackend restored to: {Axiom.backend.__class__.__name__}")
