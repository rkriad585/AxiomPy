from axiompy.matrix import Matrix
from axiompy.optimization import Optimization
from axiompy.vector import Vector


# f(x) = x^2 + 2x + 1,  f'(x) = 2x + 2
def f(x): return x ** 2 + 2 * x + 1
def grad_f(x): return 2 * x + 2
def d2f(x): return 2.0

gd = Optimization.gradient_descent(f, grad_f, start=5.0, lr=0.1, steps=100)
print(f"Gradient descent: min at x = {gd:.6f}  (expected: -1.0)")

nm = Optimization.newton_method(f, grad_f, d2f, start=5.0, steps=10)
print(f"Newton's method:  min at x = {nm:.6f}  (expected: -1.0)")

# f(x) = x^2 - 4 = 0  ->  roots at x = +/-2
def root_f(x): return x ** 2 - 4
root = Optimization.bisection(root_f, 0, 5, tol=1e-8)
print(f"\nBisection x^2 - 4: x = {root:.6f}  (expected: 2.0)")

root2 = Optimization.bisection(root_f, -5, 0, tol=1e-8)
print(f"Bisection x^2 - 4: x = {root2:.6f}  (expected: -2.0)")

# No root case
no_root = Optimization.bisection(lambda x: x ** 2 + 1, -1, 1)
print(f"Bisection x^2 + 1: {no_root}  (expected: None)")

# --- Phase 5.5 features ---

# Golden section search
golden = Optimization.golden_section(lambda x: (x - 2) ** 2, 0, 5)
print(f"\nGolden section (x-2)^2: min at x = {golden:.6f}  (expected: 2.0)")

# Conjugate gradient for SPD system Ax = b
A = Matrix([[4, 1], [1, 3]])
b_vec = Vector([1, 2])
cg = Optimization.conjugate_gradient(A, b_vec, Vector([0, 0]))
print(f"Conjugate gradient solve: x = {cg}")

# Nelder-Mead
nm_simp = Optimization.nelder_mead(lambda p: (p[0] - 1)**2 + (p[1] - 2)**2, [0, 0])
print(f"Nelder-Mead: min at {nm_simp}  (expected: [1, 2])")

# L-BFGS
lbfgs_res = Optimization.lbfgs(
    lambda p: (p[0] - 1)**2 + (p[1] - 2)**2,
    lambda p: [2 * (p[0] - 1), 2 * (p[1] - 2)],
    [0, 0]
)
print(f"L-BFGS: min at {lbfgs_res}  (expected: [1, 2])")

# Simulated annealing
sa = Optimization.simulated_annealing(lambda p: (p[0] - 3)**2 + (p[1] + 1)**2, [0, 0])
print(f"Simulated annealing: min near {sa}")
