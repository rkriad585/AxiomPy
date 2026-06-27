import math
from axiompy import Axiom
from axiompy.optimization import Optimization

# f(x) = x² + 2x + 1,  f'(x) = 2x + 2
f = lambda x: x ** 2 + 2 * x + 1
grad_f = lambda x: 2 * x + 2
d2f = lambda x: 2.0

gd = Optimization.gradient_descent(f, grad_f, start=5.0, lr=0.1, steps=100)
print(f"Gradient descent: min at x = {gd:.6f}  (expected: -1.0)")

nm = Optimization.newton_method(f, grad_f, d2f, start=5.0, steps=10)
print(f"Newton's method:  min at x = {nm:.6f}  (expected: -1.0)")

# f(x) = x² - 4 = 0  →  roots at x = ±2
root_f = lambda x: x ** 2 - 4
root = Optimization.bisection(root_f, 0, 5, tol=1e-8)
print(f"\nBisection x² - 4: x = {root:.6f}  (expected: 2.0)")

root2 = Optimization.bisection(root_f, -5, 0, tol=1e-8)
print(f"Bisection x² - 4: x = {root2:.6f}  (expected: -2.0)")

# No root case
no_root = Optimization.bisection(lambda x: x ** 2 + 1, -1, 1)
print(f"Bisection x² + 1: {no_root}  (expected: None)")
