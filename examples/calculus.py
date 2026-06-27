import math
from axiompy import Axiom
from axiompy.calculus import Calculus

f = lambda x: x ** 2

x = 3.0
deriv = Calculus.numerical_derivative(f, x)
print(f"f(x) = x^2, f'({x}) = {deriv:.6f}  (expected: 6.0)")

a, b = 0.0, 1.0
trap = Calculus.integrate_trapezoid(f, a, b, n=100)
simp = Calculus.integrate_simpson(f, a, b, n=100)
mc = Calculus.integrate_monte_carlo(f, a, b, n=50000)
exact = 1.0 / 3.0
print(f"\n∫₀¹ x² dx:")
print(f"  Trapezoid: {trap:.6f}  (error: {abs(trap - exact):.2e})")
print(f"  Simpson:   {simp:.6f}  (error: {abs(simp - exact):.2e})")
print(f"  MonteCarlo:{mc:.6f}  (error: {abs(mc - exact):.2e})")
print(f"  Exact:     {exact:.6f}")

# Multivariate gradient
g = lambda p: p[0] ** 2 + p[1] ** 3 + p[0] * p[1]
point = [1.0, 2.0]
grad = Calculus.gradient(g, point)
print(f"\n∇f(1,2) for f = x² + y³ + xy: {grad}")
print(f"  ∂f/∂x ≈ {grad[0]:.4f}  (expected: 4.0)")
print(f"  ∂f/∂y ≈ {grad[1]:.4f}  (expected: 13.0)")
