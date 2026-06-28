import math

from axiompy.calculus import Calculus


def f(x): return x ** 2

x = 3.0
deriv = Calculus.numerical_derivative(f, x)
print(f"f(x) = x^2, f'({x}) = {deriv:.6f}  (expected: 6.0)")

a, b = 0.0, 1.0
trap = Calculus.integrate_trapezoid(f, a, b, n=100)
simp = Calculus.integrate_simpson(f, a, b, n=100)
mc = Calculus.integrate_monte_carlo(f, a, b, n=50000)
exact = 1.0 / 3.0
print("\nintegral_0^1 x^2 dx:")
print(f"  Trapezoid: {trap:.6f}  (error: {abs(trap - exact):.2e})")
print(f"  Simpson:   {simp:.6f}  (error: {abs(simp - exact):.2e})")
print(f"  MonteCarlo:{mc:.6f}  (error: {abs(mc - exact):.2e})")
print(f"  Exact:     {exact:.6f}")

# Multivariate gradient
def g(p): return p[0] ** 2 + p[1] ** 3 + p[0] * p[1]
point = [1.0, 2.0]
grad = Calculus.gradient(g, point)
print(f"\ngrad f(1,2) for f = x^2 + y^3 + xy: {grad}")
print(f"  df/dx ~ {grad[0]:.4f}  (expected: 4.0)")
print(f"  df/dy ~ {grad[1]:.4f}  (expected: 13.0)")

# --- Phase 5.4 features ---

# Richardson extrapolation
def f_sin(x): return math.sin(x)
rich = Calculus.richardson_extrapolation(f_sin, 0.0, h=0.1)
print(f"\nRichardson extrapolation of sin'(0): {rich:.6f}  (expected: 1.0)")

# Gauss-Legendre quadrature
gl = Calculus.integrate_gauss_legendre(lambda x: x**2, 0, 1, n=5)
print(f"Gauss-Legendre integral x^2 [0,1]: {gl:.6f}  (expected: 0.3333)")

# Romberg integration
rom = Calculus.integrate_romberg(lambda x: x**2, 0, 1)
print(f"Romberg integral x^2 [0,1]: {rom:.6f}  (expected: 0.3333)")

# Jacobian
def fj(p): return [p[0]**2 + p[1], p[0] - p[1]**3]
J = Calculus.jacobian(fj, [1.0, 2.0])
print(f"\nJacobian of [x^2+y, x-y^3] at (1,2):\n{J}")

# Hessian
def fh(p): return p[0]**2 + 3 * p[0] * p[1] + p[1]**2
H = Calculus.hessian(fh, [1.0, 1.0])
print(f"Hessian of x^2 + 3xy + y^2 at (1,1):\n{H}")

# ODE solvers: y' = y, y(0) = 1 -> y(t) = e^t
def ode(t, y):
    return y

euler = Calculus.ode_euler(ode, [1.0], (0.0, 2.0), dt=0.01)
rk4 = Calculus.ode_rk4(ode, [1.0], (0.0, 2.0), dt=0.01)
print("\nODE y' = y, y(0) = 1 at t=2:")
print(f"  Euler: y(2) = {euler[1][-1][0]:.4f}  (expected: {math.exp(2):.4f})")
print(f"  RK4:   y(2) = {rk4[1][-1][0]:.4f}  (expected: {math.exp(2):.4f})")
