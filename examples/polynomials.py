from axiompy import Axiom
from axiompy.polynomial import Polynomial

# p(x) = 3x² + 2x + 1
p = Polynomial([1, 2, 3])
print(f"p(x) = {p}")
print(f"degree = {p.degree}")
print(f"p(2) = {p(2)}")
print(f"p(0) = {p(0)}")

q = Polynomial([1, 1])
print(f"\nq(x) = {q}")
print(f"p + q = {p + q}")
print(f"p - q = {p - q}")
print(f"p * q = {p * q}")
print(f"p * 2 = {p * 2}")

dp = p.derivative()
print(f"\np'(x) = {dp}")
print(f"p'(2) = {dp(2)}  (expected: 14)")

ip = p.integral(constant=0)
print(f"∫p dx = {ip}")
print(f"∫p(1) = {ip(1)}")

# Roots of x² - 5x + 6 = 0
poly = Polynomial([6, -5, 1])
print(f"\nRoots of {poly}: {poly.roots()}")

# Lagrange interpolation through (0,0), (1,1), (2,4)  →  f(x) = x²
xs = [0.0, 1.0, 2.0]
ys = [0.0, 1.0, 4.0]
lag = Polynomial.lagrange_interpolate(xs, ys)
print(f"\nLagrange through (0,0), (1,1), (2,4): {lag}")
print(f"  f(3) = {lag(3)}  (expected: 9.0)")
print(f"  f(0.5) = {lag(0.5)}  (expected: 0.25)")
