from axiompy.polynomial import Polynomial

# p(x) = 3x^2 + 2x + 1
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
print(f"integral p dx = {ip}")
print(f"integral p(1) = {ip(1)}")

# Roots of x^2 - 5x + 6 = 0
poly = Polynomial([6, -5, 1])
print(f"\nRoots of {poly}: {poly.roots()}")

# Lagrange interpolation through (0,0), (1,1), (2,4)  ->  f(x) = x^2
xs = [0.0, 1.0, 2.0]
ys = [0.0, 1.0, 4.0]
lag = Polynomial.lagrange_interpolate(xs, ys)
print(f"\nLagrange through (0,0), (1,1), (2,4): {lag}")
print(f"  f(3) = {lag(3)}  (expected: 9.0)")
print(f"  f(0.5) = {lag(0.5)}  (expected: 0.25)")

# --- Phase 5.8 features ---

# Polynomial division
p_div = Polynomial([1, 2, 1])
q_div = Polynomial([1, 1])
print(f"\n({p_div}) // ({q_div}) = {p_div // q_div}")
print(f"({p_div}) % ({q_div}) = {p_div % q_div}")

# Polynomial GCD
p1 = Polynomial([1, 2, 1])
p2 = Polynomial([1, 1])
print(f"gcd({p1}, {p2}) = {p1.gcd(p2)}")

# Polynomial composition
p_comp = Polynomial([0, 1])  # p(x) = x
q_comp = Polynomial([1, 1])  # q(x) = x + 1
print(f"\n{p_comp}.compose({q_comp}) = {p_comp.compose(q_comp)}  (expected: x+1)")

# Chebyshev roots
roots = Polynomial.chebyshev_roots(5)
print(f"\nChebyshev T_5 roots: {roots}")

# Least-squares fit
xs_fit = [0, 1, 2, 3, 4]
ys_fit = [0, 1, 4, 9, 16]
fitted = Polynomial.fit(xs_fit, ys_fit, degree=2)
print(f"\nLeast-squares fit to y=x^2: {fitted}")
print(f"  fitted(2) = {fitted(2):.4f}  (expected: 4.0)")
