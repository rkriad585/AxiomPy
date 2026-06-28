from axiompy import Axiom

# Gamma function
print(f"Gamma(5)     = {Axiom.gamma(5)}")              # 24
print(f"Gamma(0.5)   = {Axiom.gamma(0.5):.6f}")        # sqrt(pi)
print(f"Gamma(-0.5)  = {Axiom.gamma(-0.5):.6f}")       # -2*sqrt(pi)

# Beta function
print(f"Beta(2, 2)   = {Axiom.beta(2, 2):.6f}")        # 1/6

# Error function
print(f"erf(1)       = {Axiom.erf(1):.6f}")
print(f"erfc(1)      = {Axiom.erfc(1):.6f}")

# Bessel functions
print(f"J0(1)        = {Axiom.bessel_j(0, 1):.6f}")
print(f"J1(1)        = {Axiom.bessel_j(1, 1):.6f}")
print(f"Y0(1)        = {Axiom.bessel_y(0, 1):.6f}")

# Legendre polynomials
print(f"P2(0.5)      = {Axiom.legendre_p(2, 0.5):.6f}")
print(f"P3(0.5)      = {Axiom.legendre_p(3, 0.5):.6f}")

# Combinatorial
print(f"10!          = {Axiom.factorial(10)}")
print(f"C(10, 3)     = {Axiom.binomial(10, 3)}")
print(f"7!!          = {Axiom.double_factorial(7)}")
