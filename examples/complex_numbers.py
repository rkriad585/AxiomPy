import math

from axiompy import Axiom

# Create complex numbers
z1 = Axiom.ComplexNumber(3, 4)
z2 = Axiom.ComplexNumber(1, -2)

print(f"z1 = {z1}")
print(f"z2 = {z2}")
print(f"z1 + z2 = {z1 + z2}")
print(f"z1 * z2 = {z1 * z2}")
print(f"|z1| = {z1.modulus():.4f}")
print(f"arg(z1) = {z1.argument():.4f} rad")
print(f"conj(z1) = {z1.conjugate}")

# Polar form
z3 = Axiom.ComplexNumber.from_polar(2.0, math.pi / 3)
print(f"\nFrom polar (r=2, phi=pi/3): {z3}")

# De Moivre's power
i = Axiom.ComplexNumber(0, 1)
print(f"\ni^2 = {i.power(2)}")
print(f"i^3 = {i.power(3)}")

# Roots of unity
roots = Axiom.ComplexNumber.roots_of_unity(5)
print("\n5th roots of unity:")
for r in roots:
    print(f"  {r}  |z|={r.modulus():.2f}")

# FFT alias
x = [1, 0, -1, 0]
X = Axiom.ComplexNumber.cfft(x)
print(f"\nCFFT of {x}: {[f'{v:.2f}' for v in X]}")

# ComplexVector / ComplexMatrix
v = Axiom.ComplexVector([1+2j, 3+4j])
print(f"\nComplexVector: {v}")
m = Axiom.ComplexMatrix([[1j, 2j], [3j, 4j]])
print(f"ComplexMatrix: {m}")
