import math
from axiompy import Axiom

x = Axiom.autodiff.Variable(2.0)
y = Axiom.autodiff.Variable(3.0)
z = x * y + x
z.backward()

print("z = x*y + x")
print(f"x = {x.value}, y = {y.value}")
print(f"z = {z.value}")
print(f"dz/dx = {x.grad}  (expected: 4.0)")
print(f"dz/dy = {y.grad}  (expected: 2.0)")

# Chain rule: f(x) = sin(x^2)
x = Axiom.autodiff.Variable(1.5)
f = Axiom.autodiff.sin(x ** 2)
f.backward()
print(f"\nf(x) = sin(x^2), x = {x.value}")
print(f"f = {f.value:.4f}")
print(f"df/dx = {x.grad:.4f}  (expected: {2 * 1.5 * math.cos(2.25):.4f})")
