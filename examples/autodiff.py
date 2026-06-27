import math
from axiompy import Axiom

# Basic reverse-mode AD
x = Axiom.autodiff.Variable(2.0)
y = Axiom.autodiff.Variable(3.0)
z = x * y + x
z.backward()
print("z = x*y + x")
print(f"x = {x.value}, y = {y.value}, z = {z.value}")
print(f"dz/dx = {x.grad}  (expected: 4.0)")
print(f"dz/dy = {y.grad}  (expected: 2.0)")

# New functions: tanh, sigmoid, sqrt
x = Axiom.autodiff.Variable(0.5)
f = Axiom.autodiff.tanh(x)
f.backward()
print(f"\ntanh(0.5) = {f.value:.4f}, d/dx = {x.grad:.4f}  (expected: {1 - math.tanh(0.5)**2:.4f})")

x = Axiom.autodiff.Variable(0.0)
f = Axiom.autodiff.sigmoid(x)
f.backward()
print(f"sigmoid(0) = {f.value:.4f}, d/dx = {x.grad:.4f}  (expected: 0.25)")

x = Axiom.autodiff.Variable(4.0)
f = Axiom.autodiff.sqrt(x)
f.backward()
print(f"sqrt(4) = {f.value:.4f}, d/dx = {x.grad:.4f}  (expected: 0.25)")

# Chain rule
x = Axiom.autodiff.Variable(1.5)
f = Axiom.autodiff.sin(x ** 2)
f.backward()
print(f"\nsin(x^2), x = {x.value}")
print(f"df/dx = {x.grad:.4f}  (expected: {2 * 1.5 * math.cos(2.25):.4f})")

# Gradient descent minimizer
f = lambda x: x ** 2 + 2 * x + 1
minimum = Axiom.autodiff.gradient_descent(f, start=5.0, lr=0.1, steps=100)
print(f"\nGradient descent on x² + 2x + 1")
print(f"Minimum at x = {minimum:.4f}  (expected: -1.0)")
