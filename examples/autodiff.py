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
def f(x): return x ** 2 + 2 * x + 1
minimum = Axiom.autodiff.gradient_descent(f, start=5.0, lr=0.1, steps=100)
print("\nGradient descent on x^2 + 2x + 1")
print(f"Minimum at x = {minimum:.4f}  (expected: -1.0)")

# --- Phase 5.6 features ---

# Variable.tanh() and .sigmoid() instance methods
x = Axiom.autodiff.Variable(1.0)
t = x.tanh()
t.backward()
print(f"\nx.tanh() = {t.value:.4f}, grad = {x.grad:.4f}")

x = Axiom.autodiff.Variable(0.0)
s = x.sigmoid()
s.backward()
print(f"x.sigmoid() = {s.value:.4f}, grad = {x.grad:.4f}")

# Computational graph visualization
x = Axiom.autodiff.Variable(2.0)
y = Axiom.autodiff.Variable(3.0)
z = x * y + Axiom.autodiff.sin(x)
print("\nGraph visualization:")
print(z.to_ascii())

# Adam optimizer
def f_adam(x, y): return (x - 1)**2 + (y + 2)**2
x_opt = Axiom.autodiff.adam(f_adam, [0.0, 0.0], steps=100, lr=0.1)
print("\nAdam optimization of (x-1)^2 + (y+2)^2:")
print(f"  Minimum at: {x_opt}")

# Hessian via autodiff
def f_rosenbrock(v):
    return (1 - v[0])**2 + 100 * (v[1] - v[0]**2)**2
H = Axiom.autodiff.hessian(f_rosenbrock, [1.0, 1.0])
print(f"\nHessian of Rosenbrock at (1,1):\n{H}")
