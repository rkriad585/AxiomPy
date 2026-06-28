"""Lazy evaluation example — deferred computation for Vector and Matrix."""
from axiompy import Axiom

# Basic lazy vector operations
v = Axiom.Vector([1, 2, 3])
w = Axiom.Vector([4, 5, 6])

# Build a lazy expression tree
expr = Axiom.lazy.lazy(v) + Axiom.lazy.lazy(w)
print(f"Expression: {expr!r}")
result = expr.compute()
print(f"(v + w).compute() = {result}  (expected: Vector([5, 7, 9]))")

# Chained lazy operations
expr2 = (Axiom.lazy.lazy(v) + Axiom.lazy.lazy(w)) * 2.0 - Axiom.lazy.lazy(v)
result2 = expr2.compute()
expected = (v + w) * 2.0 - v
print(f"\n((v + w) * 2 - v).compute() = {result2}  (expected: {expected})")

# Matrix lazy operations
A = Axiom.Matrix([[1, 2], [3, 4]])
B = Axiom.Matrix([[0, 1], [1, 0]])

expr3 = Axiom.lazy.lazy(A) @ Axiom.lazy.lazy(B)
result3 = expr3.compute()
print(f"\n(A @ B).compute() = \n{result3}")
print(f"  (expected: \n{A @ B})")

# Transpose
expr4 = Axiom.lazy.lazy(A).T
result4 = expr4.compute()
print(f"\nA.T.compute() = \n{result4}")
print(f"  (expected: \n{A.T})")

# Matrix power
expr5 = Axiom.lazy.lazy(A) ** 3
result5 = expr5.compute()
print(f"\nA^3.compute() = \n{result5}")
print(f"  (expected: \n{A ** 3})")

# Negation
expr6 = -Axiom.lazy.lazy(v)
result6 = expr6.compute()
print(f"\n(-v).compute() = {result6}  (expected: {[-1, -2, -3]})")

# Scalar operations
expr7 = Axiom.lazy.lazy(v) * 10.0 + 5.0
result7 = expr7.compute()
print(f"\n(v * 10 + 5).compute() = {result7}")

# Using Axiom.lazy.compute() directly
result8 = Axiom.lazy.compute(Axiom.lazy.lazy(A) @ Axiom.lazy.lazy(v))
print(f"\nAxiom.lazy.compute(A @ v) = {result8}  (expected: {A @ v})")
