# Usage Guide

## Linear Algebra

### Vector

```python
from axiompy import Axiom

v = Axiom.Vector([1.0, 2.0, 3.0])
u = Axiom.Vector([4.0, 5.0, 6.0])

v + u         # element-wise add
v - u         # element-wise subtract
v * 2.0       # scalar multiply
v * u         # dot product (returns float)
v / 2.0       # scalar divide
v.magnitude() # Euclidean norm
v.normalize() # unit vector
v.angle_between(u)              # radians
v.angle_between(u, in_degrees=True)  # degrees
v.cross(u)    # 3-D cross product

# Indexing
v[0]          # 1.0
v.to_list()   # [1.0, 2.0, 3.0]
```

### Matrix

```python
M = Axiom.Matrix([[1, 2], [3, 4]])
N = Axiom.Matrix([[5, 6], [7, 8]])

M + N         # element-wise add
M - N         # element-wise subtract
M * 2.0       # scalar multiply
M @ N         # matrix multiply
M @ v         # matrix-vector multiply (v is Vector)
M ** 2        # matrix power

M.determinant # property
M.trace       # property
M.rank        # property (int)
M.inverse     # property (Matrix)
M.T           # transpose
M.shape       # (rows, cols)
M.to_list()   # nested list
```

### Factory methods

```python
Axiom.linalg.identity(3)          # 3×3 identity
Axiom.linalg.zeros((2, 3))        # 2×3 zeros
Axiom.linalg.ones((4, 1))         # 4×1 ones
```

## Graph Analysis

```python
g = Axiom.Graph()
g.add_edge("A", "B")
g.add_edge("A", "C")
g.add_edge("B", "C")
g.add_edge("C", "A")
g.add_edge("D", "C")

ranks = Axiom.graph_analysis.pagerank(g)
# {"C": 0.3941, "A": 0.3725, "B": 0.1958, "D": 0.0375}
```

`pagerank` accepts optional keyword arguments:
- `damping` (default `0.85`) — PageRank damping factor
- `max_iter` (default `100`) — maximum iterations
- `tol` (default `1e-6`) — convergence tolerance

## Number Theory

```python
# Chinese Remainder Theorem
#   x ≡ 2 (mod 3)
#   x ≡ 3 (mod 5)
#   x ≡ 2 (mod 7)
Axiom.number_theory.chinese_remainder_theorem([3, 5, 7], [2, 3, 2])
# 23

# Modular inverse of 7 modulo 26
Axiom.number_theory.mod_inverse(7, 26)   # 15

# Extended Euclidean algorithm
Axiom.number_theory.extended_gcd(30, 20)  # (10, 1, -1)
```

## Automatic Differentiation

AxiomPy provides a reverse-mode (backpropagation) automatic differentiation
engine.

```python
from axiompy import Axiom

x = Axiom.autodiff.Variable(2.0)
y = Axiom.autodiff.Variable(3.0)
z = x * y + x
z.backward()
x.grad  # 4.0  (dz/dx)
y.grad  # 2.0  (dz/dy)
```

Supported operators: `+`, `-`, `*`, `/`, `**` (integer power), `neg`.

Built-in functions on `Variable`:

```python
x = Axiom.autodiff.Variable(1.5)
f = Axiom.autodiff.sin(x ** 2)
f.backward()
x.grad  # df/dx = 2 * 1.5 * cos(1.5²)
```

Available functions: `sin`, `exp`, `log`.

## Electromagnetism

```python
from axiompy.electromagnetism import Electromagnetism

q = 1e-9
charges = [
    Electromagnetism.Charge(q, (0, 0, 0)),
    Electromagnetism.Charge(-q, (1, 0, 0)),
]

E = Axiom.electromagnetism.calculate_electric_field(charges, (0.5, 0.5, 0))
# Vector([25.42, 0.0, 0.0])
```

`Electromagnetism.Charge` is a namedtuple with fields `q` (coulombs) and
`position` (3-tuple). `calculate_electric_field` returns a `Vector`.

## Statistics

```python
data = Axiom.Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
Axiom.stats.covariance_matrix(data)
```

Rows are observations, columns are variables. Uses `numpy.cov` internally
with `rowvar=False`.

## Visualization

### ASCII line plot

```python
x = [i * 0.4 for i in range(20)]
y = [v**2 for v in x]
Axiom.viz.plot_ascii(x, y)
```

Optional keyword arguments: `width` (default 60), `height` (default 15).

### ASCII field plot

```python
from axiompy import Axiom

def dipole(p):
    r1 = (p[0] - 0.5, p[1], p[2])
    r2 = (p[0] + 0.5, p[1], p[2])
    E1 = Axiom.electromagnetism.calculate_electric_field(
        [Axiom.electromagnetism.Charge(1e-9, (0.5, 0, 0))], p
    )
    return E1.to_list()

# Axiom.viz.plot_field_ascii(dipole, center=(0, 0, 0), size=4)
```

## Error handling

All custom exceptions inherit from `AxiomError` (defined in `_base.py`):

```python
from axiompy._base import AxiomError

try:
    inv = Axiom.number_theory.mod_inverse(2, 4)
except AxiomError as e:
    print(e)  # "Modular inverse does not exist for 2 and 4"
```
