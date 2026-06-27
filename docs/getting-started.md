# Getting Started

## Installation

### From PyPI (recommended)

```bash
pip install axiom-math
```

### From source (development)

```bash
git clone https://github.com/rkriad585/AxiomPy.git
cd AxiomPy
uv sync          # create virtualenv + install deps
uv run python example.py  # run the demo
```

Requirements: Python ≥ 3.9, `uv` (or `pip`).

## Your first script

```python
from axiompy import Axiom

# Vectors and matrices
v = Axiom.Vector([1, 2, 3])
M = Axiom.Matrix([[4, 5, 6], [7, 8, 9]])

print(v.magnitude())          # 3.741...
print(M.determinant)          # 0.0 (singular matrix)

# Graph with PageRank
g = Axiom.Graph()
g.add_edge("A", "B")
g.add_edge("B", "C")
g.add_edge("C", "A")
print(Axiom.graph_analysis.pagerank(g))

# Automatic differentiation
x = Axiom.autodiff.Variable(3.0)
y = x ** 2 + 2 * x + 1
y.backward()
print(x.grad)                 # dy/dx = 8.0
```

## Running the examples

```bash
uv run python examples/linear_algebra.py
uv run python examples/pagerank.py
uv run python examples/autodiff.py
uv run python examples/number_theory.py
uv run python examples/electric_field.py
uv run python examples/covariance.py
uv run python examples/ascii_plot.py
```

## Building a wheel

```bash
uv build
```

The wheel is written to `dist/`.  It ships only the 12 modules under
`axiompy/`.
