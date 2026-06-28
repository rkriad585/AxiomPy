# AxiomPy: The Python Mathematics & Computation Engine

[![PyPI version](https://img.shields.io/pypi/v/axiompy.svg)](https://pypi.org/project/axiompy/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/axiompy.svg)](https://pypi.org/project/axiompy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/axiompy.svg)](https://pypi.org/project/axiompy/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14863522.svg)](https://doi.org/10.5281/zenodo.14863522)

**Author:** RK Riad Khan ([rkriad585](https://github.com/rkriad585))

---

## Overview

AxiomPy is a Python mathematics engine built from first principles. It provides
a unified API for linear algebra, graph analysis, number theory, automatic
differentiation, electromagnetism, statistics, and ASCII visualization — all
wired through a single `Axiom` singleton.

## Features

- **First-principle implementations** — algorithms are written from scratch,
  not wrapped.
- **Linear algebra** — `Vector` and `Matrix` with full operator overloading,
  cross product, determinant, inverse, trace, rank, matrix power.
- **Graph analysis** — directed graphs with PageRank algorithm.
- **Number theory** — extended GCD, modular inverse, Chinese Remainder Theorem.
- **Automatic differentiation** — reverse-mode AD engine (backpropagation)
  with `sin`, `exp`, `log` support.
- **Electromagnetism** — Coulomb electric field computation.
- **Statistics** — covariance matrix.
- **ASCII visualization** — terminal-based line and field plots.
- **Clean API** — `from axiompy import Axiom` gives access to everything.

## Installation

```bash
pip install axiom-math
```

Requirements: Python ≥ 3.9, numpy.

## Quick start

```python
from axiompy import Axiom

# Linear algebra
M = Axiom.Matrix([[1, 2], [3, 4]])
v = Axiom.Vector([5, 6])
print(M @ v)           # Vector([17.0, 39.0])
print(M.determinant)   # -2.0

# PageRank
g = Axiom.Graph()
g.add_edge("A", "B"); g.add_edge("B", "C"); g.add_edge("C", "A")
print(Axiom.graph_analysis.pagerank(g))

# Automatic differentiation
x = Axiom.autodiff.Variable(3.0)
y = x ** 2 + 2 * x + 1
y.backward()
print(x.grad)          # 8.0 (dy/dx)
```

## Documentation

Detailed guides are in the [`docs/`](docs/) directory:

| Document | Description |
|---|---|
| [Overview](docs/overview.md) | Architecture, module layout, public API reference |
| [Getting Started](docs/getting-started.md) | Installation, first script, running examples |
| [Usage Guide](docs/usage.md) | Full API reference with code snippets per module |
| [Contributing](docs/contributing.md) | Development setup, conventions, building |

## Examples

Ready-to-run example scripts are in [`examples/`](examples/):

```bash
uv run python examples/linear_algebra.py
uv run python examples/pagerank.py
uv run python examples/autodiff.py
uv run python examples/number_theory.py
uv run python examples/electric_field.py
uv run python examples/covariance.py
uv run python examples/ascii_plot.py
```

## Development

```bash
git clone https://github.com/rkriad585/AxiomPy.git
cd AxiomPy
uv sync
uv build
```

## Contributing

See [`docs/contributing.md`](docs/contributing.md) for guidelines.  Bug
reports and pull requests are welcome on the
[GitHub repository](https://github.com/rkriad585/AxiomPy).

## License

MIT — see [LICENSE](LICENSE).
