# AxiomPy: The Python Mathematics & Computation Engine

[![PyPI version](https://img.shields.io/pypi/v/axiom-math.svg)](https://pypi.org/project/axiom-math/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/axiom-math.svg)](https://pypi.org/project/axiom-math/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/axiom-math.svg)](https://pypi.org/project/axiom-math/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14863522.svg)](https://doi.org/10.5281/zenodo.14863522)

**Author:** RK Riad Khan ([rkriad585](https://github.com/rkriad585))

---

## Overview

AxiomPy is a Python mathematics engine built from first principles. It provides
a unified API for linear algebra, graph analysis, number theory, automatic
differentiation, optimization, signal processing, statistics, caching,
constants, CLI tools, and number-theoretic novelties — all wired through a
single `Axiom` singleton.

## Features

- **First-principle implementations** — algorithms written from scratch.
- **Linear algebra** — `Vector`, `Matrix`, `Tensor`, `SparseMatrix`, decompositions.
- **Graph analysis** — PageRank, MST, max flow, bipartiteness, topological sort.
- **Number theory** — CRT, Miller-Rabin, discrete log, totient, prime sieves.
- **Automatic differentiation** — reverse-mode AD with sin, exp, log, tanh.
- **Calculus** — numerical derivatives, integration, ODE solvers (RK4, adaptive).
- **Optimization** — gradient descent, Newton, Nelder-Mead, L-BFGS, simulated annealing.
- **Signal processing** — FFT, convolution, FIR filters, spectrograms.
- **Statistics** — descriptive stats, hypothesis tests, distributions, Bayesian.
- **Electromagnetism** — Coulomb fields, potentials, magnetic fields.
- **Constants** — 35 built-in math/physics/astronomy constants with search.
- **Cache** — LRU cache with TTL, hit/miss stats, async memoize, persistence.
- **CLI** — `axiompy eval`, `factors`, `convert`, `constants`, `stats`, `help`, `shell`.
- **Magic functions** — roman numerals, number words, Kaprekar, Goldbach, look-and-say, Ulam spiral, Smith numbers, emirps, and more.
- **ASCII visualization** — line plots, histograms, bar charts, field plots.
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

# Constants & CLI
Axiom.constants.PI           # 3.14159...
Axiom.constants.find("mass") # search by name
```

## CLI

```bash
axiompy eval "2 * pi * 5^2"    # evaluate expression
axiompy factors 84              # prime factorization
axiompy constants mass          # search constants
axiompy convert 42              # binary, hex, roman, words
axiompy stats                   # cache & system info
axiompy help magic              # topic help
axiompy shell                   # interactive REPL
```

## Documentation

Guides are in the [`docs/`](docs/) directory:

| Document | Description |
|---|---|
| [Tutorial](docs/tutorial.md) | Step-by-step intro for beginners |
| [CLI Reference](docs/cli.md) | Command-line tool usage |
| [Overview](docs/overview.md) | Architecture, module layout, API reference |
| [Usage Guide](docs/usage.md) | Full API with code snippets per module |
| [Getting Started](docs/getting-started.md) | Installation, first script, examples |

## Examples

Ready-to-run scripts in [`examples/`](examples/):

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

See [`docs/contributing.md`](docs/contributing.md). Bug reports and pull
requests welcome on the [GitHub repository](https://github.com/rkriad585/AxiomPy).

## License

MIT — see [LICENSE](LICENSE).
