# AxiomPy — Overview

AxiomPy is a Python mathematics engine built from first principles. It provides
a unified API for linear algebra, calculus, graph analysis, number theory,
automatic differentiation, optimization, signal processing, polynomials,
electromagnetism, statistics, and ASCII visualization — all wired through a
single `Axiom` singleton.

## Philosophy

- **Transparency.** Algorithms are implemented from the ground up.  No opaque
  black boxes — the code itself teaches how PageRank, backward-mode AD, the
  Chinese Remainder Theorem, and other algorithms work.
- **Minimal dependencies.** Only `numpy` is required.  Everything else — matrix
  inversion, eigenvalue-avoiding page-rank, automatic differentiation graphs —
  is hand-rolled.
- **Clean API.** `from axiompy import Axiom` gives you `Vector`, `Matrix`,
  `Graph`, and seven domain-specific sub-modules through properties.

## Package structure

```
axiompy/
  __init__.py          # from ._facade import Axiom
  _base.py             # AxiomError + type aliases
  _facade.py           # AxiomPy singleton wiring
  vector.py            # Vector class
  matrix.py            # Matrix class
  graph.py             # Graph + GraphAnalysis.pagerank
  linalg.py            # LinearAlgebra (identity, zeros, ones)
  stats.py             # Statistics (descriptive, regression, normal dist)
  autodiff.py          # AutoDiff Variable + backward mode
  number_theory.py     # NumberTheory (GCD, CRT, primes, totient)
  electromagnetism.py  # Electromagnetism (Coulomb E-field)
  visualization.py     # Visualization (ASCII plots)
  calculus.py          # Calculus (numerical derivative, integration)
  polynomial.py        # Polynomial class (arithmetic, roots, interpolation)
  optimization.py      # Optimization (gradient descent, Newton, bisection)
  signal.py            # Signal processing (DFT, FFT, convolution)
```

## Public API reference

| Access path | What you get |
|---|---|
| `Axiom.Vector` | `Vector` class |
| `Axiom.Matrix` | `Matrix` class |
| `Axiom.Graph` | `Graph` class |
| `Axiom.Polynomial` | `Polynomial` class |
| `Axiom.linalg` | `LinearAlgebra` instance |
| `Axiom.stats` | `Statistics` instance |
| `Axiom.graph_analysis` | `GraphAnalysis` instance |
| `Axiom.autodiff` | `AutoDiff` (namespace for `Variable`, `sin`, `exp`, `log`, `tanh`, `sigmoid`, `sqrt`) |
| `Axiom.number_theory` | `NumberTheory` instance |
| `Axiom.electromagnetism` | `Electromagnetism` instance |
| `Axiom.viz` | `Visualization` instance |
| `Axiom.calc` | `Calculus` instance |
| `Axiom.optimization` | `Optimization` instance |
| `Axiom.signal` | `Signal` instance |

## Versioning

AxiomPy follows [Semantic Versioning](https://semver.org/).  The current
version is **3.2.0** (Phase 2: calculus, polynomials, optimization, signal
processing).  All version history lives in git — no duplicate snapshot files
are shipped in the wheel.

## License

MIT — see the `LICENSE` file in the project root.
