# AxiomPy — Overview

AxiomPy is a Python mathematics engine built from first principles. It provides
a unified API for linear algebra, graph analysis, number theory, automatic
differentiation, electromagnetism, statistics, and ASCII visualization — all
wired through a single `Axiom` singleton.

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
  stats.py             # Statistics (covariance_matrix)
  autodiff.py          # AutoDiff Variable + backward mode
  number_theory.py     # NumberTheory (GCD, CRT, mod inverse)
  electromagnetism.py  # Electromagnetism (Coulomb E-field)
  visualization.py     # Visualization (ASCII plots)
```

## Public API reference

| Access path | What you get |
|---|---|
| `Axiom.Vector` | `Vector` class |
| `Axiom.Matrix` | `Matrix` class |
| `Axiom.Graph` | `Graph` class |
| `Axiom.linalg` | `LinearAlgebra` instance |
| `Axiom.stats` | `Statistics` instance |
| `Axiom.graph_analysis` | `GraphAnalysis` instance |
| `Axiom.autodiff` | `AutoDiff` (namespace for `Variable`, `sin`, `exp`, `log`) |
| `Axiom.number_theory` | `NumberTheory` instance |
| `Axiom.electromagnetism` | `Electromagnetism` instance |
| `Axiom.viz` | `Visualization` instance |

## Versioning

AxiomPy follows [Semantic Versioning](https://semver.org/).  The current
version is **3.0.0** (the restructured multi-module release).  All version
history lives in git — no duplicate snapshot files are shipped in the wheel.

## License

MIT — see the `LICENSE` file in the project root.
