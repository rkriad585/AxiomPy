# Changelog

## [3.3.0] — Phase 3: Configuration, Testing, CI, Logging
- Configuration system: `AxiomConfig` dataclass loaded from `[tool.axiompy]` in `pyproject.toml`
- Backend abstraction: `Backend` ABC with `NumpyBackend`, plugin registration
- Config-aware `Matrix`/`Vector` repr using configured precision
- Module-level logging in all 11 domain modules
- Testing infrastructure: 13 test files, 137 tests with `pytest`
- GitHub Actions CI workflow (`test.yml`) for push/PR on 3.9–3.11
- `py.typed` marker for PEP 561 compliance
- Bugfix: `to_adjacency_matrix` includes neighbor-only nodes
- Bugfix: `__rmatmul__` for `NotImplemented` protocol compliance

## [3.2.0] — Phase 2: New domains
- `calculus.py`: numerical derivative, trapezoid/simpson/Monte Carlo integration, multivariate gradient
- `polynomial.py`: `Polynomial` class with arithmetic, calculus, companion-matrix roots, Lagrange interpolation
- `optimization.py`: gradient descent, Newton's method, bisection
- `signal.py`: DFT/IDFT, FFT/IFFT (Cooley-Tukey), convolution, moving average
- All new modules wired into `Axiom` facade

## [3.1.0] — Phase 1: Core completeness
- `stats.py`: mean, median, mode, variance, std, percentile, quartiles, IQR, Pearson correlation, linear regression, normal PDF/CDF, random sampling
- `vector.py`: `__neg__`, `__abs__`, `norm(ord=…)`, `dot()`, `project_onto()`, static dot/cross
- `matrix.py`: `norm(ord=…)`, `lu_decompose`, `qr_decompose`, `cholesky_decompose`
- `linalg.py`: `solve_linear(A, b)`
- `graph.py`: directed/undirected, edge weights, BFS, DFS, Dijkstra shortest path, connected components
- `number_theory.py`: sieve, `is_prime`, `prime_factors`, `euler_totient`
- `autodiff.py`: `tanh`, `sigmoid`, `sqrt`, gradient descent helper

## [3.0.0] — Package restructuring
- Migrated from `setup.py` to `pyproject.toml` (PEP 621, hatchling)
- Split monolithic `core.py` into domain modules: `vector.py`, `matrix.py`, `linalg.py`, `stats.py`, `graph.py`, `autodiff.py`, `number_theory.py`, `electromagnetism.py`, `visualization.py`
- Unified `Axiom` singleton facade
- Deleted `_versions/` archive snapshots
- `example.py` → `examples/` directory with domain-specific scripts
- `docs/` with overview, getting-started, usage, contributing
- Infrastructure: `Makefile`, `Dockerfile`, `build.sh`, `installer.sh`, etc.
