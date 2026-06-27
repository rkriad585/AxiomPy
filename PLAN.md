# AxiomPy Development Plan

## Current state (v3.0.0)

| Module | What it has | Gap |
|---|---|---|
| `vector.py` | add, sub, mul (dot), div, cross, angle, magnitude, normalize | No neg, no abs, no scipy-style `norm(p=…)` |
| `matrix.py` | add, sub, mul, matmul, pow, det, inv, trace, rank, T | No decompositions (LU, QR, SVD, Cholesky), no `norm`, no solve |
| `linalg.py` | identity, zeros, ones | Tiny — missing most factory methods |
| `stats.py` | covariance_matrix | Only one function — no mean, median, var, std, distributions |
| `graph.py` | directed Graph, pagerank | No BFS/DFS, no shortest path, no MST, no undirected support |
| `autodiff.py` | Variable with +, -, *, /, **, sin, exp, log, backward | No tanh, sigmoid, sqrt, no higher-order grads, no optimizer |
| `number_theory.py` | extended_gcd, mod_inverse, CRT | No prime sieve, factorization, primality tests, totient |
| `electromagnetism.py` | Coulomb E-field | No B-field, no potential, no superposition helpers |
| `visualization.py` | plot_ascii, plot_field_ascii | No bar charts, no histogram, no multi-series |

---

## Phase 1 — Core completeness (current gaps)

### 1.1 Statistics (stats.py)
- [ ] `mean`, `median`, `mode`, `variance`, `std` (sample & population)
- [ ] `percentile`, `quartiles`, `iqr`
- [ ] `correlation_coefficient` (Pearson)
- [ ] `linear_regression` (simple OLS via normal equations)
- [ ] `normal_pdf`, `normal_cdf`
- [ ] `random_sample` (uniform, normal via Box-Muller)

### 1.2 Vector enhancements (vector.py)
- [ ] `__neg__`, `__abs__`
- [ ] `norm(ord=…)` — support L1, L2, L∞
- [ ] `dot` and `cross` as static methods too
- [ ] `project_onto(other)` — vector projection

### 1.3 Matrix decompositions (matrix.py or new `linalg.py`)
- [ ] `lu_decompose` → (L, U, P) , `qr_decompose` → (Q, R)
- [ ] `cholesky_decompose` (for SPD matrices)
- [ ] `solve_linear(A, b)` — solve Ax = b via LU
- [ ] `Matrix.norm(ord='fro')` — Frobenius norm

### 1.4 Graph algorithms (graph.py)
- [ ] Undirected edge support (optional `directed=False` param)
- [ ] `bfs(start)` / `dfs(start)` — traversal generators
- [ ] `shortest_path(source, target)` — Dijkstra
- [ ] `connected_components` for undirected graphs

### 1.5 Number theory (number_theory.py)
- [ ] `sieve_of_eratosthenes(n)` → list of primes up to n
- [ ] `is_prime(n)` — deterministic trial division up to √n
- [ ] `prime_factors(n)` → prime factorization
- [ ] `euler_totient(n)`

### 1.6 AutoDiff extras (autodiff.py)
- [ ] `tanh`, `sigmoid`, `sqrt`
- [ ] `Variable.__pow__` accept float (already int)
- [ ] `gradient_descent(fn, start, lr, steps)` — simple optimizer

---

## Phase 2 — New domains

### 2.1 Calculus module (`calculus.py`)
- [ ] `numerical_derivative(f, x, h=1e-6)` — central difference
- [ ] `integrate_trapezoid(f, a, b, n=100)`
- [ ] `integrate_simpson(f, a, b, n=100)`
- [ ] `integrate_monte_carlo(f, a, b, n=10000)`
- [ ] `gradient(f, point)` — numerical gradient for multivariate f

### 2.2 Polynomials (`polynomial.py`)
- [ ] `Polynomial` class with coefficients list
- [ ] `__call__`, `+`, `-`, `*`, `__eq__`
- [ ] `derivative()`, `integral()`
- [ ] `roots()` — via companion matrix eigenvalues
- [ ] `lagrange_interpolate(xs, ys)` — static factory

### 2.3 Optimization (`optimization.py`)
- [ ] `gradient_descent(f, grad_f, start, lr, steps)` — first-order
- [ ] `newton_method(f, df, d2f, start, steps)` — second-order
- [ ] `bisection(f, a, b, tol=1e-6)` — root finding

### 2.4 Signal / FFT (optional — `signal.py`)
- [ ] `dft(x)` / `idft(X)` — naive O(n²)
- [ ] `fft(x)` / `ifft(x)` — Cooley-Tukey O(n log n)
- [ ] `convolve(a, b)` — via FFT
- [ ] `moving_average(data, window)`

---

## Phase 3 — Customizability & developer experience

### 3.1 Configuration system
- [ ] `AxiomConfig` singleton or dataclass
- [ ] Overridable precision (`decimal_places` for display)
- [ ] Overridable default dtype (`float64` vs `float32`)
- [ ] Logging toggle / verbosity level
- [ ] Load config from `pyproject.toml` `[tool.axiompy]` section

```toml
[tool.axiompy]
precision = 6
dtype = "float64"
verbose = false
```

### 3.2 Numeric backend abstraction
- [ ] `Backend` ABC with `array`, `dot`, `inv`, `solve`, `det`, `fft`
- [ ] `NumpyBackend` (current, default)
- [ ] `PurePythonBackend` (for environments without numpy)
- [ ] Plugin registration: `Axiom.register_backend("pure", PurePythonBackend)`

### 3.3 Type system hardening
- [ ] `numpy.typing.NDArray` annotations in all modules
- [ ] Overload `__mul__`, `__add__` signatures
- [ ] `py.typed` marker file for PEP 561

### 3.4 Testing infrastructure
- [ ] `tests/` directory with `pytest`-compatible layout
- [ ] `test_vector.py`, `test_matrix.py`, `test_graph.py`, etc.
- [ ] CI workflow (GitHub Actions) running tests on push

### 3.5 Logging & debugging
- [ ] `import logging` — module-level logger per file
- [ ] Debug-level logging in iterative algorithms (PageRank, gradient descent)
- [ ] Optional timing/profiling decorator

---

## Phase 4 — Documentation & polish

- [ ] Docstrings for every public class and method (Google or NumPy style)
- [ ] Sphinx-based API reference (`docs/api/`)
- [ ] Type-checking CI step (`mypy` / `pyright`)
- [ ] Linter config (`.ruff.toml`) with pre-commit hook
- [ ] Changelog (`CHANGELOG.md`)

---

## How to pick what to build

1. **Phase 1 first** — fill out existing modules so they aren't one-trick ponies.
2. **Pick one module per session** — vector one day, stats the next. Small, mergable chunks.
3. **Add tests alongside code** — a `pytest` target in `pyproject.toml` makes it easy to verify.
4. **Tag milestones** — `v3.1.0` after Phase 1, `v3.2.0` after Phase 2, etc.

This plan is a living document — add, reorder, or drop items as priorities change.
