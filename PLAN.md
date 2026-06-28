# AxiomPy Development Plan

## Current state (v4.0.0)

All 11 domain modules are complete with tests, docs, linting, and CI. See `CHANGELOG.md` for version history.

---

## Phase 5 — Algorithm depth (existing modules)

### 5.1 Linear algebra depth (`linalg.py`, `matrix.py`)
- [x] `Matrix.svd_decompose()` — singular value decomposition
- [x] `Matrix.eigenvalues()` / `Matrix.eigenvectors()` — symmetric eigendecomposition
- [x] `Matrix.condition_number()` — via SVD
- [x] `Matrix.pinv()` — Moore-Penrose pseudoinverse
- [x] `linalg.least_squares(A, b)` — via normal equations or SVD
- [x] `linalg.cross_product_matrix(v)` — skew-symmetric cross-product matrix
- [x] `linalg.householder(v)` — Householder reflection

### 5.2 Graph algorithms (`graph.py`)
- [x] `minimum_spanning_tree()` — Kruskal's / Prim's
- [x] `is_bipartite()`, `bipartite_sets()`
- [x] `topological_sort()` — Kahn's algorithm (DAG only)
- [x] `max_flow(source, sink)` — Ford-Fulkerson / Edmonds-Karp
- [x] `diameter()` — longest shortest path
- [x] `pagerank` → teleport parameter as per-node dict

### 5.3 Statistics depth (`stats.py`)
- [x] Hypothesis tests: `ttest_1samp`, `ttest_ind`, `ttest_paired`, `chisquare`
- [x] ANOVA: `f_oneway`
- [x] Distributions: `uniform_pdf`, `uniform_cdf`, `exponential_pdf`, `binomial_pmf`
- [x] `covariance_matrix` → accept raw arrays, support `ddof`
- [x] `zscore`, `mad` (median absolute deviation)

### 5.4 Calculus depth (`calculus.py`)
- [x] `integrate_gauss_legendre(f, a, b, n=5)` — Gaussian quadrature
- [x] `integrate_romberg(f, a, b, tol=1e-6)` — Romberg integration
- [x] `richardson_extrapolation(f, x, h=1e-3)` — higher-order derivative
- [x] `jacobian(f, point)` — numerical Jacobian matrix
- [x] `hessian(f, point)` — numerical Hessian matrix
- [x] `ode_euler(f, y0, t_span, dt)` — forward Euler ODE solver
- [x] `ode_rk4(f, y0, t_span, dt)` — classical Runge-Kutta

### 5.5 Optimization depth (`optimization.py`)
- [x] `conjugate_gradient(A, b, x0, tol=1e-6)` — CG for SPD systems
- [x] `nelder_mead(f, start, max_iter=1000)` — simplex (derivative-free)
- [x] `lbfgs(f, grad_f, start, max_iter=100)` — limited-memory BFGS
- [x] `golden_section(f, a, b, tol=1e-6)` — 1-D line search
- [x] `simulated_annealing(f, start, temp=1.0, cooling=0.95)`

### 5.6 AutoDiff depth (`autodiff.py`)
- [x] Higher-order gradients (Hessian-vector products)
- [x] `Variable.tanh()` / `Variable.sigmoid()` as methods (not just static)
- [x] Automatic `__pow__` support for float exponents
- [x] `gradient_descent` with momentum / Adam
- [x] Computational graph visualization (to_ascii / to_dot)

### 5.7 Signal depth (`signal.py`)
- [x] `spectrogram(x, window_size, hop_size)` — STFT-based spectrogram
- [x] `autocorrelation(x)` / `cross_correlation(x, y)`
- [x] `pad_next_power_of_two(x)` — zero-pad to next power of 2
- [x] `sinc_filter(cutoff, fs, taps=51)` — FIR low-pass design
- [x] `downsample(x, factor)` / `upsample(x, factor)`

### 5.8 Polynomial depth (`polynomial.py`)
- [x] `__floordiv__`, `__mod__` — polynomial division with remainder
- [x] `gcd(other)` — polynomial GCD via Euclidean algorithm
- [x] `chebyshev_roots(n)` — roots of Chebyshev polynomial (static)
- [x] `fit(xs, ys, degree)` — least-squares polynomial fit
- [x] `compose(other)` — polynomial composition p(q(x))

### 5.9 Number theory depth (`number_theory.py`)
- [x] `miller_rabin(n, k=10)` — probabilistic primality test
- [x] `next_prime(n)` — smallest prime ≥ n
- [x] `nth_prime(n)` — n-th prime
- [x] `legendre_symbol(a, p)` / `jacobi_symbol(a, n)`
- [x] `discrete_log(g, h, p)` — baby-step giant-step
- [x] `fibonacci(n)` / `lucas(n)`

### 5.10 Electromagnetism depth (`electromagnetism.py`)
- [ ] `calculate_magnetic_field(charges, point, velocities)` — Biot-Savart
- [ ] `electric_potential(charges, point)` — scalar potential
- [ ] `dipole_moment(charges)` — net dipole moment
- [ ] Field superposition helper: `combine_fields(fields)`

### 5.11 Visualization depth (`visualization.py`)
- [ ] `plot_histogram(data, bins=10)` — ASCII histogram
- [ ] `plot_bar(labels, values)` — ASCII bar chart
- [ ] `plot_scatter(xs, ys)` — ASCII scatter plot
- [ ] `plot_ascii` → optional title, x/y axis labels
- [ ] Multi-series support (legend in margin)

---

## Phase 6 — New domains

### 6.1 Complex numbers (`complex_numbers.py`)
- [ ] `Complex` class wrapping Python `complex` with polar/rect conversion
- [ ] `conjugate()`, `modulus()`, `argument()`, `power(n)`
- [ ] `roots_of_unity(n)` — static factory
- [ ] `ComplexMatrix` / `ComplexVector` convenience subclasses
- [ ] `fft` → `cfft` alias

### 6.2 Tensors (`tensor.py`)
- [ ] `Tensor` class — n-dimensional array wrapper
- [ ] Indexing, slicing, broadcasting
- [ ] `contract(a, b, axes)` — tensor contraction (generalized dot)
- [ ] `outer(a, b)` — outer product
- [ ] `kronecker(a, b)` — Kronecker product

### 6.3 Special functions (`special.py`)
- [ ] `gamma(z)` — Lanczos approximation
- [ ] `beta(a, b)` — via gamma
- [ ] `erf(x)`, `erfc(x)` — via continued fraction or Abramowitz & Stegun
- [ ] `bessel_j(n, x)` / `bessel_y(n, x)` — Bessel functions
- [ ] `legendre_p(n, x)` — Legendre polynomials
- [ ] `factorial(n)`, `binomial(n, k)`, `double_factorial(n)`

### 6.4 Differential equations (`odes.py`)
- [ ] `solve_ivp(f, y0, t_span, method='rk4', dt=0.01)` — general ODE solver
- [ ] Methods: Euler, RK4, RK45 (adaptive), Adams-Bashforth
- [ ] `solve_bvp(f, bc, x_span, guess)` — boundary value problems
- [ ] `pendulum_odes` / `lotka_volterra_odes` — example system factories

### 6.5 Bayesian statistics (`bayesian.py`)
- [ ] `BetaBinomial`, `NormalNormal`, `PoissonGamma` — conjugate families
- [ ] `posterior(prior, likelihood, data)` — generic conjugate update
- [ ] `credible_interval(distribution, mass=0.95)`
- [ ] `mcmc_metropolis(log_pdf, start, steps=1000, proposal_std=0.1)` — MCMC sampler

### 6.6 Fractals / chaos (`fractal.py`)
- [ ] `mandelbrot(width, height, x_range, y_range, max_iter)` — Mandelbrot set
- [ ] `julia(c, width, height, x_range, y_range, max_iter)` — Julia set
- [ ] `logistic_map(r, x0, n)` — logistic map iteration
- [ ] `bifurcation_diagram(r_range, x0, n_transient, n_plot)` — ASCII bifurcation
- [ ] `lyapunov_exponent(f, x0, n)` — estimate via orbit

### 6.7 Geometry / spatial (`geometry.py`)
- [ ] `Point`, `Line`, `Plane`, `Sphere` primitives
- [ ] Distance, intersection, projection helpers
- [ ] `convex_hull(points)` — Andrew's monotone chain
- [ ] `closest_pair(points)` — divide & conquer

### 6.8 Cryptography helpers (`crypto.py`)
- [ ] `rsa_keygen(bits=1024)` — generate public/private key pair
- [ ] `rsa_encrypt(message, public_key)` / `rsa_decrypt(ciphertext, private_key)`
- [ ] `elgamal_keygen()` / `elgamal_encrypt` / `elgamal_decrypt`
- [ ] `diffie_hellman_key_exchange(p, g, private_a, private_b)`
- [ ] `sha256(message)` — pure-Python SHA-256

---

## Phase 7 — Performance & portability

### 7.1 Pure Python backend
- [ ] `PurePythonBackend` implementing `Backend` ABC without numpy
- [ ] `PurePythonBackend.array` — list-of-lists with `__getitem__` / `__setitem__`
- [ ] `PurePythonBackend.dot` — O(n³) naive implementation
- [ ] `PurePythonBackend.solve` — Gaussian elimination
- [ ] `PurePythonBackend.fft` — naive O(n²) fallback
- [ ] Test suite runs against both backends via `@pytest.fixture(params=[...])`

### 7.2 JAX / CuPy backend
- [ ] `JaxBackend` — JIT-compiled via `jax.numpy`
- [ ] GPU support for large matrix operations
- [ ] `CuPyBackend` — CUDA-accelerated via `cupy`

### 7.3 Lazy evaluation
- [ ] `__array_function__` protocol where applicable
- [ ] Expression trees (deferred computation) for `Vector` / `Matrix`
- [ ] `compute()` to materialize lazy graphs

### 7.4 Memory optimization
- [ ] `Matrix.to_sparse()` — convert to COO/CSR format
- [ ] `SparseMatrix` — lightweight sparse matrix class
- [ ] Out-of-core operations for large arrays (memory-mapped numpy)

---

## Phase 8 — Interactive & ecosystem

### 8.1 Jupyter integration
- [ ] `_repr_html_` / `_repr_latex_` for `Vector`, `Matrix`, `Polynomial`
- [ ] LaTeX output via `$$` delimiters
- [ ] Jupyter widget for live parameter exploration

### 8.2 `axiompy` CLI tool
- [ ] `axiompy shell` — interactive REPL with `Axiom` pre-imported
- [ ] `axiompy demo` — run through all example scripts
- [ ] `axiompy info` — print version, backend, config

### 8.3 Publication
- [ ] Tag `v4.0.0` and push
- [ ] PyPI release via GitHub Release (`.github/workflows/publish.yml`)
- [ ] `conda-forge` recipe (`recipe/meta.yaml`)
- [ ] Zenodo DOI for academic citation
- [ ] PyPI download stats badge in README

### 8.4 Benchmarking
- [ ] `pytest-benchmark` integration for performance regression testing
- [ ] `benchmarks/` directory with comparison to numpy/scipy
- [ ] CI benchmark step (optional, on-label only)

---

## How to pick what to build

1. **Phase 5 first** — deepen existing modules; users already import them.
2. **Pick one sub-item per session** — small, mergable chunks.
3. **Add tests alongside code** — all 137 existing tests must stay green.
4. **Tag milestones** — `v4.1.0` after any Phase-5 module, `v5.0.0` after first Phase-6 domain, etc.
5. **Backend can wait** — Phase 7 is deferred until someone requests numpy-free usage.

This plan is a living document — add, reorder, or drop items as priorities change.
