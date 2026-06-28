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
- [x] `calculate_magnetic_field(charges, point, velocities)` — Biot-Savart
- [x] `electric_potential(charges, point)` — scalar potential
- [x] `dipole_moment(charges)` — net dipole moment
- [x] Field superposition helper: `combine_fields(fields)`

### 5.11 Visualization depth (`visualization.py`)
- [x] `plot_histogram(data, bins=10)` — ASCII histogram
- [x] `plot_bar(labels, values)` — ASCII bar chart
- [x] `plot_scatter(xs, ys)` — ASCII scatter plot
- [x] `plot_ascii` → optional title, x/y axis labels
- [x] Multi-series support (legend in margin)

---

## Phase 6 — New domains

### 6.1 Complex numbers (`complex_numbers.py`)
- [x] `ComplexNumber` class wrapping Python `complex` with polar/rect conversion
- [x] `conjugate`, `modulus()`, `argument()`, `power(n)`
- [x] `roots_of_unity(n)` — static factory
- [x] `ComplexMatrix` / `ComplexVector` convenience subclasses
- [x] `fft` → `cfft` alias

### 6.2 Tensors (`tensor.py`)
- [x] `Tensor` class — n-dimensional array wrapper
- [x] Indexing, slicing, broadcasting
- [x] `contract(a, b, axes)` — tensor contraction (generalized dot)
- [x] `outer(a, b)` — outer product
- [x] `kronecker(a, b)` — Kronecker product
- [x] `einsum(subscript, *operands)` — Einstein summation

### 6.3 Special functions (`special.py`)
- [x] `gamma(z)` — Lanczos approximation
- [x] `beta(a, b)` — via gamma
- [x] `erf(x)`, `erfc(x)` — Abramowitz & Stegun approximation
- [x] `bessel_j(n, x)` / `bessel_y(n, x)` — Bessel functions (series)
- [x] `legendre_p(n, x)` — Legendre polynomials (Bonnet recurrence)
- [x] `factorial(n)`, `binomial(n, k)`, `double_factorial(n)`

### 6.4 Differential equations (`odes.py`)
- [x] `solve_ivp(f, y0, t_span, method='rk4', dt=0.01)` — general ODE solver
- [x] Methods: Euler, RK4, RK45 (adaptive), Adams-Bashforth
- [x] `solve_bvp(f, bc, x_span, guess)` — boundary value problems (shooting)
- [x] `pendulum_odes` / `lotka_volterra_odes` — example system factories

### 6.5 Bayesian statistics (`bayesian.py`)
- [x] `BetaBinomial`, `NormalNormal`, `PoissonGamma` — conjugate families
- [x] `posterior(prior, likelihood, data)` — generic conjugate update
- [x] `credible_interval(samples, mass=0.95)` — equal-tailed interval from MCMC samples
- [x] `mcmc_metropolis(log_pdf, start, steps, proposal_std)` — Metropolis MCMC sampler

### 6.6 Fractals / chaos (`fractal.py`)
- [x] `mandelbrot(width, height, x_range, y_range, max_iter)` — Mandelbrot set
- [x] `julia(c, width, height, x_range, y_range, max_iter)` — Julia set
- [x] `logistic_map(r, x0, n)` — logistic map iteration
- [x] `bifurcation_diagram(r_range, x0, n_transient, n_plot)` — bifurcation points
- [x] `lyapunov_exponent(f, x0, n)` — estimate via orbit

### 6.7 Geometry / spatial (`geometry.py`)
- [x] `Point`, `Line`, `Plane`, `Sphere` primitives
- [x] Distance, intersection, projection helpers
- [x] `convex_hull(points)` — Andrew's monotone chain
- [x] `closest_pair(points)` — divide & conquer

### 6.8 Cryptography helpers (`crypto.py`)
- [x] `rsa_keygen(bits=1024)` — generate public/private key pair
- [x] `rsa_encrypt(message, public_key)` / `rsa_decrypt(ciphertext, private_key)`
- [x] `elgamal_keygen()` / `elgamal_encrypt` / `elgamal_decrypt`
- [x] `diffie_hellman_key_exchange(p, g, private_a, private_b)`
- [x] `sha256(message)` — pure-Python SHA-256

---

## Phase 7 — Performance & portability

### 7.1 Pure Python backend
- [x] `PurePythonBackend` implementing `Backend` ABC without numpy
- [x] `PurePythonBackend.array` — list-of-lists with `__getitem__` / `__setitem__`
- [x] `PurePythonBackend.dot` — O(n³) naive implementation
- [x] `PurePythonBackend.solve` — Gaussian elimination
- [x] `PurePythonBackend.fft` — naive O(n²) fallback
- [x] Test suite runs against both backends via parametrized fixture

### 7.2 JAX / CuPy backend
- [x] `JaxBackend` — JIT-compiled via `jax.numpy`
- [x] GPU support for large matrix operations
- [x] `CuPyBackend` — CUDA-accelerated via `cupy`

### 7.3 Lazy evaluation
- [x] `__array_function__` protocol where applicable
- [x] Expression trees (deferred computation) for `Vector` / `Matrix`
- [x] `compute()` to materialize lazy graphs

### 7.4 Memory optimization
- [x] `Matrix.to_sparse()` — convert to COO/CSR format
- [x] `SparseMatrix` — lightweight sparse matrix class in COO/CSR

### 7.5 Out-of-core & numerical precision
- [x] `MmapArray` — memory-mapped array with chunked ops
- [x] `open_mmap()` / `MmapArray.from_array()` / `MmapArray.zeros()`
- [x] Chunked `matmul`, `add`, `mul` for out-of-core workloads
- [x] `Matrix.from_mmap()` — load mmap into dense Matrix
- [x] `Vector.dtype` / `Matrix.dtype` property
- [x] `Vector.astype(dtype)` / `Matrix.astype(dtype)` — dtype conversion

---

## Phase 8 — Interactive & ecosystem

### 8.1 Jupyter integration
- [x] `_repr_html_` / `_repr_latex_` for `Vector`, `Matrix`, `Polynomial`, `SparseMatrix`
- [x] LaTeX output via `$$` delimiters
- [x] Jupyter widget for live parameter exploration (`PolynomialSliders`, `MatrixExplorer`)

### 8.2 `axiompy` CLI tool
- [x] `axiompy shell` — interactive REPL with `Axiom` pre-imported
- [x] `axiompy demo` — run through all example scripts
- [x] `axiompy info` — print version, backend, config

### 8.3 Publication
- [x] Tag `v4.0.0` and push (future — run ``git tag v4.0.0 && git push --tags``)
- [x] PyPI release via GitHub Release (``.github/workflows/publish.yml``)
- [x] `conda-forge` recipe (``recipe/meta.yaml``)
- [x] Zenodo DOI metadata (``.zenodo.json``)
- [x] PyPI download stats badge in README

### 8.4 Benchmarking
- [x] `pytest-benchmark` integration for performance regression testing
- [x] `benchmarks/` directory with comparison to numpy
- [x] CI benchmark step (``.github/workflows/benchmark.yml``, optional on-label only)

---

## Phase 9 — Core math & data ecosystem

### 9.1 Basic math operations
- [x] Arithmetic: addition, subtraction, multiplication, division, integer ops
- [x] PEMDAS order-of-operations expression evaluator
- [x] Prime number utilities (is_prime, nth_prime, primes_up_to)
- [x] Fractions: create, simplify, add, subtract, multiply, divide, compare
- [x] Decimals: precision-aware decimal wrapper
- [x] Percentages: convert, apply, difference
- [x] Linear equations: solve single-variable, system of 2 equations
- [x] Exponents: power, nth_root
- [x] Square roots (with rationalized output)
- [x] Geometry formulas: perimeter, area (square, rectangle, circle, triangle)
- [x] Pythagorean theorem solver
- [x] Mean (average), median, mode
- [x] Basic probability: coin flip, dice roll, combinations, permutations
- [x] Speed/distance/time formula

### 9.2 Custom data formats & I/O
- [x] ``.axi`` custom binary/text format for AxiomPy data
- [x] CSV export/import (no external libs)
- [x] JSON export/import
- [x] TXT export (pretty-print)
- [x] ``Axiom.io`` facade with ``save()`` / ``load()``

### 9.3 Built-in constants
- [x] Math constants: ``PI``, ``E``, ``PHI``, ``TAU``, ``INF``, ``NAN``
- [x] Physics constants: ``C`` (light speed), ``G`` (gravitational), ``H`` (Planck), ``KB`` (Boltzmann), ``NA`` (Avogadro), ``EPS0``, ``MU0``, ``G0`` (standard gravity)
- [x] Accessible via ``Axiom.constants``

### 9.4 Async caching system
- [x] ``Axiom.cache`` — thread-safe LRU cache decorator
- [x] Async variants for heavy computations
- [x] Cache stats (hits, misses, size)

### 9.5 CLI extensions
- [x] ``axiompy eval "2 + 3 * 4"`` — evaluate math expressions
- [x] ``axiompy convert 42`` — show number in different formats
- [x] ``axiompy factors 84`` — prime factorization
- [x] ``axiompy constants`` — list all built-in constants

### 9.6 "Magical" functions
- [x] ``Axiom.magic.digit_sum(n)`` — sum of digits
- [x] ``Axiom.magic.digital_root(n)`` — repeated digit sum
- [x] ``Axiom.magic.is_palindrome(n)`` — palindrome check
- [x] ``Axiom.magic.reverse_number(n)`` — reverse digits
- [x] ``Axiom.magic.collatz(n)`` — Collatz sequence
- [x] ``Axiom.magic.happy_numbers(limit)`` — happy numbers
- [x] ``Axiom.magic.armstrong_number(n)`` — Armstrong/narcissistic check
- [x] ``Axiom.magic.perfect_number(n)`` — perfect number check
- [x] ``Axiom.magic.friendly_numbers(a, b)`` — friendly pair check
- [x] ``Axiom.magic.visualize_number(n)`` — ASCII visual of number properties

### 9.7 Documentation
- [x] New ``docs/basic_math.md`` — beginner-friendly guide
- [x] New ``docs/data_io.md`` — import/export guide
- [x] New ``docs/constants.md`` — constants reference
- [x] New ``docs/magical.md`` — magical functions tour
- [x] All inline docstrings written for beginner audience
- [x] ``docs/usage.md`` updated with all new sections

---

---

## Phase 10 — Beginner-friendly expansion

### 10.1 Expand built-in constants
- [ ] Add ``R`` (gas constant 8.314462618)
- [ ] Add ``E_CHARGE`` (elementary charge 1.602176634e-19)
- [ ] Add ``HBAR`` (reduced Planck 1.054571817e-34)
- [ ] Add ``M_E`` (electron mass 9.1093837015e-31)
- [ ] Add ``M_P`` (proton mass 1.67262192369e-27)
- [ ] Add ``M_N`` (neutron mass 1.67492749804e-27)
- [ ] Add ``M_MU`` (muon mass 1.883531627e-28)
- [ ] Add ``ALPHA`` (fine-structure constant 7.2973525693e-3)
- [ ] Add ``RYDBERG`` (Rydberg constant 10973731.56816)
- [ ] Add ``STEFAN_BOLTZMANN`` (σ 5.670374419e-8)
- [ ] Add ``WIEN`` (Wien displacement 2.897771955e-3)
- [ ] Add ``E_CHARGE`` (electron charge)
- [ ] Add astronomical constants: ``AU``, ``PARSEC``, ``SOLAR_MASS``, ``EARTH_MASS``, ``EARTH_RADIUS``, ``LIGHT_YEAR``
- [ ] Add time constants: ``MINUTE``, ``HOUR``, ``DAY``, ``YEAR`` (in seconds)
- [ ] Add ``list_by_category()`` — group constants by category
- [ ] Add ``find(name)`` — search constants by name substring

### 10.2 Async caching with real ``async`` support
- [ ] ``Axiom.cache.put_async(key, value, ttl=None)`` — coroutine
- [ ] ``Axiom.cache.get_async(key)`` — coroutine
- [ ] ``@Axiom.cache.async_memoize(ttl=...)`` — decorator for ``async def`` functions
- [ ] Cache statistics: ``Axiom.cache.hits``, ``Axiom.cache.misses``, ``Axiom.cache.hit_ratio``
- [ ] ``Axiom.cache.stats()`` — return dict of all stats
- [ ] ``Axiom.cache.save(path)`` / ``Axiom.cache.load(path)`` — persist cache to disk
- [ ] ``Axiom.cache.info()`` — print formatted stats

### 10.3 CLI extensions
- [ ] ``axiompy eval "2 + 3 * 4"`` — evaluate math expression (uses BasicMath PEMDAS)
- [ ] ``axiompy factors 84`` — show prime factorization
- [ ] ``axiompy constants [search]`` — list / search built-in constants
- [ ] ``axiompy convert 42`` — show number in binary, octal, hex, roman, words
- [ ] ``axiompy stats`` — show cache + system stats
- [ ] ``axiompy help [topic]`` — detailed guidance (``math``, ``cache``, ``io``, etc.)

### 10.4 More magical functions
- [ ] ``Axiom.magic.sieve_of_eratosthenes(n)`` — classic prime sieve
- [ ] ``Axiom.magic.kaprekar_routine(n)`` — Kaprekar's routine (6174)
- [ ] ``Axiom.magic.look_and_say(n)`` — look-and-say sequence
- [ ] ``Axiom.magic.ulam_spiral(n)`` — Ulam spiral coordinates
- [ ] ``Axiom.magic.narcissistic_numbers(limit)`` — all Armstrong numbers
- [ ] ``Axiom.magic.smith_numbers(limit)`` — Smith numbers
- [ ] ``Axiom.magic.emirp_numbers(limit)`` — primes that stay prime reversed
- [ ] ``Axiom.magic.goldbach_conjecture(n)`` — Goldbach partition
- [ ] ``Axiom.magic.twin_primes(limit)`` — twin prime pairs
- [ ] ``Axiom.magic.circular_primes(limit)`` — circular primes
- [ ] ``Axiom.magic.number_to_words(n)`` — spell out number in English
- [ ] ``Axiom.magic.roman_numeral(n)`` — convert to Roman numerals
- [ ] ``Axiom.magic.factorial_digit_sum(n)`` — sum of digits of n!
- [ ] ``Axiom.magic.fibonacci_spiral(n)`` — first n Fibonacci numbers
- [ ] ``Axiom.magic.automorphic_numbers(limit)`` — squares ending with itself
- [ ] ``Axiom.magic.harshad_numbers(limit)`` — divisible by digit sum

### 10.5 Documentation overhaul
- [ ] Rewrite ``docs/basic_math.md`` — beginner tone, more examples, no jargon
- [ ] Rewrite ``docs/constants.md`` — with categories, search tips, example usage
- [ ] Rewrite ``docs/io.md`` — include error handling, practical walkthrough
- [ ] Rewrite ``docs/cache.md`` — add async section, stats, persistence
- [ ] Rewrite ``docs/magic.md`` — add all new magical functions with examples
- [ ] New ``docs/cli.md`` — complete CLI reference with examples
- [ ] New ``docs/tutorial.md`` — step-by-step beginner tutorial
- [ ] Update ``docs/usage.md`` — add Phase 10 modules

### 10.6 Inline docstring refresh
- [ ] All ``_constants.py`` docstrings — add units, typical use, source references
- [ ] All ``_cache.py`` docstrings — beginner-friendly descriptions with examples
- [ ] All ``cli.py`` docstrings — explain each command clearly
- [ ] All ``_magic.py`` docstrings — full examples, parameter explanations
- [ ] All ``_basic_math.py`` docstrings — plain English, no math jargon

---

## How to pick what to build

1. **Phase 5 first** — deepen existing modules; users already import them.
2. **Pick one sub-item per session** — small, mergable chunks.
3. **Add tests alongside code** — all 137 existing tests must stay green.
4. **Tag milestones** — `v4.1.0` after any Phase-5 module, `v5.0.0` after first Phase-6 domain, etc.
5. **Backend can wait** — Phase 7 is deferred until someone requests numpy-free usage.

This plan is a living document — add, reorder, or drop items as priorities change.
