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
Axiom.linalg.identity(3)          # 3x3 identity
Axiom.linalg.zeros((2, 3))        # 2x3 zeros
Axiom.linalg.ones((4, 1))         # 4x1 ones
```

### Advanced linear algebra (Phase 5.1)

```python
# Singular value decomposition
U, S, Vt = M.svd_decompose()

# Eigen decomposition (symmetric matrices)
M.eigenvalues()
M.eigenvectors()

# Condition number (via SVD)
M.condition_number()

# Moore-Penrose pseudoinverse
M.pinv()

# Least-squares solution to Ax = b
x = Axiom.linalg.least_squares(A_mat, b_vec)

# Cross-product matrix (skew-symmetric)
Axiom.linalg.cross_product_matrix(v)

# Householder reflection
Axiom.linalg.householder(v)
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
- `damping` (default `0.85`) -- PageRank damping factor
- `max_iter` (default `100`) -- maximum iterations
- `tol` (default `1e-6`) -- convergence tolerance
- `personalization` (dict, optional) -- per-node teleport weights

### Advanced graph algorithms (Phase 5.2)

```python
# Minimum spanning tree (Kruskal's)
g.minimum_spanning_tree()

# Bipartiteness
g.is_bipartite()
g.bipartite_sets()

# Topological sort (DAG only)
g.topological_sort()

# Maximum flow (Edmonds-Karp)
g.max_flow("source", "sink")

# Graph diameter (longest shortest path)
g.diameter()
```

## Number Theory

```python
# Chinese Remainder Theorem
#   x = 2 (mod 3)
#   x = 3 (mod 5)
#   x = 2 (mod 7)
Axiom.number_theory.chinese_remainder_theorem([3, 5, 7], [2, 3, 2])
# 23

# Modular inverse of 7 modulo 26
Axiom.number_theory.mod_inverse(7, 26)   # 15

# Extended Euclidean algorithm
Axiom.number_theory.extended_gcd(30, 20)  # (10, 1, -1)
```

### Advanced number theory (Phase 5.9)

```python
# Probabilistic primality test (Miller-Rabin)
Axiom.number_theory.miller_rabin(7919)

# Prime generation
Axiom.number_theory.next_prime(100)       # 101
Axiom.number_theory.nth_prime(10)         # 29

# Quadratic residue symbols
Axiom.number_theory.legendre_symbol(2, 7)
Axiom.number_theory.jacobi_symbol(7, 15)

# Discrete logarithm (baby-step giant-step)
Axiom.number_theory.discrete_log(2, 8, 13)  # 3

# Fibonacci and Lucas numbers
Axiom.number_theory.fibonacci(10)   # 55
Axiom.number_theory.lucas(10)       # 123
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
x.grad  # df/dx = 2 * 1.5 * cos(1.5^2)
```

Available functions: `sin`, `exp`, `log`, `tanh`, `sigmoid`, `sqrt`.

Optimization helper:

```python
from axiompy import Axiom

f = lambda x: x ** 2 + 2 * x + 1
minimum = Axiom.autodiff.gradient_descent(f, start=5.0, lr=0.1, steps=100)
# -1.0
```

### Advanced autodiff (Phase 5.6)

```python
# Instance methods
x = Axiom.autodiff.Variable(1.0)
x.tanh()     # same as Axiom.autodiff.tanh(x)
x.sigmoid()

# Computational graph visualization
z.to_ascii()

# Adam optimizer
Axiom.autodiff.adam(f, [0.0, 0.0], steps=100, lr=0.1)

# Hessian via forward-over-reverse
Axiom.autodiff.hessian(f, [1.0, 1.0])
```

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

### Advanced electromagnetism (Phase 5.10)

```python
# Electric potential (scalar)
Electromagnetism.electric_potential(charges, point)

# Magnetic field via Biot-Savart (moving charges)
Electromagnetism.calculate_magnetic_field(charges, point, velocities)

# Electric dipole moment
Electromagnetism.dipole_moment(charges)

# Field superposition
Electromagnetism.combine_fields([field1, field2])
```

## Statistics

```python
data = Axiom.Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
Axiom.stats.covariance_matrix(data)
```

Rows are observations, columns are variables.

### Advanced statistics (Phase 5.3)

```python
# Hypothesis tests
Statistics.ttest_1samp([1, 2, 3, 4, 5], 3.0)
Statistics.ttest_ind([1, 2, 3], [2, 3, 4])
Statistics.ttest_paired([1, 2, 3], [2, 3, 4])
Statistics.chisquare([20, 30, 25, 25], [25, 25, 25, 25])
Statistics.f_oneway([1, 2, 3], [2, 3, 4], [3, 4, 5])

# Distribution functions
Statistics.uniform_pdf(0.5)
Statistics.uniform_cdf(0.5)
Statistics.exponential_pdf(1.0, rate=2.0)
Statistics.binomial_pmf(3, 5, 0.5)

# Robust statistics
Statistics.zscore([1, 2, 3, 4, 5])    # standardized scores
Statistics.mad([1, 2, 3, 4, 5])       # median absolute deviation

# Covariance matrix with raw arrays and ddof
Statistics.covariance_matrix([[1, 2], [3, 4], [5, 6]], ddof=0)
```

## Visualization

### ASCII line plot

```python
x = [i * 0.4 for i in range(20)]
y = [v**2 for v in x]
Axiom.viz.plot_ascii(x, y)
```

Optional keyword arguments: `width` (default 60), `height` (default 15),
`title` (str), `xlabel` (str), `ylabel` (str), `extra_series` (list of
`(x_vals, y_vals, label)` tuples for multi-series with legend).

### Advanced visualization (Phase 5.11)

```python
# Histogram (using numpy.histogram internally)
Axiom.viz.plot_histogram(data, bins=10, title="Histogram")

# Horizontal bar chart
Axiom.viz.plot_bar(["A", "B", "C"], [30, 55, 20], title="Bar chart")

# Compact scatter plot (thin wrapper around plot_ascii)
Axiom.viz.plot_scatter(x, y, title="Scatter")
```

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

## Calculus

```python
from axiompy import Axiom
from axiompy.calculus import Calculus

f = lambda x: x ** 2

# Derivative at a point (central difference)
Calculus.numerical_derivative(f, 3.0)   # 6.0

# Integration
Calculus.integrate_trapezoid(f, 0, 1)    # ~ 1/3
Calculus.integrate_simpson(f, 0, 1)      # ~ 1/3
Calculus.integrate_monte_carlo(f, 0, 1)  # ~ 1/3

# Multivariate numerical gradient
g = lambda p: p[0]**2 + p[1]**2
Calculus.gradient(g, [1.0, 2.0])  # [2.0, 4.0]
```

### Advanced calculus (Phase 5.4)

```python
# Richardson extrapolation (high-order derivative)
Calculus.richardson_extrapolation(math.sin, 0.0)

# Gaussian quadrature (n = 2..8 supported)
Calculus.integrate_gauss_legendre(lambda x: x**2, 0, 1, n=5)

# Romberg integration
Calculus.integrate_romberg(lambda x: x**2, 0, 1)

# Numerical Jacobian matrix
Calculus.jacobian(lambda p: [p[0]**2 + p[1], p[0] - p[1]**3], [1.0, 2.0])

# Numerical Hessian matrix
Calculus.hessian(lambda p: p[0]**2 + 3*p[0]*p[1] + p[1]**2, [1.0, 1.0])

# ODE solvers
Calculus.ode_euler(lambda t, y: y, 1.0, (0.0, 2.0), dt=0.01)
Calculus.ode_rk4(lambda t, y: y, 1.0, (0.0, 2.0), dt=0.01)
```

## Polynomials

```python
from axiompy.polynomial import Polynomial

p = Polynomial([1, -3, 2])     # 2x^2 - 3x + 1
p(2)                           # 3
p.degree                        # 2
p.derivative()                  # Polynomial(4x - 3)
p.integral()                    # indefinite integral

# Roots via companion matrix
p.roots()                       # [0.5, 1.0]

# Lagrange interpolation
Polynomial.lagrange_interpolate([0, 1, 2], [0, 1, 4])  # x^2
```

Also accessible as `Axiom.Polynomial`.

### Advanced polynomial features (Phase 5.8)

```python
# Division with remainder
p // q                          # quotient
p % q                           # remainder

# Polynomial GCD (Euclidean algorithm)
p.gcd(q)

# Composition p(q(x))
p.compose(q)

# Chebyshev roots
Polynomial.chebyshev_roots(5)

# Least-squares polynomial fit
Polynomial.fit([0, 1, 2, 3, 4], [0, 1, 4, 9, 16], degree=2)
```

## Optimization

```python
from axiompy import Axiom
from axiompy.optimization import Optimization

f = lambda x: x ** 2 + 2 * x + 1
df = lambda x: 2 * x + 2
d2f = lambda x: 2.0

Optimization.gradient_descent(f, df, start=5.0, lr=0.1, steps=100)  # -1.0
Optimization.newton_method(f, df, d2f, start=5.0, steps=10)          # -1.0

# Root finding
Optimization.bisection(lambda x: x**2 - 4, 0, 5)  # 2.0
```

### Advanced optimization (Phase 5.5)

```python
# Golden section search (1-D)
Optimization.golden_section(lambda x: (x-2)**2, 0, 5)

# Conjugate gradient for SPD systems
Optimization.conjugate_gradient([[4, 1], [1, 3]], [1, 2], [0, 0])

# Nelder-Mead simplex (derivative-free)
Optimization.nelder_mead(lambda p: (p[0]-1)**2 + (p[1]-2)**2, [0, 0])

# L-BFGS (limited-memory BFGS)
Optimization.lbfgs(f, grad_f, start)

# Simulated annealing
Optimization.simulated_annealing(lambda p: (p[0]-3)**2 + (p[1]+1)**2, [0, 0])
```

## Signal Processing

```python
from axiompy.signal import Signal
import math

# DFT / FFT
x = [math.sin(2 * math.pi * 2 * k / 8) for k in range(8)]
X = Signal.dft(x)             # frequency domain
recovered = Signal.idft(X)    # back to time domain

# FFT (requires power-of-2 length, falls back to DFT otherwise)
X2 = Signal.fft(x)
rec2 = Signal.ifft(X2)

# Convolution
Signal.convolve([1, 2, 3], [0, 1, 0.5])  # [0.0, 1.0, 2.5, 4.0, 1.5]

# Moving average
Signal.moving_average([1, 2, 3, 4, 5], window=3)  # [2.0, 3.0, 4.0]
```

### Advanced signal processing (Phase 5.7)

```python
# Zero-pad to next power of 2
Signal.pad_next_power_of_two([1, 2, 3])

# Autocorrelation / cross-correlation (via FFT)
Signal.autocorrelation(x)
Signal.cross_correlation(x, y)

# FIR low-pass filter design (windowed sinc)
Signal.sinc_filter(cutoff=100, fs=1000, taps=31)

# Downsample / upsample
Signal.downsample(x, factor=2)
Signal.upsample(x, factor=2)

# Spectrogram (STFT magnitude)
Signal.spectrogram(x, window_size=64, hop_size=32)
```

## Bayesian Statistics (Phase 6.5)

```python
from axiompy import Axiom

# Conjugate families
bb = Axiom.BetaBinomial(alpha=2, beta=2)          # coin bias prior
post_bb = bb.posterior(k=7, n=10)                 # 7 heads in 10 flips
post_bb.mean()                                    # posterior mean
post_bb.credible_interval()                       # 95% approx CI

nn = Axiom.NormalNormal(mu0=0, sigma0=10)         # vague prior for mean
post_nn = nn.posterior([1.2, 1.5, 0.8], sigma2=0.25)

pg = Axiom.PoissonGamma(shape=1, rate=1)
post_pg = pg.posterior([2, 0, 3, 1, 2])

# Generic conjugate update
post = Axiom.posterior(bb, 'binomial', (7, 10))

# MCMC with Metropolis
def log_pdf(x):
    return -0.5 * x[0]**2
samples = Axiom.mcmc_metropolis(log_pdf, [0.0], steps=2000, proposal_std=1.0)
intervals = Axiom.credible_interval(samples, mass=0.95)
```

## ODE Solvers (Phase 6.4)

```python
from axiompy import Axiom

# Unified IVP solver with multiple methods
rhs = lambda t, y: [y[0]]  # dy/dt = y
ts, ys = Axiom.solve_ivp(rhs, [1.0], (0, 1), method='rk4', dt=0.01)
# methods: 'euler', 'rk4', 'rk45' (adaptive), 'adams_bashforth'

# BVP via shooting method
# y'' = -y, y(0)=0, y(pi/2)=1
def bvp_rhs(t, y):
    return [y[1], -y[0]]
def bvp_bc(y_end):
    return [y_end[0] - 1.0]
ts, ys = Axiom.solve_bvp(bvp_rhs, bvp_bc, (0, math.pi/2), [0, 1])

# System factories
pend = Axiom.pendulum_odes(g=9.81, L=1.0, b=0.2)
ts, ys = Axiom.solve_ivp(pend, [0.5, 0.0], (0, 10), method='rk4')

lv = Axiom.lotka_volterra_odes(alpha=1.0, beta=0.1, gamma=1.5, delta=0.075)
ts, ys = Axiom.solve_ivp(lv, [10.0, 2.0], (0, 50), method='rk4')
```

## Special Functions (Phase 6.3)

```python
from axiompy import Axiom

# Gamma & Beta
Axiom.gamma(5)            # 24.0
Axiom.beta(2, 2)          # 0.1666...

# Error functions
Axiom.erf(1.0)            # 0.8427...
Axiom.erfc(1.0)           # 0.1573...

# Bessel functions
Axiom.bessel_j(0, 1.0)    # J0(1) ≈ 0.765
Axiom.bessel_y(0, 1.0)    # Y0(1) ≈ 0.088

# Legendre polynomials
Axiom.legendre_p(2, 0.5)  # -0.125

# Combinatorial
Axiom.factorial(10)       # 3628800
Axiom.binomial(10, 3)     # 120
Axiom.double_factorial(7) # 105 (7!! = 7*5*3*1)
```

## Tensors (Phase 6.2)

```python
from axiompy import Axiom

# Create tensors
t = Axiom.Tensor([[1, 2, 3], [4, 5, 6]])
t.shape   # (2, 3)
t.ndim    # 2
t.size    # 6

# Arithmetic (broadcasting supported)
t + 10
t * Axiom.Tensor([1, 2, 3])

# Factory methods
Axiom.Tensor.zeros(2, 3)
Axiom.Tensor.ones(4)
Axiom.Tensor.eye(3)
Axiom.Tensor.linspace(0, 1, 5)
Axiom.Tensor.arange(0, 10, 2)

# Reductions
t.sum()   # scalar Tensor
t.mean()
t.max()
t.min()
t.std()

# Tensor operations
Axiom.Tensor.contract(a, b, axes=1)      # matmul via contraction
Axiom.Tensor.outer(a, b)                 # outer product
Axiom.Tensor.kronecker(a, b)             # Kronecker product
Axiom.Tensor.einsum("ij,jk->ik", a, b)   # Einstein summation

# Reshape / transpose
t.reshape(3, 2)
t.flatten()
t.T
```

## Complex Numbers (Phase 6.1)

```python
from axiompy import Axiom

# Create complex numbers
z = Axiom.ComplexNumber(3, 4)
z_polar = Axiom.ComplexNumber.from_polar(2.0, math.pi / 3)

# Properties and methods
z.real          # 3.0
z.imag          # 4.0
z.conjugate     # ComplexNumber(3, -4)
z.modulus()     # 5.0
z.argument()    # 0.9273 rad
z.power(2)      # De Moivre power

# Roots of unity
Axiom.ComplexNumber.roots_of_unity(5)  # 5 roots

# CFFT alias (delegates to Signal.fft)
Axiom.ComplexNumber.cfft([1, 0, -1, 0])

# Complex containers
v = Axiom.ComplexVector([1+2j, 3+4j])
m = Axiom.ComplexMatrix([[1j, 2j], [3j, 4j]])
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

## Configuration

AxiomPy can be configured via `pyproject.toml` or programmatically:

```toml
[tool.axiompy]
precision = 6    # rounding precision for repr
dtype = "float64"
verbose = false   # enables debug logging
```

Programmatic configuration at runtime:

```python
from axiompy import AxiomConfig

AxiomConfig.configure(precision=3, verbose=True)
Axiom.config.precision  # 3

# Reset to defaults
AxiomConfig.reset()
```

## Numeric Backend

Backend system allows swapping the array library:

```python
from axiompy import Axiom

Axiom.backend  # NumpyBackend instance

# Switch backends (initially only "numpy" is registered)
Axiom.set_backend("numpy")
```

Custom backends can be registered:

```python
from axiompy import Backend, register_backend

class MyBackend(Backend):
    def array(self, data, dtype=None): ...
    def dot(self, a, b): ...
    # ... implement all abstract methods

register_backend("mybackend", MyBackend)
Axiom.set_backend("mybackend")
```

## Logging

Each module has a `logging.Logger` at `axiompy.<module>`. Set log level via the facade or directly:

```python
import logging
logging.getLogger("axiompy").setLevel(logging.DEBUG)
```

When `verbose = true` in the config, debug logging is enabled automatically.

## Running tests

```bash
uv sync --dev
uv run pytest tests/ -v
```
