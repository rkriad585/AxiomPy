# Basic Math (`Axiom.math`)

Beginner-friendly math operations covering arithmetic, PEMDAS, fractions, decimals, percentages, linear equations, exponents, square roots, geometry formulas, statistics, and probability.

## Quick Start

```python
from axiompy import Axiom

Axiom.math.add(5, 3)        # 8
Axiom.math.subtract(10, 4)  # 6
Axiom.math.multiply(6, 7)   # 42
Axiom.math.divide(10, 3)    # 3.333...
```

## Fractions

```python
from axiompy import Axiom

f = Axiom.Fraction(1, 3)
g = Axiom.Fraction(2, 5)
f + g  # Fraction(11, 15)
f * g  # Fraction(2, 15)
```

## All Methods

| Method | Description |
|--------|-------------|
| `add(a, b)` | Addition |
| `subtract(a, b)` | Subtraction |
| `multiply(a, b)` | Multiplication |
| `divide(a, b)` | Division |
| `power(base, exp)` | Exponentiation |
| `sqrt(x)` | Square root |
| `percent(part, whole)` | Percentage |
| `linear_eq(m, x, b)` | y = mx + b |
| `factorial(n)` | n! |
| `gcd(a, b)` | Greatest common divisor |
| `lcm(a, b)` | Least common multiple |
| `is_prime(n)` | Primality test |
| `factorize(n)` | Prime factorization |
