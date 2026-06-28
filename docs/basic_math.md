# Basic Math (`Axiom.math`)

Beginner-friendly math operations — arithmetic, PEMDAS, fractions, geometry formulas, statistics, probability, and number theory basics.

## Quick Start

```python
from axiompy import Axiom

Axiom.math.add(5, 3)        # 8
Axiom.math.sub(10, 4)       # 6
Axiom.math.mul(6, 7)        # 42
Axiom.math.div(10, 3)       # 3.333...
Axiom.math.power(2, 10)     # 1024
Axiom.math.sqrt(144)        # 12.0
```

## Fractions

```python
f = Axiom.Fraction(1, 3)
g = Axiom.Fraction(2, 5)
f + g  # Fraction(11, 15)
f * g  # Fraction(2, 15)
```

## All Methods

| Method | Description |
|--------|-------------|
| `add(a, b)` | Addition |
| `sub(a, b)` | Subtraction |
| `mul(a, b)` | Multiplication |
| `div(a, b)` | Division |
| `int_div(a, b)` | Integer division |
| `power(base, exp)` | Exponentiation |
| `sqrt(x)` | Square root |
| `nth_root(x, n)` | n-th root |
| `evaluate(expr)` | Evaluate a math expression string |
| `solve_linear(a, b)` | Solve ax + b = 0 |
| `pythagorean(a, b)` | Hypotenuse from legs |
| `pythagorean_leg(hyp, leg)` | Missing leg |
| `distance_traveled(speed, time)` | d = s × t |
| `speed(distance, time)` | s = d / t |
| `travel_time(distance, speed)` | t = d / s |
| `circle_area(r)` | πr² |
| `circle_circumference(r)` | 2πr |
| `square_area(s)` | s² |
| `square_perimeter(s)` | 4s |
| `rectangle_area(w, h)` | w × h |
| `rectangle_perimeter(w, h)` | 2(w + h) |
| `triangle_area(b, h)` | ½ × b × h |
| `percentage(part, whole)` | (part / whole) × 100 |
| `percent_of(percent, whole)` | (percent / 100) × whole |
| `percent_to_decimal(pct)` | Convert % to decimal |
| `decimal_to_percent(dec)` | Convert decimal to % |
| `mean(data)` | Arithmetic mean |
| `median(data)` | Middle value |
| `mode(data)` | Most frequent value |
| `factorial(n)` | n! |
| `is_prime(n)` | Primality test |
| `primes_up_to(n)` | List of primes ≤ n |
| `nth_prime(n)` | The n-th prime |
| `combinations(n, k)` | n choose k |
| `permutations(n, k)` | nPk |
| `coin_flip()` | Heads or tails |
| `dice_roll(sides)` | Random roll (default 6) |
