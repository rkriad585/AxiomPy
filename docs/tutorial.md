# AxiomPy Tutorial

A step-by-step introduction for beginners. No advanced math background required.

## 1. Installation

```bash
pip install axiom-math
```

Or from source:

```bash
git clone https://github.com/rkriad585/AxiomPy.git
cd AxiomPy
uv sync
```

## 2. Your First Calculation

```python
from axiompy import Axiom

# Basic arithmetic
Axiom.math.add(5, 3)        # 8
Axiom.math.div(10, 3)       # 3.333...
Axiom.math.power(2, 10)     # 1024

# Evaluate a string expression
Axiom.math.evaluate("2 + 3 * 4")  # 14.0
```

## 3. Working with Constants

```python
# Built-in constants
Axiom.constants.PI          # 3.14159...
Axiom.constants.C           # 299792458 (speed of light, m/s)
Axiom.constants.G           # 6.6743e-11 (gravitational constant)

# Find constants by name
Axiom.constants.find("mass")
# Returns: M_E, M_P, M_N, SOLAR_MASS, EARTH_MASS

# Browse by category
Axiom.constants.list_by_category()
```

## 4. Using the CLI

```bash
# Evaluate an expression
axiompy eval "3.14 * 5**2"

# Factor a number
axiompy factors 84

# List constants
axiompy constants
axiompy constants speed

# Convert number formats
axiompy convert 255

# Get help
axiompy help
axiompy help magic
```

## 5. Number Patterns

```python
# Roman numerals
Axiom.magic.roman_numeral(2024)         # 'MMXXIV'

# Number spelling
Axiom.magic.number_to_words(42)         # 'forty-two'

# Prime numbers
Axiom.magic.sieve_of_eratosthenes(30)   # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# Kaprekar routine (4-digit numbers converge to 6174)
Axiom.magic.kaprekar_routine(3524)      # [3524, 3087, 8352, 6174]

# Goldbach partitions
Axiom.magic.goldbach_conjecture(20)     # [(3, 17), (7, 13)]

# Collatz sequence
Axiom.magic.collatz(10)                 # [10, 5, 16, 8, 4, 2, 1]

# Visual summary
print(Axiom.magic.visualize_number(8128))
```

## 6. Caching

```python
# Cache a result for 10 seconds
Axiom.cache.put("key", [1, 2, 3], ttl=10.0)
Axiom.cache.get("key")          # [1, 2, 3]

# Use as a decorator
@Axiom.cache.memoize(ttl=60)
def compute(x):
    return sum(range(x))

# Check cache performance
Axiom.cache.hits
Axiom.cache.hit_ratio
Axiom.cache.stats()
```

## 7. What's Next?

- Browse the full [API Reference](api/)
- Read the [Usage Guide](usage.md) for vectors, matrices, graphs, and more
- Try the interactive shell: `axiompy shell`
