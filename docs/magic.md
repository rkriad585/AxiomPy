# Magical Functions (`Axiom.magic`)

Functional programming utilities, decorators, and number-theoretic novelty functions. Accessible via `Axiom.magic`.

## Decorators

### Timer

```python
from axiompy import Axiom

@Axiom.magic.timer
def work():
    return sum(range(1000000))

work()  # prints: # work took 12.345 ms
```

### `@magic` (cache + time)

```python
@Axiom.magic.magic(memoize=True, time_it=True)
def expensive(n):
    return sum(range(n))

expensive(1000000)  # computed + timed
expensive(1000000)  # returned from cache
```

## Functional Utilities

### Pipe

```python
Axiom.magic.pipe(5, lambda x: x * 2, lambda x: x + 1)
# 11
```

### Compose

```python
add1 = lambda x: x + 1
double = lambda x: x * 2
Axiom.magic.compose(add1, double)(5)
# f(g(x)) = add1(double(5)) = 11
```

## Number Fun

### Classic

| Method | Description |
|--------|-------------|
| `digit_sum(n)` | Sum of digits |
| `digital_root(n)` | Repeated digit sum until single digit |
| `is_palindrome(n)` | Check if number reads the same forwards/backwards |
| `reverse_number(n)` | Reverse the digits |
| `collatz(n)` | Collatz sequence starting from n |
| `happy_numbers(limit)` | All happy numbers up to limit |
| `armstrong_number(n)` | Check if n equals sum of its digits^digit_count |
| `perfect_number(n)` | Check if n equals sum of its proper divisors |
| `friendly_numbers(a, b)` | Check if a and b have the same abundancy index |
| `visualize_number(n)` | Full ASCII summary of n's properties |

### Primes & Factorization

| Method | Description |
|--------|-------------|
| `sieve_of_eratosthenes(n)` | All primes up to n |
| `emirp_numbers(limit)` | Primes that stay prime when reversed (non-palindromic) |
| `twin_primes(limit)` | Pairs of primes differing by 2 |
| `circular_primes(limit)` | Primes that stay prime under all digit rotations |
| `goldbach_conjecture(n)` | All pairs of primes summing to n (even n only) |

### Sequences & Patterns

| Method | Description |
|--------|-------------|
| `kaprekar_routine(n)` | 4-digit Kaprekar routine converging to 6174 |
| `look_and_say(n)` | First n terms of the look-and-say sequence |
| `ulam_spiral(n)` | (x, y, value) coordinates for the Ulam spiral |
| `narcissistic_numbers(limit)` | All narcissistic (Armstrong) numbers up to limit |
| `smith_numbers(limit)` | Numbers where digit sum equals prime-factor digit sum |
| `fibonacci_spiral(n)` | First n Fibonacci numbers |

### Conversion

| Method | Description |
|--------|-------------|
| `number_to_words(n)` | Spell out a number in English |
| `roman_numeral(n)` | Convert to Roman numerals (1–3999) |

### More

| Method | Description |
|--------|-------------|
| `factorial_digit_sum(n)` | Sum of digits of n! |
| `automorphic_numbers(limit)` | Numbers whose square ends with themselves |
| `harshad_numbers(limit)` | Numbers divisible by their digit sum |

## Examples

```python
Axiom.magic.roman_numeral(2024)         # 'MMXXIV'
Axiom.magic.number_to_words(42)         # 'forty-two'
Axiom.magic.sieve_of_eratosthenes(30)   # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
Axiom.magic.goldbach_conjecture(20)     # [(3, 17), (7, 13)]
Axiom.magic.look_and_say(5)             # [1, 11, 21, 1211, 111221]
Axiom.magic.automorphic_numbers(100)    # [1, 5, 6, 25, 76]
Axiom.magic.harshad_numbers(30)         # [1, ..., 10, 12, 18, 20, 21, 24, 27]
```
