# Magical Functions (`Axiom.magic`)

Functional programming utilities and decorators.

## Timer Decorator

```python
from axiompy import Axiom

@Axiom.magic.timer
def work():
    return sum(range(1000000))

work()  # prints: # work took 12.345 ms
```

## `@magic` Decorator (cache + time)

```python
@Axiom.magic.magic(memoize=True, time_it=True)
def expensive(n):
    return sum(range(n))

expensive(1000000)  # computed + timed
expensive(1000000)  # returned from cache
```

## Pipe

```python
Axiom.magic.pipe(5, lambda x: x * 2, lambda x: x + 1)
# 11
```

## Compose

```python
add1 = lambda x: x + 1
double = lambda x: x * 2
Axiom.magic.compose(add1, double)(5)
# f(g(x)) = add1(double(5)) = 11
```

## API

| Method | Description |
|--------|-------------|
| `timer(func=None, unit='ms')` | Decorator — prints execution time |
| `magic(func=None, memoize=False, time_it=False, ttl=None)` | Decorator — cache + timer |
| `pipe(value, *functions)` | Feed value through function pipeline |
| `compose(*functions)` | Right-to-left function composition |
