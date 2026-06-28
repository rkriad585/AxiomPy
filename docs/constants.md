# Constants (`Axiom.constants`)

Built-in mathematical and physical constants. Every constant is a public attribute on `Axiom.constants`.

## Quick Start

```python
from axiompy import Axiom

Axiom.constants.PI       # 3.141592653589793
Axiom.constants.E        # 2.718281828459045
Axiom.constants.C        # 299792458.0 (speed of light)
Axiom.constants.G        # 6.67430e-11 (gravitational constant)
```

## Browse & Search

List all constants by category:

```python
Axiom.constants.list_by_category()
# {'Math': [...], 'Physics': [...], 'Atomic & Particle': [...], 'Astronomy': [...], 'Time': [...]}
```

Find a constant by name (case-insensitive, partial match):

```python
Axiom.constants.find("planck")  # returns H constant
Axiom.constants.find("mass")    # returns M_E, M_P, M_N, SOLAR_MASS, EARTH_MASS
```

Check total count:

```python
len(Axiom.constants)  # 35
```

## All Constants

### Math

| Constant | Value |
|----------|-------|
| `PI` | 3.141592653589793 |
| `E` | 2.718281828459045 |
| `PHI` | 1.618033988749895 |
| `SQRT2` | 1.4142135623730951 |
| `SQRT3` | 1.7320508075688772 |
| `LN2` | 0.6931471805599453 |
| `LN10` | 2.302585092994046 |
| `LOG2E` | 1.4426950408889634 |
| `LOG10E` | 0.4342944819032518 |
| `EULER` | 0.5772156649015329 |
| `INF` | inf |
| `NAN` | nan |
| `TAU` | 6.283185307179586 |

### Physics

| Constant | Value | Unit |
|----------|-------|------|
| `C` | 299792458.0 | m/s |
| `G` | 6.6743e-11 | N·m²/kg² |
| `H` | 6.62607015e-34 | J·Hz⁻¹ |
| `HBAR` | 1.0545718176469264e-34 | J·s |
| `KB` | 1.380649e-23 | J/K |
| `NA` | 6.02214076e+23 | mol⁻¹ |
| `R` | 8.314462618 | J/(mol·K) |
| `EPS0` | 8.854187817e-12 | F/m |
| `MU0` | 1.25663706212e-06 | N/A² |
| `E_CHARGE` | 1.602176634e-19 | C |
| `G0` | 9.80665 | m/s² |

### Atomic & Particle

| Constant | Value | Unit |
|----------|-------|------|
| `M_E` | 9.1093837015e-31 | kg |
| `M_P` | 1.67262192369e-27 | kg |
| `M_N` | 1.67492749804e-27 | kg |
| `M_MU` | 1.883531627e-28 | kg |
| `ALPHA` | 0.0072973525693 | — |
| `RYDBERG` | 10973731.56816 | m⁻¹ |
| `STEFAN_BOLTZMANN` | 5.670374419e-08 | W·m⁻²·K⁻⁴ |
| `WIEN` | 0.002897771955 | m·K |

### Astronomy

| Constant | Value | Unit |
|----------|-------|------|
| `AU` | 149597870700.0 | m |
| `PARSEC` | 3.085677581491367e+16 | m |
| `LIGHT_YEAR` | 9460730472580800.0 | m |
| `SOLAR_MASS` | 1.98847e+30 | kg |
| `EARTH_MASS` | 5.9722e+24 | kg |
| `EARTH_RADIUS` | 6371000.0 | m |

### Time

| Constant | Value | Unit |
|----------|-------|------|
| `MINUTE` | 60.0 | s |
| `HOUR` | 3600.0 | s |
| `DAY` | 86400.0 | s |
| `YEAR` | 31557600.0 | s |
