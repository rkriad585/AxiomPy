"""Built-in mathematical and physical constants.

Accessible via ``Axiom.constants``.

Examples:
    >>> Axiom.constants.PI
    3.141592653589793
    >>> Axiom.constants.C
    299792458.0
"""

import math


class Constants:
    """A collection of useful math and physics constants.

    All values are read-only properties to prevent accidental mutation.
    """

    # ---- Math constants ----
    @property
    def PI(self) -> float:
        return math.pi

    @property
    def E(self) -> float:
        return math.e

    @property
    def PHI(self) -> float:
        return (1 + 5 ** 0.5) / 2

    @property
    def TAU(self) -> float:
        return 2 * math.pi

    @property
    def INF(self) -> float:
        return float("inf")

    @property
    def NAN(self) -> float:
        return float("nan")

    # ---- Physics constants (SI) ----
    @property
    def C(self) -> float:
        """Speed of light in vacuum (m/s)."""
        return 299792458.0

    @property
    def G(self) -> float:
        """Gravitational constant (m^3 kg^-1 s^-2)."""
        return 6.67430e-11

    @property
    def H(self) -> float:
        """Planck constant (J·s)."""
        return 6.62607015e-34

    @property
    def KB(self) -> float:
        """Boltzmann constant (J/K)."""
        return 1.380649e-23

    @property
    def NA(self) -> float:
        """Avogadro constant (mol^-1)."""
        return 6.02214076e23

    @property
    def EPS0(self) -> float:
        """Vacuum permittivity (F/m)."""
        return 8.854187817e-12

    @property
    def MU0(self) -> float:
        """Vacuum permeability (N/A^2)."""
        return 1.25663706212e-6

    @property
    def G0(self) -> float:
        """Standard gravity (m/s^2)."""
        return 9.80665

    def list_all(self):
        """Return a dict of all constant names and their values."""
        props = {}
        for name in dir(self):
            if name.isupper() and not name.startswith("_"):
                try:
                    props[name] = getattr(self, name)
                except Exception:
                    pass
        return props

    def __repr__(self) -> str:
        return f"Constants({len(self.list_all())} constants available)"
