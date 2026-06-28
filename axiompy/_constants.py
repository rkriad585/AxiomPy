"""Built-in mathematical and physical constants.

Accessible via ``Axiom.constants``.

Examples:
    >>> Axiom.constants.PI
    3.141592653589793
    >>> Axiom.constants.C
    299792458.0
    >>> Axiom.constants.R
    8.314462618
"""

import math


class Constants:
    """A collection of useful math and physics constants.

    All values are read-only properties to prevent accidental mutation.
    Use ``Axiom.constants.list_all()`` to see every constant,
    ``Axiom.constants.list_by_category()`` for grouped output, and
    ``Axiom.constants.find(name)`` to search by name.
    """

    # ---- Math constants ----
    @property
    def PI(self) -> float:
        r"""The ratio of a circle's circumference to its diameter.

        .. math:: \pi \approx 3.141592653589793

        Examples:
            >>> Axiom.constants.PI
            3.141592653589793
        """
        return math.pi

    @property
    def E(self) -> float:
        r"""Euler's number, base of natural logarithms.

        .. math:: e \approx 2.718281828459045

        Examples:
            >>> Axiom.constants.E
            2.718281828459045
        """
        return math.e

    @property
    def PHI(self) -> float:
        r"""Golden ratio.

        .. math:: \phi = \frac{1 + \sqrt{5}}{2} \approx 1.618033988749895

        Examples:
            >>> Axiom.constants.PHI
            1.618033988749895
        """
        return (1 + 5 ** 0.5) / 2

    @property
    def TAU(self) -> float:
        r"""Tau, equal to :math:`2\pi`.

        .. math:: \tau = 2\pi \approx 6.283185307179586

        Examples:
            >>> Axiom.constants.TAU
            6.283185307179586
        """
        return 2 * math.pi

    @property
    def INF(self) -> float:
        """Positive infinity.

        Useful for initializing minima in search algorithms.

        Examples:
            >>> Axiom.constants.INF
            inf
        """
        return float("inf")

    @property
    def NAN(self) -> float:
        """Not-a-Number (NaN).

        Results from undefined operations like 0/0.

        Examples:
            >>> import math
            >>> math.isnan(Axiom.constants.NAN)
            True
        """
        return float("nan")

    # ---- Physics constants (SI) ----
    @property
    def C(self) -> float:
        """Speed of light in vacuum (m/s).

        Exact value: 299,792,458 m/s.

        Examples:
            >>> Axiom.constants.C
            299792458.0
        """
        return 299792458.0

    @property
    def G(self) -> float:
        r"""Gravitational constant (m³·kg⁻¹·s⁻²).

        .. math:: G \approx 6.67430 \times 10^{-11}

        Examples:
            >>> Axiom.constants.G
            6.6743e-11
        """
        return 6.67430e-11

    @property
    def H(self) -> float:
        r"""Planck constant (J·s).

        .. math:: h \approx 6.62607015 \times 10^{-34}

        Examples:
            >>> Axiom.constants.H
            6.62607015e-34
        """
        return 6.62607015e-34

    @property
    def HBAR(self) -> float:
        r"""Reduced Planck constant (J·s).

        .. math:: \hbar = \frac{h}{2\pi} \approx 1.054571817 \times 10^{-34}

        Examples:
            >>> Axiom.constants.HBAR
            1.054571817e-34
        """
        return 1.0545718176469265e-34

    @property
    def KB(self) -> float:
        r"""Boltzmann constant (J/K).

        .. math:: k_B \approx 1.380649 \times 10^{-23}

        Examples:
            >>> Axiom.constants.KB
            1.380649e-23
        """
        return 1.380649e-23

    @property
    def NA(self) -> float:
        r"""Avogadro constant (mol⁻¹).

        .. math:: N_A \approx 6.02214076 \times 10^{23}

        Examples:
            >>> Axiom.constants.NA
            6.02214076e+23
        """
        return 6.02214076e23

    @property
    def R(self) -> float:
        r"""Universal gas constant (J/(mol·K)).

        .. math:: R = k_B \cdot N_A \approx 8.314462618

        Examples:
            >>> Axiom.constants.R
            8.314462618
        """
        return 8.314462618

    @property
    def EPS0(self) -> float:
        r"""Vacuum permittivity (F/m).

        .. math:: \varepsilon_0 \approx 8.854187817 \times 10^{-12}

        Examples:
            >>> Axiom.constants.EPS0
            8.854187817e-12
        """
        return 8.854187817e-12

    @property
    def MU0(self) -> float:
        r"""Vacuum permeability (N/A²).

        .. math:: \mu_0 = 4\pi \times 10^{-7} \approx 1.25663706212 \times 10^{-6}

        Examples:
            >>> Axiom.constants.MU0
            1.25663706212e-06
        """
        return 1.25663706212e-6

    @property
    def E_CHARGE(self) -> float:
        r"""Elementary charge (C).

        .. math:: e \approx 1.602176634 \times 10^{-19}

        Examples:
            >>> Axiom.constants.E_CHARGE
            1.602176634e-19
        """
        return 1.602176634e-19

    @property
    def G0(self) -> float:
        """Standard gravity (m/s²).

        Examples:
            >>> Axiom.constants.G0
            9.80665
        """
        return 9.80665

    # ---- Particle masses (kg) ----
    @property
    def M_E(self) -> float:
        r"""Electron rest mass (kg).

        .. math:: m_e \approx 9.1093837015 \times 10^{-31}

        Examples:
            >>> Axiom.constants.M_E
            9.1093837015e-31
        """
        return 9.1093837015e-31

    @property
    def M_P(self) -> float:
        r"""Proton rest mass (kg).

        .. math:: m_p \approx 1.67262192369 \times 10^{-27}

        Examples:
            >>> Axiom.constants.M_P
            1.67262192369e-27
        """
        return 1.67262192369e-27

    @property
    def M_N(self) -> float:
        r"""Neutron rest mass (kg).

        .. math:: m_n \approx 1.67492749804 \times 10^{-27}

        Examples:
            >>> Axiom.constants.M_N
            1.67492749804e-27
        """
        return 1.67492749804e-27

    @property
    def M_MU(self) -> float:
        r"""Muon rest mass (kg).

        .. math:: m_\mu \approx 1.883531627 \times 10^{-28}

        Examples:
            >>> Axiom.constants.M_MU
            1.883531627e-28
        """
        return 1.883531627e-28

    # ---- Derived dimensionless constants ----
    @property
    def ALPHA(self) -> float:
        r"""Fine-structure constant (dimensionless).

        .. math:: \alpha \approx 7.2973525693 \times 10^{-3} \approx \frac{1}{137}

        Examples:
            >>> Axiom.constants.ALPHA
            0.0072973525693
        """
        return 7.2973525693e-3

    @property
    def RYDBERG(self) -> float:
        r"""Rydberg constant (m⁻¹).

        .. math:: R_\infty \approx 10973731.56816

        Examples:
            >>> Axiom.constants.RYDBERG
            10973731.56816
        """
        return 10973731.56816

    @property
    def STEFAN_BOLTZMANN(self) -> float:
        r"""Stefan-Boltzmann constant (W·m⁻²·K⁻⁴).

        .. math:: \sigma \approx 5.670374419 \times 10^{-8}

        Examples:
            >>> Axiom.constants.STEFAN_BOLTZMANN
            5.670374419e-08
        """
        return 5.670374419e-8

    @property
    def WIEN(self) -> float:
        r"""Wien wavelength displacement constant (m·K).

        .. math:: b \approx 2.897771955 \times 10^{-3}

        Examples:
            >>> Axiom.constants.WIEN
            0.002897771955
        """
        return 2.897771955e-3

    # ---- Astronomical constants ----
    @property
    def AU(self) -> float:
        """Astronomical Unit (m).

        Mean Earth-Sun distance.

        Examples:
            >>> Axiom.constants.AU
            149597870700.0
        """
        return 149597870700.0

    @property
    def PARSEC(self) -> float:
        """Parsec (m).

        About 3.2616 light-years.

        Examples:
            >>> Axiom.constants.PARSEC
            3.085677581491367e+16
        """
        return 3.085677581491367e16

    @property
    def LIGHT_YEAR(self) -> float:
        """Light-year (m).

        Distance light travels in one Julian year.

        Examples:
            >>> Axiom.constants.LIGHT_YEAR
            9.4607304725808e+15
        """
        return 9.4607304725808e15

    @property
    def SOLAR_MASS(self) -> float:
        r"""Solar mass (kg).

        .. math:: M_\odot \approx 1.98847 \times 10^{30}

        Examples:
            >>> Axiom.constants.SOLAR_MASS
            1.98847e+30
        """
        return 1.98847e30

    @property
    def EARTH_MASS(self) -> float:
        r"""Earth mass (kg).

        .. math:: M_\oplus \approx 5.9722 \times 10^{24}

        Examples:
            >>> Axiom.constants.EARTH_MASS
            5.9722e+24
        """
        return 5.9722e24

    @property
    def EARTH_RADIUS(self) -> float:
        """Earth mean radius (m).

        Examples:
            >>> Axiom.constants.EARTH_RADIUS
            6371000.0
        """
        return 6_371_000.0

    # ---- Time constants (seconds) ----
    @property
    def MINUTE(self) -> float:
        """One minute in seconds.

        Examples:
            >>> Axiom.constants.MINUTE
            60.0
        """
        return 60.0

    @property
    def HOUR(self) -> float:
        """One hour in seconds.

        Examples:
            >>> Axiom.constants.HOUR
            3600.0
        """
        return 3600.0

    @property
    def DAY(self) -> float:
        """One day in seconds.

        Examples:
            >>> Axiom.constants.DAY
            86400.0
        """
        return 86400.0

    @property
    def YEAR(self) -> float:
        """One Julian year in seconds (365.25 days).

        Examples:
            >>> Axiom.constants.YEAR
            31557600.0
        """
        return 31557600.0

    # ---- Convenience methods ----
    def list_all(self) -> dict:
        """Return a dict of all constant names and their values.

        Examples:
            >>> consts = Axiom.constants.list_all()
            >>> "PI" in consts
            True
        """
        return self._collect()

    def list_by_category(self) -> dict:
        """Return constants grouped by category.

        Returns:
            dict with keys ``"Math"``, ``"Physics"``, ``"Atomic & Particle"``,
            ``"Astronomy"``, ``"Time"``.

        Examples:
            >>> cats = Axiom.constants.list_by_category()
            >>> list(cats.keys())
            ['Math', 'Physics', 'Atomic & Particle', 'Astronomy', 'Time']
        """
        math_names = {"PI", "E", "PHI", "TAU", "INF", "NAN"}
        physics_names = {"C", "G", "H", "HBAR", "KB", "NA", "R", "EPS0", "MU0", "E_CHARGE", "G0"}
        atomic_names = {"M_E", "M_P", "M_N", "M_MU", "ALPHA", "RYDBERG", "STEFAN_BOLTZMANN", "WIEN"}
        astro_names = {"AU", "PARSEC", "LIGHT_YEAR", "SOLAR_MASS", "EARTH_MASS", "EARTH_RADIUS"}
        time_names = {"MINUTE", "HOUR", "DAY", "YEAR"}

        all_consts = self._collect()
        return {
            "Math": {k: all_consts[k] for k in math_names if k in all_consts},
            "Physics": {k: all_consts[k] for k in physics_names if k in all_consts},
            "Atomic & Particle": {k: all_consts[k] for k in atomic_names if k in all_consts},
            "Astronomy": {k: all_consts[k] for k in astro_names if k in all_consts},
            "Time": {k: all_consts[k] for k in time_names if k in all_consts},
        }

    def find(self, name: str) -> dict:
        """Search for constants whose names contain *name* (case-insensitive).

        Args:
            name: Substring to search for (e.g. ``"mass"``, ``"planck"``).

        Returns:
            dict of matching constant names and values.

        Examples:
            >>> Axiom.constants.find("mass")  # doctest: +SKIP
            {'SOLAR_MASS': 1.98847e+30, 'EARTH_MASS': 5.9722e+24, 'M_E': 9.1e-31, ...}
        """
        name_lower = name.lower()
        return {k: v for k, v in self._collect().items() if name_lower in k.lower()}

    def _collect(self):
        props = {}
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            try:
                val = getattr(self, attr_name)
                if callable(val):
                    continue
                props[attr_name] = val
            except Exception:
                pass
        return props

    def __len__(self) -> int:
        """Return the number of constants."""
        return len(self.list_all())

    def __contains__(self, name: str) -> bool:
        """Check if a constant name exists."""
        try:
            return name in self.list_all()
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"Constants({len(self)} constants available — use .list_all() to see them)"
