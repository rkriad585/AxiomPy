import logging

from ._backend import get_backend, register_backend, set_backend
from ._config import AxiomConfig
from .autodiff import AutoDiff
from .calculus import Calculus
from .complex_numbers import ComplexMatrix, ComplexNumber, ComplexVector
from .electromagnetism import Electromagnetism
from .graph import Graph, GraphAnalysis
from .linalg import LinearAlgebra
from .matrix import Matrix
from .number_theory import NumberTheory
from .odes import lotka_volterra_odes, pendulum_odes, solve_bvp, solve_ivp
from .optimization import Optimization
from .polynomial import Polynomial
from .signal import Signal
from .special import (
    bessel_j,
    bessel_y,
    beta,
    binomial,
    double_factorial,
    erf,
    erfc,
    factorial,
    gamma,
    legendre_p,
)
from .stats import Statistics
from .tensor import Tensor
from .vector import Vector
from .visualization import Visualization

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")


class AxiomPy:
    """Primary façade for the AxiomPy mathematics library.

    Provides access to all sub-modules (:class:`~axiompy.linalg.LinearAlgebra`,
    :class:`~axiompy.stats.Statistics`, etc.) and exposes :class:`~axiompy.vector.Vector`,
    :class:`~axiompy.matrix.Matrix`, :class:`~axiompy.graph.Graph`, and
    :class:`~axiompy.polynomial.Polynomial` as class attributes.
    """

    def __init__(self):
        """Initialize the façade and instantiate all sub-module classes."""
        self._linalg = LinearAlgebra()
        self._stats = Statistics()
        self._graph_analysis = GraphAnalysis()
        self._autodiff = AutoDiff()
        self._num_theory = NumberTheory()
        self._em = Electromagnetism()
        self._viz = Visualization()
        self._calc = Calculus()
        self._opt = Optimization()
        self._sig = Signal()
        self.Vector = Vector
        self.Matrix = Matrix
        self.Graph = Graph
        self.Polynomial = Polynomial
        self.Tensor = Tensor
        self.ComplexNumber = ComplexNumber
        self.ComplexVector = ComplexVector
        self.ComplexMatrix = ComplexMatrix
        self.gamma = gamma
        self.beta = beta
        self.erf = erf
        self.erfc = erfc
        self.bessel_j = bessel_j
        self.bessel_y = bessel_y
        self.legendre_p = legendre_p
        self.factorial = factorial
        self.binomial = binomial
        self.double_factorial = double_factorial
        self.solve_ivp = solve_ivp
        self.solve_bvp = solve_bvp
        self.pendulum_odes = pendulum_odes
        self.lotka_volterra_odes = lotka_volterra_odes
        self._setup_logging()

    def _setup_logging(self):
        cfg = AxiomConfig.load()
        level = logging.DEBUG if cfg.verbose else logging.WARNING
        logging.getLogger("axiompy").setLevel(level)

    @property
    def config(self) -> AxiomConfig:
        """Return the global configuration singleton.

        Returns:
            AxiomConfig: The active configuration.
        """
        return AxiomConfig.load()

    @property
    def linalg(self):
        """Access the linear algebra sub-module.

        Returns:
            LinearAlgebra: Instance of the linear algebra API.
        """
        return self._linalg

    @property
    def stats(self):
        """Access the statistics sub-module.

        Returns:
            Statistics: Instance of the statistics API.
        """
        return self._stats

    @property
    def graph_analysis(self):
        """Access the graph analysis sub-module.

        Returns:
            GraphAnalysis: Instance of the graph analysis API.
        """
        return self._graph_analysis

    @property
    def autodiff(self):
        """Access the automatic differentiation sub-module.

        Returns:
            AutoDiff: Instance of the autodiff API.
        """
        return self._autodiff

    @property
    def number_theory(self):
        """Access the number theory sub-module.

        Returns:
            NumberTheory: Instance of the number theory API.
        """
        return self._num_theory

    @property
    def electromagnetism(self):
        """Access the electromagnetism sub-module.

        Returns:
            Electromagnetism: Instance of the electromagnetism API.
        """
        return self._em

    @property
    def viz(self):
        """Access the visualization sub-module.

        Returns:
            Visualization: Instance of the visualization API.
        """
        return self._viz

    @property
    def calc(self):
        """Access the calculus sub-module.

        Returns:
            Calculus: Instance of the calculus API.
        """
        return self._calc

    @property
    def optimization(self):
        """Access the optimization sub-module.

        Returns:
            Optimization: Instance of the optimization API.
        """
        return self._opt

    @property
    def signal(self):
        """Access the signal processing sub-module.

        Returns:
            Signal: Instance of the signal API.
        """
        return self._sig

    @property
    def backend(self):
        """Return the currently active numerical backend.

        Returns:
            Backend: The active backend instance.
        """
        return get_backend()

    def set_backend(self, name: str = "numpy"):
        """Switch the active numerical backend.

        Args:
            name: Backend identifier (default ``"numpy"``).
        """
        set_backend(name)

    def register_backend(self, name: str, backend_cls):
        """Register a custom backend implementation.

        Args:
            name: Identifier for the backend.
            backend_cls: Class implementing the :class:`Backend` interface.
        """
        register_backend(name, backend_cls)


Axiom = AxiomPy()
