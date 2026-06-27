from .linalg import LinearAlgebra
from .stats import Statistics
from .graph import GraphAnalysis
from .autodiff import AutoDiff
from .number_theory import NumberTheory
from .electromagnetism import Electromagnetism
from .visualization import Visualization
from .vector import Vector
from .matrix import Matrix
from .graph import Graph


class AxiomPy:
    def __init__(self):
        self._linalg = LinearAlgebra()
        self._stats = Statistics()
        self._graph_analysis = GraphAnalysis()
        self._autodiff = AutoDiff()
        self._num_theory = NumberTheory()
        self._em = Electromagnetism()
        self._viz = Visualization()
        self.Vector = Vector
        self.Matrix = Matrix
        self.Graph = Graph

    @property
    def linalg(self):
        return self._linalg

    @property
    def stats(self):
        return self._stats

    @property
    def graph_analysis(self):
        return self._graph_analysis

    @property
    def autodiff(self):
        return self._autodiff

    @property
    def number_theory(self):
        return self._num_theory

    @property
    def electromagnetism(self):
        return self._em

    @property
    def viz(self):
        return self._viz


Axiom = AxiomPy()
