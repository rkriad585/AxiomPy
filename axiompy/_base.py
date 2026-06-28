from typing import TypeVar

MatrixData = list[list[float]]
VectorData = list[float]
GraphNode = TypeVar('GraphNode')

class AxiomError(Exception):
    """Base exception for all AxiomPy errors.

    All custom exceptions in the package inherit from this class.
    """
