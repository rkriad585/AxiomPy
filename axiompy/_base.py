import numpy as np
from typing import List, TypeVar

MatrixData = List[List[float]]
VectorData = List[float]
GraphNode = TypeVar('GraphNode')

class AxiomError(Exception):
    pass
