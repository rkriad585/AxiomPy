import numpy as np
from typing import Dict, List, Tuple, Generic
from collections import defaultdict
from ._base import GraphNode
from .matrix import Matrix
from .vector import Vector

class Graph(Generic[GraphNode]):
    def __init__(self):
        self.adj: Dict[GraphNode, List[GraphNode]] = defaultdict(list)

    def add_edge(self, u, v):
        self.adj[u].append(v)

    def to_adjacency_matrix(self) -> Tuple[Matrix, Dict[GraphNode, int]]:
        nodes = sorted(list(self.adj.keys()))
        node_map = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        for u, neighbors in self.adj.items():
            for v in neighbors:
                if v in node_map:
                    adj_matrix[node_map[u], node_map[v]] = 1
        return Matrix(adj_matrix), node_map


class GraphAnalysis:
    @staticmethod
    def pagerank(graph: Graph, damping: float = 0.85,
                 max_iter: int = 100, tol: float = 1e-6) -> Dict[GraphNode, float]:
        M_matrix, node_map = graph.to_adjacency_matrix()
        n = M_matrix.shape[0]
        if n == 0:
            return {}

        out_degree = M_matrix._data.sum(axis=1, keepdims=True)
        dangling_nodes = np.where(out_degree == 0)[0]
        for node_idx in dangling_nodes:
            M_matrix._data[node_idx, :] = 1 / n

        M_hat = np.divide(
            M_matrix._data,
            M_matrix._data.sum(axis=1, keepdims=True),
            out=np.zeros_like(M_matrix._data),
            where=M_matrix._data.sum(axis=1, keepdims=True) != 0,
        )

        teleport = np.full((n, n), 1 / n)
        M = damping * M_hat.T + (1 - damping) * teleport

        ranks = Vector(np.full(n, 1 / n))
        for _ in range(max_iter):
            prev_ranks = ranks
            ranks = Vector(M @ ranks._data)
            if (ranks - prev_ranks).magnitude() < tol:
                break

        inv_node_map = {i: node for node, i in node_map.items()}
        return {inv_node_map[i]: rank for i, rank in enumerate(ranks.to_list())}
