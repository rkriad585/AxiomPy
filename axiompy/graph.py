import heapq
import numpy as np
from typing import Dict, List, Tuple, Generic, Optional, Set
from collections import defaultdict
from ._base import GraphNode
from .matrix import Matrix
from .vector import Vector


class Graph(Generic[GraphNode]):
    def __init__(self, directed: bool = True):
        self.adj: Dict[GraphNode, List[GraphNode]] = defaultdict(list)
        self.weights: Dict[Tuple[GraphNode, GraphNode], float] = {}
        self.directed = directed

    def add_edge(self, u, v, weight: float = 1.0):
        self.adj[u].append(v)
        self.weights[(u, v)] = weight
        if not self.directed:
            self.adj[v].append(u)
            self.weights[(v, u)] = weight

    def neighbors(self, node) -> List[GraphNode]:
        return self.adj.get(node, [])

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

    def bfs(self, start: GraphNode) -> List[GraphNode]:
        visited = []
        queue = [start]
        seen = {start}
        while queue:
            node = queue.pop(0)
            visited.append(node)
            for nb in self.adj.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        return visited

    def dfs(self, start: GraphNode) -> List[GraphNode]:
        visited = []
        stack = [start]
        seen = {start}
        while stack:
            node = stack.pop()
            visited.append(node)
            for nb in self.adj.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return visited

    def shortest_path(self, source: GraphNode, target: GraphNode) -> Tuple[Optional[List[GraphNode]], float]:
        if source not in self.adj and source not in self.weights:
            return None, float('inf')
        pq = [(0.0, source)]
        distances: Dict[GraphNode, float] = {source: 0.0}
        prev: Dict[GraphNode, Optional[GraphNode]] = {source: None}
        settled = set()
        while pq:
            d, node = heapq.heappop(pq)
            if node in settled:
                continue
            settled.add(node)
            if node == target:
                path = []
                while node is not None:
                    path.append(node)
                    node = prev[node]
                return path[::-1], d
            for nb in self.adj.get(node, []):
                if nb in settled:
                    continue
                nd = d + self.weights.get((node, nb), 1.0)
                if nd < distances.get(nb, float('inf')):
                    distances[nb] = nd
                    prev[nb] = node
                    heapq.heappush(pq, (nd, nb))
        return None, float('inf')

    def connected_components(self) -> List[List[GraphNode]]:
        all_nodes = set(self.adj.keys())
        for u, neighbors in self.adj.items():
            all_nodes.update(neighbors)
        visited: Set[GraphNode] = set()
        components = []
        for node in all_nodes:
            if node not in visited:
                component = []
                stack = [node]
                while stack:
                    cur = stack.pop()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    component.append(cur)
                    for nb in self.adj.get(cur, []):
                        if nb not in visited:
                            stack.append(nb)
                components.append(component)
        return components


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
