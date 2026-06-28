import heapq
import logging
from collections import defaultdict
from typing import Generic, Optional

import numpy as np

from ._base import GraphNode
from .matrix import Matrix
from .vector import Vector

logger = logging.getLogger(__name__)


class Graph(Generic[GraphNode]):
    """A generic graph implementation supporting directed and undirected edges."""

    def __init__(self, directed: bool = True):
        """Initialize an empty graph.

        Args:
            directed (bool): Whether the graph is directed (default True).
        """
        self.adj: dict[GraphNode, list[GraphNode]] = defaultdict(list)
        self.weights: dict[tuple[GraphNode, GraphNode], float] = {}
        self.directed = directed

    def add_edge(self, u, v, weight: float = 1.0):
        """Add an edge between nodes u and v.

        Args:
            u: The source node.
            v: The target node.
            weight (float): The edge weight (default 1.0).
        """
        self.adj[u].append(v)
        self.weights[(u, v)] = weight
        if not self.directed:
            self.adj[v].append(u)
            self.weights[(v, u)] = weight

    def neighbors(self, node) -> list[GraphNode]:
        """Return the neighbors of the given node.

        Args:
            node: The node whose neighbors to retrieve.

        Returns:
            List[GraphNode]: A list of neighboring nodes.
        """
        return self.adj.get(node, [])

    def to_adjacency_matrix(self) -> tuple[Matrix, dict[GraphNode, int]]:
        """Convert the graph to an adjacency matrix.

        Returns:
            Tuple[Matrix, Dict[GraphNode, int]]: A tuple of the adjacency matrix and
                a mapping from nodes to matrix indices.
        """
        all_nodes: set[GraphNode] = set(self.adj.keys())
        for neighbors in self.adj.values():
            all_nodes.update(neighbors)
        nodes = sorted(list(all_nodes))
        node_map = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        for u, neighbors in self.adj.items():
            for v in neighbors:
                adj_matrix[node_map[u], node_map[v]] = 1
        return Matrix(adj_matrix), node_map

    def bfs(self, start: GraphNode) -> list[GraphNode]:
        """Perform breadth-first search from the start node.

        Args:
            start (GraphNode): The starting node.

        Returns:
            List[GraphNode]: Nodes visited in BFS order.
        """
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

    def dfs(self, start: GraphNode) -> list[GraphNode]:
        """Perform depth-first search from the start node.

        Args:
            start (GraphNode): The starting node.

        Returns:
            List[GraphNode]: Nodes visited in DFS order.
        """
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

    def shortest_path(self, source: GraphNode, target: GraphNode) -> tuple[Optional[list[GraphNode]], float]:
        """Find the shortest path between source and target using Dijkstra's algorithm.

        Args:
            source (GraphNode): The source node.
            target (GraphNode): The target node.

        Returns:
            Tuple[Optional[List[GraphNode]], float]: A tuple of the path (list of nodes)
                and the total distance. Returns (None, inf) if no path exists.
        """
        if source not in self.adj and source not in self.weights:
            return None, float('inf')
        pq = [(0.0, source)]
        distances: dict[GraphNode, float] = {source: 0.0}
        prev: dict[GraphNode, Optional[GraphNode]] = {source: None}
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

    def minimum_spanning_tree(self) -> 'Graph[GraphNode]':
        """Compute the minimum spanning tree using Kruskal's algorithm.

        The graph must be undirected.

        Returns:
            Graph[GraphNode]: A new graph containing the MST edges.
        """
        logger.debug("MST of graph with %d nodes", len(self.adj))
        all_nodes = set(self.adj.keys())
        for u, neighbors in self.adj.items():
            all_nodes.update(neighbors)
        edges = [(w, u, v) for (u, v), w in self.weights.items()]
        edges.sort()
        parent = {n: n for n in all_nodes}
        rank = {n: 0 for n in all_nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
            return True

        mst = Graph(directed=False)
        for w, u, v in edges:
            if union(u, v):
                mst.add_edge(u, v, weight=w)
                if len(mst.weights) // 2 == len(all_nodes) - 1:
                    break
        return mst

    def is_bipartite(self) -> bool:
        """Check whether the graph is bipartite.

        Returns:
            bool: True if the graph is bipartite.
        """
        all_nodes = set(self.adj.keys())
        for u, neighbors in self.adj.items():
            all_nodes.update(neighbors)
        color: dict[GraphNode, int] = {}
        for node in all_nodes:
            if node not in color:
                color[node] = 0
                queue = [node]
                while queue:
                    cur = queue.pop(0)
                    for nb in self.adj.get(cur, []):
                        if nb not in color:
                            color[nb] = 1 - color[cur]
                            queue.append(nb)
                        elif color[nb] == color[cur]:
                            return False
        return True

    def bipartite_sets(self) -> tuple[set[GraphNode], set[GraphNode]]:
        """Return the two bipartite sets.

        Returns:
            Tuple[Set[GraphNode], Set[GraphNode]]: The two color classes.

        Raises:
            AxiomError: If the graph is not bipartite.
        """
        from ._base import AxiomError
        all_nodes = set(self.adj.keys())
        for u, neighbors in self.adj.items():
            all_nodes.update(neighbors)
        color: dict[GraphNode, int] = {}
        for node in all_nodes:
            if node not in color:
                color[node] = 0
                queue = [node]
                while queue:
                    cur = queue.pop(0)
                    for nb in self.adj.get(cur, []):
                        if nb not in color:
                            color[nb] = 1 - color[cur]
                            queue.append(nb)
                        elif color[nb] == color[cur]:
                            raise AxiomError("Graph is not bipartite")
        left = {n for n, c in color.items() if c == 0}
        right = {n for n, c in color.items() if c == 1}
        return left, right

    def topological_sort(self) -> list[GraphNode]:
        """Perform topological sort using Kahn's algorithm.

        The graph must be a directed acyclic graph (DAG).

        Returns:
            List[GraphNode]: Nodes in topological order.

        Raises:
            AxiomError: If the graph contains a cycle.
        """
        from ._base import AxiomError
        in_degree: dict[GraphNode, int] = {}
        all_nodes = set(self.adj.keys())
        for u, neighbors in self.adj.items():
            all_nodes.update(neighbors)
        for n in all_nodes:
            in_degree[n] = 0
        for u, neighbors in self.adj.items():
            for v in neighbors:
                in_degree[v] = in_degree.get(v, 0) + 1
        queue = [n for n in all_nodes if in_degree.get(n, 0) == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for nb in self.adj.get(node, []):
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)
        if len(result) != len(all_nodes):
            raise AxiomError("Graph contains a cycle; topological sort not possible")
        return result

    def max_flow(self, source: GraphNode, sink: GraphNode) -> tuple[float, dict[tuple[GraphNode, GraphNode], float]]:
        """Compute the maximum flow using the Edmonds-Karp algorithm.

        Args:
            source: The source node.
            sink: The sink node.

        Returns:
            Tuple[float, Dict[Tuple[GraphNode, GraphNode], float]]: The max flow value
            and a flow dictionary mapping edges to flow amounts.
        """
        logger.debug("Max flow from %s to %s", source, sink)
        flow: dict[tuple[GraphNode, GraphNode], float] = {}
        all_nodes = set(self.adj.keys())
        for u, neighbors in self.adj.items():
            all_nodes.update(neighbors)
        for (u, v), w in self.weights.items():
            flow[(u, v)] = 0.0
            flow[(v, u)] = 0.0
        total_flow = 0.0

        def residual_capacity(u, v):
            if (u, v) in self.weights:
                return self.weights[(u, v)] - flow.get((u, v), 0.0)
            if (v, u) in self.weights:
                return flow.get((v, u), 0.0)
            return 0.0

        def bfs_path():
            parent = {source: None}
            queue = [source]
            while queue:
                cur = queue.pop(0)
                if cur == sink:
                    path = []
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    return path[::-1]
                for nb in self.adj.get(cur, []):
                    if nb not in parent and residual_capacity(cur, nb) > 1e-12:
                        parent[nb] = cur
                        queue.append(nb)
                for (u, v), _ in self.weights.items():
                    if v == cur and u not in parent:
                        cap = residual_capacity(u, v)
                        if cap > 1e-12:
                            parent[u] = v
                            queue.append(u)
            return None

        while True:
            path = bfs_path()
            if path is None:
                break
            bottleneck = float('inf')
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                cap = residual_capacity(u, v)
                if cap < bottleneck:
                    bottleneck = cap
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                flow[(u, v)] = flow.get((u, v), 0.0) + bottleneck
                flow[(v, u)] = flow.get((v, u), 0.0) - bottleneck
            total_flow += bottleneck
        return total_flow, {k: v for k, v in flow.items() if v != 0.0}

    def diameter(self) -> float:
        """Compute the graph diameter (longest shortest path).

        Uses BFS from each node for unweighted graphs;
        uses Dijkstra from each node for weighted graphs.

        Returns:
            float: The diameter (longest shortest path distance).
        """
        logger.debug("Computing diameter")
        all_nodes = set(self.adj.keys())
        for u, neighbors in self.adj.items():
            all_nodes.update(neighbors)
        node_list = list(all_nodes)
        diam = 0.0
        for start in node_list:
            distances: dict[GraphNode, float] = {start: 0.0}
            queue = [start]
            while queue:
                cur = queue.pop(0)
                for nb in self.adj.get(cur, []):
                    nd = distances[cur] + self.weights.get((cur, nb), 1.0)
                    if nb not in distances or nd < distances[nb]:
                        distances[nb] = nd
                        queue.append(nb)
            for d in distances.values():
                if d > diam:
                    diam = d
        return diam

    def connected_components(self) -> list[list[GraphNode]]:
        """Find all connected components in the graph.

        Returns:
            List[List[GraphNode]]: A list of components, each containing a list of nodes.
        """
        all_nodes = set(self.adj.keys())
        for u, neighbors in self.adj.items():
            all_nodes.update(neighbors)
        visited: set[GraphNode] = set()
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
    """Graph analysis algorithms including PageRank."""

    @staticmethod
    def pagerank(graph: Graph, damping: float = 0.85,
                 max_iter: int = 100, tol: float = 1e-6,
                 personalization: Optional[dict[GraphNode, float]] = None) -> dict[GraphNode, float]:
        """Compute the PageRank of each node in the graph.

        Args:
            graph (Graph): The input graph.
            damping (float): The damping factor (default 0.85).
            max_iter (int): Maximum number of iterations (default 100).
            tol (float): Convergence tolerance (default 1e-6).
            personalization: Optional per-node teleport weights. Nodes not
                specified get zero teleport probability. If ``None``, uniform
                teleport is used (default ``None``).

        Returns:
            Dict[GraphNode, float]: A mapping from node to its PageRank score.
        """
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

        if personalization is not None:
            teleport_vec = np.zeros(n)
            for node, w in personalization.items():
                if node in node_map:
                    teleport_vec[node_map[node]] = w
            teleport_sum = teleport_vec.sum()
            if teleport_sum > 0:
                teleport_vec /= teleport_sum
            else:
                teleport_vec.fill(1.0 / n)
            teleport = np.tile(teleport_vec, (n, 1))
        else:
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
