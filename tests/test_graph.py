import pytest
from axiompy import Axiom


class TestGraph:
    def test_add_edge(self):
        g = Axiom.Graph()
        g.add_edge("A", "B")
        assert "B" in g.neighbors("A")

    def test_undirected(self):
        g = Axiom.Graph(directed=False)
        g.add_edge("A", "B")
        assert g.neighbors("A") == ["B"]
        assert g.neighbors("B") == ["A"]

    def test_bfs(self):
        g = Axiom.Graph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        order = g.bfs("A")
        assert order[0] == "A"
        assert set(order) == {"A", "B", "C", "D"}

    def test_dfs(self):
        g = Axiom.Graph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        order = g.dfs("A")
        assert set(order) == {"A", "B", "C", "D"}

    def test_shortest_path(self):
        g = Axiom.Graph()
        g.add_edge("A", "B", weight=1)
        g.add_edge("B", "C", weight=2)
        g.add_edge("A", "C", weight=10)
        path, dist = g.shortest_path("A", "C")
        assert path == ["A", "B", "C"]
        assert dist == pytest.approx(3.0)

    def test_shortest_path_no_path(self):
        g = Axiom.Graph()
        g.add_edge("A", "B")
        path, dist = g.shortest_path("A", "C")
        assert path is None
        assert dist == float('inf')

    def test_connected_components(self):
        g = Axiom.Graph(directed=False)
        g.add_edge(1, 2)
        g.add_edge(3, 4)
        comps = g.connected_components()
        assert len(comps) == 2

    def test_to_adjacency_matrix(self):
        g = Axiom.Graph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        mat, node_map = g.to_adjacency_matrix()
        assert mat.shape == (3, 3)

    def test_pagerank(self):
        g = Axiom.Graph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        g.add_edge("D", "C")
        ranks = Axiom.graph_analysis.pagerank(g)
        assert abs(sum(ranks.values()) - 1.0) < 1e-6
        assert set(ranks.keys()) == {"A", "B", "C", "D"}

    def test_pagerank_personalization(self):
        g = Axiom.Graph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        ranks = Axiom.graph_analysis.pagerank(g, personalization={"A": 1.0})
        assert abs(sum(ranks.values()) - 1.0) < 1e-6

    def test_minimum_spanning_tree(self):
        g = Axiom.Graph(directed=False)
        g.add_edge("A", "B", weight=4)
        g.add_edge("A", "C", weight=2)
        g.add_edge("B", "C", weight=1)
        g.add_edge("B", "D", weight=5)
        mst = g.minimum_spanning_tree()
        total_weight = sum(mst.weights.values()) / 2
        assert total_weight == pytest.approx(8.0)

    def test_is_bipartite(self):
        g = Axiom.Graph(directed=False)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 4)
        assert g.is_bipartite() is True

    def test_is_not_bipartite(self):
        g = Axiom.Graph(directed=False)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        assert g.is_bipartite() is False

    def test_bipartite_sets(self):
        g = Axiom.Graph(directed=False)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        left, right = g.bipartite_sets()
        assert len(left) + len(right) == 3
        assert left & right == set()
        assert "A" in left or "A" in right
        assert "B" in left or "B" in right
        assert "C" in left or "C" in right

    def test_bipartite_sets_not_bipartite(self):
        import pytest as _pytest
        g = Axiom.Graph(directed=False)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        with _pytest.raises(Exception):
            g.bipartite_sets()

    def test_topological_sort(self):
        g = Axiom.Graph(directed=True)
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        order = g.topological_sort()
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_topological_sort_cycle(self):
        import pytest as _pytest
        g = Axiom.Graph(directed=True)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        with _pytest.raises(Exception):
            g.topological_sort()

    def test_max_flow(self):
        g = Axiom.Graph(directed=True)
        g.add_edge("S", "A", weight=10)
        g.add_edge("S", "B", weight=5)
        g.add_edge("A", "B", weight=15)
        g.add_edge("A", "T", weight=10)
        g.add_edge("B", "T", weight=10)
        flow_value, flow_dict = g.max_flow("S", "T")
        assert flow_value == pytest.approx(15.0)

    def test_diameter_unweighted(self):
        g = Axiom.Graph(directed=False)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 4)
        g.add_edge(4, 5)
        diam = g.diameter()
        assert diam == pytest.approx(4.0)

    def test_diameter_weighted(self):
        g = Axiom.Graph(directed=False)
        g.add_edge(1, 2, weight=3)
        g.add_edge(2, 3, weight=4)
        diam = g.diameter()
        assert diam == pytest.approx(7.0)
