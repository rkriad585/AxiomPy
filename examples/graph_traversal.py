from axiompy import Axiom

# Directed graph
g = Axiom.Graph(directed=True)
g.add_edge("A", "B")
g.add_edge("A", "C")
g.add_edge("B", "D")
g.add_edge("C", "D")
g.add_edge("D", "E")
g.add_edge("E", "A")

print("Directed graph edges:")
for u in g.adj:
    for v in g.adj[u]:
        print(f"  {u} -> {v}")

print("\nBFS from A:", g.bfs("A"))
print("DFS from A:", g.dfs("A"))

path, dist = g.shortest_path("A", "E")
print(f"Shortest path A-E: {path} (distance={dist})")

# Undirected graph with weighted edges
print("\n--- Undirected weighted graph ---")
ug = Axiom.Graph(directed=False)
ug.add_edge("X", "Y", 4)
ug.add_edge("Y", "Z", 3)
ug.add_edge("X", "Z", 10)
ug.add_edge("Z", "W", 2)

path, dist = ug.shortest_path("X", "W")
print(f"Shortest path X-W: {path} (distance={dist})")

print("Connected components:", ug.connected_components())

# --- Phase 5.2 features ---

# Minimum spanning tree
mst = ug.minimum_spanning_tree()
print("\nMinimum spanning tree edges:", mst)

# Bipartite check
bg = Axiom.Graph(directed=False)
bg.add_edge(1, 2)
bg.add_edge(2, 3)
bg.add_edge(3, 1)
print(f"\nTriangle is bipartite: {bg.is_bipartite()}")

bg2 = Axiom.Graph(directed=False)
bg2.add_edge(1, 2)
bg2.add_edge(2, 3)
print(f"Path is bipartite: {bg2.is_bipartite()}")
print(f"  Sets: {bg2.bipartite_sets()}")

# Topological sort
dag = Axiom.Graph(directed=True)
dag.add_edge("A", "B")
dag.add_edge("A", "C")
dag.add_edge("B", "D")
dag.add_edge("C", "D")
print(f"\nTopological sort: {dag.topological_sort()}")

# Max flow
flow_g = Axiom.Graph(directed=True)
flow_g.add_edge("s", "a", 10)
flow_g.add_edge("s", "b", 5)
flow_g.add_edge("a", "b", 15)
flow_g.add_edge("a", "t", 10)
flow_g.add_edge("b", "t", 10)
print(f"\nMax flow s-t: {flow_g.max_flow('s', 't')}")

# Diameter
print(f"Diameter: {ug.diameter()}")
