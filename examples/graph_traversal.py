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
print(f"Shortest path A→E: {path} (distance={dist})")

# Undirected graph with weighted edges
print("\n--- Undirected weighted graph ---")
ug = Axiom.Graph(directed=False)
ug.add_edge("X", "Y", 4)
ug.add_edge("Y", "Z", 3)
ug.add_edge("X", "Z", 10)
ug.add_edge("Z", "W", 2)

path, dist = ug.shortest_path("X", "W")
print(f"Shortest path X→W: {path} (distance={dist})")

print("Connected components:", ug.connected_components())
