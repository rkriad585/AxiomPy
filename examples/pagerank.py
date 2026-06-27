from axiompy import Axiom

g = Axiom.Graph()
g.add_edge("A", "B")
g.add_edge("A", "C")
g.add_edge("B", "C")
g.add_edge("C", "A")
g.add_edge("D", "C")

ranks = Axiom.graph_analysis.pagerank(g)
print("PageRank scores:")
for node, rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
    print(f"  {node}: {rank:.4f}")
