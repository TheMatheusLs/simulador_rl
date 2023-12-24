import networkx as nx
import matplotlib.pyplot as plt

# Criar um grafo para representar a topologia da NSFNET
nfsnet = nx.Graph()

# Adicionar nós
nfsnet.add_nodes_from([1, 2, 3, 4, 5, 6])

# Adicionar arestas (conexões)
edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (4, 6)]
nfsnet.add_edges_from(edges)

# Definir posições para os nós (para um layout mais claro)
pos = {
    1: (0, 0),
    2: (1, 1),
    3: (1, -1),
    4: (2, 0),
    5: (3, 1),
    6: (3, -1)
}

# Plotar o grafo
nx.draw(nfsnet, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')
plt.title("Topologia NSFNET")
plt.show()
