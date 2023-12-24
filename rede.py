import networkx as nx

# Criação da topologia
nfsnet = nx.Graph()

# Adição de nós (representando os equipamentos)
nfsnet.add_nodes_from([1, 2, 3, 4, 5])

# Adição de enlaces (representando as conexões)
nfsnet.add_edge(1, 2, distance=10)
nfsnet.add_edge(1, 3, distance=15)
nfsnet.add_edge(2, 3, distance=8)
nfsnet.add_edge(2, 4, distance=12)
nfsnet.add_edge(3, 4, distance=6)
nfsnet.add_edge(4, 5, distance=9)

# Algoritmo de roteamento (usando menor caminho)
def elastic_routing(source, destination):
    return nx.shortest_path(nfsnet, source, destination, weight='distance')

# Exemplo de utilização
source = 1
destination = 5
path = elastic_routing(source, destination)
print(f"Rota de {source} para {destination}: {path}")
