import networkx as nx
import heapq

def k_shortest_paths(G, source, target, k):
    paths = []
    heap = [(0, [source])]
    
    while heap and len(paths) < k:
        (cost, path) = heapq.heappop(heap)
        current_node = path[-1]
        
        if current_node == target:
            paths.append((cost, path))
        else:
            for next_node in G[current_node]:
                if next_node not in path:
                    heapq.heappush(heap, (cost + G[current_node][next_node]['weight'], path + [next_node]))
    
    return paths

# Criar um grafo de exemplo
G = nx.Graph()
G.add_weighted_edges_from([(1, 2, 10), (1, 3, 15), (2, 3, 8), (2, 4, 12), (3, 4, 6), (4, 5, 9)])

source = 1
target = 5
k = 3

k_shortest = k_shortest_paths(G, source, target, k)
for i, (cost, path) in enumerate(k_shortest, start=1):
    print(f"Rota {i}: Custo {cost}, Caminho {path}")
