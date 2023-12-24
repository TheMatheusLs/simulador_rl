# Cria a topologia de rede NSFNet usando o NetworkX
import networkx as nx
import numpy as np
import heapq

from OpticalSimFasterEnv.Settings import *

class Generic():
    def __init__(self, num_of_nodes: int, num_of_slots: int, lengths: list):

        self.num_of_nodes = num_of_nodes
        self.num_of_links = 0
        self.num_of_slots = num_of_slots
        
        self.graph = nx.Graph()

        # Adiciona os nós (representando os equipamentos)
        self.graph.add_nodes_from(range(self.num_of_nodes))

        # Criando um dicionário com um mapa para acessar os links a partir de uma tupla (source, destination)
        self.links_map = {}

        for source, destination, length in lengths:

            # Adiciona os enlaces (representando as conexões) (source, destination, length)
            self.graph.add_edge(source, destination, weight=length)

            self.num_of_links += 1

            self.links_map[(source, destination)] = self.num_of_links - 1

            # Adiciona os enlaces (representando as conexões) (destination, source, length) se for bidirecional
            if IS_BIDIRECTIONAL:
                self.graph.add_edge(destination, source, weight=length)

                self.num_of_links += 1

                self.links_map[(destination, source)] = self.num_of_links - 1

        # Matriz de links (linhas) e slots (colunas)
        self.all_optical_links = np.zeros((self.num_of_links, self.num_of_slots), dtype=np.bool_)


    def __str__(self):
        return_text = "Links\t Slots\n"

        for i in range(self.num_of_links):
            return_text += f"{i}\t {[status for status in self.all_optical_links[i]]}\n"

        return return_text


    def get_num_of_nodes(self):
        return self.num_of_nodes
    
    
    def get_num_of_links(self):
        return self.num_of_links
    

    def get_num_of_slots(self):
        return self.num_of_slots
    

    def get_all_optical_links(self):
        return self.all_optical_links
    

    def get_graph(self):
        return self.graph
    

    # Algoritmo de roteamento (usando menor caminho)
    def dijkstra_routing(self, source, destination):
        return nx.shortest_path(self.graph, source, destination, weight='weight')


    def k_shortest_paths(self, source, destination, k_routes):
        paths = []
        heap = [(0, [source])]
        
        while heap and len(paths) < k_routes:
            (cost, path) = heapq.heappop(heap)
            current_node = path[-1]
            
            if current_node == destination:
                paths.append((cost, path))
            else:
                for next_node in self.graph[current_node]:
                    if next_node not in path:
                        heapq.heappush(heap, (cost + self.graph[current_node][next_node]['weight'], path + [next_node]))
        
        only_paths = []
        for i in range(len(paths)):
            only_paths.append(paths[i][1] if len(paths[i][1]) > 1 else None)

        return only_paths
    

    def print_graph(self):

        import matplotlib.pyplot as plt

        # Plotar o grafo
        nx.draw(self.graph, pos=self.node_positions, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_color='black', font_weight='bold')
        plt.title("Topology Generic")
        plt.show()


    def allocate_slots(self, route_path, slots):

        for source, destination in zip(route_path[:-1], route_path[1:]):
            self.all_optical_links[self.links_map[(source, destination)], slots] = True

    
    def deallocate_slots(self, route_path, slots):

        for source, destination in zip(route_path[:-1], route_path[1:]):
            self.all_optical_links[self.links_map[(source, destination)], slots] = False

