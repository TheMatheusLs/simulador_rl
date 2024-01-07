import numpy as np

def find_slots(network, routes_path: list, demands: int) -> np.array:

    # Executa o algoritmo First-Fit. Encontra o primeiro slot disponível para alocar a demanda e retorna o conjunto de slots. Se não houver slots disponíveis no primeiro slot da primeira rota tenta no primeiro slot da segunda rota e assim por diante. Se não houver slots disponíveis em nenhuma rota retorna um conjunto vazio.

    all_links = network.get_all_optical_links()

    links_mapping = network.links_map

    for slot in range(len(all_links[0]) - demands + 1):
        for route_path in routes_path:

            # Retorna o índice dos links na matriz de links conforme a origem e destino da rota
            route_links = [links_mapping[route_path[i], route_path[i + 1]] for i in range(len(route_path) - 1)]

            # Realiza uma operação lógica OR entre as linhas resultantes de links[route_path]
            availability_vector = all_links[route_links].any(axis=0)


            # Sendo os slots disponíveis aqueles que são False, encontraremos a primeira sequência de False com tamanho igual a demanda
            if not availability_vector[slot: slot + demands].any():
                return route_path, np.arange(slot, slot + demands)
        
    return None, np.array([])