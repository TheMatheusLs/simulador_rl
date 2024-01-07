import numpy as np

def find_slots(network, routes_path: list, demands: int) -> np.array:

    # Executa o algoritmo First-Fit. Encontra o primeiro slot disponível para alocar a demanda e retorna o conjunto de slots. Se não houver slots disponíveis na primeira rota tenta na segunda e assim por diante. Se não houver slots disponíveis em nenhuma rota retorna um conjunto vazio.

    all_links = network.get_all_optical_links()

    links_mapping = network.links_map

    for route_path in routes_path:

        # Retorna o índice dos links na matriz de links conforme a origem e destino da rota
        route_links = [links_mapping[route_path[i], route_path[i + 1]] for i in range(len(route_path) - 1)]

        # Realiza uma operação lógica OR entre as linhas resultantes de links[route_path]
        availability_vector = all_links[route_links].any(axis=0)

        # Sendo os slots disponíveis aqueles que são False, encontraremos a primeira sequência de False com tamanho igual a demanda
        for i in range(len(availability_vector) - demands + 1):
            if not availability_vector[i:i + demands].any():
                return route_path, np.arange(i, i + demands)

    return None, np.array([])
