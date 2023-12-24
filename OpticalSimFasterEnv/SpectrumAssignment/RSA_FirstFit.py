import numpy as np

def find_slots(links: np.array, routes_path: list, demands: int) -> np.array:

    # Executa o algoritmo First-Fit. Encontra o primeiro slot disponível para alocar a demanda e retorna o conjunto de slots. Se não houver slots disponíveis na primeira rota tenta na segunda e assim por diante. Se não houver slots disponíveis em nenhuma rota retorna um conjunto vazio.

    for route_path in routes_path:

        # Realiza uma operação lógica OR entre as linhas resultantes de links[route_path]
        availability_vector = links[route_path].any(axis=0)

        # Sendo os slots disponíveis aqueles que são False, encontraremos a primeira sequência de False com tamanho igual a demanda
        for i in range(len(availability_vector) - demands + 1):
            if not availability_vector[i:i + demands].any():
                return route_path, np.arange(i, i + demands)

    return None, np.array([])
