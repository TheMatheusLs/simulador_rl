import numpy as np

def find_slots(links: np.array, routes_path: list, demands: int) -> np.array:

    # Executa o algoritmo First-Fit. Encontra o primeiro slot disponível para alocar a demanda e retorna o conjunto de slots. Se não houver slots disponíveis no primeiro slot da primeira rota tenta no primeiro slot da segunda rota e assim por diante. Se não houver slots disponíveis em nenhuma rota retorna um conjunto vazio.

    for slot in range(len(links[0]) - demands + 1):
        for route_path in routes_path:
            # Realiza uma operação lógica OR entre as linhas resultantes de links[route_path]
            availability_vector = links[route_path].any(axis=0)


            # Sendo os slots disponíveis aqueles que são False, encontraremos a primeira sequência de False com tamanho igual a demanda
            if not availability_vector[slot: slot + demands].any():
                return route_path, np.arange(slot, slot + demands)

        res = set([(s, d) for route in routes_path
               for s, d in zip(route[:-1], route[1:])])
    return res, np.array([])
