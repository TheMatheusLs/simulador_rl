import numpy as np
from collections import Counter
#np.random.seed(42)

from OpticalSimFasterEnv.Topology.NSFNet import NSFNet
from OpticalSimFasterEnv.SpectrumAssignment import RSA_FirstFit, SAR_FistFit

RSA_ACTION, SAR_ACTION = 0, 1

import gymnasium as gym
from gymnasium import spaces

class Demand:
    def __init__(self, demand_ID: int, demand_class: int, slots: list[int], route, simulation_time: float, departure_time: float):
        self.demand_ID = demand_ID
        self.demand_class = demand_class
        self.slots = slots
        self.route = route

        self.arrival_time = simulation_time
        self.departure_time = departure_time

    def deallocate(self, network):
        slots_remaining = network.deallocate_slots(self.route, self.slots)
        self.slots = []

class Enviroment(gym.Env):
    def __init__(self, network_load, k_routes,
                 number_of_slots = 64, penalty=-1000,
                 seed=None,
                 is_dynamic_network=True) -> None:
        self.number_of_slots = number_of_slots
        self.network_load = network_load
        self.k_routes = k_routes
        self.penalty = penalty
        self.is_dynamic_network = is_dynamic_network
        self.seed = seed
        self.busy_links = Counter()

        # Cria a topologia de rede NSFNet
        self.network = NSFNet(num_of_slots = number_of_slots)

        self.allRoutes = [[self.network.k_shortest_paths(origin, source, k_routes) for source in range(self.network.get_num_of_nodes())] for origin in range(self.network.get_num_of_nodes())]

        # Imprime todas as rotas possíveis
        # for i in range(self.network.get_num_of_nodes()):
        #     for j in range(self.network.get_num_of_nodes()):
        #         if i != j:
        #             print(f"Routes from {str(i).zfill(2)} to {str(j).zfill(2)}:", self.allRoutes[i][j])


        self.nbr_nodes = self.network.get_num_of_nodes()

        self.random_generator = np.random.default_rng(self.seed)

        # Define o espaço de ações para a saída do algoritmo. Como
        # nosso estado de ação é 0 e 1. Usamos o Discrete(2) para
        # definir o espaço de ações.
        # O ATRIBUTO ABAIXO APARENTEMENTE NÃO ESTÁ SENDO USADO NO
        # MOMENTO, E POR ISSO ESTÁ COMENTADO
        # self.action_space = spaces.Discrete(2)

        # O ATRIBUTO ABAIXO APARENTEMENTE NÃO ESTÁ SENDO USADO NO
        # MOMENTO, E POR ISSO ESTÁ COMENTADO
        # self.observation_space = spaces.MultiBinary(self.nbr_nodes * 2 + self.number_of_slots * 42)

        # Cria um mapa de codificação para os estados de origem e
        # destino em one hot encoding
        self.source_destination_map = np.eye(self.nbr_nodes)

        self.reward_episode = 0
        self.reward_by_step = []

    def reset(self, options = None):
        self.reward_by_step.append(self.reward_episode)
        self.reward_episode = 0

        # Cria a topologia de rede NSFNet
        self.network = NSFNet(num_of_slots = self.number_of_slots)

        self.simulation_time = 0.0
        self.is_available_slots = False
        self.total_number_of_blocks = 0
        self.last_request = 0

        # Gera uma matriz aleatória para cada par origem-destino
        # com as ações
        random_start_actions = self.random_generator.integers(0, 2,
                            size=(self.nbr_nodes, self.nbr_nodes))

        # Lista para armazenar as demandas ativas na rede
        self.list_of_demands = []

        return self.get_observation(), {}

    def get_source_destination(self):
        source = self.random_generator.integers(0, self.nbr_nodes)
        destination = self.random_generator.integers(0, self.nbr_nodes)
        while source == destination:
            destination = self.random_generator.integers(0, self.nbr_nodes)

        return source, destination

    def keep_or_remove(self, demand):
        if demand.departure_time <= self.simulation_time:
            slots_remaining = demand.deallocate(self.network)
            return False
        return True

    def step(self, action):
        source, destination = self.source, self.destination

        self.isAvailableSlots = False

        # Adiciona um incremento ao tempo de simulacao conforme a
        # carga da rede
        self.simulation_time += self.random_generator.exponential(1/self.network_load)

        # Remove as demandas que expiraram, no caso dinâmico
        if self.is_dynamic_network:
            self.list_of_demands = \
                [demand for demand in self.list_of_demands
                 if self.keep_or_remove(demand)]

        # Sorteia uma demanda de slots entre [2,3,6]
        demand_class = self.random_generator.choice([3, 6, 11])

        # Executa o First-Fit para o algoritmo RSA
        if action == 0:
            route, slots = RSA_FirstFit.find_slots(self.network.get_all_optical_links(), self.allRoutes[source][destination], demand_class)
        elif action == 1:
            route, slots = SAR_FistFit.find_slots( self.network.get_all_optical_links(), self.allRoutes[source][destination], demand_class)

        # Calcula o tempo de partida da demanda (tempo atual + tempo
        # de duração da demanda)
        departure_time = self.simulation_time + \
            self.random_generator.exponential(1)

        # Verifica se o conjunto de slots é diferente de vazio
        if slots.size != 0:
            self.isAvailableSlots = True

            # Cria a demanda e seus atributos para serem utilizados
            # na alocação
            demand = Demand(self.last_request, demand_class, slots,
                            route, self.simulation_time,
                            departure_time)

            self.list_of_demands.append(demand)

            # Realiza a alocação dos slots na matriz de links
            slots_remaining = self.network.allocate_slots(route, slots)
        else:
            self.isAvailableSlots = False
            self.total_number_of_blocks += 1
            self.busy_links.update(route)

        self.last_request += 1

        self.reward_episode = 1 if self.isAvailableSlots else\
            self.penalty

        return self.get_observation(), self.reward_episode, \
            not self.isAvailableSlots, False, {}

    def code_nodes(self, src, dest):
        return dest + src*self.nbr_nodes

    def decode_nodes(self, coded_nodes):
        return (coded_nodes//self.nbr_nodes,
                coded_nodes % self.nbr_nodes)

    def get_observation(self):

        source, destination = self.get_source_destination()

        self.source = source
        self.destination = destination

        # return np.concatenate([self.source_destination_map[source],
        #                        self.source_destination_map[destination],
        #                        self.network.get_all_optical_links().reshape(-1)])
        # return self.code_nodes(source, destination)
        return self.code_nodes(source, destination)


    def plot_reward(self):
        import matplotlib.pyplot as plt
        plt.plot(self.reward_by_step, label='Reward by Step')

        # Média móvel dos últimos 50 episódios
        mean_reward = np.convolve(self.reward_by_step, np.ones((50,))/50, mode='valid')
        plt.plot(mean_reward, label='Mean of last 50 episodes')

        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.title('Reward by Episode')
        plt.grid(True)
        plt.show()
