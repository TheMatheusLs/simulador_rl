
from OpticalSimFasterEnv.Enviroment.Manager import Enviroment
import numpy as np

# Cria o ambiente de simulação
env = Enviroment(network_load = 300, k_routes = 3, number_of_slots = 128)

# Reseta o ambiente
state, _ = env.reset()

# Soma das recompensas
reward_sum = 0.0

# RSA 
chromosome = np.zeros(14 * 14, dtype = np.int32)

# SAR
#chromosome = np.ones(env.number_of_nodes * env.number_of_nodes, dtype = np.int32)

MAX_REQS = 100000

blocking = 0
for _ in range(MAX_REQS):

    # Retorna a ação escolhida pelo indivíduo para o par de nós atual
    action = chromosome[state[0] * 14 + state[1]]

    next_state, reward, done, tru, info = env.step(action)

    if reward == -1:
        blocking += 1

    state = next_state

print('Blocking: ', blocking)
print('Blocking Probability: ', blocking / MAX_REQS)