import numpy as np
import matplotlib.pyplot as plt
from OpticalSimFasterEnv.Enviroment.Manager import Enviroment, RSA_ACTION, SAR_ACTION


if __name__ == "__main__":

    # Cria o ambiente de simulação
    env = Enviroment(network_load = 100, k_routes = 3)

    count = 0
    for _ in range(2):
        # # Reseta o ambiente
        initial_state, _ = env.reset()
        src, dst = initial_state

        reward_sum = 0.0

        done = False

        action = RSA_ACTION

        # Cria a tabela para as estatísticas
        stats_table = np.zeros((2, env.nbr_nodes, env.nbr_nodes, 3))
        runs = 0
        while not done and runs < 30000:
            runs += 1
            next_state, reward, done, tru, info = env.step(action)
            stats_table[action, src, dst, 0] += reward
            stats_table[action, src, dst, 1] += reward*reward
            stats_table[action, src, dst, 2] += 1
            src, dst = next_state

            reward_sum += reward
            count += 1
            if count == 10000:
                count = 0
                print("Reward:", reward_sum)

        print("Ended simulation by " +
              ("termination" if done
               else "number of runs exceeded."))
        print("Reward sum:", reward_sum)
        state_list = [(stats_table[0, i, j, 0], i, j)
                      for i in range(env.nbr_nodes)
                      for j in range(env.nbr_nodes)]
        state_list.sort(reverse=True)
        print("\n".join(f"{src:2d} - {dst:2d}: {value}"
                        for value, src, dst in state_list[:30]))
