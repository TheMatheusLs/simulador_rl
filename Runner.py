import numpy as np
import matplotlib.pyplot as plt
from OpticalSimFasterEnv.Enviroment.Manager import Enviroment, RSA_ACTION, SAR_ACTION


if __name__ == "__main__":

    # Cria o ambiente de simulação
    env = Enviroment(network_load = 100, k_routes = 3)

    # Cria a tabela para as estatísticas. Na última dimensão, índice 0
    # é para a atual média, índice 1 para armazenar a média dos
    # quadrados dos rewards (eventualmente, para ser usado para
    # calcular a variância), e o último índice é para o contador de
    # ocorrências
    stats_table = np.zeros((2, env.nbr_nodes, env.nbr_nodes, 3))
    gamma = 0.05

    for r in range(50):
        if r & 511 == 0:
            print(f"Round {r:4d}")
        # # Reseta o ambiente
        (src, dst), _ = env.reset()

        done, action = False, SAR_ACTION

        runs, count, rewards, states_trail = 0, 0, [], [(src, dst, action)]
        while not done and runs < 30000:
            runs += 1
            (src, dst), reward, done, tru, info = env.step(action)
            ## Update action (to be done later)
            states_trail.append((src, dst, action))

            rewards.append(reward)
            count += 1
            if count == 10000:
                count = 0
                print(f"Runs in iter {r}: {runs}")

        value = 0
        for reward, (src, dst, action) in zip(rewards[::-1], states_trail[::-1]):
            stats_table[action, src, dst, 1] += reward*reward
            stats_table[action, src, dst, 2] += 1
            value = gamma*value + reward
            avg_value = stats_table[action, src, dst, 0] + \
                (value - stats_table[action, src, dst, 0])/stats_table[action, src, dst, 2]
            stats_table[action, src, dst, 0] = avg_value

        if not done:
            print(f"Ended simulation #{r} by " +
                  ("termination" if done
                   else "number of runs exceeded."))
            print("Runs:", runs)

    state_list = [(stats_table[action, i, j, 0], stats_table[action, i, j, 2], action, i, j)
                  for action in [0, 1]
                  for i in range(env.nbr_nodes)
                  for j in range(env.nbr_nodes) if i != j and stats_table[action, i, j, 2] > 0]
    state_list.sort(reverse=True)
    print("\n".join(f"{src:2d} - {dst:2d} / {['RSA', 'SAR'][action]} ({int(nbr):3d}): {round(value, 3)}"
                    for value, nbr, action, src, dst in state_list[:30]))
    print()
    print("\n".join(f"{src:2d} - {dst:2d} / {['RSA', 'SAR'][action]} ({int(nbr):3d}): {round(value, 3)}"
                    for value, nbr, action, src, dst in state_list[-100:]))
