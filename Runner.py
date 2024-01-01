import numpy as np
import matplotlib.pyplot as plt
from Ex3_5.Manager import Enviroment


NBR_ACTIONS, NBR_STATES, NBR_STATS = 4, 25, 3
DIR_MAP = {0: "W", 1: "N", 2: "S", 3: "E"}

if __name__ == "__main__":

    # Cria o ambiente de simulação
    env = Enviroment()

    # Cria a tabela para as estatísticas. Na última dimensão, índice 0
    # é para a atual média, índice 1 para armazenar a média dos
    # quadrados dos rewards (eventualmente, para ser usado para
    # calcular a variância), e o último índice é para o contador de
    # ocorrências
    stats_table = np.zeros((NBR_ACTIONS, NBR_STATES, NBR_STATS))
    gamma = 0.9

    for r in range(500):
        if r & 511 == 0:
            print(f"Round {r:4d}")
        # # Reseta o ambiente
        st, _ = env.reset()

        action = np.random.randint(4)
        runs, count, rewards, states_trail = 0, 0, [], []
        for runs in range(10000):
            states_trail.append((st, action))
            st, reward, _, _, _ = env.step(action)
            rewards.append(reward)
            action = np.random.randint(4)

        value = 0
        for reward, (st, action) in zip(rewards[::-1], states_trail[::-1]):
            stats_table[action, st, 1] += reward*reward
            stats_table[action, st, 2] += 1
            value = gamma*value + reward
            avg_value = stats_table[action, st, 0] + \
                (value - stats_table[action, st, 0])/stats_table[action, st, 2]
            stats_table[action, st, 0] = avg_value

    state_list = [(stats_table[action, st, 0],
                   stats_table[action, st, 2],
                   action, st)
                  for action in range(NBR_ACTIONS)
                  for st in range(NBR_STATES)
                  if stats_table[action, st, 2] > 0]
    state_list.sort(reverse=True)
    print("\n".join(f"{st:2d} / {DIR_MAP[action]} ({int(nbr):3d}): {round(value, 3)}"
                    for value, nbr, action, st in state_list))

    print(f"{(stats_table[:,:,0].sum(0)/4).round(1).reshape((5, 5))}")
