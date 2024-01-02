import numpy as np
import matplotlib.pyplot as plt

from Ex3_5.Manager import Enviroment
from RL.policy import EquiprobablePolicy, EpisodicTablePolicyUpdater
from RL.simulators import EpisodicSimulator

# A policy that uses equally likely actions and a table to store the
# statistics
class MyPolicy(EpisodicTablePolicyUpdater,
               EquiprobablePolicy):
    def __init__(self, nbr_states, nbr_actions, gamma):
        EpisodicTablePolicyUpdater.__init__(self, nbr_states,
                                            nbr_actions, gamma)
        EquiprobablePolicy.__init__(self, nbr_states, nbr_actions)

# Cria o ambiente de simulação e a executa
NBR_ACTIONS, NBR_STATES = 4, 25

env = Enviroment()
policy = MyPolicy(NBR_STATES, NBR_ACTIONS, gamma=0.9)

sim = EpisodicSimulator(env, policy, episode_size=250,
                        report_active=True, report_at=10000)
sim.run(100000)

# Report some statistics to match the figure shown in example 3.5
DIR_MAP = {0: "W", 1: "N", 2: "S", 3: "E"}
stats_table = policy.stats
state_list = [(stats_table[st, action, 0],
               stats_table[st, action, 1],
               st, action)
              for action in range(NBR_ACTIONS)
              for st in range(NBR_STATES)
              if stats_table[st, action, 1] > 0]
state_list.sort(reverse=True)
print("\n".join(f"{st:2d} / {DIR_MAP[action]} ({int(nbr):3d}): {round(value, 3)}"
                for value, nbr, st, action in state_list))

print(f"{(stats_table[:,:,0].sum(axis=1)/4).round(1).reshape((5, 5))}")
