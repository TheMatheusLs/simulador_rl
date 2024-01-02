import numpy as np
import matplotlib.pyplot as plt

from Ex3_5.Manager import Enviroment as Env_Ex3_5
from OpticalSimFasterEnv.Enviroment.Manager import Enviroment as Env_Optical
from OpticalSimFasterEnv.Enviroment.Manager import RSA_ACTION, SAR_ACTION

from RL.policy import EquiprobablePolicy, \
    EpisodicTablePolicyUpdater, FixedActionPolicy
from RL.simulator import EpisodicSimulator

# Some variables for configuration
USE_OPTICAL_ENV, USE_SAR_ACTION = False, True

# A policy that uses equally likely actions and a table to store the
# statistics
class Policy_Ex3_5(EpisodicTablePolicyUpdater,
                   EquiprobablePolicy):
    def __init__(self, nbr_states, nbr_actions, gamma):
        EpisodicTablePolicyUpdater.__init__(self, nbr_states,
                                            nbr_actions, gamma)
        EquiprobablePolicy.__init__(self, nbr_states, nbr_actions)

class Policy_Optical(EpisodicTablePolicyUpdater,
                   EquiprobablePolicy):
    def __init__(self, nbr_states, nbr_actions, gamma, value):
        EpisodicTablePolicyUpdater.__init__(self, nbr_states,
                                            nbr_actions, gamma=0.9)
        FixedActionPolicy.__init__(self, nbr_states, nbr_actions,
                                   value)

# Cria o ambiente de simulação e a executa
if USE_OPTICAL_ENV:
    NBR_ACTIONS, NBR_STATES = 2, env.nbr_nodes**2
    NBR_RUNS, EPISODE_SIZE = 1000, 30000
    env = Env_Optical()
    policy = Policy_Optical(NBR_STATES, NBR_ACTIONS, gamma=0.9,
                            value=SAR_ACTION if USE_SAR_ACTION else
                            RSA_ACTION)
else:
    NBR_ACTIONS, NBR_STATES = 4, 25
    NBR_RUNS, EPISODE_SIZE = 100000, 250
    env = Env_Ex3_5()
    policy = Policy_Ex3_5(NBR_STATES, NBR_ACTIONS, gamma=0.9)

sim = EpisodicSimulator(env, policy, episode_size=EPISODE_SIZE,
                        report_active=True, report_at=10000)
sim.run(NBR_RUNS)

# Report some statistics to match the figure shown in example 3.5
if USE_OPTICAL_ENV:
    pass
else:
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
