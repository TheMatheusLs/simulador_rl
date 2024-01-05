import numpy as np
import time

from Ex3_5.Manager import Enviroment as Env_Ex3_5
from OpticalSimFasterEnv.Enviroment.Manager import Enviroment as Env_Optical
from OpticalSimFasterEnv.Enviroment.Manager import RSA_ACTION, SAR_ACTION

from RL.policy import EquiprobablePolicy, DeterministicPolicy,\
    FixedActionPolicy, EpsilonPolicy, EpisodicTablePolicyUpdater,\
    EpisodicTablePolicyActor
from RL.simulator import EpisodicSimulator

# Some variables for configuration
USE_OPTICAL_ENV, USE_SAR_ACTION = True, True
USE_DETERMINISTIC_POLICY = True

# A policy that uses equally likely actions and a table to store the
# statistics
if USE_DETERMINISTIC_POLICY:
    class Policy_Optical(EpisodicTablePolicyUpdater,
                         DeterministicPolicy):
        def __init__(self, nbr_states, nbr_actions, gamma, actions):
            EpisodicTablePolicyUpdater.__init__(self, nbr_states,
                                                nbr_actions, gamma)
            DeterministicPolicy.__init__(self, nbr_states,
                                         nbr_actions, actions)

    class Policy_Ex3_5(EpisodicTablePolicyUpdater,
                       DeterministicPolicy):
        def __init__(self, nbr_states, nbr_actions, gamma, actions):
            EpisodicTablePolicyUpdater.__init__(self, nbr_states,
                                                nbr_actions, gamma)
            DeterministicPolicy.__init__(self, nbr_states,
                                         nbr_actions, actions)
else:
    class Policy_Optical(EpisodicTablePolicyUpdater,
                         EpisodicTablePolicyActor):
        def __init__(self, nbr_states, nbr_actions, gamma):
            EpisodicTablePolicyUpdater.__init__(self, nbr_states,
                                                nbr_actions, gamma)
            EpisodicTablePolicyActor.__init__(self, nbr_states,
                                              nbr_actions)

    class Policy_Ex3_5(EpisodicTablePolicyUpdater,
                       EpisodicTablePolicyActor):
        def __init__(self, nbr_states, nbr_actions, gamma):
            EpisodicTablePolicyUpdater.__init__(self, nbr_states,
                                                nbr_actions, gamma)
            EpisodicTablePolicyActor.__init__(self, nbr_states,
                                              nbr_actions)

blocks_list = []
def store_blocks(run, step):
    global blocks_list
    blocks_list.append(step)

blocks_list2 = {}
def store_blocks2(run, step):
    global blocks_list2
    blocks_list2.setdefault(run, []).append(step)

# Cria o ambiente de simulação e a executa
if USE_OPTICAL_ENV:
    env = Env_Optical(network_load = 100, k_routes = 3)
    NBR_ACTIONS, NBR_STATES = 2, env.nbr_nodes**2
    NBR_RUNS, EPISODE_SIZE, REPORT_EVERY = 5, 500000, 2
    if USE_DETERMINISTIC_POLICY:
        actions = [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0]
        # actions = np.ones(196)
        policy = Policy_Optical(NBR_STATES, NBR_ACTIONS, gamma=0.9,
                                actions=actions)
    else:
        explorer = EquiprobablePolicy(NBR_STATES, NBR_ACTIONS)
        exploiter = Policy_Optical(NBR_STATES, NBR_ACTIONS, gamma=0.9)
        policy = EpsilonPolicy(NBR_STATES, NBR_ACTIONS,
                               epsilon=0.03, explorer=explorer,
                               exploiter=exploiter)
else:
    NBR_ACTIONS, NBR_STATES = 4, 25
    NBR_RUNS, EPISODE_SIZE, REPORT_EVERY = 20000, 500, 10000
    env = Env_Ex3_5()
    if USE_DETERMINISTIC_POLICY:
        actions = [3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 3, 1, 0, 0, 0, 3, 1, 0, 0, 0, 3, 1, 0, 0, 0]
        actions = [3, 3, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
        policy = Policy_Ex3_5(NBR_STATES, NBR_ACTIONS, gamma=0.9,
                              actions=actions)
    else:
        explorer = EquiprobablePolicy(NBR_STATES, NBR_ACTIONS)
        exploiter = Policy_Ex3_5(NBR_STATES, NBR_ACTIONS, gamma=0.9)
        policy = EpsilonPolicy(NBR_STATES, NBR_ACTIONS,
                               epsilon=0.03, explorer=explorer,
                               exploiter=exploiter)

print(f"""*******************************
Running simulation with:
\tNbr of runs: {NBR_RUNS}
\tEpisode size: {EPISODE_SIZE}
*******************************""")
start = time.time()
EpisodicSimulator(env, policy, episode_size=EPISODE_SIZE,
                  terminate_on_eoe=False, report_active=True,
                  report_every=REPORT_EVERY,
                  process_done_cb=store_blocks2).run(NBR_RUNS)
print(f"Elapsed time: {time.time() - start}")

# Report some statistics to match the figure shown in example 3.5
stats_table = policy.stats
state_list = [(stats_table[st, action, 0],
               stats_table[st, action, 1],
               st, action)
              for action in range(NBR_ACTIONS)
              for st in range(NBR_STATES)
              if stats_table[st, action, 1] > 0]
state_list.sort(reverse=True)
if USE_OPTICAL_ENV:
    ACTION_MAP = {0: "RSA", 1: "SAR"}
    print("\n".join(f"{st % env.nbr_nodes:2d} - {st // env.nbr_nodes:2d} / ({ACTION_MAP[action]})\
({int(nbr):5d}): {value}"
                    for value, nbr, st, action in state_list[:30]))
    print()
    print("\n".join(f"{st % env.nbr_nodes:2d} - {st // env.nbr_nodes:2d} / ({ACTION_MAP[action]})\
({int(nbr):5d}): {value}"
                    for value, nbr, st, action in state_list[-30:]))
else:
    DIR_MAP = {0: "W", 1: "N", 2: "S", 3: "E"}
    print("\n".join(f"{st:2d} / {DIR_MAP[action]} ({int(nbr):3d}): {round(value, 3)}"
                    for value, nbr, st, action in state_list))

    print(f"{(stats_table[:,:,0].sum(axis=1)/4).round(3).reshape((5, 5))}")

if __name__ == "__main__":
    blocks_list2_a = blocks_list2

    import matplotlib.pyplot as plt

    plt.hist([v for v, _, _, _ in state_list], bins=25)
    plt.show()


    print([len(l) for l in blocks_list2.values()])

    blks = np.zeros(200000)
    blks[np.array(blocks_list2[4])] = 1
    plt.plot(np.convolve(blks, np.ones(150)))


    _ = plt.hist(blocks_list2[2], bins=20)
    plt.show()

    lmx = np.array(blocks_list2[4])
    lmx = lmx[2:] - lmx[1:-1]
    _ = plt.hist(lmx, bins=40, range=(0, 500))
    plt.show()

    llx = np.array(list(zip(lmx[:-1], lmx[1:])))
    plt.scatter(llx[:, 0], llx[:, 1])
    plt.show()

    lmn = np.array(blocks_list2[2])
    lmn = lmn[2:] - lmn[1:-1]
    _ = plt.hist(lmn, bins=40, range=(0, 1500))
    plt.show()
