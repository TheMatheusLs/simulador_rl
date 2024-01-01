import gymnasium as gym
import numpy as np


class Enviroment(gym.Env):
    # List with the mapping between an index representing the input
    # state and an output tuple with the 4 possible tuples of the kind
    # (state, reward), one for each action. The actions' mapping is
    # "W" -> 0, "N" -> 1, "S" -> 2, "E" -> 3
    STATE_TABLE = [((0, -1), (0, -1), (5, 0), (1, 0)),
                   ((21, 10), (21, 10), (21, 10), (21, 10)),
                   ((1, 0), (2, -1), (7, 0), (3, 0)),
                   ((13, 5), (13, 5), (13, 5), (13, 5)),
                   ((3, 0), (4, -1), (9, 0), (4, -1)),

                   ((5, -1), (0, 0), (10, 0), (6, 0)),
                   ((5, 0), (1, 0), (11, 0), (7, 0)),
                   ((6, 0), (2, 0), (12, 0), (8, 0)),
                   ((7, 0), (3, 0), (13, 0), (9, 0)),
                   ((8, 0), (4, 0), (14, 0), (9, -1)),

                   ((10, -1), (5, 0), (15, 0), (11, 0)),
                   ((10, 0), (6, 0), (16, 0), (12, 0)),
                   ((11, 0), (7, 0), (17, 0), (13, 0)),
                   ((12, 0), (8, 0), (18, 0), (14, 0)),
                   ((13, 0), (9, 0), (19, 0), (14, -1)),

                   ((15, -1), (10, 0), (20, 0), (16, 0)),
                   ((15, 0), (11, 0), (21, 0), (17, 0)),
                   ((16, 0), (12, 0), (22, 0), (18, 0)),
                   ((17, 0), (13, 0), (23, 0), (19, 0)),
                   ((18, 0), (14, 0), (24, 0), (19, -1)),

                   ((20, -1), (15, 0), (20, -1), (21, 0)),
                   ((20, 0), (16, 0), (21, -1), (22, 0)),
                   ((21, 0), (17, 0), (22, -1), (23, 0)),
                   ((22, 0), (18, 0), (23, -1), (24, 0)),
                   ((23, 0), (19, 0), (24, -1), (24, -1)),
        ]
    

    def __init__(self):
        pass

    def reset(self):
        self.cur_state = np.random.randint(25)
        return self.cur_state, {}

    def step(self, action):
        self.cur_state, reward = self.STATE_TABLE[self.cur_state][action]
        return self.cur_state, reward, False, False, {}
