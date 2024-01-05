import random
import numpy as np


class Policy:
    def __init__(self, nbr_states, nbr_actions):
        self.nbr_states, self.nbr_actions = nbr_states, nbr_actions

    def reset(self):
        pass

    def action(self, state):
        return 0

    def update(self, state, action, value):
        pass

    def final_update(self):
        pass

class FixedActionPolicy(Policy):
    def __init__(self, nbr_states, nbr_actions, action=0):
        Policy.__init__(self, nbr_states, nbr_actions)
        self.fixed_action = action

    def action(self, state):
        return self.fixed_action

class DeterministicPolicy(Policy):
    def __init__(self, nbr_states, nbr_actions, actions):
        Policy.__init__(self, nbr_states, nbr_actions)
        self.actions = np.array([a for a in actions],
                                dtype=np.int32)

    def action(self, state):
        return self.actions[state]

class EquiprobablePolicy(Policy):
    def __init__(self, nbr_states, nbr_actions):
        Policy.__init__(self, nbr_states, nbr_actions)

    def action(self, state):
        return random.randrange(self.nbr_actions)

class EpsilonPolicy(Policy):
    def __init__(self, nbr_states, nbr_actions, epsilon,
                 explorer, exploiter):
        Policy.__init__(self, nbr_states, nbr_actions)
        self.epsilon = epsilon
        self.explorer, self.exploiter = explorer, exploiter
        # Workaround to have easy access to the stats tables
        self.stats = exploiter.stats

    def action(self, state):
        if random.random() < self.epsilon:
            return self.explorer.action(state)
        else:
            return self.exploiter.action(state)

    def update(self, state, action, reward):
        self.exploiter.update(state, action, reward)
        self.explorer.update(state, action, reward)

    def final_update(self):
        self.exploiter.final_update()
        self.explorer.final_update()

    def reset(self):
        self.exploiter.reset()
        self.explorer.reset()

class AdaptiveEpsilonPolicy(EpsilonPolicy):
    def __init__(self, nbr_states, nbr_actions, epsilon_curve,
                 explorer, exploiter):
        try:
            eps = int(epsilon_curve)
        except TypeError:
            try:
                self.ptr = iter(epsilon_curve)
            except TypeError as err:
                print("ERROR: epsilon_curve is not an integer nor a sequence")
                print(err)
            try:
                self.eps = next(self.ptr)
            except StopIteration:
                print("ERROR: epsilon_curve is empty")

        self.not_at_end = True

        EpsilonPolicy.__init__(self, nbr_states, nbr_actions, eps,
                         explorer, exploiter)

    def action(self, state):
        rc = EpsilonPolicy.action(self, state)

        if self.not_at_end:
            try:
                self.epsilon = next(self.ptr)
            except StopIteration:
                self.not_at_end = False

        return rc

class EpisodicTablePolicy(Policy):
    def __init__(self, nbr_states, nbr_actions):
        Policy.__init__(self, nbr_states, nbr_actions)

        # Cria o array de estatísticas. O último índice é para
        # armazenar a média e o contador
        self.reset_stats()
        self.reset()

    def reset(self):
        self.state_trail, self.rewards = [], []

    def reset_stats(self):
        self.stats = np.zeros((self.nbr_states, self.nbr_actions, 2))
        self.best_actions = np.zeros(self.nbr_states,
                                     dtype=np.int32)

    def internal_stats_update(self):
        pass

class EpisodicTablePolicyUpdater(EpisodicTablePolicy):
    def __init__(self, nbr_states, nbr_actions, gamma):
        EpisodicTablePolicy.__init__(self, nbr_states, nbr_actions)
        self.gamma = gamma

    def update(self, state, action, reward):
        self.state_trail.append((state, action))
        self.rewards.append(reward)

    def final_update(self):
        stats, value = self.stats, 0
        for reward, (st, action) in zip(self.rewards[::-1],
                                        self.state_trail[::-1]):
            stats[st, action, 1] += 1
            value = self.gamma*value + reward
            avg_value = stats[st, action, 0] + \
                (value - stats[st, action, 0])/stats[st, action, 1]
            stats[st, action, 0] = avg_value

        self.internal_stats_update()

class EpisodicTablePolicyActor(EpisodicTablePolicy):
    def __init__(self, nbr_states, nbr_actions):
        EpisodicTablePolicy.__init__(self, nbr_states, nbr_actions)

    def action(self, state):
        return self.best_actions[state]

    def internal_stats_update(self):
        self.best_actions = self.stats[:, :, 0].argmax(axis=1)

class FreeRunningTablePolicy(Policy):
    def __init__(self, nbr_states, nbr_actions):
        Policy.__init__(self, nbr_states, nbr_actions)

    def update(self, state, action, reward):
        pass

#        self.idx_map = dict(zip(range(self.nbr_actions),
#                                [0]*self.nbr_actions))

if __name__ == "__main__":
    NBR_ACTIONS, NBR_STATES, EPSILON = 10, 20, 0.05

    policy = FixedActionPolicy(NBR_STATES, NBR_ACTIONS, 1)
    print(f"Action = {policy.action(0)}")
    print(f"Action = {policy.action(1)}")

    policy = EquiprobablePolicy(NBR_STATES, NBR_ACTIONS)
    actions = [policy.action(0) for _ in range(50)]
    print(f"Actions: {actions[:10]}")
    print(f"Average: {sum(actions)/len(actions)}")

    class MyPolicy(EpisodicTablePolicyUpdater,
                   FixedActionPolicy):
        def __init__(self, a, nbr_states, nbr_actions, gamma):
            EpisodicTablePolicyUpdater.__init__(self, nbr_states, nbr_actions, gamma)
            FixedActionPolicy.__init__(self, nbr_states, nbr_actions, a)

    policy2 = MyPolicy(3, 4, 5, 0.9)
    print(f"Action is {policy2.action(0)}")
    policy2.update(1, 2, 3)
    policy2.final_update()

    exploiter = MyPolicy(3, NBR_STATES, NBR_ACTIONS, 0.9)
    policy = EpsilonPolicy(NBR_STATES, NBR_ACTIONS, EPSILON,
                           exploiter=exploiter, explorer=policy)
    print(f"Action = {policy.action(0)}")
    print(f"Action = {policy.action(0)}")
    policy.update(0, 0, 1)
