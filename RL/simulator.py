class EpisodicSimulator:
    def __init__(self, env, policy, episode_size,
                 report_active=False, report_at=100000):
        self.env, self.policy = env, policy
        self.episode_size = episode_size
        self.report_active, self.report_at = report_active, report_at

    def run(self, nbr_runs):
        for r in range(nbr_runs):
            if self.report_active and \
               r % self.report_at == 0:
                print(f"Round {r:4d}")

            st, _ = self.env.reset()

            action = self.policy.action(st)
            self.policy.reset()
            for runs in range(self.episode_size):
                old_st = st
                st, reward, _, _, _ = self.env.step(action)
                self.policy.update(old_st, action, reward)
                action = self.policy.action(st)

            self.policy.final_update()

if __name__ == "__main__":
    class MyEnvironment:
        def reset(self):
            return 0, None

        def step(self, action):
            return 0, 1, None, None, None

    class MyPolicy:
        def reset(self):
            pass

        def action(self, state):
            return state

        def update(self, state, action, value):
            pass

        def final_update(self):
            pass

    sim = EpisodicSimulator(MyEnvironment(), MyPolicy(),
                            episode_size=250,
                            report_active=True, report_at=10000)
    sim.run(100000)
