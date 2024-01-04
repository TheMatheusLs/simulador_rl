class EpisodicSimulator:
    def __init__(self, env, policy, episode_size,
                 report_active=False, report_every=100000,
                 terminate_on_eoe=False,
                 process_done_cb=lambda x, y: None):
        self.env, self.policy = env, policy
        self.episode_size = episode_size
        self.report_active, self.report_every = report_active, report_every
        # The flag below indicates the run should terminate on
        # EndOfEpisode (or, if the EoE doesn't happen, upon
        # episode_size steps
        self.terminate_on_eoe = terminate_on_eoe
        self.process_done = process_done_cb

    def run(self, nbr_runs):
        for run in range(nbr_runs):
            if self.report_active and \
               run % self.report_every == 0:
                print(f"Round {run:4d}")

            st, _ = self.env.reset()
            action = self.policy.action(st)
            self.policy.reset()
            for step in range(self.episode_size):
                old_st = st
                st, reward, done, _, _ = self.env.step(action)
                self.policy.update(old_st, action, reward)
                action = self.policy.action(st)
                if done:
                    self.process_done(run, step)
                    if self.terminate_on_eoe:
                        break

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
                            report_active=True, report_every=10000)
    sim.run(100000)
