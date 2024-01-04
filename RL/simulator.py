class EpisodicSimulator:
    def __init__(self, env, policy, episode_size,
                 report_active=False, report_every=100000,
                 terminate_on_eoe=False,
                 process_done_cb=None,
                 callback_every=100000,
                 callback_fun=None):
        self.env, self.policy = env, policy
        self.episode_size = episode_size
        self.report_active, self.report_every = report_active, report_every
        # The flag below indicates the run should terminate on
        # EndOfEpisode (or, if the EoE doesn't happen, upon
        # episode_size steps
        self.terminate_on_eoe = terminate_on_eoe
        self.process_done = process_done_cb
        # Function callback to be called every so often between runs
        # of the simulation
        self.cb_every, self.cb_fun = callback_every, callback_fun

    def run(self, nbr_runs):
        for run in range(nbr_runs):
            if self.report_active and \
               run % self.report_every == 0:
                print(f"Round {run:4d}")

            if self.cb_fun and (run + 1) % self.cb_every == 0:
                self.cb_fun(run)

            st, _ = self.env.reset()
            action = self.policy.action(st)
            self.policy.reset()
            for step in range(self.episode_size):
                old_st = st
                st, reward, done, _, _ = self.env.step(action)
                self.policy.update(old_st, action, reward)
                action = self.policy.action(st)
                if done:
                    if self.process_done:
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
                            report_active=True, report_every=50,
                            callback_every=1,
                            callback_fun=lambda x: print(f"Run: {x}"))
    sim.run(100)
