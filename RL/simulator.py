class EpisodicSimulator:
    def __init__(self, env, policy, episode_size,
                 report_active=False, report_every=100000,
                 terminate_on_eoe=False,
                 process_done_cb=None,
                 periodic_cb_period=100000,
                 periodic_cb_fun=None,
                 eor_cb_fun=None):
        self.env, self.policy = env, policy
        self.episode_size = episode_size
        self.report_active, self.report_every = report_active, report_every
        # The flag below indicates the run should terminate on
        # EndOfEpisode (or, if the EoE doesn't happen, upon
        # episode_size steps
        self.terminate_on_eoe = terminate_on_eoe
        self.process_done = process_done_cb
        # Function callback to be called every so often in the
        # simulation
        self.periodic_cb_period = periodic_cb_period
        self.periodic_cb_fun = periodic_cb_fun
        self.periodic_cb_ctr = periodic_cb_period
        # Callback function to be called at the end of each simulation
        # run
        self.eor_cb_fun = eor_cb_fun

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
                self.policy.update(old_st, action, st, reward)
                action = self.policy.action(st)
                if done:
                    if self.process_done:
                        self.process_done(run, step)
                    if self.terminate_on_eoe:
                        break
                if self.periodic_cb_fun:
                    self.periodic_cb_ctr -= 1
                    if self.periodic_cb_ctr == 0:
                        self.periodic_cb_ctr = self.periodic_cb_period
                        self.periodic_cb_fun(self.policy,
                                             run, step,
                                             state=old_st,
                                             action=action,
                                             new_state=st,
                                             reward=reward)

            self.policy.final_update()
            if self.eor_cb_fun:
                self.eor_cb_fun(self, run)


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

        def update(self, state, action, new_state, value):
            pass

        def final_update(self):
            pass

    def periodic_cb(sim, run, step, **kw):
        print(f"Run: {run}, step: {step}")
    sim = EpisodicSimulator(MyEnvironment(), MyPolicy(),
                            episode_size=250,
                            report_active=True, report_every=5,
                            periodic_cb_period=125,
                            periodic_cb_fun=periodic_cb)
    sim.run(10)
