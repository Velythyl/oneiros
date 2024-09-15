class EveryN:
    def __init__(self, every, hook):
        self.every = every
        self.last_did_every = 0
        self.count = 0
        self.hook = hook

    def step(self, n_steps):
        assert n_steps >= 0
        self.count = n_steps
        if self.count > self.last_did_every:
            if self.hook is not None:
                self.hook(nsteps=self.count)
            self.last_did_every += self.every
            return True

class EveryN2:
    def __init__(self, hook_steps, hooks):

        self.num_hooks = len(hooks)
        self.hook_steps = hook_steps
        self.every_lastdid = {i: 0 for i, _ in enumerate(hooks)}
        self.hooks = hooks

        self.last_did_every = 0
        self.count = 0

    def step(self, n_steps, agent):
        assert n_steps >= 0
        self.count = n_steps

        ret_dict = {}
        for i in range(self.num_hooks):
            every = self.hook_steps[i]
            hook = self.hooks[i]

            if self.count > self.every_lastdid[i]:
                ret = hook(nsteps=self.count, agent=agent)
                if ret is not None:
                    if isinstance(ret, dict):
                        ret_dict.update(ret)
                    else:
                        raise AssertionError("Hook's return type must be None or Dict")

                self.every_lastdid[i] += every
        return ret_dict