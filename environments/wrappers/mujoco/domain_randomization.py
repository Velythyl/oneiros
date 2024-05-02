import gymnasium
import numpy as np
from gym import Wrapper

def make_sampler(base, percent_below, percent_above):
    base_low = base * percent_below
    base_high = base * percent_above

    if (base_low == base_high).all():
        def identity():
            return base
        return identity
    else:
        def sampler():
            return np.random.uniform(base_low, base_high)
        return sampler

def marshall_str(string):
    if isinstance(string, str):
        return {
            "none": None,
            "true": True,
            "false": False
        }[string.lower()]
    return string

class MujocoDomainRandomization(Wrapper):
    def __init__(self, env, percent_below, percent_above, do_on_reset, do_on_N_step, do_at_creation):
        super().__init__(env)

        self.sampler = make_sampler(env.unwrapped.model.body_mass, percent_below, percent_above)

        self.do_at_creation = marshall_str(do_at_creation)
        self.done_at_creation = False
        self.do_on_reset = marshall_str(do_on_reset)

        do_on_N_step = marshall_str(do_on_N_step)
        if isinstance(do_on_N_step, int):
            val = do_on_N_step
            def _thunk():
                return val
        elif isinstance(do_on_N_step, tuple) or isinstance(do_on_N_step, list):
            def _thunk():
                return np.random.uniform(do_on_N_step[0], do_on_N_step[1])
        self.do_on_N_step = _thunk
        self.current_do_on_N_step = None

    def do_dr(self):
        new_masses = self.sampler()
        self.env.unwrapped.model.body_mass = new_masses
        print(new_masses)

    def reset(self, **kwargs):
        if self.do_at_creation and not self.done_at_creation:
            self.do_dr()
            self.done_at_creation = True

        if self.do_on_reset is not None and self.do_on_reset:
            self.do_dr()

        if self.do_on_N_step:
            self.current_do_on_N_step = self.do_on_N_step()

        ret = super(MujocoDomainRandomization, self).reset(**kwargs)
        self.step_count = 0
        return ret

    def step(self, action):
        if self.current_do_on_N_step is not None and self.current_do_on_N_step > 0 and self.step_count > 0 and (self.step_count % self.current_do_on_N_step) == 0:
            self.do_dr()
            self.step_count = 0
            self.current_do_on_N_step = self.do_on_N_step()

        ret = super(MujocoDomainRandomization, self).step(action)
        if len(ret) == 4:
            done = ret[-2]
        else:
            done = ret[-3] or ret[-2]

        if done:
            if self.do_on_N_step:
                self.current_do_on_N_step = self.do_on_N_step()
            self.step_count = 0

        self.step_count += 1

        return ret


if __name__ == "__main__":
    env = gymnasium.make("Ant-v4", max_episode_steps=1000, autoreset=True)
    x = env.unwrapped.model.body_mass
    print(x)
    print(np.unique(x, return_counts=True))
    exit()


    def eval(low, high):
        baselines = []

        for _ in range(1000):
            np.random.seed(1)
            env = gymnasium.make("Ant-v4", max_episode_steps=1000, autoreset=True)
            env = MujocoDomainRandomization(env, low, high, do_on_reset=False, do_on_N_step=10, do_at_creation=True)
            x = env.reset(seed=1)
            for i in range(20):
                x = env.step(action=env.action_space.sample() * 0)[0]
            baselines.append(x)

        baselines = np.vstack(baselines)
        baselines_mean = baselines.mean(axis=0)
        baselines_std = baselines.std(axis=0)
        return baselines_mean[5], baselines_std[5]

    print(eval(1., 1.)) # EQ
    print(eval(1., 1000.)) # DIFF
    print(eval(1., 1.)) # EQ
