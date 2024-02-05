import brax.io.torch
import jax
from gym import Wrapper


class RMAWrapper(Wrapper):
    def __init__(self, env, ppo_agent, z_dim):
        super().__init__(env)

        self.nets = []
        for net in net_and_opt[:,0]:
            self.nets.append(net)
        self.opts = []
        for opts in net_and_opt[:,1]:
            self.opts.append(opts)

        # https://ashish-kmr.github.io/a-rma/ 3 phases
        # normal rma: 2 phases

        self.rma = len(self.nets) == 2
        self.a_rma = len(self.nets) == 3
        assert self.rma != self.a_rma and self.rma or self.a_rma



    def step(self, action):
        return super(RMAWrapper, self).step(action)

    def reset(self, **kwargs):
        return super(RMAWrapper, self).reset(**kwargs)

    def get_e_privileged(self):
        # for use in phase 1
        ground_truth = self.get_current_skrs_vals()
        ground_truth = jax.vmap(self.skrs_util.true2free)(ground_truth)
        ground_truth = brax.io.torch.jax_to_torch(ground_truth)
        return ground_truth

    def freeze_phase1(self):
        # Here, we assume that the PPO agent that's being trained had self.nets[0] as part of its parameters.
        # That's super important

    def phase2_update(self):


