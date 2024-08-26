import gym
import numpy as np
import torch
from tqdm import tqdm


class WidowRandomPosition(gym.Wrapper):
    def __init__(self, env, brax_or_mujoco, widow_random_steps, device):
        super().__init__(env)
        self.brax_or_mujoco = brax_or_mujoco
        self.widow_random_steps = widow_random_steps
        self.device = device

    def randomize_position(self):
        #acts = torch.from_numpy(
        #    np.random.uniform(low=-np.pi, high=np.pi, size=(WIDOW_RANDOM_STEPS, *base_envs[-1].action_space.shape))
        #).to(multiplex_env_cfg.device[0]).detach()
        #acts.requires_grad = False
        #acts = acts.to(torch.float32)
        #acts = self.cos_act(acts)

        for i in tqdm(range(self.widow_random_steps), "Randomizing initial position..."):
            act = torch.from_numpy(
                np.random.uniform(low=-np.pi, high=np.pi, size=self.env.action_space.shape)
            ).to(self.device).detach()
            act = self.cos_act(act)
            ret = self.inner_step(act)

        return ret

    def reset(self):
        ret = super().reset()
        ret = self.randomize_position()[0]
        return ret

    def cos_act(self, action):
        action[-2:] = 0
        return torch.cos(action)

    def scale_act(self, action):
        return  action

    def inner_step(self, act):
        return super().step(act)

    def step(self, act):
        act = self.scale_act(self.cos_act(act))
        obs, rew, done, info = self.inner_step(act)

        if done[0] == True:
            # only randomize when step encounters a DONE flag if we are a brax env; otherwise reset was already called and mujoco already randomized
            assert torch.all(done) == True
            obs, _, _, info = self.randomize_position()

        return obs, rew, done, info