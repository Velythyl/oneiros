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

        NUM_OBS = self.observation_space.shape[1] + self.action_space.shape[1]
        self.obs_space_shape = (self.observation_space.shape[0], NUM_OBS)
        self.observation_space = gym.spaces.Box(low=np.ones(self.obs_space_shape) * -np.inf,
                                                high=np.ones(self.obs_space_shape) * np.inf)

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
        ret = self.append_act(ret, None)

        self.last_act = torch.zeros(self.env.action_space.shape).to(self.device)
        return self.noise_obs(ret)

    def append_act(self, obs, act=None):
        if act is None:
            act = torch.zeros(self.env.action_space.shape).to(self.device)
        ret = torch.concatenate((obs, act), axis=1)
        return ret

    def cos_act(self, action):
        #act = torch.cos(action)
        act = action
        act[:,-1] = 0
        return act

    def scale_act(self, action):
        noise = torch.normal(torch.zeros_like(action), 0.05).to(self.device)
        action = action + noise
        #action = action.clip(-1, 1)

        #if self.brax_or_mujoco is False:
        #    action = action * 0.75

        return action

    def inner_step(self, act):
        return super().step(act)

    def noise_obs(self, obs):
        noise = torch.normal(torch.zeros_like(obs), 0.005).to(self.device)
        return obs + noise

    def step(self, act):
        initial_act = act
        cosact = self.cos_act(act)
        act = self.scale_act(cosact)

        act = act.clip(-0.1, 0.1)
        act = act + self.last_act
        self.last_act = act

        obs, rew, done, info = self.inner_step(act)

        if done[0] == True:
            # only randomize when step encounters a DONE flag if we are a brax env; otherwise reset was already called and mujoco already randomized
            assert torch.all(done) == True
            obs, _, _, info = self.randomize_position()
            self.last_act = torch.zeros(self.env.action_space.shape).to(self.device)

        obs = self.append_act(obs, initial_act)
        obs = self.noise_obs(obs)
        return obs, rew, done, info