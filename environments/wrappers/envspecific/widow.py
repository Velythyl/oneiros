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

        #for i in tqdm(range(self.widow_random_steps), "Randomizing initial position..."):
        #    act = torch.from_numpy(
        #        np.random.uniform(low=-np.pi, high=np.pi, size=self.env.action_space.shape)
        #    ).to(self.device).detach()
        #    act[:,-1] = 0
        #    #act = self.act_cos_and_nogripper(act)
        #    ret = self.inner_step(act)

        act = torch.from_numpy(
            np.random.uniform(low=-np.pi, high=np.pi, size=self.env.action_space.shape)
        ).to(self.device).detach()
        act[:, -1] = 0

        act = act / 4
        temp_act = torch.zeros_like(act) + act
        for i in tqdm(range(self.widow_random_steps), "Randomizing initial position..."):
            #temp_act += act
            if i == self.widow_random_steps // 4:
                temp_act += act
            if i == self.widow_random_steps // 2:
                temp_act += act
            if i == 3 * (self.widow_random_steps // 4):
                temp_act += act
            ret = self.inner_step(temp_act)

        return ret

    def reset(self):
        ret = super().reset()
        ret = self.randomize_position()[0]

        self.action_state = self.get_current_pos(ret)
        ret = self.append_act(ret, self.action_state)

        return self.obs_noise(ret)

    def get_current_pos(self, obs):
        return torch.clone(obs[:,:7])

    def append_act(self, obs, act):
        #if act is None:
        #    act = torch.zeros(self.env.action_space.shape).to(self.device)
        ret = torch.concatenate((obs, act), axis=1)
        return ret

    def act_cos_clip(self, action, desired_range=0.1):
        act = torch.cos(action)
        act = act * desired_range # fits (-1,1) to (-0.1,0.1)
        return act

    def act_noise(self, action):
        noise = torch.normal(torch.zeros_like(action), 0.0005).to(self.device)
        action = action + noise
        return action

    def inner_step(self, act):
        return super().step(act)

    def obs_noise(self, obs):
        noise = torch.normal(torch.zeros_like(obs), 0.005).to(self.device)
        return obs + noise

    def step(self, action):
        delta_action = self.act_cos_clip(action)
        delta_action = self.act_noise(delta_action)
        self.action_state = self.action_state + delta_action
        self.action_state[:,-1] = 0     # no gripper

        obs, rew, done, info = self.inner_step(self.action_state)

        if done[0] == True:
            # only randomize when step encounters a DONE flag if we are a brax env; otherwise reset was already called and mujoco already randomized
            assert torch.all(done) == True
            obs, _, _, info = self.randomize_position()
            self.action_state = self.get_current_pos(obs)

        obs = self.append_act(obs, self.action_state)
        obs = self.obs_noise(obs)
        return obs, rew, done, info