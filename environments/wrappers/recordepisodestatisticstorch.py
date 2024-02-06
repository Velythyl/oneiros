import gym
import numpy as np
import torch


class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device, num_envs):
        super().__init__(env)
        self.num_envs = num_envs
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        assert len(kwargs) == 0, "If non empty kwargs, might be a reset mask, in which case we're screwed because we dont have logic for partial resets. TODO."

        observations = self.env.reset()
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)

        _rewards = rewards#.to('cpu')
        _dones = dones#.to('cpu')

        self.episode_returns += _rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - _dones
        self.episode_lengths *= (1 - _dones).int()

        if dones.any():
            ep_tot_rew = self.returned_episode_returns[dones.bool()].cpu().numpy()
            ep_tot_len = self.returned_episode_lengths[dones.bool()].float().cpu().numpy()

            ep_avg_tot_rew = np.mean(ep_tot_rew)
            ep_std_tot_rew = np.std(ep_tot_rew)
            ep_avg_tot_len = np.mean(ep_tot_len)
            ep_std_tot_len = np.std(ep_tot_len)

            infos["/ep_avg_tot_rew"] = ep_avg_tot_rew
            infos["/ep_std_tot_rew"] = ep_std_tot_rew
            infos["/ep_avg_tot_len"] = ep_avg_tot_len
            infos["/ep_std_tot_len"] = ep_std_tot_len

        return (
            observations,
            rewards,
            dones,
            infos,
        )
