# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import time
from typing import Union, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from gym import spaces

from src.algs.alg import _Alg
from src.utils.dict_list import DictList
from src.utils.tiny_logger import TinyLogger


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor

class ReplayBuffer:

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            in_device: Union[torch.device, str] = "cpu",
            out_device: Union[torch.device, str] = "cuda",
            n_envs: int = 1,
            dtype=torch.float32
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = observation_space.shape

        self.action_dim = action_space.shape[-1]
        self.pos = 0
        self.full = False

        self.in_device = in_device
        self.out_device = out_device

        self.n_envs = n_envs

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        self.observations = torch.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=dtype, device=in_device)
        self.next_observations = torch.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=dtype, device=in_device)
        self.actions = torch.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=dtype, device=in_device)
        self.rewards = torch.zeros((self.buffer_size, self.n_envs), dtype=dtype, device=in_device)
        self.dones = torch.zeros((self.buffer_size, self.n_envs), dtype=dtype, device=in_device)


    def to_out_device(self, array: torch.Tensor, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.clone(array).to(self.out_device)
        return array.to(self.out_device)

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def add(
            self,
            obs: torch.Tensor,
            next_obs: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            done: torch.Tensor,
    ) -> None:
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = torch.clone(obs)

        self.next_observations[self.pos] = torch.clone(next_obs)

        self.actions[self.pos] = torch.clone(action)
        self.rewards[self.pos] = torch.clone(reward)
        self.dones[self.pos] = torch.clone(done)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = torch.randint(low=0, high=upper_bound, size=(batch_size,))
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: torch.Tensor) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = torch.randint(low=0, high=self.n_envs, size=(len(batch_inds),))

        next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices]).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_out_device, data)))

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        import numpy as np
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        import numpy as np
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action_logprob_mean(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_action(self, x):
        return self.get_action_logprob_mean(x)[0]

class Agent(nn.Module):
    def __init__(self, actor, q1, q2, t_q1, t_q2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.t_q1 = t_q1
        self.t_q2 = t_q2

class SAC(_Alg):
    #args = parse_args()
    def __init__(self, train_envs, all_hooks, q_lr, policy_lr, autotune, alpha, buffer_size, total_timesteps, learning_starts, batch_size, policy_frequency, target_network_frequency, tau, gamma, **kwargs):
        # env setup
        super().__init__(train_envs, all_hooks)

        envs = self.train_envs
        device = self.device
        self.total_timesteps = int(total_timesteps)
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)

        self.alpha = alpha
        self.autotune = autotune
        self.q_lr = q_lr

        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency

        self.tau = tau
        self.gamma = gamma

        max_action = float(envs.single_action_space.high[0])

        self.actor = Actor(envs).to(device)
        self.qf1 = SoftQNetwork(envs).to(device)
        self.qf2 = SoftQNetwork(envs).to(device)
        self.qf1_target = SoftQNetwork(envs).to(device)
        self.qf2_target = SoftQNetwork(envs).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.agent = Agent(self.actor, self.qf1, self.qf2, self.qf1_target, self.qf2_target) # todo rework everything to directly access this data

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr)


    def train(self):
        # Automatic entropy tuning
        if self.autotune:
            target_entropy = -torch.prod(torch.Tensor(self.train_envs.single_action_space.shape).to(self.device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=self.q_lr)
        else:
            alpha = self.alpha

        self.rb = ReplayBuffer(
            self.buffer_size,
            self.train_envs.single_observation_space,
            self.train_envs.single_action_space,
            in_device=self.device, # use in_device="cpu" if you don't have a huge beefy GPU with tons of RAM
            out_device=self.device,
            n_envs=self.train_envs.num_envs
        )

        envs = self.train_envs
        device = self.device
        def make_random_action_sampler():
            low = torch.tensor(self.train_envs.single_action_space.low, device=device)
            high = torch.tensor(self.train_envs.single_action_space.high, device=device)

            def sampler():
                def do_one_env(i):
                    return (low - high) * torch.rand(self.train_envs.single_action_space.shape[-1], device=device) + high
                ret = torch.vmap(do_one_env, randomness="different")(torch.arange(self.train_envs.num_envs))
                return ret
            return sampler

        random_action_sampler = make_random_action_sampler()
        self.start_time = time.time()


        # TRY NOT TO MODIFY: start the game
        obs = envs.reset()
        wandb_log_returns = None
        global_step = 0
        single_global_step = 0
        logger = TinyLogger()
        print("======= START ========")

        while global_step < self.total_timesteps:
        # for global_step in range(self.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.learning_starts:
                actions = random_action_sampler()
            else:
                actions, _, _ = self.actor.get_action_logprob_mean(obs.float())
                actions = actions.detach()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, infos = envs.step(actions)
            global_step += self.train_envs.num_envs
            single_global_step += 1

            logger.info(done=terminations, info=infos, global_step=global_step)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = torch.clone(next_obs)
            self.rb.add(obs, real_next_obs, actions, rewards, terminations)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.learning_starts:
                data = self.rb.sample(self.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action_logprob_mean(data.next_observations)
                    qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if global_step % self.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        self.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self.actor.get_action_logprob_mean(data.observations)
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        if self.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action_logprob_mean(data.observations)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                # update the target networks
                if global_step % self.target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                if single_global_step % 100 == 0:
                    wandb_logs = {
                        "losses/qf1_values": qf1_a_values.mean().item(),
                    "losses/qf2_values": qf2_a_values.mean().item(),
                    "losses/qf1_loss": qf1_loss.item(),
                    "losses/qf2_loss": qf2_loss.item(),
                    "losses/qf_loss": qf_loss.item() / 2.0,
                    "losses/actor_loss": actor_loss.item(),
                    "losses/alpha": alpha,
                    }
                    if self.autotune:
                        wandb_logs["losses/alpha_loss"] = alpha_loss.item()

                    logger.log(
                        self.merge_logs(wandb_logs, global_step)
                    )

            wandb.log(logger.output())
            logger.reset()

    def merge_logs(self, wandb_logs, global_step):
        hook_start = time.time()
        wandb_logs.update(self.all_hooks.step(global_step, agent=self.actor))
        hook_end = time.time()
        self.time_spent_hooking += hook_end - hook_start

        sps = int(global_step / (
                (time.time() - self.start_time) - self.time_spent_hooking
        ))

        wandb_logs["charts/sps"] = sps
        print("sps:", sps)
        return wandb_logs

