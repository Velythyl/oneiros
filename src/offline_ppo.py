import functools
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.distributions.normal import Normal
from tqdm import tqdm

from src.utils.eval import evaluate
from src.utils.every_n import EveryN, EveryN2
from src.utils.record import record


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action(self, x, action=None):
        return self.get_action_and_value(x, action)[0].detach()


class PPO:
    def __init__(self, train_envs, replay_buffer, all_hooks, learning_rate, num_steps, total_timesteps, num_minibatches, anneal_lr, gamma, gae_lambda, update_epochs, clip_coef, norm_adv, clip_vloss, ent_coef, vf_coef, max_grad_norm, target_kl, **kwargs):
        # kwargs is unused, just there so we can splat cfg

        self.device = train_envs.device

        self.train_envs = train_envs
        self.all_hooks = all_hooks

        self.agent = Agent(self.train_envs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)

        self.num_steps = num_steps
        self.num_envs = train_envs.num_env
        self.total_timesteps = total_timesteps
        self.anneal_lr = anneal_lr
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.clip_coef = clip_coef
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.replay_buffer = replay_buffer


    def train(self):
        N_TRAIN_STEPS = 75_000_000

        for GLOBAL_STEP in tqdm(range(0, N_TRAIN_STEPS, self.batch_size)):
            batch = next(iter(self.replay_buffer))

            x = []
            for b in batch:
                x.append(b.to(self.device))

            wandb_logs = self._train(tuple(x))
            wandb_logs.update(self.all_hooks.step(GLOBAL_STEP, agent=self.agent))
            wandb.log(wandb_logs)
        print("Training done")


    def _train(self, batch):
        envs = self.train_envs
        device = self.device

        state, bc_action, next_state, reward, not_done = batch # self.replay_buffer.sample(self.batch_size)

        batch_size = int(self.num_envs * self.num_steps)
        minibatch_size = int(batch_size // self.num_minibatches)

        #obs = torch.zeros((self.num_steps, self.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
        #actions = torch.zeros((self.num_steps, self.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(device)
        #logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        #rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        #dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        #values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        num_updates = self.total_timesteps // batch_size

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            from src.utils.dict_list import DictList
            wandb_log_returns = DictList()

            # ALGO LOGIC: action logic
            with torch.no_grad():
                actions, logprobs, _, value = self.agent.get_action_and_value(state)
                values = value.flatten()

            # TRY NOT TO MODIFY: execute the game and log data.
            if False: #next_done.any():
                episodic_return = info['r'][next_done.bool()].cpu().numpy()
                episodic_length = info['l'][next_done.bool()].float().cpu().numpy()
                wandb_log_returns["perf/ep_r"] = episodic_return
                wandb_log_returns["perf/ep_l"] = episodic_length
                episodic_return = episodic_return.mean()
                episodic_length = episodic_length.mean()
                print(
                    f"global_step={global_step}, episodic_return={episodic_return}, episodic_length={episodic_length}")

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_state).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(self.update_epochs):
                b_inds = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break


            # TRY NOT TO MODIFY: record rewards for plotting purposes
            wandb_logs = {
                "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "charts/SPS": int(global_step / (time.time() - start_time))
            }

            wandb_logs.update(self.all_hooks.step(global_step, agent=self.agent))

            print("SPS:", int(global_step / (time.time() - start_time)))
            wandb.log(wandb_logs)