import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.distributions.normal import Normal

from src.algs.alg import _Alg
from src.algs.get_nn_for_spaces import get_nets
from src.utils.tiny_logger import TinyLogger


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_current_obs(env_obs_stack):
    return env_obs_stack[:,0,:]

def concat_priv_info(x, priv_info):
    def concat(_x, _priv_info):
        return torch.hstack((_x, _priv_info))
    return torch.vmap(concat)(x, priv_info)

class RMAAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        priv_info_size = envs.ONEIROS_METADATA.priv_info_size

        self.phase = 1

        LATENT_SIZE = 16

        assert len(envs.ONEIROS_METADATA.single_observation_space) == 2 # assert is a framestack

        self.critic_encoder = get_nets((priv_info_size,), (LATENT_SIZE,), for_actor=False)

        self.phase_1_actor_encoder = get_nets((priv_info_size,), (LATENT_SIZE,), for_actor=True)
        self.phase_2_actor_encoder = get_nets(envs.ONEIROS_METADATA.single_observation_space, (LATENT_SIZE,), for_actor=True)

        final_nets_input_space = (envs.ONEIROS_METADATA.single_observation_space[-1] + LATENT_SIZE,)

        self.critic = get_nets(final_nets_input_space, (1,), for_actor=False)

        self.actor_mean = get_nets(final_nets_input_space, envs.ONEIROS_METADATA.single_action_space, for_actor=True)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.ONEIROS_METADATA.single_action_space)))

    def bump_phase(self):
        assert self.phase == 1
        self.phase = 2

        for param in self.phase_1_actor_encoder.parameters():
            param.requires_grad = False

    #def freeze_phase_2_encoder(self):
    #    for param in self.phase_2_actor_encoder.parameters():
    #        param.requires_grad = False

    #def unfreeze_phase_2_encoder(self):
    #    for param in self.phase_2_actor_encoder.parameters():
    #        param.requires_grad = False

    def get_value(self, x, priv_info):
        x = get_current_obs(x)

        priv_info = self.critic_encoder(priv_info)
        x = concat_priv_info(x, priv_info)

        return self.critic(x)

    def get_action_and_value(self, x, priv_info, action=None):
        assert self.phase in (1,2)
        if self.phase == 1:
            z = self.phase_1_actor_encoder(priv_info)
        elif self.phase == 2:
            z = self.phase_2_actor_encoder(x)

        x = get_current_obs(x)
        x = concat_priv_info(x, z)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action(self, x, priv_info, action=None):
        return self.get_action_and_value(x, priv_info, action)[0].detach()


class RMA(_Alg):
    def __init__(self, train_envs, all_hooks, learning_rate, num_steps, total_timesteps, num_minibatches, anneal_lr,
                 gamma, gae_lambda, update_epochs, clip_coef, norm_adv, clip_vloss, ent_coef, vf_coef, max_grad_norm,
                 target_kl, **kwargs):
        # kwargs is unused, just there so we can splat cfg

        super().__init__(train_envs, all_hooks)

        self.agent = RMAAgent(self.train_envs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)

        self.phase_2_optimizer = optim.Adam(self.agent.phase_2_actor_encoder.parameters(), lr=learning_rate, eps=1e-5)
        from torch.nn import MSELoss
        self.phase_2_loss = MSELoss()

        self.num_steps = num_steps
        self.num_envs = train_envs.num_envs
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

    def _update(self, obs, actions, logprobs, rewards, dones, values, advantages):
        pass

    def train(self):
        envs = self.train_envs
        device = self.device


        batch_size = int(self.num_envs * self.num_steps)
        minibatch_size = max(int(batch_size // self.num_minibatches), int(batch_size //2))

        obs = torch.zeros((self.num_steps, self.num_envs) + envs.ONEIROS_METADATA.single_observation_space, dtype=torch.float).to(device)
        priv_info = torch.zeros((self.num_steps, self.num_envs, envs.ONEIROS_METADATA.priv_info_size), dtype=torch.float).to(device)
        actions = torch.zeros((self.num_steps, self.num_envs) + envs.ONEIROS_METADATA.single_action_space, dtype=torch.float).to(device)
        logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        returns = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

        # TRY NOT TO MODIFY: start the game
        single_global_step = 0
        global_step = 0
        self.start_time = time.time()
        next_obs = envs.reset()
        next_priv_info = envs.get_priv()
        next_done = torch.zeros(self.num_envs, dtype=torch.float).to(device)
        num_updates = self.total_timesteps // batch_size
        logger = TinyLogger()

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            from src.utils.dict_list import DictList
            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs
                obs[step] = next_obs
                priv_info[step] = next_priv_info
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs, next_priv_info)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards[step], next_done, info = envs.step(action)
                next_priv_info = envs.get_priv()
                single_global_step += 1

                if (single_global_step % 200) == 0:
                    print("\n\n\t\t~~200 GLOBAL STEPS~~\n\n")
                logger.info(next_done, info, global_step=global_step)

            if self.agent.phase == 1:
                # bootstrap value if not done
                with torch.no_grad():
                    next_value = self.agent.get_value(next_obs, next_priv_info).reshape(1, -1)
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
            b_obs = obs.reshape((-1,) + envs.ONEIROS_METADATA.single_observation_space)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.ONEIROS_METADATA.single_action_space)
            b_priv_info = priv_info.reshape((-1,) + (envs.ONEIROS_METADATA.priv_info_size,))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            if self.agent.phase == 1:
                # Optimizing the policy and value network
                clipfracs = []
                for epoch in range(self.update_epochs):
                    b_inds = torch.randperm(batch_size, device=device)
                    for start in range(0, batch_size, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_priv_info[mb_inds], b_actions[mb_inds])
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
                    "phase/phase": np.random.randint(0,10) / 10
                }

            elif self.agent.phase == 2:
                phase_2_num_updates = 0
                tot_phase_2_loss = 0
                for epoch in range(self.update_epochs):
                    b_inds = torch.randperm(batch_size, device=device)
                    for start in range(0, batch_size, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]

                        target = self.agent.phase_1_actor_encoder(b_priv_info[mb_inds])
                        pred = self.agent.phase_2_actor_encoder(b_obs[mb_inds])

                        self.phase_2_optimizer.zero_grad()
                        loss = self.phase_2_loss(pred, target)
                        loss.backward()
                        self.phase_2_optimizer.step()

                        tot_phase_2_loss += loss.detach().cpu().item()
                        phase_2_num_updates += 1
                wandb_logs = {
                    "losses/phase_2_loss": tot_phase_2_loss,
                    "phase/phase": 1 + np.random.randint(0, 10) / 10
                }

            logger.log(self.merge_logs(wandb_logs, global_step))
            wandb.log(logger.output())
            logger.reset()

        if self.agent.phase == 1:
            print()
            print("PHASE 1 DONE. STARTING PHASE 2.")
            print()
            self.agent.bump_phase()
            self.total_timesteps = self.total_timesteps // 2
            return self.train()
        elif self.agent.phase == 2:
            print("PHASE 2 DONE.")
            return

    def merge_logs(self, wandb_logs, global_step):
        hook_start = time.time()
        wandb_logs.update(self.all_hooks.step(global_step, agent=self.agent))
        hook_end = time.time()
        self.time_spent_hooking += hook_end - hook_start

        sps = int(global_step / (
                (time.time() - self.start_time) - self.time_spent_hooking
        ))

        wandb_logs["charts/sps"] = sps
        wandb_logs["charts/global_step"] = global_step
        print("sps:", sps)
        print("global_step:", global_step)
        return wandb_logs
