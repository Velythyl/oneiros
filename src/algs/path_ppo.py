import time

import brax.io.torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from flax import struct
from torch.distributions.normal import Normal
from tqdm import tqdm

from src.algs.alg import _Alg
from src.algs.eren_yeager import Paths, _Data
from src.utils.tiny_logger import TinyLogger


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




import jax.numpy as jp
@struct.dataclass
class PPOData(_Data):
    obs: jp.array
    action: jp.array
    logprob: jp.array
    rew: jp.array
    done: jp.array
    value: jp.array

def make_ppodata(**kwargs):
    def t2j(x_torch):
        shape = x_torch.shape
        x_torch_flat = torch.flatten(x_torch)
        x_jax = brax.io.torch.torch_to_jax(x_torch_flat)
        return x_jax.reshape(shape)

    dico = {}
    for key, val in kwargs.items():
        val = t2j(val)
        dico[key] = val

    return PPOData(
        **dico
    )
def extract_ppodata(ppodata):
    ret = {}
    for key, val in vars(ppodata).items():
        ret[key] = brax.io.torch.jax_to_torch(val)
    return PPOData(**ret)

def keep_non_leafs(ppodata):
    ret = {}
    for key, val in vars(ppodata).items():
        ret[key] = val[:-1]
    ret["rew"] = ppodata.rew[1:]
    return PPOData(**ret)

class Path_PPO(_Alg):
    def __init__(self, train_envs, all_hooks, learning_rate, num_steps, total_timesteps, minibatch_size, anneal_lr,
                 gamma, gae_lambda, update_epochs, clip_coef, norm_adv, clip_vloss, ent_coef, vf_coef, max_grad_norm,
                 target_kl, **kwargs):
        # kwargs is unused, just there so we can splat cfg

        super().__init__(train_envs, all_hooks)

        self.agent = Agent(self.train_envs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)

        self.num_steps = num_steps
        self.num_envs = train_envs.num_envs
        self.total_timesteps = total_timesteps
        self.anneal_lr = anneal_lr
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
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

        # todo maybe run random actions for however many steps, to find out what num_steps means in terms of batch_size? idk

        buf = Paths.init(
            PPOData(
                obs=jp.zeros((1, self.num_envs) + envs.single_observation_space.shape),
                action = jp.zeros((1, self.num_envs) + envs.single_action_space.shape),
                logprob = jp.zeros((1, self.num_envs)),
                rew = jp.zeros((1, self.num_envs)),
                done = jp.zeros((1, self.num_envs)),
                value = jp.zeros((1, self.num_envs)),
            )
        )

        # TRY NOT TO MODIFY: start the game
        single_global_step = 0
        global_step = 0
        self.start_time = time.time()
        next_obs = envs.reset()

        id_proto = jp.ones((self.num_envs, envs.num_history, 1))
        envs.add_stack("id", id_proto)
        id_proto = jp.ones((self.num_envs, 1))

        next_done = torch.zeros(self.num_envs, dtype=torch.float).to(device)
        next_rewards = torch.zeros_like(next_done)
        #num_updates = self.total_timesteps // batch_size
        logger = TinyLogger()

        while global_step < self.total_timesteps:
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                raise NotImplementedError()
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            buf = buf.clear()
            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs

                obs = next_obs
                dones = next_done
                rewards = next_rewards

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values = value.flatten()
                actions = action
                logprobs = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                envs.add_to_stack("id", buf.main_branch_end * id_proto)
                next_obs, next_rewards, next_done, info = envs.step(action)
                single_global_step += 1

                if (single_global_step % 200) == 0:
                    print("\n\n\t\t~~RESAMPLED~~\n\n")
                logger.info(next_done, info, global_step=global_step)

                buf = buf.grow(
                    make_ppodata(
                        obs=obs,
                        action=actions,
                        logprob=logprobs,
                        rew=rewards,
                        done=dones,
                        value=values,
                        #is_leaf=False   # useless
                    )
                )

            buf = buf.grow(
                make_ppodata(
                    obs=next_obs,
                    done=next_done,
                    rew=next_rewards,
                    #is_leaf=True,   # all stuff after this is useless
                    action=actions, # dummy
                    logprob=logprobs,# dummy
                    value=values# dummy
                )
            )

            def add_extra_traindata():
                extra_traindata = envs.get_traindata()
                if extra_traindata["next_obs"] is None:
                    return buf, 0

                num_particles = extra_traindata["next_obs"].shape[1]   # num frames for each envs
                history_len = extra_traindata["next_obs"].shape[2]   # num frames for each envs

                def splat(arr):
                    #         (env,particle) (env,history) (particles*history, num_envs, dim)
                    arr = arr.transpose(0,1).transpose(1,2)#
                    if len(arr.shape) == 4:
                        arr = arr.reshape(num_particles * history_len, envs.num_envs, -1)
                    else:
                        arr = arr.reshape(num_particles * history_len, envs.num_envs)
                    return torch.clone(arr)
                next_obs = splat(extra_traindata["next_obs"])
                rewards = splat(extra_traindata["reward"])
                dones = splat(extra_traindata["done"])
                ids = brax.io.torch.torch_to_jax(splat(extra_traindata["id"]).squeeze()[:,0]) # along env dim, all ==

                num_stars = ids.shape[0]

                def fake(arr):
                    return torch.ones_like(arr[None].expand((num_stars, *arr.shape))) * -1

                data = {
                    "obs": next_obs,
                    "rew": rewards,
                    "done": dones,
                    "logprob": fake(logprobs),
                    "action": fake(actions),
                    "value": fake(values)
                }

                new_buf = buf.star(ids, make_ppodata(**data))
                return new_buf, num_stars * envs.num_envs
            buf, added = add_extra_traindata()
            global_step += added

            # todo this whole thing is fucked :)
            wandb_logs = self.update(buf)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            logger.log(self.merge_logs(wandb_logs, global_step))
            wandb.log(logger.output())
            logger.reset()

    def update(self, buf):
        obs, logprobs, actions, advantages, returns, values = self.get_path_advantages(buf)

        # flatten the batch
        b_obs = obs.reshape((-1,) + self.train_envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.train_envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        DEBUG = True
        if DEBUG:
            for i, tensor in enumerate((b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)):
                # has nan?
                if torch.isnan(tensor).any():
                    print("NAN YES!")
                    print(i)
                    print(tensor.shape)
                    exit(-1)
                # has nan?
                if not torch.isfinite(tensor).any():
                    print("INF YES!")
                    print(i)
                    print(tensor.shape)
                    exit(-1)

        batch_size = b_obs.shape[0]  # int(self.num_envs * self.num_steps)
        minibatch_size = self.minibatch_size  # int(batch_size // self.num_minibatches)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in tqdm(range(self.update_epochs)):
            b_inds = torch.randperm(batch_size, device=self.device)
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

        wandb_logs = {
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
        }
        return wandb_logs

    def get_path_advantages(self, buf):
        with torch.no_grad():
            path_obs = []
            path_action = []
            path_logprob = []
            path_rew = []
            path_value = []
            path_done = []
            path_advantages = []
            path_returns = []

            buf_paths = buf.get_data_paths()
            buf.clear()
            for path in tqdm(buf_paths):
                path = extract_ppodata(path)
                next_obs = path.obs[-1]
                next_done = path.done[-1]

                path = keep_non_leafs(path)
                num_elements_in_path = path.obs.shape[0]
                rewards = path.rew
                value = path.value
                done = path.done

                path_obs.append(path.obs)
                path_action.append(path.action)
                path_logprob.append(path.logprob)
                path_rew.append(path.rew)
                path_done.append(path.done)
                path_value.append(path.value)

                # bootstrap value if not done
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                num_steps = path.action.shape[0]
                for t in reversed(range(num_elements_in_path)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - done[t + 1]
                        nextvalues = value[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - value[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + value
                path_advantages.append(advantages)
                path_returns.append(returns)

            ret = (path_obs, path_logprob, path_action, path_advantages, path_returns, path_value)
            def conc(tens):
                return torch.concatenate(tens, dim=0)
            return list(map(conc, ret))


    def merge_logs(self, wandb_logs, global_step):
        hook_start = time.time()
        wandb_logs.update(self.all_hooks.step(global_step, agent=self.agent))
        hook_end = time.time()
        self.time_spent_hooking += hook_end - hook_start

        sps = int(global_step / (
                (time.time() - self.start_time) - self.time_spent_hooking
        ))

        wandb_logs["charts/sps"] = sps
        print("sps:", sps)
        return wandb_logs


