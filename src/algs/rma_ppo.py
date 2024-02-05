import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.distributions.normal import Normal

from environments.make_env import get_skrs
from src.algs.alg import _Alg
from src.mb_rma.syskeyrange_logic import SysKeyRangeUtils
from src.utils.tiny_logger import TinyLogger


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

class EnvParamsEncoder(nn.Module):
    def __init__(self,  num_params_encoder,
                        latent_dim,
                        hidden_dims=[128, 128],
                        init_noise_std=1.0,
                        activation='elu',
                        **kwargs):
        if kwargs:
            print("EnvParamsEncoder.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(EnvParamsEncoder, self).__init__()

        activation = get_activation(activation)

        # Extrinsic params encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(num_params_encoder, hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                encoder_layers.append(nn.Linear(hidden_dims[l], latent_dim))
            else:
                encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)
        print(f"Encoder MLP: {self.encoder}")

        # # Noise. TODO?
        # self.std = nn.Parameter(init_noise_std * torch.ones(latent_dim))
        # self.distribution = None
        # # disable args validation for speedup
        # Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self, env_params):
        return self.encoder(env_params)

class AdaptationModule(nn.Module):
    def __init__(self, num_obs,
                 num_actions,
                 latent_dim,
                 hidden_dims=[64, 64],
                 mlp_output_dim=32,
                 num_temporal_steps=32,
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("AdaptationModule.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(AdaptationModule, self).__init__()

        activation = get_activation(activation)

        # MLP
        mlp_layers = []
        mlp_layers.append(nn.Linear(num_obs + num_actions, hidden_dims[0]))
        mlp_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                mlp_layers.append(nn.Linear(hidden_dims[l], mlp_output_dim))
            else:
                mlp_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                mlp_layers.append(activation)
        self.mlp = nn.Sequential(*mlp_layers)
        print(f"Adaptation module MLP: {self.mlp}")

        # Temporal CNN
        self.num_temporal_steps = num_temporal_steps
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(mlp_output_dim, 32, kernel_size=8, stride=3, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=0))
        # Add non linearities here?
        self.linear = nn.Linear(32, latent_dim)
        print(f"Adaptation module temporal CNN: {self.temporal_cnn}")

        # # Noise. TODO?
        # self.std = nn.Parameter(init_noise_std * torch.ones(latent_dim))
        # self.distribution = None
        # # disable args validation for speedup
        # Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self, state_action_history):
        if len(state_action_history.shape) == 3:
            return self.forward_batch(state_action_history)
        elif len(state_action_history.shape) == 2:
            return self.forward_single(state_action_history)
        else:
            raise ValueError(f"state_action_history has invalid shape: {state_action_history.shape}")

    def forward_batch(self, state_action_history):
        # state_action_history (num_envs, num_temporal_steps, num_obs + num_actions)
        mlp_output = self.mlp(state_action_history)
        mlp_output = mlp_output.permute(0, 2, 1)
        # mlp_output (num_envs, num_temporal_steps, mlp_output_dim)
        cnn_output = self.temporal_cnn(mlp_output)
        # cnn_output (num_envs, 32, 1)
        latent = self.linear(cnn_output.flatten(1))
        # latent (num_envs, latent_dim)
        return latent

    def forward_single(self, state_action_history):
        mlp_output = self.mlp(state_action_history)
        mlp_output = mlp_output.permute(1, 0)
        cnn_output = self.temporal_cnn(mlp_output)
        return self.linear(cnn_output.flatten(0))

class Agent(nn.Module):
    def __init__(self, envs, num_params_latent):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + num_params_latent, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + num_params_latent, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_latents(self):
        raise AssertionError("agent.get_latents() was not filled in; fill this in by assigning to it in the PPO class.")

    def augment_input(self, x, history, groudtruth_env_params):
        latents = self.get_latents(history, groudtruth_env_params)
        concat = torch.concat((x, latents), dim=0)
        return concat

    def get_value(self, x, history, groundtruth_env_params):
        return self.critic(self.augment_input(x, history, groundtruth_env_params))

    def get_action_and_value(self, x, history, groundtruth_env_params, action=None):
        action_mean = self.actor_mean(self.augment_input(x, history, groundtruth_env_params))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(self.augment_input(x, history, groundtruth_env_params))

    def get_action(self, x, history, groundtruth_env_params, action=None):
        return self.get_action_and_value(x, history, groundtruth_env_params, action)[0].detach()


import torch.nn.functional as F


class PPO(_Alg):
    def __init__(self, train_envs, all_hooks, learning_rate, num_steps, total_timesteps, num_minibatches, anneal_lr,
                 gamma, gae_lambda, update_epochs, clip_coef, norm_adv, clip_vloss, ent_coef, vf_coef, max_grad_norm,
                 target_kl, num_params_latent, **kwargs):
        # kwargs is unused, just there so we can splat cfg

        super().__init__(train_envs, all_hooks)

        self.num_steps = num_steps
        self.num_envs = train_envs.num_envs

        skrs_util = SysKeyRangeUtils(get_skrs(None,env=train_envs))
        self.num_groundtruth_env_params = skrs_util.vskr.free_dim
        self.num_params_latent = num_params_latent

        self.adaptation_module = AdaptationModule(
            num_obs=np.array(train_envs.single_observation_space.shape).prod(),
            num_actions=np.prod(train_envs.single_action_space.shape),
            latent_dim=self.num_params_latent)
        self.privileged_module = EnvParamsEncoder(self.num_groundtruth_env_params, self.num_params_latent)
        self.adaptation_module_optimizer = optim.Adam(self.adaptation_module.parameters(), lr=learning_rate, eps=1e-5)

        self.learning_rate = learning_rate
        def instantiate_agent_and_optimizer(with_encoder):
            self.agent = Agent(self.train_envs, self.num_params_latent).to(self.device)
            if with_encoder:
                self.optimizer = optim.Adam([{"params": self.agent.parameters()}, {"params": self.privileged_module.parameters()}], lr=learning_rate, eps=1e-5)
            else:
                self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        self.instantiate_agent_and_optimizer = instantiate_agent_and_optimizer
        self.instantiate_agent_and_optimizer(with_encoder=True)

        def gt_latents(groundtruth_env_params):
            return self.privileged_module.forward(groundtruth_env_params)

        def adaptation_latents(history):
            return self.adaptation_module.forward(history)

        self.phase = 0
        self.phase = self.incr_phase()

        def get_phase_latents(history, groundtruth_env_params):
            if self.phase == 1:
                return gt_latents(groundtruth_env_params)
            if self.phase == 2:
                return adaptation_latents(history)
            if self.phase == 3:
                return adaptation_latents(history)
        self.get_phase_latents = get_phase_latents
        self.agent.get_latents = get_phase_latents

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

    def incr_phase(self):
        new_phase = self.phase + 1
        if new_phase == 1:
            self.adaptation_module = self.adaptation_module.eval()
            self.privileged_module = self.privileged_module.train()
            self.agent = self.agent.train()

        if new_phase == 2:
            self.privileged_module = self.privileged_module.eval()
            self.adaptation_module = self.adaptation_module.train()
            self.agent = self.agent.eval()

        if new_phase == 3:
            self.instantiate_agent_and_optimizer(with_encoder=False)  # completely resets agent weights
            self.privileged_module = self.privileged_module.eval()
            self.adaptation_module = self.adaptation_module.eval()
            self.agent = self.agent.train()

        if new_phase > 3:
            raise Exception("No RMA or A-RMA phase exists beyond 3.")
        self.phase = new_phase
        return self.phase

    def update_adaptation_module(self, batch_history, batch_groundtruth_env_params):
        mean_mse_loss = 0
        for env_params_batch, state_action_history_batch in (batch_history, batch_groundtruth_env_params):
            encoded_params_adaptation_module = self.adaptation_module(state_action_history_batch)
            encoded_params_encoder = self.privileged_module(batch_groundtruth_env_params).detach()
            mse_loss = F.mse_loss(encoded_params_adaptation_module, encoded_params_encoder)

            # Gradient step
            self.adaptation_module_optimizer.zero_grad()
            mse_loss.backward()
            # Use this?
            # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.adaptation_module_optimizer.step()

            mean_mse_loss += mse_loss.item()

        num_updates = batch_history.shape[0]
        mean_mse_loss /= num_updates
        return mean_mse_loss

    def maybe_update(self, obs, actions, logprobs, rewards, dones, values, advantages):
        pass

    def train(self):
        envs = self.train_envs
        device = self.device

        batch_size = int(self.num_envs * self.num_steps)
        minibatch_size = max(int(batch_size // self.num_minibatches), int(batch_size //2))

        obs = torch.zeros((self.num_steps, self.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
        actions = torch.zeros((self.num_steps, self.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(device)
        logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(device)
        advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

        # TRY NOT TO MODIFY: start the game
        single_global_step = 0
        global_step = 0
        self.start_time = time.time()
        next_obs = envs.reset()
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
            wandb_log_returns = DictList()
            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards[step], next_done, info = envs.step(action)
                single_global_step += 1

                if (single_global_step % 200) == 0:
                    print("\n\n\t\t~~RESAMPLED~~\n\n")
                logger.info(next_done, info, global_step=global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
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

            if self.phase == 2:
                adaptation_mse_loss = self.update_adaptation_module(...)
                wandb_logs = {
                    "losses/adaptation_mse_loss": adaptation_mse_loss,
                }
            else:
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

                wandb_logs = {
                    "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                }

            logger.log(self.merge_logs(wandb_logs, global_step))
            wandb.log(logger.output())
            logger.reset()

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
