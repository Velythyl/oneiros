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

from src.utils.every_n import EveryN
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

class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["obs"]


class PPO:
    
    def __init__(
        self,
        config,
        device,
        train_env,
        eval_env,
        video_env
    ):    
        print(config)
        print("init")

        self.config = config
        self.envs = train_env
        self.seed=self.config["seed"]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.device = device
        self.config["batch_size"] = int(self.config["num_envs"] * self.config["num_steps"])
        self.config["minibatch_size"] = int(self.config["batch_size"] // self.config["num_minibatches"])

        self.agent = Agent(self.envs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config["learning_rate"], eps=1e-5)

        RUN_DIR = "/".join(wandb.run.dir.split("/")[:-1])
        #RUN_DIR = f"videos"
        self.eval_hooks = []
        if eval_env is None:
            eval_env = []
        for _env in eval_env:
            self.eval_hooks.append(EveryN(5_000_000, functools.partial(eval, eval_envs=_env, agent=self.agent)))
        self.video_hooks = []
        if video_env is None:
            video_env = []
        for _env in video_env:
            self.video_hooks.append(EveryN(25_000_000, functools.partial(record, video_envs=_env, agent=self.agent, RUN_DIR=RUN_DIR)))
        self.hooks = self.video_hooks + self.eval_hooks


        
        # ALGO Logic: Storage setup
        self.obs = torch.zeros((self.config["num_steps"], self.config["num_envs"]) + self.envs.single_observation_space.shape, dtype=torch.float).to(self.device)
        self.actions = torch.zeros((self.config["num_steps"], self.config["num_envs"]) + self.envs.single_action_space.shape, dtype=torch.float).to(self.device)
        self.logprobs = torch.zeros((self.config["num_steps"], self.config["num_envs"]), dtype=torch.float).to(self.device)
        self.rewards = torch.zeros((self.config["num_steps"], self.config["num_envs"]), dtype=torch.float).to(self.device)
        self.dones = torch.zeros((self.config["num_steps"], self.config["num_envs"]), dtype=torch.float).to(self.device)
        self.values = torch.zeros((self.config["num_steps"], self.config["num_envs"]), dtype=torch.float).to(self.device)
        self.advantages = torch.zeros_like(self.rewards, dtype=torch.float).to(self.device)
        


    def train(self,total_timesteps):
        print(f"Total time step is {total_timesteps}")
        # print("Trained")

        self.start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        next_obs = self.envs.reset()
        next_done = torch.zeros(self.config["num_envs"], dtype=torch.float).to(self.device)
        num_updates = total_timesteps // self.config["batch_size"]
        # video_filenames = set()


        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            # if self.config["anneal_lr"]:
            if False:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.config["learning_rate"]
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.config["num_steps"]-1):

                global_step += 1 * self.config["num_envs"]
                self.obs[step] = next_obs
                # import pdb; pdb.set_trace()
                # print(self.dones[step])
                # print(next_done)
                self.dones[step] = next_done

                if True and global_step > 10000: # todo uncomment
                    for hook in self.hooks:
                        hook.step(global_step)
                    pass




                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                # next_obs, reward, terminated, truncated, infos = self.env.step(action.cpu().numpy())
                

                next_obs, self.rewards[step], next_done, info = self.envs.step(action)
                if next_done.any():
                    _next_done = next_done.to('cpu')
                    episodic_return = info['r'][_next_done.bool()]
                    episodic_length = info['l'][_next_done.bool()]
                    episodic_return = episodic_return.mean().item()
                    episodic_length = episodic_length.float().mean().item()
                    print(
                        f"global_step={global_step}, episodic_return={episodic_return}, episodic_length={episodic_length}")

                    wandb.log({
                        "steps": global_step,
                        "charts/episodic_return": episodic_return,
                        "charts/episodic_length": episodic_length,
                        # "charts/episodic_return": info["episode"]["r"],
                        # "charts/episodic_length": info["episode"]["l"],                        
                                })

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                self.advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.config["num_steps"])):
                    if t == self.config["num_steps"] - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.config["gamma"] * nextvalues * nextnonterminal - self.values[t]
                    self.advantages[t] = lastgaelam = delta + self.config["gamma"] * self.config["gae_lambda"] * nextnonterminal * lastgaelam
                returns = self.advantages + self.values

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = self.advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(self.config["update_epochs"]):
                b_inds = torch.randperm(self.config["batch_size"], device=self.device)
                for start in range(0, self.config["batch_size"], self.config["minibatch_size"]):
                    end = start + self.config["minibatch_size"]
                    mb_inds = b_inds[start:end]
                    
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.config["clip_coef"]).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.config["norm_adv"]:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config["clip_coef"], 1 + self.config["clip_coef"])
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.config["clip_vloss"]:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.config["clip_coef"],
                            self.config["clip_coef"],
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.config["ent_coef"] * entropy_loss + v_loss * self.config["vf_coef"]

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config["max_grad_norm"])
                    self.optimizer.step()

                if self.config["target_kl"] is not None:
                    if approx_kl > self.config["target_kl"]:
                        break

            wandb.log({
                "steps": global_step,
                "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                # "losses/explained_variance": explained_var,
                "charts/SPS": int(global_step / (time.time() - self.start_time)),
                })




            print("SPS:", int(global_step / (time.time() - self.start_time)))

        print("Trained")

            # if self.config["track"] and self.config["capture_video"]:
            #     for filename in os.listdir(f"videos/{self.config['exp_name']}"):
            #         if filename not in video_filenames and filename.endswith(".mp4"):
            #             wandb.log({f"videos": wandb.Video(f"videos/{self.config['exp_name']}/{filename}")})
            #             video_filenames.add(filename)





    def save(self, path):
        # print("WIP")
        torch.save(self.agent.state_dict(),path)


    def load(self,path):
        self.agent.load_state_dict(torch.load(path))




