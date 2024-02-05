import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import wandb






def policy(config, env):
    agent=Actor_Critic(config=config,env=env)
    return agent


# def load_policy(model_path, env=None, log_path=None, device=None):
#     env.reset()
#     agent = PPO.load(path=model_path, env=env,tensorboard_log=log_path, device=device)
    
#     print(type(agent))
#     print("Return Loaded Agent")
    
#     return agent





class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias







class Actor_Critic:
    
    def __init__(
        self,
        config,
        env,
    ):    

        print(config)
        print("init")

        self.config = config
        self.env = env        
        self.seed=self.config["seed"]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.device = self.config["device"]


        self.actor = Actor(env).to(self.device)
        self.qf1 = QNetwork(env).to(self.device)
        self.qf1_target = QNetwork(env).to(self.device)
        self.target_actor = Actor(env).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(list(qf1.parameters()), lr=self.config["learning_rate"])
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.config["learning_rate"])
        self.start_time = time.time()


    # env.single_observation_space.dtype = np.float32
    # rb = ReplayBuffer(
    #     self.config["buffer_size"],
    #     env.single_observation_space,
    #     env.single_action_space,
    #     self.device,
    #     handle_timeout_termination=True,
    # )


 



    def train(self,total_timesteps):
        print("Total time step is {total_timesteps}")
        print("Trained")


        obs = self.env.reset()
        for global_step in range(total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.config["learning_starts"]:
                actions = np.array([self.env.single_action_space.sample() for _ in range(self.env.num_envs)])
            else:
                with torch.no_grad():
                    actions = self.actor(torch.Tensor(obs).to(self.device))
                    actions += torch.normal(0, self.actor.action_scale * self.config["exploration_noise"])
                    actions = actions.cpu().numpy().clip(self.env.single_action_space.low, self.env.single_action_space.high)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = self.env.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

                    wandb.log({
                        "steps": global_step,
                        "charts/episodic_return": info["episode"]["r"],
                        "charts/episodic_length": info["episode"]["l"],
                    })

                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, done in enumerate(dones):
                if done:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            # rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.config["learning_starts"]:
                data = rb.sample(self.config["batch_size"])
                with torch.no_grad():
                    next_state_actions = self.target_actor(data.next_observations)
                    qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.config["gamma"] * (qf1_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                self.q_optimizer.zero_grad()
                qf1_loss.backward()
                self.q_optimizer.step()

                if global_step % self.config["policy_frequency"] == 0:
                    actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                        target_param.data.copy_(self.config["tau"] * param.data + (1 - self.config["tau"]) * target_param.data)
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.config["tau"] * param.data + (1 - self.config["tau"]) * target_param.data)

                if global_step % 100 == 0:
                    wandb.log({
                        "steps": global_step,
                        "losses/qf1_loss": qf1_loss.item(),
                        "losses/actor_loss": actor_loss.item(),
                        "losses/qf1_values": qf1_a_values.mean().item(),
                        "charts/SPS": int(global_step / (time.time() - self.start_time)),
                               })

                    print("SPS:", int(global_step / (time.time() - self.start_time)))




    def save(self, path):
        print("WIP")

#         # Copy parameter list so we don't mutate the original dict
#         data = self.__dict__.copy()

#         # Exclude is union of specified parameters (if any) and standard exclusions
#         if exclude is None:
#             exclude = []
#         exclude = set(exclude).union(self._excluded_save_params())

#         # Do not exclude params if they are specifically included
#         if include is not None:
#             exclude = exclude.difference(include)

#         state_dicts_names = ["policy", "policy.optimizer"]
#         torch_variable_names = []
    
#         all_pytorch_variables = state_dicts_names + torch_variable_names
            
#         for torch_var in all_pytorch_variables:
#             # We need to get only the name of the top most module as we'll remove that
#             var_name = torch_var.split(".")[0]
#             # Any params that are in the save vars must not be saved by data
#             exclude.add(var_name)

#         # Remove parameter entries of parameters which are to be excluded
#         for param_name in exclude:
#             data.pop(param_name, None)

#         # Build dict of torch variables
#         pytorch_variables = None
#         if torch_variable_names is not None:
#             pytorch_variables = {}
#             for name in torch_variable_names:
#                 attr = recursive_getattr(self, name)
#                 pytorch_variables[name] = attr

#         # Build dict of state_dicts
#         params_to_save = self.get_parameters()
        
#         save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)







