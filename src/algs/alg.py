import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.distributions.normal import Normal

class _Alg:
    def __init__(self, train_envs, all_hooks):
        # kwargs is unused, just there so we can splat cfg

        self.device = train_envs.device
        self.train_envs = train_envs
        self.all_hooks = all_hooks
        self.time_spent_hooking = 0

        # todo when you make a new subclass: self.agent = ???
        # todo when you make a new subclass: ensure that self.agent.get_action(obs) returns the actions for obs, where obs is a batched obs

    def train(self):
        # todo when you make a new subclass: self.start_time = time
        # todo when you make a new subclass: call every few updates: self.write_wandb(logs that are always there, logs that might not exist, global step)
        raise NotImplementedError()

    def write_wandb(self, wandb_logs, wandb_log_returns, global_step):
        hook_start = time.time()
        wandb_logs.update(self.all_hooks.step(global_step, agent=self.agent))
        if wandb_log_returns:
            wandb_logs.update(wandb_log_returns.getmean())
        hook_end = time.time()
        self.time_spent_hooking += hook_end - hook_start

        sps = int(global_step / (
                (time.time() - self.start_time) - self.time_spent_hooking
        ))

        wandb_logs["charts/sps"] = sps
        print("sps:", sps)
        wandb.log(wandb_logs)
