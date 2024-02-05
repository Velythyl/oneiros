import brax.io.torch
import jax
import numpy as np
import torch
import wandb
from tqdm import tqdm
import jax.numpy as jp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs    # noqa
import plotly.tools as tls

NUM_STEPS = 499

def evaluate(nsteps, eval_envs, agent):

    collected_infos = {}
    ep_rew = []
    ep_len = []

    from environments.wrappers.framesave import update_nsteps
    update_nsteps(nsteps)

    next_obs = eval_envs.reset()
    for _ in tqdm(range(0, NUM_STEPS)): # todo this should match train_env's max ep len to some extent
        with torch.no_grad():
            action = agent.get_action(next_obs)
        next_obs, _, next_done, info = eval_envs.step(action)

        for key, val in info.items():
            if key in collected_infos:
                collected_infos[key].append(val)
            else:
                collected_infos[key] = [val]

    wandb_logs = {}

    def make_mbrma_plot(key, log_yscale=False):
        # key is of shape "mbrma/meanparticle_{skr key id}

        def prep_y(y_key):
            y = collected_infos[y_key]
            assert isinstance(y, list)

            if isinstance(y[0], jax.Array):
                y = [brax.io.torch.jax_to_torch(ys) for ys in y]
            if isinstance(y[0], float):
                y = [torch.tensor(ys) for ys in y]
            y = torch.vstack(y).squeeze().cpu().numpy()

            if len(y.shape) == 2:
                y = np.mean(y, axis=1)

            return y

        x = np.arange(NUM_STEPS)
        y = prep_y(key)


        if y.shape[0] != NUM_STEPS:
            SUB_STEPS = NUM_STEPS // y.shape[0]
            x = x[0::SUB_STEPS][:y.shape[0]]

        fig, ax1 = plt.subplots()

        ax1.plot(
            x,
            y,
            label=key
        )

        #title = key

        """
        RESAMPLING_KEY = "mbrma/all_resampled"
        if RESAMPLING_KEY in collected_infos:
            moments_where_resampled = collected_infos[RESAMPLING_KEY]
            for i, moment in enumerate(moments_where_resampled):
                if moment != 0:
                    ax1.axvline(x=i, color='b', label='r\e \sim U')

                    title = f"Eval (resampling e every {i} steps"""

        if log_yscale:
            ax1.set_yscale("log")
        #plt.xlabel('Timestep')
        #ax1.set_ylabel(f'{key.split("/")[-1]}')
        #plt.title(title)
        return tls.mpl_to_plotly(fig)   # todo this strips the vertical lines for some reason

    for key in collected_infos.keys():
        if key.endswith("#rew"):
            collected_rews = collected_infos[key]
            collected_rews = torch.vstack(collected_rews)

            _key = key.split("#")[0]
            wandb_logs[f"perf/EVAL_{_key}_avg_rew"] = torch.mean(collected_rews)
            wandb_logs[f"perf/EVAL_{_key}_tot_rew"] = torch.sum(collected_rews)
            wandb_logs[f"perf/EVAL_{_key}_ep_len"] = collected_rews.shape[0]

            wandb_logs[f"EVAL_EP/{key}"] = make_mbrma_plot(key, log_yscale=False)

        # make_plot
        #wandb_logs[key] = np.array(collected_infos[key]).mean()
        if "mbrma/meanparticle" in key:
            wandb_logs[f"EVAL_EP/{key}"] = make_mbrma_plot(key, log_yscale=False)
        if "mbrma/fitness" in key:
            wandb_logs[f"EVAL_EP/{key}"] = make_mbrma_plot(key, False)
        #if key.endswith("#rew"):

    return wandb_logs
