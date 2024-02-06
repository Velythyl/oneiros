import torch
from plotly.offline import download_plotlyjs  # noqa
from tqdm import tqdm


def evaluate(nsteps, eval_envs, agent, NUM_STEPS):

    collected_infos = {}

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

    for key in collected_infos.keys():
        if key.endswith("#rew"):
            collected_rews = collected_infos[key]
            collected_rews = torch.vstack(collected_rews)

            _key = key.split("#")[0]
            wandb_logs[f"EVAL_{_key}/avg_rew"] = torch.mean(collected_rews)
            wandb_logs[f"EVAL_{_key}/tot_rew"] = torch.sum(collected_rews)
            wandb_logs[f"EVAL_{_key}/ep_len"] = collected_rews.shape[0]


        # make_plot
        #wandb_logs[key] = np.array(collected_infos[key]).mean()
        #if "mbrma/meanparticle" in key:
        #    wandb_logs[f"EVAL_EP/{key}"] = make_mbrma_plot(key, log_yscale=False)
        #if "mbrma/fitness" in key:
            #wandb_logs[f"EVAL_EP/{key}"] = make_mbrma_plot(key, False)
        #if key.endswith("#rew"):

    return wandb_logs
