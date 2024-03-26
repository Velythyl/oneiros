"""
Reads the configuration file and carries out the instructions according to the arguments passed using the settings in the configuration file.
"""
import logging
import os
import random

# import gym
import hydra
import numpy as np
import torch
from omegaconf import omegaconf

from environments.config_utils import envkey_runname_multienv, marshall_multienv_cfg, make_powerset_cfgs, \
    envkey_tags_multienv
from src.algs.ppo import PPO

logging.basicConfig(level=logging.INFO)

def get_save_path():
    try:
        import wandb
        save_path = "/".join(wandb.run.dir.split("/")[:-1])
        if save_path == "":
            raise Exception()
    except:
        save_path = "/tmp"
        print("WARNING: SAVING TO /tmp BECAUSE WANDB ISN'T RUNNING")
    return save_path

def do_exp(cfg):
    import wandb

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"

    RUN_NAME = envkey_runname_multienv(cfg.multienv)
    TAGS = envkey_tags_multienv(cfg.multienv)

    run = wandb.init(
        # entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=f"{RUN_NAME}",  # todo
        save_code=True,
        settings=wandb.Settings(start_method="thread"),
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        tags=TAGS
        # mode="disabled"
    )

    device = cfg.multienv.device

    seed = cfg.wandb["seed"]
    if seed == -1:
        seed = random.randint(0, 20000)
        cfg.wandb["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True # Potential Variable

    from environments.make_env import make_sim2sim
    train_envs, all_hooks = make_sim2sim(cfg.multienv, seed, get_save_path())

    logging.info("==========Begin trainning the Agent==========")

    agent = PPO(device=device, train_envs=train_envs, all_hooks=all_hooks, **(cfg.rl))

    agent.train()

    model_save_path = os.path.join(get_save_path(), 'model.pkl')


    logging.info("NOT SAVING MODEL>>>>>")
    #logging.info(f"==========Saving model to {model_save_path}==========")
    #if cfg.train.alg == "ppo":
    #torch.save(agent.agent.state_dict(), get_save_path() + "/saved_agent.pth")

    logging.info("==========Trainning Completed==========")
    wandb.finish()


@hydra.main(version_base=None, config_path="config", config_name="conf")
def main(cfg):
    if cfg.multienv.do_powerset is False:
        return do_exp(cfg)
    # ELSE...
    print("DOING POWERSET EVALUATION. NOTE: {eval_envs} WILL BE IGNORED")

    for new_cfg in make_powerset_cfgs(cfg):
        do_exp(new_cfg)

if __name__ == '__main__':
    main()
