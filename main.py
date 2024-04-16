"""
Reads the configuration file and carries out the instructions according to the arguments passed using the settings in the configuration file.
"""
import logging
import os
import random

# import gym
from subprocess import Popen
from time import sleep

import hydra
import numpy as np
import torch
from omegaconf import omegaconf

from environments.config_utils import envkey_runname_multienv, marshall_multienv_cfg, make_powerset_cfgs, \
    envkey_tags_multienv
from src.algs.ppo import PPO
from src.algs.sac import SAC
from src.utils import wandbcsv

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

    seed = cfg.wandb["seed"]
    if seed == -1:
        seed = random.randint(0, 20000)
        cfg.wandb["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"

    RUN_NAME = envkey_runname_multienv(cfg.multienv)
    TAGS = envkey_tags_multienv(cfg.multienv)

    wandbcsv.encapsulate(other_metadata={"seed": cfg.wandb.seed, "RL_ALG": cfg.rl.alg}, pd_attrs=dict(vars(cfg)))

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

    # torch.backends.cudnn.deterministic = True # Potential Variable

    from environments.make_env import make_sim2sim
    train_envs, all_hooks, close_all_envs = make_sim2sim(cfg.multienv, seed, get_save_path())

    logging.info("==========Begin trainning the Agent==========")

    if cfg.rl.alg == "ppo":
        agent = PPO(device=device, train_envs=train_envs, all_hooks=all_hooks, **(cfg.rl))
    elif cfg.rl.alg == "sac":
        agent = SAC(device=device, train_envs=train_envs, all_hooks=all_hooks, **(cfg.rl))


    agent.train()

    close_all_envs()

    #model_save_path = os.path.join(get_save_path(), 'model.pkl')

    #logging.info("NOT SAVING MODEL>>>>>")
    #logging.info(f"==========Saving model to {model_save_path}==========")
    #if cfg.train.alg == "ppo":
    #torch.save(agent.agent.state_dict(), get_save_path() + "/saved_agent.pth")

    logging.info("==========Trainning Completed==========")

    wandb.log_artifact(wandbcsv.get_pd_artifact())
    wandbcsv.finish()
    wandb.finish()

    exit()




@hydra.main(version_base=None, config_path="config", config_name="conf")
def main(cfg):
    import sys
    if cfg.multienv.do_powerset is False:
        return do_exp(cfg)
    # ELSE...
    print("DOING POWERSET EVALUATION. NOTE: {eval_envs} WILL BE IGNORED")

    all_powerset_cfgs = make_powerset_cfgs(cfg)
    if cfg.multienv.do_powerset_id == "None":
        for new_cfg in all_powerset_cfgs:
            import subprocess
            command = f"python3 main.py multienv.do_powerset_id={new_cfg.multienv.do_powerset_id}"
            process = subprocess.Popen(command, shell=True)

            while process.poll() is None:
                sleep(10)

    else:
        for new_cfg in all_powerset_cfgs:
            if new_cfg.multienv.do_powerset_id != cfg.multienv.do_powerset_id:
                continue

            do_exp(new_cfg)


if __name__ == '__main__':
    main()
