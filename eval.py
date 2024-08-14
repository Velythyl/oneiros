"""
Reads the configuration file and carries out the instructions according to the arguments passed using the settings in the configuration file.
"""
import logging
import os
import random

# import gym
import shutil
import uuid
from subprocess import Popen
from time import sleep

import hydra
import numpy as np
import torch
from omegaconf import omegaconf, OmegaConf

from environments.config_utils import envkey_runname_multienv, marshall_multienv_cfg, make_powerset_cfgs, \
    envkey_tags_multienv
from src.algs.ppo import PPO
from src.algs.rma import RMA
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
    ENV = cfg.multienv.train.env_key[0].split("-")[-1]
    print("ENV:", ENV)

    wandbcsv.encapsulate(
        other_metadata={"seed": cfg.wandb.seed, "RL_ALG": cfg.rl.alg, "FRAMESTACK": cfg.multienv.train.framestack,
                        "ENV": ENV}, pd_attrs=dict(vars(cfg)))

    run = wandb.init(
        # entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=f"{RUN_NAME}",  # todo
        save_code=False,
        settings=wandb.Settings(start_method="thread"),
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        tags=TAGS,
        #mode="disabled"
    )

    cfg_yaml = OmegaConf.to_yaml(cfg)
    print(cfg_yaml)
    with open(f"{wandb.run.dir}/hydra_config.yaml", "w") as f:
        f.write(cfg_yaml)

    device = cfg.multienv.device

    # torch.backends.cudnn.deterministic = True # Potential Variable

    from environments.make_env import make_sim2sim
    cfg.multienv.train.num_env = [1]*len(cfg.multienv.train.num_env)
    cfg.multienv.eval.num_env = [1] * len(cfg.multienv.train.num_env)

    train_envs, all_hooks, close_all_envs = make_sim2sim(cfg.multienv, seed, get_save_path())

    logging.info("==========Begin trainning the Agent==========")

    def playback_pt(i, pt_file):
        from torch import jit
        agent = jit.load(pt_file, map_location="cpu").cuda()

        class Agent:
            def __init__(self, net):
                self.net = net

            def get_action(self, obs):
                def get_act(obs):
                    try:
                        return agent(obs)
                    except:
                        pass

                    return agent(obs, None)
                act = get_act(obs)
               # print(act)
                return act

        all_hooks.step(i, Agent(agent))


    playback = cfg.multienv.eval_playback
    if playback.endswith(".pt"):
        todo_files = [playback]
    else:
        todo_files = []
        for root, dirs, files in os.walk(playback):
            for file in files:
                if file.endswith(".pt"):
                    todo_files += [os.path.join(root, file)]

    for i, playback_file in enumerate(todo_files):
        playback_pt(i, playback_file)

        playback_file_name = playback_file.split("/")[-1].split(".")[0]

        for root, dirs, files in os.walk(wandb.run.dir):
            for file in files:
                if file.startswith(f"{i+1}_.pt"):
                    os.rename(
                        os.path.join(root, file),
                        os.path.join(root,
                                     file.replace(f"{i+1}_", f"{playback_file_name}_")
                                     )
                    )


    close_all_envs()

    print("Saving PD artifact")
    wandb.log_artifact(wandbcsv.get_pd_artifact())
    print("Saving local PD")
    wandbcsv.finish()
    exit()


@hydra.main(version_base=None, config_path="config", config_name="conf_playback")
def main(cfg):
    def fix_cfg_jank(cfg):
        omegaconf2dict = lambda c: OmegaConf.to_container(c)
        dict2omegaconf = lambda c: OmegaConf.create(c)

        train_env = cfg.train_env
        eval_env = cfg.eval_env
        env_method = cfg.env_method

        cfg = omegaconf2dict(cfg)

        cfg["multienv"]["train"] = omegaconf2dict(train_env)
        cfg["multienv"]["eval"] = omegaconf2dict(eval_env)

        cfg["multienv"]["train"].update(omegaconf2dict(env_method))
        cfg["multienv"]["eval"].update(omegaconf2dict(env_method))

        if env_method.alg == "ppo":
            cfg["rl"] = cfg["ppo"]
        elif env_method.alg == "rma":
            cfg["rl"] = cfg["rma"]

        del cfg["train_env"]
        del cfg["eval_env"]
        cfg = dict2omegaconf(cfg)
        return cfg

    old_cfg = cfg
    cfg = fix_cfg_jank(cfg)

    assert cfg.multienv.do_powerset is not False and cfg.multienv.do_powerset is not None

    # ELSE...
    print("DOING POWERSET EVALUATION. NOTE: {eval_envs} WILL BE IGNORED")

    if cfg.multienv.do_powerset_id == "None":
        pass
    else:
        if isinstance(cfg.multienv.do_powerset_id, str):
            if "-" in cfg.multienv.do_powerset_id:
                cfg.multienv.do_powerset_id = list(map(int, cfg.multienv.do_powerset_id.split("-")))
            elif ":" in cfg.multienv.do_powerset_id:
                low, high = list(map(int, cfg.multienv.do_powerset_id.split(":")))
                cfg.multienv.do_powerset_id = (np.arange(high)[low:high]).tolist()
        elif isinstance(cfg.multienv.do_powerset_id, int):
            cfg.multienv.do_powerset_id = [cfg.multienv.do_powerset_id]

        assert isinstance(cfg.multienv.do_powerset_id, omegaconf.ListConfig)

    def subproc_launch_exp(powerset_id):
        import subprocess

        id = uuid.uuid4()
        with open(f"/tmp/{id}.yaml", "w") as f:
            OmegaConf.save(old_cfg, f)

        command = f"python3 main.py --config-path=/tmp --config-name={id} multienv.do_powerset_id={powerset_id} "
        process = subprocess.Popen(command, shell=True)

        while process.poll() is None:
            sleep(10)

        try:
            os.remove(f"/tmp/{id}.yaml")
        except:
            pass

    all_powerset_cfgs = make_powerset_cfgs(cfg)
    if cfg.multienv.do_powerset_id == "None":
        for new_cfg in all_powerset_cfgs:
            print(f"DOING POWERSET ID {new_cfg.multienv.do_powerset_id}")
            subproc_launch_exp(new_cfg.multienv.do_powerset_id)

    else:
        for new_cfg in all_powerset_cfgs:
            if new_cfg.multienv.do_powerset_id not in cfg.multienv.do_powerset_id:
                continue

            if len(cfg.multienv.do_powerset_id) == 1:
                print(f"DOING POWERSET ID {new_cfg.multienv.do_powerset_id}")
                do_exp(new_cfg)
            else:
                subproc_launch_exp(new_cfg.multienv.do_powerset_id)


if __name__ == '__main__':
    main()
