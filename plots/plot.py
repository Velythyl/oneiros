import copy
import dataclasses
import functools
import json
import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from vapeplot import vapeplot
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def _run_shell(cmd):
    return subprocess.check_output(cmd, shell=True, universal_newlines=True, executable='/bin/bash')


@dataclasses.dataclass
class STATICS:
    ENVS: str = ("Brax MJX", "Brax Positional", "Brax Spring", "Brax Generalized", "Mujoco")
    KEY_TRAIN_ENVS: str = "TRAIN_ENVS"
    KEY_NUM_TRAIN_ENVS: str = "NUM_TRAIN_ENVS"
    KEY_RUN_DIR: str = "RUN_DIR"
    KEY_METHOD: str = "METHOD"

STATICS = STATICS()

def env_rename(logged_envname):
    return {
        "braxmjx": STATICS.ENVS[0],
        "braxpositional": STATICS.ENVS[1],
        "braxgeneralized": STATICS.ENVS[3],
        "braxspring": STATICS.ENVS[2],
        "mujoco": STATICS.ENVS[4],

        "bpos": STATICS.ENVS[1],
        "bgen": STATICS.ENVS[3],
        "bmjx": STATICS.ENVS[0],
        "bspring": STATICS.ENVS[2]
    }[logged_envname]

def name_dr_env(dr_env_config):
    BELOW = dr_env_config["dr_percent_below"]
    ABOVE = dr_env_config["dr_percent_above"]
    ENV = env_rename(dr_env_config["env_key"].split("-")[0])

    FINAL_NAME = f"{ENV} ({BELOW}-{ABOVE})"
    return FINAL_NAME


def env_col_renamer(envname_dicostr):
    dico = envname_dicostr.split(" ")[-1]
    dico, ENDING = dico.split("/")
    dico = json.loads(dico)

    FINAL_NAME = f"{name_dr_env(dico)}/{ENDING}"
    return FINAL_NAME

def list_eval_cols(frame):
    eval_cols = [col for col in frame.columns if col.startswith('EVAL_')]

    if len(eval_cols) > 0:
        return eval_cols

    eval_cols = [col for col in frame.columns if col.startswith('EVAL ')]
    return eval_cols


def list_train_cols(frame):
    train_cols = []

    for col in frame.columns:
        for possible_env in ["braxspring", "braxpositional", "braxmjx", "braxgeneralized", "mujoco"]:
            if col.startswith(possible_env):
                train_cols.append(col)

    return train_cols

@functools.lru_cache
def get_pd(run_dir):

    frame = pandas.read_csv(f"{run_dir}/pd_logs.csv")
    frame[STATICS.KEY_RUN_DIR] = run_dir.split("/")[-1]

    with open(f"{run_dir}/logs/debug.log") as f:
        lines = f.readlines()

        config = None
        for line in lines:
            if line.startswith("config:"):
                config = line
                break
        assert config is not None
        config = eval(line.split("config:")[-1].strip())
    frame["CONFIG"] = json.dumps(config)

    rename_train_cols = {col: env_col_renamer(col) for col in list_train_cols(frame)}
    frame.rename(columns=rename_train_cols, inplace=True)

    TRAIN_ENVIRONMENTS = list(set(map(lambda c: c.split("/")[0], list(rename_train_cols.values()))))
    TRAIN_environments = " + ".join(list(sorted(TRAIN_ENVIRONMENTS)))
    frame[STATICS.KEY_TRAIN_ENVS] = TRAIN_environments


    TRAIN_num = len(TRAIN_environments.split("-"))
    frame[STATICS.KEY_NUM_TRAIN_ENVS] = TRAIN_num

    eval_col_replace = {col: f"EVAL {env_col_renamer(col)}" for col in list_eval_cols(frame)}
    frame = frame.rename(columns=eval_col_replace)

    # method key
    from omegaconf import OmegaConf
    config = OmegaConf.create(config)

    IS_VECFRAMESTACK = config.multienv.train.framestack[0] > 0 and (config.multienv.train.mat_framestack_instead[0] == False)
    IS_MATFRAMESTACK = config.multienv.train.framestack[0] > 0 and (config.multienv.train.mat_framestack_instead[0] == True)
    IS_NOT_FRAMESTACK = not IS_MATFRAMESTACK and not IS_VECFRAMESTACK

    IS_DOMAIN_RANDOMIZATION = config.multienv.train.dr_percent_below[0] < config.multienv.train.dr_percent_above[0]

    IS_VANILLA = IS_NOT_FRAMESTACK and not IS_DOMAIN_RANDOMIZATION

    method = None
    if IS_VECFRAMESTACK:
        method = "Vector Framestack"
    elif IS_MATFRAMESTACK:
        method = "Matrix Framestack"
    elif IS_VANILLA:
        method = "Vanilla"

    if IS_DOMAIN_RANDOMIZATION:
        method = f"DR + {method}"
    frame[STATICS.KEY_METHOD] = method

    return frame


def list_runs(runs_dir):
    return os.listdir(runs_dir)

@functools.lru_cache
def get_concat_pd(runs_dir):
    USE_CHECKPOINT = True
    if USE_CHECKPOINT:
        try:
            return pandas.read_feather(f"{runs_dir}/concat_pd_checkpoint.chckpnt")

        except:
            pass

    frames = []
    eval_cols = []

    for subdir in list_runs(runs_dir):
        if subdir.endswith(".chckpnt"):
            continue
        try:
            frames += [ get_pd(f"{runs_dir}/{subdir}") ]
            eval_cols.append(set(list_eval_cols(frames[-1])))

            if len(eval_cols) > 1:
                assert eval_cols[-1] == eval_cols[-2]
        except FileNotFoundError:
            print(f"FILE NOT FOUND: {runs_dir}")


    df = pandas.concat(frames)
    df["RUNS_DIR"] = runs_dir
    df = df.copy()

    df.to_feather(f"{runs_dir}/concat_pd_checkpoint.chckpnt")

    return df

def prep_env_checkpoints(runs_dir):
    checkpoints = {
        "ant": False,
        "hopper": False,
        "walker2d": False,
        "inverted_double_pendulum": False,
        "reacher": False
    }

    def read_env(env):
        return pandas.read_feather(f"{runs_dir}/{env}.chckpnt")

    def write_env(env, df):
        df.to_feather(f"{runs_dir}/{env}.chckpnt")

    for subdir in tqdm(list_runs(runs_dir)):
        if subdir.endswith(".chckpnt"):
            continue

        frame = get_pd(f"{runs_dir}/{subdir}")

        FOUND_ENV = None
        for env in checkpoints.keys():
            if env in " ".join(list(frame.columns)):
                FOUND_ENV = env
                break
        assert FOUND_ENV is not None

        if not checkpoints[FOUND_ENV]:
            checkpoints[FOUND_ENV] = []

        checkpoints[FOUND_ENV] += [frame]

    for key, val in checkpoints.items():
        write_env(key, pandas.concat(val))


def get_concatdf_for_env(runs_dir, env):

    def read_env(env):
        return pandas.read_feather(f"{runs_dir}/{env}.chckpnt")

    def write_env(env, df):
        df.to_feather(f"{runs_dir}/{env}.chckpnt")

    return read_env(env)

@functools.lru_cache
def get_eval_tot_rew(DIR):
    big_df = get_concat_pd(DIR)

    EVAL_related_cols = list(filter(lambda c: c.startswith("EVAL_"), list(big_df.columns)))
    EVAL_environments = list(set(map(lambda c: c.split("/")[0], EVAL_related_cols)))
    EVAL_tot_rew = list(filter(lambda c: c.endswith("/tot_rew"), EVAL_related_cols))

    return EVAL_tot_rew

#@functools.lru_cache
def list_trainenvs(DIR):
    if isinstance(DIR, str):
        pd = get_concat_pd(DIR)
        return list(set(pd[[STATICS.KEY_TRAIN_ENVS]].values.reshape(-1).tolist()))
    else:
        pd = DIR
        return pd[[STATICS.KEY_TRAIN_ENVS]].values.reshape(-1).tolist()

@functools.lru_cache
def gen_dup_pairs(DIR):
    def extract_env(string):
        return string.split(" (")[0]

    ret = []
    for train in list_trainenvs(DIR):

        train_envs = list(map(extract_env, train.split(" + ")))

        for eval in list_eval_cols(get_concat_pd(DIR)):

            eval_env = extract_env(eval.split("EVAL ")[-1])

            if eval_env in train_envs:
                ret.append((train, eval))

    return ret

def get_totrew_cols(frame):
    COLS_TO_KEEP = list(filter(lambda c: c.endswith("/tot_rew"), frame.columns))
    return COLS_TO_KEEP


@functools.lru_cache
def make_heatmap(DIR, KEY, DEDUP=False):
    big_df = get_concat_pd(DIR)

    POSSIBLE_KEYS = [STATICS.KEY_TRAIN_ENVS, STATICS.KEY_NUM_TRAIN_ENVS]
    assert KEY in POSSIBLE_KEYS

    COLS_TO_KEEP = [STATICS.KEY_TRAIN_ENVS, STATICS.KEY_RUN_DIR, STATICS.KEY_METHOD]
    COLS_TO_KEEP_EVAL = []
    COLS_TO_KEEP_TRAIN = []
    for col in big_df.columns:
        for env in STATICS.ENVS:
            #if col.startswith(env):
            #    COLS_TO_KEEP.append(col)
            #    COLS_TO_KEEP_TRAIN.append(col)
            if col.startswith(f"EVAL {env}"):
                COLS_TO_KEEP_EVAL.append(col)
                COLS_TO_KEEP.append(col)

    COLS_TO_KEEP = list(set(COLS_TO_KEEP))

    cropped_df = big_df[[*COLS_TO_KEEP]].dropna()
    only_last_eval = cropped_df.groupby(STATICS.KEY_RUN_DIR).tail(1)
    only_last_eval = only_last_eval[[STATICS.KEY_TRAIN_ENVS, STATICS.KEY_METHOD, *COLS_TO_KEEP_EVAL]]

    avg_runs = only_last_eval
    #avg_runs = only_last_eval.groupby([STATICS.KEY_TRAIN_ENVS]).mean().reset_index()

    if KEY == STATICS.KEY_TRAIN_ENVS:
        def sort_heatmap_trainenvs(heatmap_df):
            ALL_TRAIN_ENVS = list_trainenvs(heatmap_df)
            ALL_TRAIN_ENVS_SAVED = copy.deepcopy(ALL_TRAIN_ENVS)
            ALL_TRAIN_ENVS.sort()
            ALL_TRAIN_ENVS.sort(key=lambda x: len(x.split(" + ")))
            NEW_INDICES = []
            for x in ALL_TRAIN_ENVS:
                NEW_INDICES.append(ALL_TRAIN_ENVS_SAVED.index(x))
                ALL_TRAIN_ENVS_SAVED[NEW_INDICES[-1]] = None

            heatmap_df = heatmap_df.iloc[NEW_INDICES]
            return heatmap_df

        avg_runs = sort_heatmap_trainenvs(avg_runs)

    if DEDUP:
        for train, eval in gen_dup_pairs(DIR):
            avg_runs.loc[avg_runs[STATICS.KEY_TRAIN_ENVS] == train, eval] = np.nan

    if KEY == STATICS.KEY_NUM_TRAIN_ENVS:
        names = avg_runs[STATICS.KEY_TRAIN_ENVS].tolist()
        col_num_envs = list(map(lambda x: len(x.split(" + ")), names))
        avg_runs[KEY] = col_num_envs

        avg_runs = avg_runs.drop(STATICS.KEY_TRAIN_ENVS, axis=1)

    KEY_MAP = {
        STATICS.KEY_TRAIN_ENVS: "Training environment",
        STATICS.KEY_NUM_TRAIN_ENVS: "Number of training environments"
    }
    avg_runs = avg_runs.rename_axis([KEY_MAP[KEY]])
    avg_runs = avg_runs.reindex(sorted(avg_runs.columns), axis=1)

    def do_heatmapp(avg_runs, for_method):
        if KEY == STATICS.KEY_TRAIN_ENVS:
            avg_runs = avg_runs.groupby(STATICS.KEY_TRAIN_ENVS, sort=False).mean()
        elif KEY == STATICS.KEY_NUM_TRAIN_ENVS:
            avg_runs = avg_runs.groupby(KEY).mean().dropna()

        COLS_TO_KEEP = list(filter(lambda c: c.endswith("/tot_rew"), avg_runs.columns))
        avg_runs = avg_runs[[*COLS_TO_KEEP]]

        fig, ax = plt.subplots()
        seaborn.heatmap(avg_runs, annot=True, xticklabels=True, yticklabels=True, ax=ax)
        ax.set_title(f"{for_method}'s average eval performance " + ("(raw)" if not DEDUP else "(dedup)"))
        ax.set_xlabel("Evaluation environment")
        plt.show()

    for method in avg_runs[STATICS.KEY_METHOD].unique():
        do_this_one = avg_runs[avg_runs[STATICS.KEY_METHOD] == method]
        do_this_one = do_this_one.drop(STATICS.KEY_METHOD, axis=1)
        do_heatmapp(do_this_one, for_method=method)

    return avg_runs

def make_manyeval_linechart(DIR, DEDUP):
    avg_runs = make_heatmap(DIR, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=DEDUP)

    avg_runs = avg_runs[[STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD, *get_totrew_cols(avg_runs)]]

    avg_runs = avg_runs.set_index(STATICS.KEY_NUM_TRAIN_ENVS)

    fig, ax = plt.subplots()
    seaborn.lineplot(data=avg_runs, ax=ax)

    ax.set_title("Transfer prowess "+ ("(raw)" if not DEDUP else "(dedup)"))
    ax.set_xlabel(f"Number of training simulators" )
    ax.set_ylabel("Average total evaluation performance")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

def make_avgeval_linechart(DIR, DEDUP):
    avg_runs = make_heatmap(DIR, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=DEDUP)

    avg_runs = avg_runs[[STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD, *get_totrew_cols(avg_runs)]]

    avg_runs = avg_runs.groupby([STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD]).mean()
    #avg_runs = avg_runs.set_index(STATICS.KEY_NUM_TRAIN_ENVS).mean(axis=1)

    fig, ax = plt.subplots()
    seaborn.lineplot(avg_runs, ax=ax)

    ax.set_title("Transfer prowess averaged over all simulators "+ ("(raw)" if not DEDUP else "(dedup)"))
    ax.set_xlabel(f"Number of training simulators" )
    ax.set_ylabel("Average total evaluation performance")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

def make_avgeval_allavg_linechart(DIR, DEDUP):
    avg_runs = make_heatmap(DIR, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=DEDUP)

    avg_runs = avg_runs[[STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD, *get_totrew_cols(avg_runs)]]

    avg_runs = pandas.melt(avg_runs, id_vars=[STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD], value_vars=get_totrew_cols(avg_runs), var_name='Original_Column', value_name="NEW_COL")
    avg_runs = avg_runs.drop('Original_Column', axis=1)


    #avg_runs = avg_runs.groupby([STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD]).mean()
    #avg_runs = avg_runs.mean(axis=1).reset_index()
    #avg_runs = avg_runs.set_index(STATICS.KEY_NUM_TRAIN_ENVS).mean(axis=1)

    fig, ax = plt.subplots()
    seaborn.lineplot(avg_runs, x=STATICS.KEY_NUM_TRAIN_ENVS, y="NEW_COL", ax=ax, hue=STATICS.KEY_METHOD)

    ax.set_title("Transfer prowess averaged over all simulators "+ ("(raw)" if not DEDUP else "(dedup)"))
    ax.set_xlabel(f"Number of training simulators" )
    ax.set_ylabel("Average total evaluation performance")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


if __name__ == "__main__":
    DIR = "./runs"
    ENV = "ant"

    prep_env_checkpoints(DIR)
    ant_df = get_concatdf_for_env(DIR, ENV)


    make_heatmap(DIR, KEY=STATICS.KEY_TRAIN_ENVS, DEDUP=False)
    make_heatmap(DIR, KEY=STATICS.KEY_TRAIN_ENVS, DEDUP=True)

    make_heatmap(DIR, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=False)
    make_heatmap(DIR, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=True)

    #make_manyeval_linechart(DIR, DEDUP=False)
    #make_manyeval_linechart(DIR, DEDUP=True)

    make_avgeval_allavg_linechart(DIR, DEDUP=False)
    make_avgeval_allavg_linechart(DIR, DEDUP=True)
    i=0
   # make_avgeval_linechart(DIR, DEDUP=False)
    #make_avgeval_linechart(DIR, DEDUP=True)


    exit()
