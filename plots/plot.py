import copy
import dataclasses
import functools
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn
from tqdm import tqdm
from vapeplot import vapeplot

def _run_shell(cmd):
    return subprocess.check_output(cmd, shell=True, universal_newlines=True, executable='/bin/bash')


@dataclasses.dataclass
class STATICS:
    ENVS: str = ("bgen", "bpos", "bspring", "bmjx", "mujoco")
    KEY_TRAIN_ENVS: str = "TRAIN_ENVS"
    KEY_NUM_TRAIN_ENVS: str = "NUM_TRAIN_ENVS"
STATICS = STATICS()

@functools.lru_cache
def get_pd(run_dir):
    frame = pandas.read_csv(f"{run_dir}/pd_logs.csv")
    frame["RUN_DIR"] = run_dir.split("/")[-1]

    TRAIN_tags = list(filter(lambda c: c.startswith("TRAIN "), list(frame.columns)))
    TRAIN_environments = list(set(map(lambda c: c.split("TRAIN ")[1], TRAIN_tags)))
    TRAIN_environments = "-".join(list(sorted(TRAIN_environments[0].split("-"))))
    frame[STATICS.KEY_TRAIN_ENVS] = TRAIN_environments

    TRAIN_num = len(TRAIN_environments.split("-"))
    frame[STATICS.KEY_NUM_TRAIN_ENVS] = TRAIN_num

    #EVAL_related_cols = list(filter(lambda c: c.startswith("EVAL_"), list(frame.columns)))
    #EVAL_environments = list(set(map(lambda c: c.split("/")[0], EVAL_related_cols)))

    return frame

def list_runs(runs_dir):
    return os.listdir(runs_dir)

@functools.lru_cache
def get_concat_pd(runs_dir):
    frames = []
    for subdir in list_runs(runs_dir):
        frames += [ get_pd(f"{runs_dir}/{subdir}") ]

    df = pandas.concat(frames)
    df["RUNS_DIR"] = runs_dir

    return df

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
    ret = []
    for train in list_trainenvs(DIR):
        for eval in STATICS.ENVS:
            if eval in train:
                ret.append((train, eval))
    return ret

def make_heatmap(DIR, KEY, DEDUP=False):
    big_df = get_concat_pd(DIR)

    POSSIBLE_KEYS = [STATICS.KEY_TRAIN_ENVS, STATICS.KEY_NUM_TRAIN_ENVS]
    assert KEY in POSSIBLE_KEYS

    def rename_eval_cols(df):
        RENAME_MAPPING = {
            k: k.split("EVAL_")[1].split("-")[0] for k in get_eval_tot_rew(DIR)
        }
        df = df.rename(columns=RENAME_MAPPING)
        RENAME_MAPPING = {
            "braxgeneralized": "bgen",
            "braxpositional": "bpos",
            "braxspring": "bspring",
            "braxmjx": "bmjx",
            "mujoco": "mujoco"
        }
        df = df.rename(columns=RENAME_MAPPING)
        return df

    big_df = rename_eval_cols(big_df)

    cropped_df = big_df[[STATICS.KEY_TRAIN_ENVS, *STATICS.ENVS]].dropna()

    only_last_eval = cropped_df.groupby(STATICS.KEY_TRAIN_ENVS).tail(1)
    avg_runs = only_last_eval.groupby([STATICS.KEY_TRAIN_ENVS]).mean().reset_index()

    if KEY == STATICS.KEY_TRAIN_ENVS:
        def sort_heatmap_trainenvs(heatmap_df):
            ALL_TRAIN_ENVS = list_trainenvs(heatmap_df)
            ALL_TRAIN_ENVS_SAVED = copy.deepcopy(ALL_TRAIN_ENVS)
            ALL_TRAIN_ENVS.sort()
            ALL_TRAIN_ENVS.sort(key=lambda x: len(x.split("-")))
            NEW_INDICES = []
            for x in ALL_TRAIN_ENVS:
                NEW_INDICES.append(ALL_TRAIN_ENVS_SAVED.index(x))

            heatmap_df = heatmap_df.iloc[NEW_INDICES]
            return heatmap_df

        avg_runs = sort_heatmap_trainenvs(avg_runs)
    avg_runs = avg_runs.set_index(STATICS.KEY_TRAIN_ENVS)

    if DEDUP:
        for train, eval in gen_dup_pairs(DIR):
            avg_runs.at[train, eval] = np.nan

    if KEY == STATICS.KEY_NUM_TRAIN_ENVS:
        col_num_envs = list(map(lambda x: len(x.split("-")), avg_runs.index.tolist()))
        avg_runs[KEY] = col_num_envs
        avg_runs = avg_runs.groupby(KEY)
        avg_runs = avg_runs.mean().dropna()

    KEY_MAP = {
        STATICS.KEY_TRAIN_ENVS: "Training environment",
        STATICS.KEY_NUM_TRAIN_ENVS: "Number of training environments"
    }
    avg_runs = avg_runs.rename_axis([KEY_MAP[KEY]])

    avg_runs = avg_runs.reindex(sorted(avg_runs.columns), axis=1)

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots()
    seaborn.heatmap(avg_runs, annot=True, xticklabels=True, yticklabels=True, ax=ax)
    ax.set_title(f"{KEY_MAP[KEY]} average performance " + ("(raw)" if not DEDUP else "(dedup)"))
    ax.set_xlabel("Evaluation environment")
    plt.show()


def make_numenv_heatmap(DIR, DEDUP):
    NUM_TRAIN_ENVS = STATICS.KEY_NUM_TRAIN_ENVS
    return

def make_trainenv_heatmap(DIR, DEDUP):
    return make_heatmap(DIR, KEY=STATICS.KEY_TRAIN_ENVS, DEDUP=DEDUP)

if __name__ == "__main__":
    DIR = "./runs"

    make_heatmap(DIR, KEY=STATICS.KEY_TRAIN_ENVS, DEDUP=False)
    make_heatmap(DIR, KEY=STATICS.KEY_TRAIN_ENVS, DEDUP=True)

    make_heatmap(DIR, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=False)
    make_heatmap(DIR, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=True)

    exit()
