import copy
import dataclasses
import functools
import gc
import json
import os
import re
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
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

#@functools.lru_cache
def get_pd(run_dir):

    frame = pandas.read_csv(f"{run_dir}/pd_logs.csv")
    frame[STATICS.KEY_RUN_DIR] = run_dir.split("/")[-1]

    import yaml

    with open(f"{run_dir}/files/hydra_config.yaml") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    #with open(f"{run_dir}/logs/debug.log") as f:
    #    lines = f.readlines()
    #
    #    config = None
    #    for line in lines:
    #        if line.startswith("config:"):
    #            config = line
    #            break
    #    assert config is not None
    #    config = eval(line.split("config:")[-1].strip())
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

    #IS_VANILLA = IS_NOT_FRAMESTACK and not IS_DOMAIN_RANDOMIZATION

    method = None
    if IS_VECFRAMESTACK:
        method = "Vector Framestack"
    elif IS_MATFRAMESTACK:
        method = "Matrix Framestack"

    if config.rl.alg == "rma":
        assert IS_MATFRAMESTACK
        method = "RMA (PPO)"
    else:
        if method is None:
            method = f"{str(config.rl.alg).upper()}"
        else:
            method = f"{method} ({str(config.rl.alg).upper()})"

    if IS_DOMAIN_RANDOMIZATION:
        method = f"(WITH DR); {method}"
    else:
        method = f"(WITHOUT DR); {method}"
    frame[STATICS.KEY_METHOD] = method

    COLS_TO_KEEP_ENDSWITH = [
        STATICS.KEY_TRAIN_ENVS,
        STATICS.KEY_NUM_TRAIN_ENVS,
        STATICS.KEY_METHOD,
        STATICS.KEY_RUN_DIR,
        "/tot_rew",
        "/avg_rew",
        "/rew",
    ]

    FOUND_ENV = None
    for env in ENVNAMES:
        if env in " ".join(list(frame.columns)):
            FOUND_ENV = env
            break
    assert FOUND_ENV is not None

    frame = frame[[col for col in frame.columns if any(col.endswith(suffix) for suffix in COLS_TO_KEEP_ENDSWITH)]]

    #if "hopper" in FOUND_ENV:
    #    shutil.rmtree(run_dir)

    return frame, FOUND_ENV


def list_runs(runs_dir):
    ret = os.listdir(runs_dir)
    ret = list(filter(lambda x: not x.endswith(".zip"), ret))
    return ret

ENVNAMES = ["ant", "hopper", "walker2d", "inverted_double_pendulum", "reacher", "widow"]


def read_env(runs_dir, env):
    return pandas.read_feather(f"{runs_dir}/{env}.chckpnt") #df.read_csv(f"{runs_dir}/{env}.chckpnt") #  pandas.read_feather(f"{runs_dir}/{env}.chckpnt")


def write_env(runs_dir, env, df):
    #df.to_csv(f"{runs_dir}/{env}.chckpnt")
    df.to_feather(f"{runs_dir}/{env}.chckpnt")


def prep_env_checkpoints(runs_dir):
    CHECKPOINTS = {key: False for key in ENVNAMES}


    for subdir in tqdm(list_runs(runs_dir)):
        if subdir.endswith(".chckpnt"):
            continue

        try:
            frame, FOUND_ENV = get_pd(f"{runs_dir}/{subdir}")
        except FileNotFoundError as e:
            assert e.filename.endswith("pd_logs.csv")
            continue

        #FOUND_ENV = None
        #for env in CHECKPOINTS.keys():
        #    if env in " ".join(list(frame.columns)):
        #        FOUND_ENV = env
        #        break
        #assert FOUND_ENV is not None

        if not CHECKPOINTS[FOUND_ENV]:
            CHECKPOINTS[FOUND_ENV] = []

        CHECKPOINTS[FOUND_ENV] += [frame]

    for key, val in CHECKPOINTS.items():
        if val is False:
            continue
        print("Saving checkpoint...")
        gc.collect()
        write_env(runs_dir, key, pandas.concat(val))


def get_concatdf_for_env(runs_dir, env):


    #def read_env(env):
    #    return pandas.read_feather(f"{runs_dir}/{env}.chckpnt")

    #def write_env(env, df):
    #    df.to_feather(f"{runs_dir}/{env}.chckpnt")

    try:
        return read_env(runs_dir, env)
    except:
        return False


def list_trainenvs(DIR):
    pd = DIR
    return pd[[STATICS.KEY_TRAIN_ENVS]].values.reshape(-1).tolist()

@functools.lru_cache
def gen_dup_pairs(DIR, ENV):
    big_df = get_concatdf_for_env(DIR, ENV)

    def extract_env(string):
        return string.split(" (")[0]

    ret = []
    for train in list_trainenvs(big_df):

        train_envs = list(map(extract_env, train.split(" + ")))

        for eval in list_eval_cols(big_df):

            eval_env = extract_env(eval.split("EVAL ")[-1])

            if eval_env in train_envs:
                ret.append((train, eval))

    return ret

def get_totrew_cols(frame):
    COLS_TO_KEEP = list(filter(lambda c: c.endswith("/tot_rew"), frame.columns))
    return COLS_TO_KEEP

def rename_rew_cols(frame):
    renames = {col: col.split("/")[0] for col in get_totrew_cols(frame)}
    return frame.rename(columns=renames)


def filter_df(df):  # todo fixme make flag
    # List of substrings to search for
    substrings = ["LOW", "MID", "HIGH"]

    # Filter columns containing any of the substrings
    cols_with_substrings = [col for col in df.columns if any(sub in col for sub in substrings)]

    # Further filter to keep only columns that contain "MID D"
    filtered_cols = [col for col in cols_with_substrings if "MID D" in col]

    # Combine filtered columns with other columns
    final_cols = [col for col in df.columns if col not in cols_with_substrings] + filtered_cols

    # Create a new DataFrame with the selected columns
    filtered_df = df[final_cols]

    filtered_df.columns = filtered_df.columns.str.replace('  MID D', '')

    return filtered_df

def make_heatmap(DIR, ENV, KEY, DEDUP=False, AGG_METHOD=False, do_plots=True, ONLY_TRAIN=False, FIVE_GRID=False):
    big_df = get_concatdf_for_env(DIR, ENV)

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
    only_last_eval = cropped_df.groupby(STATICS.KEY_RUN_DIR).tail(3) #.head(2).groupby(STATICS.KEY_RUN_DIR).tail(1) #.dropna().tail(1)
    only_last_eval = only_last_eval[[STATICS.KEY_TRAIN_ENVS, STATICS.KEY_METHOD, *COLS_TO_KEEP_EVAL]]

    avg_runs = only_last_eval
    #avg_runs = only_last_eval.groupby([STATICS.KEY_TRAIN_ENVS]).mean().reset_index()

    def get_vmin_vmax(frame_pre_modifs):
        frame_pre_modifs = frame_pre_modifs.copy()
        totrew_cols = get_totrew_cols(frame_pre_modifs)
        #totrew_cols = [col  for col in totrew_cols if "0.5-10.0" in col]
        frame_pre_modifs = pandas.melt(frame_pre_modifs, id_vars=[STATICS.KEY_TRAIN_ENVS, STATICS.KEY_METHOD],
                                       value_vars=totrew_cols, var_name='Original_Column', value_name="NEW_COL")
        frame_pre_modifs = frame_pre_modifs.drop('Original_Column', axis=1)
        HEATMAP_VMAX = frame_pre_modifs.NEW_COL.max()
        HEATMAP_VMIN = frame_pre_modifs.NEW_COL.min()

        if ENV == "reacher" and do_plots:
            for column in avg_runs.columns:
                try:
                    avg_runs[column] = np.clip(avg_runs[column], -10000, HEATMAP_VMAX)
                    HEATMAP_VMIN = -10000
                except:
                    pass


        return frame_pre_modifs, HEATMAP_VMIN, HEATMAP_VMAX
    HEATMAP_MESH, HEATMAP_VMIN, HEATMAP_VMAX = get_vmin_vmax(avg_runs)

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

    if ONLY_TRAIN:
        assert DEDUP
        assert KEY == STATICS.KEY_NUM_TRAIN_ENVS

    if DEDUP:

        # Get the values in column 'A'
        a_values = avg_runs[STATICS.KEY_TRAIN_ENVS].values

        # Create a boolean mask for the entire DataFrame
        mask = np.zeros_like(avg_runs.values, dtype=bool)

        # Check each value in 'A' against the column names
        for i, value in enumerate(a_values):
            mask_row = []

            value = list(map(lambda v: v.split(" (")[0], value.split(" + ")))
            for col in avg_runs.columns:
                if col == STATICS.KEY_TRAIN_ENVS:
                    mask_row.append(False)
                    continue

                FOUND_VAL = False
                for val in value:
                    if val in col:
                        FOUND_VAL = True
                mask_row.append(FOUND_VAL * (not ONLY_TRAIN))


            mask[i, :] = mask_row #[value.split(" (")[0] in col for col in avg_runs.columns]

        #if ONLY_TRAIN:
        #    mask = ~mask

        avg_runs = avg_runs.mask(mask)

        #for index, row in df.iterrows():
        #    for col in list_eval_cols(df):
        #        if row[STATICS.KEY_TRAIN_ENVS].split(" (")[0] in col:
        #            df.at[index, col] = np.nan

        #for train, eval in gen_dup_pairs(DIR, ENV):
        #    avg_runs.loc[avg_runs[STATICS.KEY_TRAIN_ENVS] == train, eval] = np.nan

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
    avg_runs = avg_runs.reindex(sorted(avg_runs.columns,  key=lambda col: " ".join([col.split(" (")[0], col.split("-")[-1]])), axis=1)


    TRAIN_COL_REPLACE = {
        "Brax Spring": "BS",
        "Brax Positional": "BP",
        "Brax MJX": "BM",
        "Brax Generalized": "BG",
        "Mujoco": "M "
    }
    def cleanup_trainenvs(frame):
        frame.index = frame.index.map(lambda x: re.sub(r' \(.+?-.+?\)', '', x.strip()).replace("  ", " "))
        # frame['TRAIN_ENVS'] = frame['TRAIN_ENVS'].apply(lambda x: re.sub(r' \(.+?-.+?\)', '', x.strip()).replace("  ", " "))

        def replace_trainenv(trainenv):
            trainenv = trainenv.split(" + ")
            trainenv = list(map(lambda t: TRAIN_COL_REPLACE[t], trainenv))
            return ", ".join(trainenv)

        frame.index = frame.index.map(
            lambda x: replace_trainenv(x))
        return frame

    def cleanup_evalenvs(frame):
        RANGES = {
            "0.005-0.5": " LOW D",
            #"0.375-0.751": " LOW P",
            "0.2525-0.2535": " LOW P",
            "1.0-1.0": "  MID P",
            "0.5-10.0": "  MID D",
            "0.5-2.0": "  MID D",
            "15.0-15.1": "HIGH P",
            "10.0-20.0": "HIGH D",
            "2.0-4.0": "HIGH D",
            "3.0-3.1": "HIGH P",
        }

        EVAL_COL_REPLACE = list_eval_cols(frame)
        EVAL_COL_REPLACE_2 = {}
        for eval in EVAL_COL_REPLACE:
            new_name = eval
            for key, val in TRAIN_COL_REPLACE.items():
                new_name = re.sub(f"{key}", val, new_name)
            new_name = new_name.replace("EVAL ", "")

            for RANGE, NAME in RANGES.items():
                new_name = new_name.replace(f"({RANGE})", NAME)

            EVAL_COL_REPLACE_2[eval] = new_name
        frame = frame.rename(columns=EVAL_COL_REPLACE_2)
        frame = rename_rew_cols(frame)
        return frame

    def do_heatmapp(avg_runs, title, vmin, vmax):
        if KEY == STATICS.KEY_TRAIN_ENVS:
            avg_runs = avg_runs.groupby(STATICS.KEY_TRAIN_ENVS, sort=False).mean()
        elif KEY == STATICS.KEY_NUM_TRAIN_ENVS:
            avg_runs = avg_runs.groupby(KEY).mean().dropna()

        COLS_TO_KEEP = list(filter(lambda c: c.endswith("/tot_rew"), avg_runs.columns))
        avg_runs = avg_runs[[*COLS_TO_KEEP]]

        if KEY == STATICS.KEY_TRAIN_ENVS:
            avg_runs = cleanup_trainenvs(avg_runs)
        avg_runs = cleanup_evalenvs(avg_runs)
        avg_runs = rename_rew_cols(avg_runs)

        if ENV == "reacher" and do_plots:
            for column in avg_runs.columns:
                try:
                    avg_runs[column] = np.clip(avg_runs[column], -10000, vmax)
                    vmin = -10000
                except:
                    pass



        avg_runs = filter_df(avg_runs)

        if FIVE_GRID:
            avg_runs = avg_runs[~avg_runs.index.str.contains('\,')]


        fig, ax = plt.subplots(figsize=(20,20) if not FIVE_GRID else (5,5))
        ax.figure.tight_layout()
        #cmap = vapeplot.cmap('jazzcup')
        #cmap = seaborn.color_palette("coolwarm", as_cmap=True)

        def create_checkerboard(shape, block_size):
            pattern = np.indices(shape).sum(axis=0) % 2
            checkerboard = np.kron(pattern, np.ones((block_size, block_size)))
            return checkerboard
        checkerboard = create_checkerboard((avg_runs.shape[0] * 15, avg_runs.shape[1]*15), 1)
        extent = [0, avg_runs.shape[1], 0, avg_runs.shape[0]]
        from PIL import Image
        #ax.imshow(checkerboard, cmap=ListedColormap(['white', 'black']), alpha=1.0, extent=extent, origin="upper")

        ax.imshow(np.array(Image.open("./crosslines.png")), cmap=ListedColormap(['white', 'black']), alpha=1.0, extent=extent)

        # Define multiple colors
        flag_colors = [
            "#a30162",
            '#b55690',
            '#d162a4',
            '#ffffff',
            "#ff9a56",
            "#ef7626",
            "#ef7626"]
        # Create a colormap
        jazz_colors = vapeplot.cmap("jazzcup").colors
        sea_colors = vapeplot.cmap("seapunk").colors
        colors = [jazz_colors[0],sea_colors[-1],jazz_colors[-2], jazz_colors[-1],  flag_colors[-1]]
        colors = vapeplot.cmap("vaporwave").colors
        colors = colors[len(list(colors)) // 2 + 1:]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        cmap.set_bad(color='none')

        heatmap = seaborn.heatmap(avg_runs, annot=False, xticklabels=True, yticklabels=True, ax=ax, cmap=cmap, cbar=False, cbar_kws={'shrink': 0.4}, vmin=vmin, vmax=vmax)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        mesh = ax.pcolormesh(HEATMAP_MESH[["NEW_COL"]].values, cmap=cmap)
        divider = make_axes_locatable(ax)
        cbar_size = (0.1 / avg_runs.shape[-1]) * 100
        #print(cbar_size)
        cax = divider.append_axes("right", size=f"{cbar_size}%", pad=0.05)
        # Get the images on an axis
        #ax.collections[-1].colorbar.remove()
        cb = plt.colorbar(mesh, cax=cax)

        vrange = (cb.vmax - cb.vmin)
        ypad = 0.05 * vrange
        cb.set_ticks([cb.vmin + ypad, cb.vmax - ypad])  # Set the ticks at the min and max values
        cb.set_ticklabels(['Low', 'High'])  # Set custom tick labels
        #cb.ax.xaxis.set_tick_params(pad=-15)
        cb.outline.set_visible(False)
        #ax.xticks(rotation=90)
        #plt.yticks(rotation=0)
        mesh.remove()

        ax.set_title(title)
        ax.set_xlabel("Evaluation simulator")
        ax.set_ylabel(f"Training simulator{'(s)' if not FIVE_GRID else ''}")
        #ax.subplots_adjust(bottom=4)
        plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.3)

        os.makedirs(f"saved_plots/{ENV}/{KEY}/dedups/", exist_ok=True)
        os.makedirs(f"saved_plots/{ENV}/{KEY}/raws/", exist_ok=True)
        if DEDUP:
            plt.savefig(f"saved_plots/{ENV}/{KEY}/dedups/isagg_{AGG_METHOD}_isfivegrid_{FIVE_GRID}_{title}.pdf", dpi=900, bbox_inches='tight')
        else:
            plt.savefig(f"saved_plots/{ENV}/{KEY}/raws/isagg_{AGG_METHOD}_isfivegrid_{FIVE_GRID}_{title}.pdf", dpi=900, bbox_inches='tight')

        #plt.savefig(f"saved_plots/{title}.pdf", dpi=900, bbox_inches='tight')
        plt.close(fig)
        #plt.show()
        x=0


    if FIVE_GRID:
        assert AGG_METHOD



    if do_plots:
        if AGG_METHOD:
            do_this_one = avg_runs.copy()

            WITH_DR = do_this_one.loc[do_this_one[STATICS.KEY_METHOD].str.contains("WITH DR")]
            WITHOUT_DR = do_this_one.loc[do_this_one[STATICS.KEY_METHOD].str.contains("WITHOUT DR")]

            for do_this_one, STRING in zip([WITH_DR, WITHOUT_DR], ["with DR", "without DR"]):
                do_this_one = do_this_one.drop(STATICS.KEY_METHOD, axis=1)
                do_this_one = do_this_one.groupby(KEY, sort=False).mean().reset_index()

                if KEY == STATICS.KEY_TRAIN_ENVS:
                    if DEDUP:
                        if FIVE_GRID:
                            title = f" {ENV} ({STRING})"
                        else:
                            title = f"{ENV} ({STRING})"
                    else:
                        if FIVE_GRID:
                            title = f" {ENV} ({STRING})"
                        else:
                            title = f"{ENV} ({STRING})"

                    #title = f"{ENV}: Performance on held out Simulator transfer averaged over all methods " + ("(raw)" if not DEDUP else "(dedup)")
                elif KEY == STATICS.KEY_NUM_TRAIN_ENVS:
                    assert FIVE_GRID == False
                    if DEDUP:
                        title = f"{ENV}: Performance {STRING} on held out simulators (test set)"
                    else:
                        title = f"{ENV}: Performance {STRING} on all simulators (train & test set)"

                    #title = f"{ENV}: Aggregated simulator transfer averaged over all methods " + ("(raw)" if not DEDUP else "(dedup)")

                do_heatmapp(do_this_one, title=title, vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
        else:
            for method in tqdm(avg_runs[STATICS.KEY_METHOD].unique(), desc=f"Heatmaps KEY={KEY}, DEDUP={DEDUP}, AGG={AGG_METHOD}"):
                do_this_one = avg_runs[avg_runs[STATICS.KEY_METHOD] == method]
                do_this_one = do_this_one.drop(STATICS.KEY_METHOD, axis=1)
                if KEY == STATICS.KEY_TRAIN_ENVS:
                    if DEDUP:
                        title = f"{ENV}: ({method}) Performance per simulator on held out simulators (test set)"
                    else:
                        title = f"{ENV}: ({method}) Performance per simulator  on all simulators (train & test set)"

                    #title = f"{ENV}: {method}'s simulator transfer " + ("(raw)" if not DEDUP else "(dedup)")
                elif KEY == STATICS.KEY_NUM_TRAIN_ENVS:
                    if DEDUP:
                        title = f"{ENV}: ({method}) Performance on held out simulators (test set)"
                    else:
                        title = f"{ENV}: ({method}) Performance on all simulators (train & test set)"

                    #title = f"{ENV}: {method}'s aggregated simulator transfer " + ("(raw)" if not DEDUP else "(dedup)")
                do_heatmapp(do_this_one, title=title, vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)

    avg_runs = avg_runs[[KEY, STATICS.KEY_METHOD, *get_totrew_cols(avg_runs)]]
    return cleanup_evalenvs(avg_runs)

def make_manyeval_linechart(DIR, DEDUP):
    avg_runs = make_heatmap(DIR, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=DEDUP)

    avg_runs = avg_runs[[STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD, *get_totrew_cols(avg_runs)]]

    avg_runs = avg_runs.set_index(STATICS.KEY_NUM_TRAIN_ENVS)
    avg_runs = rename_rew_cols(avg_runs)

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

    avg_runs = rename_rew_cols(avg_runs)
    #avg_runs = avg_runs.set_index(STATICS.KEY_NUM_TRAIN_ENVS).mean(axis=1)

    fig, ax = plt.subplots()
    seaborn.lineplot(avg_runs, ax=ax)

    ax.set_title("Transfer prowess averaged over all simulators "+ ("(raw)" if not DEDUP else "(dedup)"))
    ax.set_xlabel(f"Number of training simulators" )
    ax.set_ylabel("Average total evaluation performance")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


def make_linechart_per_sim(DIR,ENV, DEDUP, DO_LEGEND):
    avg_runs = make_heatmap(DIR, ENV, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=DEDUP, do_plots=False)

   # avg_runs = avg_runs[[STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD, *get_totrew_cols(avg_runs)]]



    avg_runs = filter_df(avg_runs)

    avg_runs = pandas.melt(avg_runs, id_vars=[STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD], value_vars=set(avg_runs.columns) - set([STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD]), var_name='Original_Column', value_name="NEW_COL")

    DONES = set()
    for method in avg_runs["METHOD"].unique():
        if method in DONES:
            continue

        method = method.split(";")[-1].strip().split("(")[0].strip()
        DONES.add(method)
        do_this_one = avg_runs[avg_runs["METHOD"].str.contains(method)]

        WITH_DR = do_this_one[avg_runs["METHOD"].str.contains('WITH DR')]
        WITHOUT_DR = do_this_one[avg_runs["METHOD"].str.contains('WITHOUT DR')]

        pal = vapeplot.palette("vaporwave")
        pal = [pal[0], pal[len(pal) // 4], pal[3 * (len(pal) //4)], pal[-1]]

        mallsoft = vapeplot.palette("mallsoft")


        fig, ax = plt.subplots()

        pal = {
            "(WITHOUT DR); Vector Framestack (PPO)": pal[0],
            "Vector Framestack (PPO)": pal[0],
            "(WITHOUT DR); RMA (PPO)": pal[1],
            "RMA (PPO)": pal[1],
            "(WITHOUT DR); Matrix Framestack (PPO)": pal[2],
            "Matrix Framestack (PPO)": pal[2],
            "(WITHOUT DR); PPO": pal[3],
            "PPO": pal[3],
            "BP ": mallsoft[0],
            "BS ": mallsoft[1],
            "BM ": mallsoft[2],
            "BG ": mallsoft[3],
            "M  ": mallsoft[5]

        }

        seaborn.lineplot(WITHOUT_DR,
                         x=STATICS.KEY_NUM_TRAIN_ENVS,
                         y="NEW_COL",
                         ax=ax,
                         hue="Original_Column",
                         palette=pal,#vapeplot.palette("avanti"),
                         #linestyle="dashed",
                         style=True,
                         dashes=[(2,10)],
                         legend=False
                         )

        #WITH_DR["METHOD"] = WITH_DR["METHOD"].apply(lambda m: m.split("; ")[-1])
        full_line_lineplot = seaborn.lineplot(WITH_DR,
                         x=STATICS.KEY_NUM_TRAIN_ENVS,
                         y="NEW_COL",
                         ax=ax,
                         hue="Original_Column",
                         palette=pal,#vapeplot.palette("avanti"),
                         #linestyle="dashed"
                        legend=DO_LEGEND
                         )
        #full_line_lineplot.legend(
        #    list(set(list(map(lambda m: m.split("); ")[-1], WITH_DR.METHOD.tolist()))))
        #)
        #full_line_lineplot.get_legend_handles_labels()

        if DO_LEGEND:
            # Get existing handles and labels from the legend
            handles, labels = ax.get_legend_handles_labels()

            # Create custom legend entries
            custom_lines = [
                plt.Line2D([0], [0], color='black', lw=2, linestyle='-', label='With DR'),
                plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='Without DR')
            ]

            # Add custom entries to handles and labels
            handles.extend(custom_lines)
            labels.extend(['With DR', 'Without DR'])

            # Update legend with new handles and labels
            ax.legend(handles, labels)


        #handles = full_line_lineplot.get_legend_handles_labels()
        #new_handles = [handles[0][4:], handles[1][4:]]

        #if DEDUP and not ONLY_TRAIN:
        #    title = f"{ENV}: Performance on held out simulators (test set)"
        #elif DEDUP and ONLY_TRAIN:
        #    title = f"{ENV}: Performance on training simulators (train set)"
        #else:
        #    title = f"{ENV}: Performance on all simulators (train & test set)"


        title = f"{ENV}: sim+sim for {method}"
        ax.set_title(title)
        ax.set_xlabel(f"Number of training simulators" )
        ax.set_ylabel("Average total evaluation reward")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())

        #labels = list(set(list(map(lambda m: m.split("); ")[-1], WITH_DR.METHOD.tolist()))))
        #plt.legend(title='Method', loc='upper left', labels=labels, handles=new_handles[0])

        os.makedirs(f"saved_plots/{ENV}/line_per_sim", exist_ok=True)
        plt.savefig(f"saved_plots/{ENV}/line_per_sim/{title}_legend{DO_LEGEND}.pdf", bbox_inches="tight")

        plt.close(fig) #plt.show()


def make_avgeval_allavg_linechart(DIR,ENV, DEDUP, ONLY_TRAIN, DO_LEGEND):
    avg_runs = make_heatmap(DIR, ENV,KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=DEDUP, do_plots=False, ONLY_TRAIN=ONLY_TRAIN)

    #avg_runs = avg_runs[[STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD, *get_totrew_cols(avg_runs)]]

    avg_runs = pandas.melt(avg_runs, id_vars=[STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD], value_vars=set(avg_runs.columns) - set([STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD]), var_name='Original_Column', value_name="NEW_COL")
    avg_runs = avg_runs.drop('Original_Column', axis=1)


    avg_runs = filter_df(avg_runs)


    #avg_runs = avg_runs.groupby([STATICS.KEY_NUM_TRAIN_ENVS, STATICS.KEY_METHOD]).mean()
    #avg_runs = avg_runs.mean(axis=1).reset_index()
    #avg_runs = avg_runs.set_index(STATICS.KEY_NUM_TRAIN_ENVS).mean(axis=1)

    #WITH_DR = avg_runs[["WITH DR" in avg_runs.METHOD]]
    WITH_DR = avg_runs[avg_runs["METHOD"].str.contains('WITH DR')]
    WITHOUT_DR = avg_runs[avg_runs["METHOD"].str.contains('WITHOUT DR')]

    pal = vapeplot.palette("vaporwave")
    pal = [pal[0], pal[len(pal) // 4], pal[3 * (len(pal) //4)], pal[-1]]

    fig, ax = plt.subplots()

    pal = {
        "(WITHOUT DR); Vector Framestack (PPO)": pal[0],
        "Vector Framestack (PPO)": pal[0],
        "(WITHOUT DR); RMA (PPO)": pal[1],
        "RMA (PPO)": pal[1],
        "(WITHOUT DR); Matrix Framestack (PPO)": pal[2],
        "Matrix Framestack (PPO)": pal[2],
        "(WITHOUT DR); PPO": pal[3],
        "PPO": pal[3],
    }

    seaborn.lineplot(WITHOUT_DR,
                     x=STATICS.KEY_NUM_TRAIN_ENVS,
                     y="NEW_COL",
                     ax=ax,
                     hue=STATICS.KEY_METHOD,
                     palette=pal,#vapeplot.palette("avanti"),
                     # linestyle="dashed",
                     style=True,
                     dashes=[(2, 10)],
                     legend=False
                     )

    WITH_DR["METHOD"] = WITH_DR["METHOD"].apply(lambda m: m.split("; ")[-1])
    full_line_lineplot = seaborn.lineplot(WITH_DR,
                     x=STATICS.KEY_NUM_TRAIN_ENVS,
                     y="NEW_COL",
                     ax=ax,
                     hue=STATICS.KEY_METHOD,
                     palette=pal,#vapeplot.palette("avanti"),
                     #linestyle="dashed"
                    legend=DO_LEGEND
                     )
    #full_line_lineplot.legend(
    #    list(set(list(map(lambda m: m.split("); ")[-1], WITH_DR.METHOD.tolist()))))
    #)
    #full_line_lineplot.get_legend_handles_labels()

    if DO_LEGEND:
        # Get existing handles and labels from the legend
        handles, labels = ax.get_legend_handles_labels()

        # Create custom legend entries
        custom_lines = [
            plt.Line2D([0], [0], color='black', lw=2, linestyle='-', label='With DR'),
            plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='Without DR')
        ]

        # Add custom entries to handles and labels
        handles.extend(custom_lines)
        labels.extend(['With DR', 'Without DR'])

        # Update legend with new handles and labels
        ax.legend(handles, labels)


    #handles = full_line_lineplot.get_legend_handles_labels()
    #new_handles = [handles[0][4:], handles[1][4:]]

    if DEDUP and not ONLY_TRAIN:
        title = f"{ENV}: Test set"
    elif DEDUP and ONLY_TRAIN:
        title = f"{ENV}: Train set"
    else:
        title = f"{ENV}: Train & Test set"


    #title = f"{ENV}: Aggregated simulator transfer "+ ("(raw)" if not DEDUP else "(dedup)")
    ax.set_title(title)
    ax.set_xlabel(f"Number of training simulators" )
    ax.set_ylabel("Average total evaluation reward")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())

    #labels = list(set(list(map(lambda m: m.split("); ")[-1], WITH_DR.METHOD.tolist()))))
    #plt.legend(title='Method', loc='upper left', labels=labels, handles=new_handles[0])

    os.makedirs(f"saved_plots/{ENV}", exist_ok=True)
    plt.savefig(f"saved_plots/{ENV}/{title}_legend{DO_LEGEND}.pdf", bbox_inches="tight")

    plt.close(fig) #plt.show()



if __name__ == "__main__":
    DIR = "./runs"
    ENV = "ant"

    prep_env_checkpoints(DIR)
    exit()

    """
    DID_ONE_LEGEND = False
    
    ant = make_heatmap(DIR, "ant", KEY=STATICS.KEY_TRAIN_ENVS, DEDUP=False, AGG_METHOD=True, do_plots=False)
    walker = make_heatmap(DIR, "walker2d", KEY=STATICS.KEY_TRAIN_ENVS, DEDUP=False, AGG_METHOD=True, do_plots=False)

   # ant = ant.drop(columns="METHOD", axis=1)
   #walker = walker.drop(columns="METHOD", axis=1)

    ant_vector = ant[ant.METHOD.str.contains("Vector")].drop(columns="METHOD", axis=1).groupby("TRAIN_ENVS").mean()
    walker_vector = walker[walker.METHOD.str.contains("Vector")].drop(columns="METHOD", axis=1).groupby("TRAIN_ENVS").mean()

    ant_matrix = ant[ant.METHOD.str.contains("Matrix")].drop(columns="METHOD", axis=1).groupby("TRAIN_ENVS").mean()
    walker_matrix = walker[walker.METHOD.str.contains("Matrix")].drop(columns="METHOD", axis=1).groupby("TRAIN_ENVS").mean()

    ant_rma = ant[ant.METHOD.str.contains("RMA")].drop(columns="METHOD", axis=1).groupby("TRAIN_ENVS").mean()
    walker_rma = walker[walker.METHOD.str.contains("RMA")].drop(columns="METHOD", axis=1).groupby("TRAIN_ENVS").mean()

    ant_ppo = ant[ant.METHOD.str.contains("\; PPO")].drop(columns="METHOD", axis=1).groupby("TRAIN_ENVS").mean()
    walker_ppo = walker[walker.METHOD.str.contains("\; PPO")].drop(columns="METHOD", axis=1).groupby("TRAIN_ENVS").mean()

    MISSING_VECTOR = set(ant_vector.index) - set(walker_vector.index)

    MISSING_MATRIX = set(ant_matrix.index) - set(walker_matrix.index)

    MISSING_PPO = set(ant_ppo.index) - set(walker_ppo.index)

    MISSING_RMA = set(ant_rma.index) - set(walker_rma.index)
"""



    DO_ENVS = ["hopper", "widow"] #, "ant", "walker", "reacher"]
    #DO_ENVS = ["reacher"]

    for ENV in ENVNAMES:
        FOUND = False
        for DO_IT in DO_ENVS:
            if DO_IT in ENV:
                FOUND = True
                break
        if not FOUND:
            continue

        df = get_concatdf_for_env(DIR, ENV)
        if df is False:
            continue



        #make_avgeval_allavg_linechart(DIR, ENV, DEDUP=False, ONLY_TRAIN=False)

       # exit()

        font_size = 20
        params = {
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "axes.titlesize": font_size,
            "text.usetex": True,
            "figure.figsize": [10, 10],
        }

        matplotlib.rcParams.update(params)


        #make_heatmap(DIR, ENV, KEY=STATICS.KEY_TRAIN_ENVS, DEDUP=False, AGG_METHOD=True)
        make_heatmap(DIR, ENV, KEY=STATICS.KEY_TRAIN_ENVS, DEDUP=False, AGG_METHOD=True, FIVE_GRID=True)
        #make_heatmap(DIR, ENV, KEY=STATICS.KEY_TRAIN_ENVS, DEDUP=True, AGG_METHOD=True)
        #make_heatmap(DIR, ENV, KEY=STATICS.KEY_TRAIN_ENVS, DEDUP=False)
        #make_heatmap(DIR,ENV, KEY=STATICS.KEY_TRAIN_ENVS, DEDUP=True)



        #make_heatmap(DIR,ENV, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=False)
        #make_heatmap(DIR, ENV, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=True)


        #make_heatmap(DIR, ENV, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=False, AGG_METHOD=True)
        #make_heatmap(DIR, ENV, KEY=STATICS.KEY_NUM_TRAIN_ENVS, DEDUP=True, AGG_METHOD=True)

        #make_manyeval_linechart(DIR, DEDUP=False)
        #make_manyeval_linechart(DIR, DEDUP=True)

        #make_avgeval_allavg_linechart(DIR,ENV, DEDUP=False)
        #make_avgeval_allavg_linechart(DIR, ENV, DEDUP=True)

       # make_avgeval_linechart(DIR, DEDUP=False)
        #make_avgeval_linechart(DIR, DEDUP=True)

        font_size = 25
        params = {
            "axes.labelsize": font_size,
            "legend.fontsize": 15,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "axes.titlesize": font_size,
            "text.usetex": True,
            "figure.figsize": [5, 5],
        }

        matplotlib.rcParams.update(params)


        make_linechart_per_sim(DIR, ENV, DEDUP=True, DO_LEGEND=True)
        make_linechart_per_sim(DIR, ENV, DEDUP=True, DO_LEGEND=False)

        make_avgeval_allavg_linechart(DIR, ENV, DEDUP=True, ONLY_TRAIN=False, DO_LEGEND=True)
        make_avgeval_allavg_linechart(DIR, ENV, DEDUP=True, ONLY_TRAIN=True, DO_LEGEND=True)
        make_avgeval_allavg_linechart(DIR, ENV, DEDUP=True, ONLY_TRAIN=False, DO_LEGEND=False)
        make_avgeval_allavg_linechart(DIR, ENV, DEDUP=True, ONLY_TRAIN=True, DO_LEGEND=False)

        # exit()

    exit()
