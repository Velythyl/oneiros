import copy
import dataclasses
from typing import Union, List

import numpy as np
from omegaconf import omegaconf


def envkey_tags_multienv(multienv_cfg):
    def envkey2shortkey(envkey):
        envkey = envkey.split("-")[0]
        return {
            "braxpositional": "bpos",
            "braxmjx": "bmjx",
            "braxgeneralized": "bgen",
            "braxspring": "bspring",
            "mujoco": "mujoco"
        }[envkey]

    multienv_cfg = marshall_multienv_cfg(multienv_cfg)

    train_envs = envkey_multiplex(multienv_cfg.train)
    eval_envs = envkey_multiplex(multienv_cfg.eval)

    envname = train_envs[0].split("-")[1]

    train_envs = list(map(envkey2shortkey, train_envs))
    PREFIX_train_envs = list(map(lambda name: f"train-{name}", train_envs))
    eval_envs = list(map(envkey2shortkey, eval_envs))
    PREFIX_eval_envs = list(map(lambda name: f"eval-{name}", eval_envs))

    return PREFIX_train_envs + PREFIX_eval_envs + [f"eval num = {len(eval_envs)}", f"train num = {len(train_envs)}",
                                                   f"envname {envname}", f"TRAIN {'-'.join(train_envs)}"]
        #,
        #                                           f"EVAL {'-'.join(eval_envs)}"]


def envkey_runname_multienv(multienv_cfg):
    def envkey2shortkey(envkey):
        envkey = envkey.split("-")[0]
        return {
            "braxpositional": "bpos",
            "braxmjx": "bmjx",
            "braxgeneralized": "bgen",
            "braxspring": "bspring",
            "mujoco": "mujoco"
        }[envkey]

    multienv_cfg = marshall_multienv_cfg(multienv_cfg)

    train_envs = envkey_multiplex(multienv_cfg.train)
    eval_envs = envkey_multiplex(multienv_cfg.eval)

    envname = train_envs[0].split("-")[1]

    train_envs = "-".join(list(map(envkey2shortkey, train_envs)))
    eval_envs = "-".join(list(map(envkey2shortkey, eval_envs)))

    return f"{envname}_TRAIN_{train_envs}" #_EVAL_{eval_envs}"


def envkey_multiplex(multiplex_cfg):
    return multiplex_cfg.env_key


def num_multiplex(multiplex_cfg):
    return len(multiplex_cfg.env_key)


def keys_multiplex(mutiplex_cfg):
    return vars(mutiplex_cfg)["_content"].keys()


def slice_multiplex(multiplex_cfg, index):
    EVAL_CONFIG = copy.deepcopy(multiplex_cfg)

    for k in keys_multiplex(multiplex_cfg):
        EVAL_CONFIG[k] = multiplex_cfg[k][index]

    return EVAL_CONFIG


def monad_multiplex(multiplex_cfg):
    EVAL_CONFIG = copy.deepcopy(multiplex_cfg)
    for k in keys_multiplex(multiplex_cfg):
        EVAL_CONFIG[k] = [multiplex_cfg[k]]
    return EVAL_CONFIG


def splat_multiplex(multiplex_cfg):
    ret = []
    for i in range(num_multiplex(multiplex_cfg)):
        ret += [slice_multiplex(multiplex_cfg, i)]
    return ret

def make_powerset_cfgs(full_cfg):
    from itertools import chain, combinations

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    envs_to_powerset = marshall_multienv_cfg(full_cfg.multienv).train
    TOT_NUMBER_TRAIN_ENVS = sum(envs_to_powerset.num_env)

    assert len(envs_to_powerset.env_key) > 1

    pw_ind = powerset(list(range(len(envs_to_powerset.env_key))))
    pw_ind = filter(lambda x: len(x) > 0, pw_ind)
    pw_ind = list(pw_ind)
    FULL_PW_IND = copy.deepcopy(pw_ind)

    MIN_POWERSET_LEN = full_cfg.multienv.min_powerset_len
    MAX_POWERSET_LEN = full_cfg.multienv.max_powerset_len

    MIN_POWERSET_LEN = -np.inf if MIN_POWERSET_LEN == "-inf" or MIN_POWERSET_LEN == "None" else MIN_POWERSET_LEN
    MAX_POWERSET_LEN = np.inf if MAX_POWERSET_LEN == "inf" or MAX_POWERSET_LEN == "None" else MAX_POWERSET_LEN

    pw_ind = list(filter(lambda x: len(x) >= MIN_POWERSET_LEN, pw_ind))
    pw_ind = list(filter(lambda x: len(x) <= MAX_POWERSET_LEN, pw_ind))

    MAX_IND = list(sorted(pw_ind, key=lambda x: len(x)))[-1]

    def left_out(ind):
        ret = set(MAX_IND) - set(ind)
        return tuple(ret)

    def myround(x, base=10):
        return int(base * round(x / base))

    envs_to_powerset = vars(envs_to_powerset)["_content"]

    def collect_ind(ind):
        new_train = {}

        for key, val in envs_to_powerset.items():
            acc = []
            for i in ind:
                acc.append(val[i])
            new_train[key] = acc

        new_train["powerset_metadata"] = ind
        return new_train

    LIST_OF_CFGS = []
    for ind in pw_ind:
        OG_CFG = copy.deepcopy(vars(full_cfg)["_content"])

        new_train = collect_ind(ind)
        new_eval = collect_ind(MAX_IND)

        new_train["num_env"] = [myround(TOT_NUMBER_TRAIN_ENVS // len(ind)) for _ in new_train["num_env"]]
        OG_CFG["multienv"]["train"] = new_train
        OG_CFG["multienv"]["eval"] = gen_eval_gamut(new_eval) #new_eval

        OG_CFG["multienv"]["do_powerset_id"] = FULL_PW_IND.index(ind)

        new_cfg = omegaconf.OmegaConf.create(OG_CFG)
        LIST_OF_CFGS.append(new_cfg)
    return LIST_OF_CFGS

def gen_eval_gamut(eval_cfgs):
    final_eval_cfgs = []

    dr_do_on_N_step = "20:80"
    dr_do_on_reset = True
    dr_do_at_creation = False

    keys = None
    for cur_eval_cfg in splat_multiplex(omegaconf.OmegaConf.create(eval_cfgs)):
        keys = list(cur_eval_cfg.keys())

        GET_LOW = cur_eval_cfg["dr_percent_below"]
        GET_HIGH = cur_eval_cfg["dr_percent_above"]

        SET_LOW = 0.5
        SET_HIGH = 10.
        assert (GET_LOW == GET_HIGH == 1.0) or (GET_LOW == SET_LOW and GET_HIGH == SET_HIGH)

        SCALE_HIGH = 2
        SCALE_LOW = 0.01

        # DR is assumed to be (0.5, 10.)
        DR_PERCENTS = [(1., 1.), (SET_LOW, SET_HIGH)] # IN BOUND
        DR_PERCENTS += [(SET_HIGH, 2*SET_HIGH),  ((SCALE_HIGH*SET_HIGH + SET_HIGH) / 2, (SCALE_HIGH* SET_HIGH + SET_HIGH) / 2 + 0.1)]   # OUTSIDE UP BOUND
        DR_PERCENTS += [(SCALE_LOW * SET_LOW, SET_LOW), ((SET_LOW * SET_LOW+ SET_LOW) / 2, (SET_LOW * SET_LOW + SET_LOW) + 0.001)]   # OUTSIDE LOW BOUND

        for percents in DR_PERCENTS:
            COPY = copy.deepcopy(cur_eval_cfg)

            COPY["dr_do_on_N_step"] = dr_do_on_N_step if not (percents[0] == percents[1] == 1.0) else 0
            COPY["dr_do_on_reset"] = dr_do_on_reset if not (percents[0] == percents[1] == 1.0) else False
            COPY["dr_do_at_creation"] = dr_do_at_creation if not (percents[0] == percents[1] == 1.0) else False
            COPY["dr_percent_below"] = percents[0]
            COPY["dr_percent_above"] = percents[1]

            final_eval_cfgs += [COPY]

    # now merge 'em
    dico = {key: [] for key in keys}
    for key in keys:
        for final_eval_cfg in final_eval_cfgs:
            dico[key].append(final_eval_cfg[key])

    return dico


def marshall_multienv_cfg(multienv_cfg):
    multienv_cfg = copy.deepcopy(multienv_cfg)

    for key in ["num_env", "max_episode_length", "action_repeat", "framestack"]:
        if multienv_cfg[key] not in ["None", None]:
            multienv_cfg.train[key] = multienv_cfg[key]
            multienv_cfg.eval[key] = multienv_cfg[key]

    def coerce_traineval(train_or_eval_cfg):

        def coerce_possible_list(key):
            val = train_or_eval_cfg[key]

            if isinstance(val, str):
                if "," in val:
                    val = val.split(",")
                    val = list(map(lambda x: x.strip(), val))
                else:
                    val = [val]
            elif isinstance(val, int) or isinstance(val, float):
                val = [val]

            VALID = True
            for x in val:
                if isinstance(x, str) or isinstance(x, int) or isinstance(x, float):
                    continue
                else:
                    VALID = False
                    break

            if not VALID:
                raise ValueError(f"Input is wrong for key {key}")

            train_or_eval_cfg[key] = val

        counts = {}
        for key in vars(train_or_eval_cfg)["_content"].keys():
            coerce_possible_list(key)
            counts[key] = len(train_or_eval_cfg[key])

        if num_multiplex(train_or_eval_cfg) == 1:
            for key, c in counts.items():
                if c != 1:
                    raise ValueError("When there's only one env in multienv, all other params must also be of len 1")
        else:
            NUM_MULTIENV = num_multiplex(train_or_eval_cfg)
            assert NUM_MULTIENV > 1

            for key, count in counts.items():
                if key == "env_key":
                    continue

                if count == NUM_MULTIENV:
                    continue

                assert count == 1
                train_or_eval_cfg[key] = [*train_or_eval_cfg[key]] * NUM_MULTIENV

    coerce_traineval(multienv_cfg.train)
    coerce_traineval(multienv_cfg.eval)

    return multienv_cfg


def cfg_envkey_startswith(cfg, name):
    return cfg.env_key.startswith(name)


def build_dr_dataclass(cfg):
    DO_DR = True

    if cfg.dr_percent_below == cfg.dr_percent_above == 1.0:
        DO_DR = False

    if (not cfg.dr_do_on_reset) and (not cfg.dr_do_on_N_step) and (not cfg.dr_do_at_creation):
        DO_DR = False

    if isinstance(cfg.dr_do_on_N_step, str):
        do_on_N_step = tuple(list(map(int, map(lambda x: x.strip(), cfg.dr_do_on_N_step.split(":")))))
        assert len(do_on_N_step) == 2
    else:
        assert isinstance(cfg.dr_do_on_N_step, int)
        do_on_N_step = cfg.dr_do_on_N_step

    return DR_Config(
        DO_DR,
        cfg.dr_percent_below,
        cfg.dr_percent_above,
        cfg.dr_do_on_reset,
        do_on_N_step,
        cfg.dr_do_at_creation
    )


@dataclasses.dataclass
class DR_Config:
    DO_DR: bool
    percent_below: float
    percent_above: float
    do_on_reset: bool
    do_on_N_step: Union[List, int]
    do_at_creation: bool
