import copy

from omegaconf import omegaconf


def envkey_runname_multienv(multienv_cfg):
    multienv_cfg = marshall_multienv_cfg(multienv_cfg)

    train = "-".join(envkey_multiplex(multienv_cfg.train))
    eval = "-".join(envkey_multiplex(multienv_cfg.eval))

    return f"TRAIN_{train}_EVAL_{eval}"

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

    def left_out(ind):
        MAX_ITEM = list(sorted(pw_ind, key=lambda x: len(x)))[-1]
        ret = set(MAX_ITEM) - set(ind)
        return tuple(ret)

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
        OG_CFG = vars(full_cfg)["_content"]

        new_train = collect_ind(ind)
        new_eval = collect_ind(left_out(ind))

        new_train["num_env"] = [TOT_NUMBER_TRAIN_ENVS // len(ind) for _ in new_train["num_env"]]
        OG_CFG["multienv"]["train"] = new_train
        OG_CFG["multienv"]["eval"] = new_eval
        new_cfg = omegaconf.OmegaConf.create(OG_CFG)
        LIST_OF_CFGS.append(new_cfg)
    return LIST_OF_CFGS


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
            elif isinstance(val, int):
                val = [val]

            VALID = True
            for x in val:
                if isinstance(x, str) or isinstance(x, int):
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
