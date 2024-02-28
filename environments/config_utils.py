import copy



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
