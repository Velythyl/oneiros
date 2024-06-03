import os
import re
import shutil
import sys
import uuid
from io import StringIO

import hydra
import yaml


@hydra.main(version_base=None, config_path="/tmp/hydra_splat", config_name="spoofconf")
def main(cfg):
    pass


def spoof_config():
    import sys

    register_folders = []

    paths = []
    for arg in sys.argv:
        if "=" in arg:
            arg, default = arg.split("=")

            if "." in arg:
                argpath = arg.split(".")
            elif "@" in arg:
                register_folders.append((arg.split("@")[0], arg.split("@")[1], default))
                continue
            else:
                argpath = [arg]

            paths.append(argpath)

    # print(paths[0])

    def recurs(acc, path_tail):

        key = path_tail[0]

        if key not in acc:
            acc[key] = {}

        acc = acc[key]

        if len(path_tail) == 1:
            # only elem is the key we just did...
            return

        return recurs(acc, path_tail[1:])

    spoof_dict = {}
    for path in paths:
        recurs(spoof_dict, path)

    def dfs(dico):
        for key in list(dico.keys()):
            if len(dico[key]) == 0:
                dico[key] = None
            else:
                dfs(dico[key])

    dfs(spoof_dict)

    tmp_id = uuid.uuid4()
    spoof_dict["hydra"] = {"sweeper": {"max_batch_size": 16}, "sweep": {"dir": f"/tmp/{tmp_id}"},
                           "run": {"dir": f"/tmp/{tmp_id}"}}

    os.makedirs("/tmp/hydra_splat", exist_ok=True)
    for folder, key, default in register_folders:
        spoof_dict["defaults"] = [{f"{folder}@{key}": default.split(",")[0]}]

        os.makedirs(f"/tmp/hydra_splat/{folder}", exist_ok=True)

        for defaul in default.split(","):
            with open(f"/tmp/hydra_splat/{folder}/{defaul}.yaml", "w") as f:
                f.write("temp: None")

    with open('/tmp/hydra_splat/spoofconf.yaml', 'w') as outfile:
        yaml.dump(spoof_dict, outfile)

    return tmp_id


def parse_hydra_output(hydra_output):
    lines = hydra_output.split("\n")[1:]

    dico = {}
    for line in lines:
        if line is None or len(line) == 0:
            continue

        if re.compile(r"Launching \d+ jobs locally").search(line):
            continue

        line = line.split("#")[-1]
        key, args = line.split(" : ")
        dico[key] = args

    return dico


if __name__ == '__main__':

    last_argv = sys.argv[-1]
    if last_argv.startswith("KEY="):
        KEY = last_argv.split("KEY=")[-1]
        sys.argv = sys.argv[:-1]
        RUN_MODE = True
    else:
        RUN_MODE = False

    tmp_id = spoof_config()

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    main()
    sys.stdout = old_stdout
    hydra_output = mystdout.getvalue()

    hydra_output_multirun = parse_hydra_output(hydra_output)

    if RUN_MODE:
        selected_multirun = hydra_output_multirun[KEY]
        print(selected_multirun)
    else:
        print(len(hydra_output_multirun))

    DEBUG = True
    if DEBUG:
        print(hydra_output_multirun)

    try:
        shutil.rmtree(f"/tmp/{tmp_id}")
    except:
        pass
