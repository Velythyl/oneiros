
def rundir():
    import wandb
    RUN_DIR = "/".join(wandb.run.dir.split("/")[:-1])
    return RUN_DIR