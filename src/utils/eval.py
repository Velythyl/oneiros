import torch
import wandb
from plotly.offline import download_plotlyjs  # noqa
from tqdm import tqdm

from src.algs.rma import RMAAgent


def evaluate(nsteps, eval_envs, agent, NUM_STEPS, DO_VIDEO):
    collected_infos = {}

    print(f"Evaluating in {eval_envs}")

    def render(env):
        if DO_VIDEO:
            frame = env.render()[0]
            return frame
        else:
            return None

    frame_list = []

    next_obs = eval_envs.reset()
    frame_list.append(render(eval_envs))

    for _ in tqdm(range(0, NUM_STEPS)): # todo this should match train_env's max ep len to some extent
        with torch.no_grad():
            if isinstance(agent, RMAAgent):
                action = agent.get_action(next_obs, eval_envs.get_priv())
            else:
                action = agent.get_action(next_obs)
        next_obs, _, next_done, info = eval_envs.step(action)

        for key, val in info.items():
            if key in collected_infos:
                collected_infos[key].append(val)
            else:
                collected_infos[key] = [val]

        frame_list.append(render(eval_envs))

    wandb_logs = {}

    for key in collected_infos.keys():
        if key.endswith("#rew"):
            collected_rews = collected_infos[key]
            collected_rews = torch.vstack(collected_rews)

            _key = key.split("#")[0]
            wandb_logs[f"EVAL_{_key}/avg_rew"] = torch.mean(collected_rews)
            wandb_logs[f"EVAL_{_key}/tot_rew"] = torch.sum(collected_rews)
            wandb_logs[f"EVAL_{_key}/ep_len"] = collected_rews.shape[0]

    assert eval_envs.ONEIROS_METADATA.prefix != "MULTIPLEX"

    if DO_VIDEO:
        FPS = 30
        import cv2
        SHAPE = frame_list[0].shape
        out = cv2.VideoWriter(f'{wandb.run.dir}/{nsteps}_{eval_envs.ONEIROS_METADATA.prefix}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (SHAPE[1], SHAPE[0]))
        for frame in frame_list:
            out.write(frame)
        out.release()

    return wandb_logs


if __name__ == "__main__":
    import brax
    from brax.envs.wrappers.torch import TorchWrapper

    from environments.customenv.braxcustom.widow_reacher import WidowReacher
    from environments.customenv.mujococustom.widow_reacher import WidowReacher

    env = brax.envs.create(env_name="widow", episode_length=1000, backend="mjx",
                           batch_size=2, no_vsys=False)


    env = DomainRandWrapper(env,
                            percent_below=0.5,
                            percent_above=2.0,
                            do_on_reset=False,
                            do_on_N_step=sample_num,
                            do_at_creation=False,
                            seed=2
                            )
    env = VectorGymWrapper(env, seed=2)
    env = WritePrivilegedInformationWrapper(env)
    env = TorchWrapper(env, device="cuda")

    import jax.numpy as jp

    env.reset()
    for i in range(1000):
        env.step(torch.ones(2, env.action_space.shape[-1]).to("cuda") * 0)
        print(i)

    exit()