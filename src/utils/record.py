import torch
from tqdm import tqdm



def record(nsteps, video_envs, agent, RUN_DIR, NUM_STEPS):
    video_envs = video_envs

    if video_envs is None:
        return
    # reset env
    next_obs = video_envs.reset()
    frame_list = []

    # collect frames
    x = video_envs.render()[0]  # we're dealing with a MultiPlexEnv with 1 sub-env
    for _ in tqdm(range(0, NUM_STEPS)):
        with torch.no_grad():
            action = agent.get_action(next_obs)
        frame_list.append(x)
        next_obs, _, done, info = video_envs.step(action)
        x = video_envs.render()[0]

    FPS = 30
    import cv2
    out = cv2.VideoWriter(f'{RUN_DIR}/{nsteps}_{video_envs.prefix}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (x.shape[0], x.shape[1]))
    for frame in frame_list:
        out.write(frame)
    out.release()

    return None
