import torch
from tqdm import tqdm



def record(nsteps, video_envs, agent, RUN_DIR, NUM_STEPS):
    video_envs = video_envs

    if video_envs is None:
        return
    # reset env
    next_obs = video_envs.reset()
    frame_list = []

    def render():
        x = video_envs.render()
        return x

    # collect frames
    x = render()
    for _ in tqdm(range(0, NUM_STEPS)): # todo this should match the max ep len , something like (max_ep_len // 2 -1 )
        with torch.no_grad():
            action = agent.get_action(next_obs)
        frame_list.append(x)
        next_obs, _, done, info = video_envs.step(action)
        # print(done.any())
        x = video_envs.render()

    FPS = 30
    import cv2
    out = cv2.VideoWriter(f'{RUN_DIR}/{nsteps}_{video_envs.prefix}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (x.shape[0], x.shape[1]))
    for frame in frame_list:
        out.write(frame)
    out.release()

    return None
