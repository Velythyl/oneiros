import torch
from gym import Wrapper

singleton = [None]

def update_nsteps(nsteps):
    if singleton[0] is not None:
        singleton[0].global_steps = nsteps

class FrameSave(Wrapper):
    def __init__(self, env, episode_length, device, rundir, num_frames):
        super().__init__(env)
        self.device = device
        self.rundir = rundir
        self.num_frames = num_frames    # num frames used in the framestack. This is just to pad the saved buffer on disk to simulate empty frames

        BUFFER_SIZE = episode_length # number of frames to save in memory before comitting to disk (saves on i/o)
        self.buffer = torch.ones((self.observation_space.shape[0], BUFFER_SIZE, self.observation_space.shape[1]), dtype=torch.float32, requires_grad=False, device=device) * -torch.inf
        self.buffer_index = torch.zeros((self.observation_space.shape[0],), dtype=torch.int64, requires_grad=False, device=device)
        self.n_buffers_saved = 0
        self.env_indices = torch.arange(self.observation_space.shape[0], requires_grad=False, dtype=torch.int64, device=device)

 #       if singleton[0] is not None:
#            raise Exception()
     #   singleton[0] = self
        self.global_steps = 0

    def reset_stacks(self, env_mask):
        if env_mask is None:
            env_mask = torch.ones(self.env.num_envs, dtype=torch.bool, requires_grad=False, device=self.device)

        else:
            env_mask = env_mask.bool()
            if torch.any(env_mask):  # at least one env has to have been reset
                buffer_to_save = self.buffer[env_mask]

                padding_shape = [*buffer_to_save.shape]
                padding_shape[1] = self.num_frames-1
                padding = torch.zeros(tuple(padding_shape), dtype=torch.float32, requires_grad=False, device=self.device)
                buffer_to_save = torch.concat((padding, buffer_to_save), dim=1)

                torch.save(buffer_to_save, f"{self.rundir}/buffer.pth") # todo
                self.n_buffers_saved += 1

        self.buffer[env_mask] = -torch.inf
        self.buffer_index[env_mask] = 0

    def reset(self, **kwargs):
        self.reset_stacks(None)
        return super(FrameSave, self).reset(**kwargs)

    def add_to_buffer(self, obs):
        self.buffer[self.env_indices, self.buffer_index] = obs
        self.buffer_index += 1

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self.reset_stacks(done)
        self.add_to_buffer(obs)

        return obs, rew, done, info

