import torch
from gym import Wrapper


class ReplayBufferSave(Wrapper):
    def __init__(self, env, buffer_length, device, rundir):
        super().__init__(env)
        self.device = device
        self.rundir = rundir

        buffer_length = int(buffer_length)
        BUFFER_SIZE = buffer_length
        self.buffer_length = buffer_length

        num_envs = self.observation_space.shape[0]
        def buf_factory(shape):
            return (
                    torch.ones(
                        (num_envs, BUFFER_SIZE, shape[-1]),
                        dtype=torch.float32, requires_grad=False, device=device) * -torch.inf
            )
        self.buffers = {
            "action": buf_factory(self.action_space.shape),
            "obs": buf_factory(self.observation_space.shape),
            "done": buf_factory((1,)),
            "rew": buf_factory((1,))
        }

        self.buffer_index = torch.zeros((self.observation_space.shape[0],), dtype=torch.int32, requires_grad=False, device=device)
        self.env_indices = torch.arange(self.observation_space.shape[0], requires_grad=False, dtype=torch.int32, device=device)
        self.n_buffers_saved = 0

    def maybe_flush(self):
        if (self.buffer_index >= self.buffer_length).all():  # at least one env has to have been reset

            buffers_on_cpu = {}
            for key, buf in self.buffers.items():
                assert not (buf == -torch.inf).any()
                buffers_on_cpu[key] = buf.cpu().detach()
                buf.fill_(-torch.inf)
            torch.save(buffers_on_cpu, f"{self.rundir}/rb_{self.n_buffers_saved}.pth")  # todo
            print("ReplayBufferSave saved a batch of buffers")
            self.n_buffers_saved += 1
            self.buffer_index.fill_(0)

    def reset(self, **kwargs):
        self.maybe_flush()
        obs = super(ReplayBufferSave, self).reset(**kwargs)
        self.last_obs = obs[:,:]
        return obs

    def add_to_buffer(self, **kwargs):
        for key, val in kwargs.items():
            buffer = self.buffers[key]

            if len(val.shape) == 1:
                val = val.unsqueeze(-1)

            buffer[self.env_indices, self.buffer_index] = val
        self.buffer_index += 1

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self.add_to_buffer(obs=self.last_obs, rew=rew, done=done, action=action)
        self.last_obs = obs
        self.maybe_flush()

        return obs, rew, done, info

