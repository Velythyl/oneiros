import gym


class RenderWrap(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def render(self, *args, **kwargs):
        return super(RenderWrap, self).render(mode='rgb_array')
