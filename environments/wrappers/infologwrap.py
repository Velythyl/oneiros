import gym


class InfoLogWrap(gym.Wrapper):
    def __init__(self, env, prefix):
        super().__init__(env)
        self.prefix = prefix

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        new_info = {}
        for key, value in info.items():
            new_info[f"{self.prefix}#{key}"] = value
        new_info[f"{self.prefix}#rew"] = rew
        return obs, rew, done, new_info
