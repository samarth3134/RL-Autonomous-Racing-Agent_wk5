import gymnasium as gym


class RacingEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        super().__init__(gym.make("CarRacing-v3", render_mode=render_mode))