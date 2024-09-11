import gymnasium as gym


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, max_t=200):
        super().__init__(env)

        self.max_t = max_t
        self.timestep = 0

    def reset(self, *args, **kwargs):
        self.timestep = 0
        obs, _ = self.env.reset(**kwargs)  # reset seeds
        return obs, {}

    def step(self, action):

        self.timestep += 1
        ob, reward, term, _, info = self.env.step(action.copy())  # copy is used in TD-MPC2 wrapper
        if self.timestep >= self.max_t:  # inequality is safer than equality
            trun = True
        else:
            trun = False

        return ob, reward, term, trun, info
