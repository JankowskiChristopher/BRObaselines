import logging

import gymnasium as gym

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GymnasiumWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

        logger.info(f"Created GymnasiumWrapper with action space: {self.action_space} "
                    f"of shape {self.action_space.shape}\n"
                    f"and observation space: {self.observation_space} "
                    f"of shape {self.observation_space.shape}")

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        return super().step(action)
