import logging
from typing import Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MANISKILL_TASKS = {
    'lift-cube': "LiftCube-v0",
    'pick-cube': "PickCube-v0",
    'stack-cube': "StackCube-v0",
    'pick-ycb': "PickSingleYCB-v0",
    'turn-faucet': "TurnFaucet-v0",
}


class Maniskill2Wrapper(gym.Wrapper):
    """
    Wrapper for ManiSkill2 environments.
    """

    def __init__(self, env, max_t: int = 200):
        super().__init__(env)
        self.timestep = 0
        self.max_t = max_t

        logger.info(f"Created Maniskill2 wrapper with action space: {self.action_space} "
                    f"of shape {self.action_space.shape}\n"
                    f"and observation space: {self.observation_space} "
                    f"of shape {self.observation_space.shape}")

    def reset(self, *args, **kwargs):
        self.timestep = 0
        obs, _ = self.env.reset(**kwargs)  # example on GitHub accepts seed here.
        return obs, {}

    def step(self, action):

        self.timestep += 1
        ob, reward, term, trun, info = self.env.step(action.copy())
        if self.timestep == self.max_t:
            trun = True
        else:
            trun = False

        return ob, reward, term, trun, info

    @staticmethod
    def get_camera_pos(env_name: str):
        if env_name in ["lift-cube", "pick-cube", "stack-cube"]:
            return "pd_ee_delta_pos"
        if env_name in ["pick-ycb", "turn-faucet"]:
            return "pd_ee_delta_pose"
        raise ValueError(f"Unknown ManiSkill2 task: {env_name}")
