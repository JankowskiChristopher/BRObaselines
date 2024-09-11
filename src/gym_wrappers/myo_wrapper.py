import logging

import gym
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MYOSUITE_TASKS = {
    'myo-test': 'myoElbowPose1D6MRandom-v0',
    'myo-reach': 'myoHandReachFixed-v0',
    'myo-reach-hard': 'myoHandReachRandom-v0',
    'myo-pose': 'myoHandPoseFixed-v0',
    'myo-pose-hard': 'myoHandPoseRandom-v0',
    'myo-obj-hold': 'myoHandObjHoldFixed-v0',
    'myo-obj-hold-hard': 'myoHandObjHoldRandom-v0',
    'myo-key-turn': 'myoHandKeyTurnFixed-v0',
    'myo-key-turn-hard': 'myoHandKeyTurnRandom-v0',
    'myo-pen-twirl': 'myoHandPenTwirlFixed-v0',
    'myo-pen-twirl-hard': 'myoHandPenTwirlRandom-v0',
}


class MyoSuiteWrapper(gym.Wrapper):
    _initialized = False

    def __init__(self, env: gym.Env, max_t: int = 100, seed: int = 0):
        super().__init__(env)
        self.seed(seed)
        self.env = env
        self.camera_id = 'hand_side_inter'
        self._max_t = max_t
        self._elapsed_steps = 0
        self._seed = seed

        if not MyoSuiteWrapper._initialized:
            MyoSuiteWrapper._initialized = True
            logger.info(f"Created MyoSuiteWrapper with action space: {self.action_space} "
                        f"of shape {self.action_space.shape}\n"
                        f"and observation space: {self.observation_space} "
                        f"of shape {self.observation_space.shape}")

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['success'] = info['solved']
        obs = obs.astype(np.float32)
        self._elapsed_steps += 1
        if self._elapsed_steps == self._max_t:
            trun = True
        else:
            trun = False
        return obs, reward, False, trun, info

    def reset(self, seed=None, **kwargs):
        # Seed is passed to trick CrossQ to not convert to Gym21 but Gym26 compatibility class.
        self._elapsed_steps = 0
        self._seed = seed
        self.seed(self._seed)
        return super().reset(), {}

    @property
    def unwrapped(self):
        return self

    @property
    def render_mode(self):
        # CrossQ compatibility. Not that good, but working.
        return None
