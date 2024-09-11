import logging
from enum import Enum
from typing import Optional, Tuple

import dm_env
import gymnasium as gym
import numpy as np
from dm_control import composer, suite
from dm_control.rl import control
from dm_control.suite.wrappers import action_scale

TimeStep = Tuple[np.ndarray, float, bool, dict]


class EnvType(Enum):
    """The environment type."""

    COMPOSER = 0
    RL_CONTROL = 1


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DMCWrapper(gym.Wrapper):
    _initialized = False  # Used to avoid logging multiple times
    float32_mode = False

    def __init__(self, domain_name: str, task_name: str, seed: int) -> None:
        # Do not call super() as it brakes.
        assert seed is not None, "Seed passed to constructor of DMCWrapper is None. Please pass a seed for training."
        env = suite.load(domain_name=domain_name,
                         task_name=task_name,
                         task_kwargs={"random": seed})
        self._env = action_scale.Wrapper(env, minimum=-1., maximum=1.)

        self._observation_space, self._action_space = self._create_obs_and_action_spaces()
        self._domain_name = domain_name
        self._task_name = task_name
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "multi_camera"],
            "render_fps": self._env.control_timestep() * 1000,
        }  # Used to avoid errors in CrossQ
        self.env_type = self._find_env_type(self._env)

        if not DMCWrapper._initialized:
            DMCWrapper._initialized = True
            logger.info(
                f"Created DMCWrapper with action space: {self._action_space} of shape {self._action_space.shape}\n"
                f"and observation space: {self._observation_space} with shape {self._observation_space.shape}")

    def set_float32_mode(self):
        """
        Some algorithms based on Stable Baselines3 prefer float32 mode in actions and observations.
        """
        logger.info("Setting DMCWrapper to float32 mode")
        DMCWrapper.float32_mode = True

    @property
    def env(self):
        return self._env

    def __getattr__(self, item):
        return getattr(self._env, item)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _create_obs_and_action_spaces(self):
        obs_shp = []
        for v in self._env.observation_spec().values():
            try:
                shp = np.prod(v.shape)
            except:
                shp = 1
            obs_shp.append(shp)
        obs_shp = (int(np.sum(obs_shp)),)
        act_shp = self._env.action_spec().shape
        observation_space = gym.spaces.Box(
            low=np.full(
                obs_shp,
                -np.inf,
                dtype=np.float32),
            high=np.full(
                obs_shp,
                np.inf,
                dtype=np.float32),
            dtype=np.float32,
        )
        action_space_dtype = self._env.action_spec().dtype
        action_space = gym.spaces.Box(
            low=np.full(act_shp, self._env.action_spec().minimum),
            high=np.full(act_shp, self._env.action_spec().maximum),
            dtype=action_space_dtype)

        return observation_space, action_space

    def _obs_to_array(self, obs):
        if DMCWrapper.float32_mode:
            return np.concatenate([v.flatten().astype(np.float32) for v in obs.values()])

        return np.concatenate([v.flatten() for v in obs.values()])

    def step(self, action: np.ndarray):
        assert self.action_space.contains(
            action), f"Action {action}, {action.shape} is not in action space {self.action_space}"

        time_step = self._env.step(action)
        reward = time_step.reward or 0
        term = time_step.last()
        trun = False
        obs = self._obs_to_array(time_step.observation)

        info = {}
        if term and time_step.discount == 1.0:
            info['TimeLimit.truncated'] = True
            trun = True

        self._check_observation_not_none(obs, action, "step")
        logger.debug(f"Returning obs: {obs}, reward: {reward}, term: {term}, trun: {trun}, info: {info}")

        return obs, reward, term, trun, info

    def reset(self, *args, **kwargs):
        time_step = self._env.reset()  # Reset here does not seed
        self.seed(kwargs.get('seed', None))  # We seed here. None as in some places CrossQ does not seed the env.
        obs = self._obs_to_array(time_step.observation)

        self._check_observation_not_none(obs, None, "reset")
        logger.debug(f"Returning obs: {obs}")
        return obs, {}

    @property
    def np_random(self) -> np.random.RandomState:
        """This should be np.random.Generator but dm-control uses np.random.RandomState."""
        if self.env_type is EnvType.RL_CONTROL:
            return self._env.task._random
        else:
            return self._env._random_state

    @np_random.setter
    def np_random(self, value: np.random.RandomState):
        if self.env_type is EnvType.RL_CONTROL:
            self._env.task._random = value
        else:
            self._env._random_state = value

    def seed(self, seed: Optional[int]):
        if seed is not None:
            self.np_random = np.random.RandomState(seed=seed)

    def _find_env_type(self, env) -> EnvType:
        """Tries to discover env types, in particular for environments with wrappers."""
        if isinstance(env, composer.Environment):
            return EnvType.COMPOSER
        elif isinstance(env, control.Environment):
            return EnvType.RL_CONTROL
        else:
            assert isinstance(env, dm_env.Environment)

            if hasattr(env, "_env"):
                return self._find_env_type(
                    env._env  # pyright: ignore[reportGeneralTypeIssues]
                )
            elif hasattr(env, "env"):
                return self._find_env_type(
                    env.env  # pyright: ignore[reportGeneralTypeIssues]
                )
            else:
                raise AttributeError(
                    f"Can't know the dm-control environment type, actual type: {type(env)}"
                )

    @property
    def unwrapped(self):
        return self

    def _check_observation_not_none(self, obs: np.ndarray, action: Optional[np.ndarray], function_name: str):
        if np.any(np.isnan(obs)):
            logger.info(f"Observation: {obs} contains nans in {function_name}.")
            if action is None:
                logger.info(f"action is None or was not passed.")
            else:
                logger.info(f"action: {action}")
            raise ValueError("Observation is None. This is not expected. Please check the environment.")
