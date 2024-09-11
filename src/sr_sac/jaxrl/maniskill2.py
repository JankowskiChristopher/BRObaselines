from typing import Optional, Tuple, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Maniskill2Env(gym.Env):
    """
    Wrapper for ManiSkill2 environments.
    Based on official Colab:
    https://colab.research.google.com/github/haosulab/ManiSkill2/blob/main/examples/tutorials/2_reinforcement_learning.ipynb#scrollTo=NNT4Fi6ZPTa4
    """

    def __init__(self, env_name: str, num_envs: int, seed: int, max_episode_steps: Optional[int] = None):
        logger.info(f"Creating {num_envs} ManiSkill2 environments of type {env_name} with num seeds {seed}.")

        np.random.seed(seed)
        self.num_seeds = num_envs
        self.timesteps = np.ones(num_envs)

        self.obs_mode = "state"
        self.control_mode = "pd_ee_delta_pose"
        self.reward_mode = "normalized_dense"  # this the default reward mode which is a dense reward scaled to [0, 1]

        seeds = np.random.randint(low=0, high=int(1e6), size=(self.num_seeds,))

        import mani_skill2.envs  # tutorial states that important when multiprocessing
        self.envs = [gym.make(env_name,
                              obs_mode=self.obs_mode,
                              reward_mode=self.reward_mode,
                              control_mode=self.control_mode,
                              max_episode_steps=max_episode_steps,
                              render_mode="cameras") for _ in seeds]
        # TODO idk for now
        self.action_space = spaces.Box(low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
                                       high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
                                       shape=(len(self.envs), self.envs[0].action_space.shape[0]),
                                       dtype=self.envs[0].action_space.dtype)
        self.observation_space = self.envs[0].observation_space

    def _reset_idx(self, idx: int) -> np.ndarray:
        obs, _ = self.envs[idx].reset()
        return obs

    def reset_where_done(self, observations: np.ndarray, terms: np.ndarray, truns: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        resets = np.zeros(terms.shape)
        for j, (term, trun) in enumerate(zip(terms, truns)):
            if term or trun:
                observations[j], terms[j], truns[j] = self._reset_idx(j), False, False
                resets[j] = 1
                self.timesteps[j] = 1
        return observations, terms, truns, resets

    def reset(self, seed: Optional[int] = None, options: dict[str, Any] | None = None) -> np.ndarray:
        obs = []
        for env in self.envs:
            ob, _ = env.reset()
            obs.append(ob)
        return np.stack(obs)

    def generate_masks(self, terms: np.ndarray, truns: np.ndarray) -> np.ndarray:
        masks = []
        for term, trun in zip(terms, truns):
            if not term or trun:
                mask = 1.0
            else:
                mask = 0.0
            masks.append(mask)
        masks = np.array(masks)
        return masks

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs, rews, terms, truns, goals = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, reward, term, trun, info = env.step(action)
            obs.append(ob)
            rews.append(reward)
            terms.append(term)
            truns.append(trun)
            goals.append(info['success'])
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns), np.stack(goals)
