# import copy
from functools import partial
from typing import Dict, Optional, Tuple, Union, no_type_check

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import (is_image_space,
                                                    maybe_transpose)
from stable_baselines3.common.utils import is_vectorized_observation


class BaseJaxPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["return_logprob"])
    def sample_action(actor_state, obervations, key, return_logprob=False):
        if hasattr(actor_state, "batch_stats"):
            dist = actor_state.apply_fn({"params": actor_state.params, "batch_stats": actor_state.batch_stats},
                                        obervations, train=False)
        else:
            dist = actor_state.apply_fn(actor_state.params, obervations)
        action = dist.sample(seed=key)

        if not return_logprob:
            return action
        else:
            return action, dist.log_prob(action)

    @staticmethod
    @partial(jax.jit, static_argnames=["return_logprob"])
    def select_action(actor_state, obervations, return_logprob=False):
        if hasattr(actor_state, "batch_stats"):
            dist = actor_state.apply_fn({"params": actor_state.params, "batch_stats": actor_state.batch_stats},
                                        obervations, train=False)
        else:
            dist = actor_state.apply_fn(actor_state.params, obervations)
        action = dist.mode()

        if not return_logprob:
            return action
        else:
            return action, dist.log_prob(action)

    @no_type_check
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # self.set_training_mode(False)

        observation, vectorized_env = self.prepare_obs(observation)

        actions = self._predict(observation, deterministic=deterministic)

        # Convert to numpy, and reshape to the original action shape
        actions = np.array(actions).reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Clip due to numerical instability
                actions = np.clip(actions, -1, 1)
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)  # type: ignore[call-overload]

        return actions, state

    def prepare_obs(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(self.observation_space, spaces.Dict)
            # Minimal dict support: flatten
            keys = list(self.observation_space.keys())
            vectorized_env = is_vectorized_observation(observation[keys[0]], self.observation_space[keys[0]])

            # Add batch dim and concatenate
            observation = np.concatenate(
                [observation[key].reshape(-1, *self.observation_space[key].shape) for key in keys],
                axis=1,
            )

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(self.observation_space, spaces.Dict):
            assert isinstance(observation, np.ndarray)
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))  # type: ignore[misc]

        assert isinstance(observation, np.ndarray)
        return observation, vectorized_env

    def set_training_mode(self, mode: bool) -> None:
        # self.actor.set_training_mode(mode)
        # self.critic.set_training_mode(mode)
        self.training = mode
