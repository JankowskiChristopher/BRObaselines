from typing import Tuple

import gym
import numpy as np

from jaxrl.datasets.dataset import Batch, Batch2
import jax
import jax.numpy as jnp

from functools import partial
import os
import pickle


class ParallelReplayBuffer:
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int, num_seeds: int):
        self.observations = np.empty((num_seeds, capacity, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.actions = np.empty((num_seeds, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.masks = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.dones_float = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.next_observations = np.empty((num_seeds, capacity, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.n_parts = 4
        
        #self.obs_means, self.obs_vars = np.zeros((num_seeds, 1, observation_space.shape[-1])), np.ones((num_seeds, 1, observation_space.shape[-1]))
        #self.rew_means, self.rew_vars = np.zeros((num_seeds, 1)), np.ones((num_seeds, 1))
        
    def update_obs_stats(self):
        self.obs_means = self.observations[:,:self.size,:].mean(1, keepdims=True)
        self.obs_vars = self.observations[:,:self.size,:].var(1, keepdims=True).sqrt() + 1e-3

    def update_rew_stats(self):
        self.rew_means = self.rewards[:,:self.size].mean(1, keepdims=True)
        self.rew_vars = self.rewards[:,:self.size].var(1, keepdims=True).sqrt() + 1e-3
        
    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.dones_float[:, self.insert_index] = done_float
        self.next_observations[:, self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_parallel(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[:, indx],
                     actions=self.actions[:, indx],
                     rewards=self.rewards[:, indx],
                     masks=self.masks[:, indx],
                     dones=self.dones_float[:, indx],
                     next_observations=self.next_observations[:, indx])

    def sample_parallel_multibatch(self, batch_size: int, num_batches: int) -> Batch:
        indxs = np.random.randint(self.size, size=(num_batches, batch_size))
        return Batch(observations=self.observations[:, indxs],
                     actions=self.actions[:, indxs],
                     rewards=self.rewards[:, indxs],
                     masks=self.masks[:, indxs],
                     dones=self.dones_float[:, indxs],
                     next_observations=self.next_observations[:, indxs])
    
    def sample_state(self, batch_size: int) -> np.ndarray:
        indx = np.random.randint(self.size, size=batch_size)
        return self.observations[:, indx]

    def save(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[:, i*chunk_size : (i+1)*chunk_size],
                self.actions[:, i*chunk_size : (i+1)*chunk_size],
                self.rewards[:, i*chunk_size : (i+1)*chunk_size],
                self.masks[:, i*chunk_size : (i+1)*chunk_size],
                self.dones_float[:, i*chunk_size : (i+1)*chunk_size],
                self.next_observations[:, i*chunk_size : (i+1)*chunk_size]
            ]

            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))
        # Save also size and insert_index
        pickle.dump((self.size, self.insert_index), open(os.path.join(save_dir, 'buffer_info'), 'wb'))

    def load(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))

            self.observations[:, i*chunk_size : (i+1)*chunk_size], \
            self.actions[:, i*chunk_size : (i+1)*chunk_size], \
            self.rewards[:, i*chunk_size : (i+1)*chunk_size], \
            self.masks[:, i*chunk_size : (i+1)*chunk_size], \
            self.dones_float[:, i*chunk_size : (i+1)*chunk_size], \
            self.next_observations[:, i*chunk_size : (i+1)*chunk_size] = data_chunk
        self.size, self.insert_index = pickle.load(open(os.path.join(save_dir, 'buffer_info'), 'rb'))

class ParallelReplayBufferNStep:
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int, num_seeds: int):
        self.observations = np.empty((num_seeds, capacity, observation_space.shape[-1]),
                                     dtype=observation_space.dtype)
        self.actions = np.empty((num_seeds, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.masks = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.dones_float = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.next_observations = np.empty((num_seeds, capacity, observation_space.shape[-1]),
                                          dtype=observation_space.dtype)
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.n_parts = 4

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.dones_float[:, self.insert_index] = done_float
        self.next_observations[:, self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_parallel(self, batch_size: int, n_step: int = 1, discount: float = 0.99) -> Batch:
        indx = np.random.randint(self.size-n_step+1, size=batch_size)
        obs = self.observations[:, indx]
        actions = self.actions[:, indx]
        rewards = self.rewards[:, indx]
        n_step_mask = np.ones_like(rewards)
        truns = self.dones_float[:, indx]
        masks = self.masks[:, indx]
        discounts = np.ones_like(rewards)
        new_obs = self.next_observations[:, indx]
        if n_step > 1:
            for i in range(1, n_step):
                n_step_mask *= masks * (1 - truns)
                discounts += n_step_mask
                rewards += discount ** i * self.rewards[:, indx+i] * n_step_mask
                masks = self.masks[:, indx+i]
                terms = self.dones_float[:, indx+i]
                new_obs = np.expand_dims(n_step_mask, -1) * self.next_observations[:, indx+i] + (1 - np.expand_dims(n_step_mask, -1)) * new_obs
            masks = n_step_mask
        else:
            new_obs = self.next_observations[:, indx]   
        discounts = discount ** discounts
        return Batch(observations=obs,
                     actions=actions,
                     rewards=rewards,
                     masks=masks,
                     dones=terms,
                     next_observations=new_obs), discounts

    def sample_parallel_multibatch(self, batch_size: int, num_batches: int, n_step: int = 1, discount: float = 0.99) -> Batch:
        indxs = np.random.randint(self.size - n_step + 1, size=(num_batches, batch_size))
        obs = self.observations[:, indxs]
        actions = self.actions[:, indxs]
        rewards = self.rewards[:, indxs]
        n_step_mask = np.ones_like(rewards)
        terms = self.dones_float[:, indxs]
        masks = self.masks[:, indxs]
        discounts = np.ones_like(rewards)
        new_obs = self.next_observations[:, indxs]
        if n_step > 1:
            for i in range(1, n_step):
                n_step_mask *= masks * (1 - terms)
                discounts += n_step_mask
                rewards += discount ** i * self.rewards[:, indxs+i] * n_step_mask
                masks = self.masks[:, indxs+i]
                terms = self.dones_float[:, indxs+i]
                new_obs = np.expand_dims(n_step_mask, -1) * self.next_observations[:, indxs+i] + (1 - np.expand_dims(n_step_mask, -1)) * new_obs
            masks = n_step_mask
        else:
            new_obs = self.next_observations[:, indxs]   
        discounts = discount ** discounts
        return Batch(observations=obs,
                     actions=actions,
                     rewards=rewards,
                     masks=masks,
                     dones=terms,
                     next_observations=new_obs), discounts
    
    def sample_state(self, batch_size: int) -> np.ndarray:
        indx = np.random.randint(self.size, size=batch_size)
        return self.observations[:, indx]
