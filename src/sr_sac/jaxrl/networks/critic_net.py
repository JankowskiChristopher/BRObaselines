"""Implementations of algorithms for continuous control."""

from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import MLP, MLP_LN, MLP_SN, MLP_SN2, MLPClassic, MLPClassic2


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1
    categorical: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLPClassic(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=True, output_nodes=self.output_nodes, categorical=self.categorical)(inputs)
        if self.output_nodes == 1:
            return jnp.squeeze(critic, -1)
        else:
            return critic
        
class Critic2(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1
    categorical: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLPClassic2(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=True, output_nodes=self.output_nodes, categorical=self.categorical)(inputs)
        if self.output_nodes == 1:
            return jnp.squeeze(critic, -1)
        else:
            return critic

class CriticLN(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1
    categorical: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP_LN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=True, output_nodes=self.output_nodes, categorical=self.categorical)(inputs)
        if self.output_nodes == 1:
            return jnp.squeeze(critic, -1)
        else:
            return critic
        
class CriticSN(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1
    categorical: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP_SN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=True, output_nodes=self.output_nodes, categorical=self.categorical)(inputs, training=training)
        if self.output_nodes == 1:
            return jnp.squeeze(critic, -1)
        else:
            return critic
        
class CriticSN2(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1
    categorical: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP_SN2(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=True, output_nodes=self.output_nodes, categorical=self.categorical)(inputs, training=training)
        if self.output_nodes == 1:
            return jnp.squeeze(critic, -1)
        else:
            return critic
        
class DoubleCritic(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1
    categorical: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes, categorical=self.categorical)(observations, actions)
        critic2 = Critic(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes, categorical=self.categorical)(observations, actions)
        return critic1, critic2
    
class DoubleCritic2(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1
    categorical: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic2(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes, categorical=self.categorical)(observations, actions)
        critic2 = Critic2(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes, categorical=self.categorical)(observations, actions)
        return critic1, critic2

class DoubleCriticLN(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1
    categorical: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = CriticLN(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes, categorical=self.categorical)(observations, actions)
        critic2 = CriticLN(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes, categorical=self.categorical)(observations, actions)
        return critic1, critic2


class DoubleCriticSN(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1
    categorical: bool = False

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = CriticSN(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes, categorical=self.categorical)(observations, actions, training=training)
        critic2 = CriticSN(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes, categorical=self.categorical)(observations, actions, training=training)
        return critic1, critic2

class DoubleCriticSN2(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1
    categorical: bool = False

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = CriticSN2(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes, categorical=self.categorical)(observations, actions, training=training)
        critic2 = CriticSN2(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes, categorical=self.categorical)(observations, actions, training=training)
        return critic1, critic2

class EnsembleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions):

        VmapCritic = nn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        qs = VmapCritic(self.hidden_dims, activations=self.activations)(states, actions)
        return qs