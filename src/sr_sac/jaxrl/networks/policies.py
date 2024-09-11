import functools
from typing import Optional, Sequence, Tuple, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from jaxrl.networks.common import MLPClassic, MLPClassic2, MLP_LN, MLP_SN, MLP, Params, PRNGKey, default_init

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0

class DecoupledPolicy(nn.Module):
    hidden_dims: int
    action_dim: int
    scale_means: float = 0.01
    scale_stds: float = 0.01
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    model_version: int = 0
    depth: int = 1
    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 means: jnp.ndarray,
                 stds: jnp.ndarray,
                 std_multiplier: float,
                 return_params: bool = True) -> jnp.ndarray:
        
        inputs = jnp.concatenate([observations, means], -1)
        if self.model_version == 0:
            outputs = MLP_LN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(inputs)
        if self.model_version == 1:
            outputs = MLP_LN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(inputs)
        if self.model_version == 2:
            outputs = MLP_SN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(inputs)
        if self.model_version == 3:
            outputs = MLPClassic(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(inputs)
        if self.model_version == 4:
            outputs = MLPClassic2(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(inputs)
        action_shift = nn.Dense(self.action_dim, kernel_init=default_init(scale=self.scale_means), use_bias=False)(outputs)
        log_mult = nn.Dense(self.action_dim, kernel_init=default_init(scale=self.scale_stds), use_bias=False)(outputs)
        log_mult_min = -1.0
        log_mult_max = 1.0
        log_mult = log_mult_min + (log_mult_max - log_mult_min) * 0.5 * (1 + nn.tanh(log_mult))
        std_mult = jnp.exp(log_mult) * std_multiplier
        optimistic_means = means + action_shift
        optimistic_stds = stds * std_mult
        base_dist = tfd.MultivariateNormalDiag(loc=optimistic_means, scale_diag=optimistic_stds)
        if return_params is False:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        else:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh()), optimistic_means, optimistic_stds

class DecoupledPolicySimple(nn.Module):
    hidden_dims: int
    action_dim: int
    scale_means: float = 0.01
    scale_stds: float = 0.01
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    model_version: int = 0
    depth: int = 1
    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 means: jnp.ndarray,
                 stds: jnp.ndarray,
                 std_multiplier: float,
                 return_params: bool = True) -> jnp.ndarray:
        
        inputs = jnp.concatenate([observations, means], -1)
        if self.model_version == 0:
            outputs = MLP_LN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(inputs)
        if self.model_version == 1:
            outputs = MLP_LN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(inputs)
        if self.model_version == 2:
            outputs = MLP_SN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(inputs)
        if self.model_version == 3:
            outputs = MLPClassic(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(inputs)
        if self.model_version == 4:
            outputs = MLPClassic2(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(inputs)
        action_shift = nn.Dense(self.action_dim, kernel_init=default_init(scale=self.scale_means), use_bias=False)(outputs)
        optimistic_means = means + action_shift
        mult = jnp.ones(1) * std_multiplier
        optimistic_stds = stds * mult
        base_dist = tfd.MultivariateNormalDiag(loc=optimistic_means, scale_diag=optimistic_stds)
        if return_params is False:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        else:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh()), optimistic_means, optimistic_stds

class MSEPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray, temperature: float = 1.0, training: bool = False) -> jnp.ndarray:
        outputs = MLP(self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate)(
            observations, training=training
        )

        actions = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        return nn.tanh(actions)


class NormalTanhPolicy(nn.Module):
    hidden_dims: int
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    model_version: int = 0
    depth: int = 1

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0, training: bool = False, return_params: bool = False
    ) -> tfd.Distribution:
        if self.model_version == 0:
            outputs = MLP_LN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(observations)
        if self.model_version == 1:
            outputs = MLP_LN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(observations)
        if self.model_version == 2:
            outputs = MLP_SN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(observations)
        if self.model_version == 3:
            outputs = MLPClassic(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(observations)
        if self.model_version == 4:
            outputs = MLPClassic2(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(observations)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim, kernel_init=default_init(self.log_std_scale))(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        # log_stds = jnp.clip(log_stds, log_std_min, log_std_max)
        # suggested by Ilya for stability
        log_stds = log_std_min + (log_std_max - log_std_min) * 0.5 * (1 + nn.tanh(log_stds))

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        stds = jnp.exp(log_stds)
        stds = stds * temperature
        base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=stds)
        if self.tanh_squash_distribution:
            if return_params is False:
                return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
            else:
                return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh()), means, stds
        else:
            return base_dist

class NormalTanhPolicyMix(nn.Module):
    hidden_dims: int
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    model_version: int = 0
    depth: int = 1
    
    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False,
                 return_params: bool = False,
                 mix_parameter: float = 0.8) -> tfd.Distribution:
        
        if self.model_version == 0:
            outputs = MLP_LN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(observations)
        if self.model_version == 1:
            outputs = MLP_LN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(observations)
        if self.model_version == 2:
            outputs = MLP_SN(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(observations)
        if self.model_version == 3:
            outputs = MLPClassic(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(observations)
        if self.model_version == 4:
            outputs = MLPClassic2(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=256, categorical=False)(observations)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(self.log_std_scale))(outputs)
        log_stds_independent = self.param('log_stds', nn.initializers.zeros, (self.action_dim, ))
        
        shape_log_stds = log_stds.shape
        shape_log_stds_independent = log_stds_independent.shape
        
        if len(shape_log_stds) > len(shape_log_stds_independent):
            log_stds_independent = jnp.expand_dims(log_stds_independent, 0)
            shape_log_stds_independent = log_stds_independent.shape
            if len(shape_log_stds) > len(shape_log_stds_independent):
                log_stds_independent = jnp.expand_dims(log_stds_independent, 0)
                shape_log_stds_independent = log_stds_independent.shape
                
        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        #log_stds = jnp.clip(log_stds, log_std_min, log_std_max)
        # suggested by Ilya for stability
        log_stds = log_std_min + (log_std_max - log_std_min) * 0.5 * (1 + nn.tanh(log_stds))
        log_stds_independent = log_std_min + (log_std_max - log_std_min) * 0.5 * (1 + nn.tanh(log_stds_independent))

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)
            
        stds = jnp.exp(log_stds)
        stds_independent = jnp.exp(log_stds_independent)
        stds = (1.0 - mix_parameter) * stds + mix_parameter * stds_independent
        stds = stds * temperature
        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=stds)
        if self.tanh_squash_distribution:
            if return_params is False:
                return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
            else:
                return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh()), means, stds
        else:
            return base_dist

@functools.partial(jax.jit, static_argnames=("actor_def", "distribution"))
@functools.partial(jax.vmap, in_axes=(0, None, 0, 0, None, None))
def _sample_activations(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
) -> Tuple[jnp.ndarray]:
    if distribution == "det":
        return rng, actor_def.apply({"params": actor_params}, observations, temperature)
    else:
        filter_rep = lambda l, _: l.name is not None and 'act' in l.name
        dist, feat = actor_def.apply(
            {"params": actor_params},
            observations,
            temperature,
            mutable=['intermediates'],
            capture_intermediates=filter_rep,
        )
        return feat['intermediates']

def sample_activations(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
) -> Tuple[jnp.ndarray]:
    return _sample_activations(rng, actor_def, actor_params, observations, temperature, distribution)

@functools.partial(jax.jit, static_argnames=("actor_def", "distribution", "return_logprob"))
@functools.partial(jax.vmap, in_axes=(0, None, 0, 0, None, None, None))
def _sample_actions(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
    return_logprob: bool = False,
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({"params": actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    actions = dist.sample(seed=key)
    if return_logprob == True:
        logprobs = dist.log_prob(actions)
        return rng, actions, logprobs
    else:
        return rng, actions


def sample_actions(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
    return_logprob: bool = False,
) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_def, actor_params, observations, temperature, distribution, return_logprob)

@functools.partial(jax.jit, static_argnames=('actor_o_def', 'actor_c_def'))
@functools.partial(jax.vmap, in_axes=(0, None, 0, None, 0, 0, None, None))
def _sample_actions_o(
        rng: PRNGKey,
        actor_o_def: nn.Module,
        actor_o_params: Params,
        actor_c_def: nn.Module,
        actor_c_params: Params,
        observations: np.ndarray,
        std_multiplier: float,
        temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    _, mu_c, std_c = actor_c_def.apply({'params': actor_c_params}, observations=observations, temperature=temperature, return_params=True)
    dist, mu_o, std_o = actor_o_def.apply({'params': actor_o_params}, observations=observations, means=mu_c, stds=std_c, std_multiplier=std_multiplier)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)

def sample_actions_o(
        rng: PRNGKey,
        actor_o_def: nn.Module,
        actor_o_params: Params,
        actor_c_def: nn.Module,
        actor_c_params: Params,
        observations: np.ndarray,
        std_multiplier: float,
        temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions_o(rng, actor_o_def, actor_o_params, actor_c_def, actor_c_params, observations, std_multiplier, temperature)


@functools.partial(jax.jit, static_argnames=('actor_def'))
@functools.partial(jax.vmap, in_axes=(None, 0, 0))
def _get_distribution_parameters(
        actor_def: nn.Module,
        actor_params: Params,
        observations: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    _, mu, std = actor_def.apply({'params': actor_params}, observations,
                           temperature=1.0, return_params=True)
    return mu, std

def get_distribution_parameters(
        actor_def: nn.Module,
        actor_params: Params,
        observations: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return _get_distribution_parameters(actor_def, actor_params, observations)
