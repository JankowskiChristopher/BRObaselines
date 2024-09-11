"""Implementations of algorithms for continuous control."""
import copy
import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os

import flax.linen as nn
from jaxrl import weight_recyclers
from jaxrl.bandit import Bandit

from jaxrl.agents.dac_simple import temperature
from jaxrl.agents.dac_simple.actor import update as update_actor
from jaxrl.agents.dac_simple.actor import update_optimistic as update_actor_optimistic
from jaxrl.agents.dac_simple.actor import update_optimistic_quantile as update_actor_optimistic_quantile
from jaxrl.agents.dac_simple.actor import update_quantile as update_actor_quantile
from jaxrl.agents.dac_simple.actor import update_categorical as update_actor_categorical
from jaxrl.agents.dac_simple.critic import target_update, validate_batch, critic_activations
from jaxrl.agents.dac_simple.critic import update as update_critic
from jaxrl.agents.dac_simple.critic import update_quantile as update_critic_quantile
from jaxrl.agents.dac_simple.critic import update_categorical as update_critic_categorical
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey

@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None))
def _update(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model, temp: Model, pessimism: Model, 
    actor_o: Model, optimism: Model, regularizer: Model,
    batch: Batch, discount: float, tau: float, target_entropy: float, pessimism_value: jnp.ndarray, critic_regularization: int, network_regularization: int, use_sam: bool,
    distributional: bool, n_atoms: int, quantile_taus: jnp.ndarray, v_min: float, v_max: float, support: jnp.ndarray, soft_critic: bool, std_multiplier: float, action_dim: int, kl_target: float, pessimism_value_: float
):

    rng, key = jax.random.split(rng)
    if distributional == 2:
        new_critic, critic_info = update_critic_categorical(
            key,
            actor,
            critic,
            target_critic,
            temp,
            pessimism,
            batch,
            discount,
            pessimism_value,
            critic_regularization,
            network_regularization,
            soft_critic=soft_critic,
            use_sam=use_sam,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            support=support
        )
    if distributional == 1:
        new_critic, critic_info = update_critic_quantile(
            key,
            actor,
            critic,
            target_critic,
            temp,
            pessimism,
            batch,
            discount,
            pessimism_value,
            critic_regularization,
            network_regularization,
            soft_critic=soft_critic,
            use_sam=use_sam,
            n_atoms=n_atoms,
            taus=quantile_taus
        )
    if distributional == 0:
        new_critic, critic_info = update_critic(
            key,
            actor,
            critic,
            target_critic,
            temp,
            pessimism,
            batch,
            discount,
            pessimism_value,
            critic_regularization,
            network_regularization,
            soft_critic=soft_critic,
            use_sam=use_sam
        )
    new_target_critic = target_update(new_critic, target_critic, tau)
    if (critic_regularization == 0) or (critic_regularization == 1) or (critic_regularization == 2):
        new_pessimism = pessimism
        pessimism_info = {}
    if critic_regularization == 3:
        new_pessimism, pessimism_info = temperature.update_pessimism(pessimism, critic_info['errors'])
    rng, key = jax.random.split(rng)
    if distributional == 2:
        new_actor, actor_info = update_actor_categorical(key, actor, new_critic, temp, new_pessimism, batch, pessimism_value, critic_regularization, use_sam=use_sam, support=support)
    if distributional == 1:
        new_actor, actor_info = update_actor_quantile(key, actor, new_critic, temp, new_pessimism, batch, pessimism_value, critic_regularization, use_sam=use_sam)
        rng, key = jax.random.split(rng)
        new_actor_o, actor_o_info = update_actor_optimistic_quantile(key, new_actor, actor_o, new_critic, optimism, regularizer, batch, std_multiplier, action_dim, use_sam=use_sam)

    if distributional == 0:
        new_actor, actor_info = update_actor(key, actor, new_critic, temp, new_pessimism, batch, pessimism_value, critic_regularization, use_sam=use_sam)
        rng, key = jax.random.split(rng)
        new_actor_o, actor_o_info = update_actor_optimistic(key, new_actor, actor_o, new_critic, optimism, regularizer, batch, std_multiplier, action_dim, use_sam=use_sam)

    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'], target_entropy)
    new_optimism, optimism_info = temperature.update_optimism(optimism, actor_o_info['kl'], kl_target, pessimism_value_)
    new_regularizer, regularizer_info = temperature.update_regularizer(regularizer, actor_o_info['kl'], kl_target)
    
    return rng, new_actor, new_critic, new_target_critic, new_temp, new_pessimism, new_actor_o, new_optimism, new_regularizer, {
        **critic_info,
        **actor_info,
        **alpha_info,
        **pessimism_info,
        **actor_o_info,
        **optimism_info,
        **regularizer_info
    }

@functools.partial(
    jax.jit,
    static_argnames=(
        "discount",
        "tau",
        "target_entropy",
        "num_updates",
        "critic_regularization",
        "network_regularization",
        "use_sam",
        "distributional",
        "n_atoms", 
        "v_min",
        "v_max",
        "soft_critic",
        "std_multiplier",
        "action_dim",
        "kl_target",
        "pessimism_value_"
    ),
)
def _do_multiple_updates(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    pessimism: Model,
    actor_o: Model,
    optimism: Model,
    regularizer: Model,
    batches: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
    step,
    num_updates: int,
    pessimism_value: jnp.ndarray,
    critic_regularization: int,
    network_regularization: int,
    use_sam: bool,
    distributional: int,
    n_atoms: int,
    quantile_taus: jnp.ndarray, 
    v_min: float, 
    v_max: float, 
    support: jnp.ndarray,
    soft_critic: bool,
    std_multiplier: float,
    action_dim: int,
    kl_target: float,
    pessimism_value_: float
):
    def one_step(i, state):
        step, rng, actor, critic, target_critic, temp, pessimism, actor_o, optimism, regularizer, info = state
        step = step + 1
        new_rng, new_actor, new_critic, new_target_critic, new_temp, new_pessimism, new_actor_o, new_optimism, new_regularizer, info = _update(
            rng,
            actor,
            critic,
            target_critic,
            temp,
            pessimism,
            actor_o,
            optimism,
            regularizer,
            jax.tree_map(lambda x: jnp.take(x, i, axis=1), batches),
            discount,
            tau,
            target_entropy,
            pessimism_value,
            critic_regularization,
            network_regularization,
            use_sam,
            distributional,
            n_atoms,
            quantile_taus,
            v_min,
            v_max,
            support,
            soft_critic,
            std_multiplier,
            action_dim,
            kl_target,
            pessimism_value_
        )
        return step, new_rng, new_actor, new_critic, new_target_critic, new_temp, new_pessimism, new_actor_o, new_optimism, new_regularizer, info

    step, rng, actor, critic, target_critic, temp, pessimism, actor_o, optimism, regularizer, info = one_step(
        0, (step, rng, actor, critic, target_critic, temp, pessimism, actor_o, optimism, regularizer, {})
    )
    return jax.lax.fori_loop(1, num_updates, one_step, (step, rng, actor, critic, target_critic, temp, pessimism, actor_o, optimism, regularizer, info))

class DACSimpleLearner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        num_seeds: int = 5,
        critic_regularization: int = 0,
        network_regularization: int = 0,
        updates_per_step: int = 16,
        critic_depth: int = 1,
        critic_size: int = 256,
        actor_type: int = 1,
        actor_depth: int = 1,
        actor_size: int = 256,
        use_sam: bool = False,
        use_swish: bool = False,
        use_redo: bool = False,
        use_reset: bool = True,
        sn_type: int = 0,
        distributional: int = 0,
        n_atoms: int = 100,
        v_min_cat: float = 0.0,
        v_max_cat: float = 200,
        soft_critic: bool = True,
        kl_target: float = 0.1,
        std_multiplier: float = 0.5,
        init_optimism: float = 1.0,
        init_regularizer: float = 0.25
    ) -> None:
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        
        self.actor_type = actor_type
        self.critic_depth = critic_depth
        self.critic_size = critic_size
        self.use_sam = use_sam
        self.use_swish = use_swish
        self.use_redo = use_redo
        self.use_reset = use_reset
        self.sn_type = sn_type
        self.distributional = distributional
        self.n_atoms = n_atoms
        self.actor_type = actor_type
        self.soft_critic = soft_critic
        self.std_multiplier = std_multiplier
        action_dim = actions.shape[-1]
        self.action_dim = float(action_dim)
        self.actor_depth = actor_depth
        self.actor_size = actor_size
        self.kl_target = kl_target
        
        self.log_temp_min = -10.0
        self.log_temp_max = 7.5
        self.init_optimism = self.calculate_init_values(init_optimism, self.log_temp_min, self.log_temp_max)
        self.init_regularizer = self.calculate_init_values(init_regularizer, self.log_temp_min, self.log_temp_max)
        
        self.v_min = v_min_cat
        self.v_max = v_max_cat
        self.support = jnp.linspace(v_min_cat, v_max_cat, num=self.n_atoms)[None, ...]
        taus_ = jnp.arange(0, n_atoms+1) / n_atoms
        self.quantile_taus = ((taus_[1:] + taus_[:-1]) / 2.0)[None, ...]
        self.categorical = False
        if self.distributional == 2:
            self.categorical = True
        
        self.seeds = jnp.arange(seed, seed + num_seeds)

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.discount = discount

        reset_list = [15001, 50001]
        reset_step = (2500000 // 10 + 1)
        while reset_step < 1000000:
            reset_list.append(reset_step)
            reset_step += 2500000 // 10
        reset_list = reset_list + [1000001, 1500001]
        self.reset_list = reset_list
        
        self.num_seeds = num_seeds
        if critic_regularization == 0 or critic_regularization == 3:
            self.pessimism_value = jnp.zeros(num_seeds).reshape(num_seeds, 1)
            self.pessimism_value_ = 0.0
        if critic_regularization == 1:
            self.pessimism_value = jnp.ones(num_seeds).reshape(num_seeds, 1)
            self.pessimism_value_ = 1.0
        if critic_regularization == 2:
            self.controller = Bandit(num_seeds=num_seeds)
            pessimism_value = self.controller.sample()
            self.pessimism_value = jnp.array(pessimism_value).reshape(num_seeds, 1)

        self.critic_regularization = critic_regularization
        self.network_regularization = network_regularization

        if self.use_swish:
            activations = nn.swish
        else:
            activations = nn.relu

        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, temp_key, pessimism_key, actor_o_key, optimism_key, regularizer_key = jax.random.split(rng, 8)
            
            
            
            actor_def = policies.NormalTanhPolicy(self.actor_size, action_dim, activations=nn.relu, model_version=self.actor_type, depth=self.actor_depth)
            if self.actor_type == 1:
                actor = Model.create(actor_def, inputs=[actor_key, observations], tx=optax.adamw(learning_rate=actor_lr))
            if self.actor_type != 1:
                actor = Model.create(actor_def, inputs=[actor_key, observations], tx=optax.adam(learning_rate=actor_lr))
            
            
            
            
            if (self.distributional == 1) or (self.distributional == 2):
                output_nodes = self.n_atoms
            else:
                output_nodes = 1
            if network_regularization == 0:
                critic_def = critic_net.DoubleCriticLN(hidden_dims=critic_size, depth=critic_depth, activations=activations, output_nodes=output_nodes, categorical=self.categorical)
                critic = Model.create(critic_def, inputs=[critic_key, observations, actions], tx=optax.adam(learning_rate=critic_lr))

            if network_regularization == 1:
                critic_def = critic_net.DoubleCriticLN(hidden_dims=critic_size, depth=critic_depth, activations=activations, output_nodes=output_nodes, categorical=self.categorical)
                critic = Model.create(critic_def, inputs=[critic_key, observations, actions], tx=optax.adamw(learning_rate=critic_lr))
            
            if network_regularization == 2:
                if sn_type == 0:
                    critic_def = critic_net.DoubleCriticSN(hidden_dims=critic_size, depth=critic_depth, activations=activations, output_nodes=output_nodes, categorical=self.categorical)
                else:
                    critic_def = critic_net.DoubleCriticSN2(hidden_dims=critic_size, depth=critic_depth, activations=activations, output_nodes=output_nodes, categorical=self.categorical)
                critic = Model.create(critic_def, inputs=[critic_key, observations, actions], tx=optax.adam(learning_rate=critic_lr))

            if network_regularization == 3:
                critic_def = critic_net.DoubleCritic(hidden_dims=critic_size, depth=critic_depth, activations=activations, output_nodes=output_nodes, categorical=self.categorical)
                critic = Model.create(critic_def, inputs=[critic_key, observations, actions], tx=optax.adam(learning_rate=critic_lr))

            if network_regularization == 4:
                critic_def = critic_net.DoubleCritic2(hidden_dims=critic_size, depth=critic_depth, activations=activations, output_nodes=output_nodes, categorical=self.categorical)
                critic = Model.create(critic_def, inputs=[critic_key, observations, actions], tx=optax.adam(learning_rate=critic_lr))
                
            target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions])

            temp = Model.create(
                temperature.Temperature(init_temperature), inputs=[temp_key], tx=optax.adam(learning_rate=temp_lr, b1=0.5)
            )
            pessimism = Model.create(
                temperature.Pessimism(), inputs=[pessimism_key], tx=optax.adam(learning_rate=temp_lr, b1=0.5)
            )
            optimism = Model.create(temperature.TemperatureOffset(init_value=self.init_optimism, log_temp_min=self.log_temp_min, log_temp_max=self.log_temp_max),
                                inputs=[optimism_key],
                                tx=optax.adam(learning_rate=3e-5, b1=0.5))
            regularizer = Model.create(temperature.TemperatureOffset(init_value=self.init_regularizer, log_temp_min=self.log_temp_min, log_temp_max=self.log_temp_max),
                                inputs=[regularizer_key],
                                tx=optax.adam(learning_rate=3e-5, b1=0.5))
            
            
            actor_o_def = policies.DecoupledPolicy(hidden_dims=self.actor_size, action_dim=action_dim, activations=nn.relu, model_version=self.actor_type, depth=self.actor_depth)
            actor_o = Model.create(actor_o_def,
                                 inputs=[actor_o_key, observations, actions, actions, self.std_multiplier],
                                 tx=optax.chain(optax.adam(learning_rate=actor_lr)))
            
            actor_o_def = policies.DecoupledPolicySimple(self.actor_size, action_dim, activations=nn.relu, model_version=self.actor_type, depth=self.actor_depth)
            if self.actor_type == 1:
                actor_o = Model.create(actor_o_def, inputs=[actor_o_key, observations, actions, actions, self.std_multiplier], tx=optax.adamw(learning_rate=actor_lr))
            if self.actor_type != 1:
                actor_o = Model.create(actor_o_def, inputs=[actor_o_key, observations, actions, actions, self.std_multiplier], tx=optax.adam(learning_rate=actor_lr))
            return actor, critic, target_critic, temp, pessimism, actor_o, optimism, regularizer, rng

        self.init_models = jax.jit(jax.vmap(_init_models))
        self.actor, self.critic, self.target_critic, self.temp, self.pessimism, self.actor_o, self.optimism, self.regularizer, self.rng = self.init_models(self.seeds)
        self.step = 1

        if self.use_redo:
            # Actor
            mlp_model_actor_name = [key for key in self.actor.params.keys() if "MLP" in key and "/" not in key][0]
            actor_heads = [key for key in self.actor.params.keys() if not ("MLP" in key) and "/" not in key]
            self.weight_recycler_actor = weight_recyclers.NeuronRecycler(
                [
                    f"{mlp_model_actor_name}/{elem}"
                    for elem in list(self.actor.params[mlp_model_actor_name].keys())
                    if "Dense" in elem
                ]+[actor_heads],
                reset_period=100000,#2560000 // updates_per_step,
                # reset_period=1,
                # recycle_rate=0.9,
                reset_start_step=0,
            )
          
            # Critics
            self.critics_names = list(self.critic.params.keys())
            mlp_model_critic_name = [
                key for key in self.critic.params[self.critics_names[0]].keys() if "MLP" in key and "/" not in key
            ][0]
            self.weight_recyclers_critic = []
            for critic_name in self.critics_names:
                self.weight_recyclers_critic.append(
                    weight_recyclers.NeuronRecycler(
                        [
                            f"{critic_name}/{mlp_model_critic_name}/{elem}"
                            for elem in list(self.critic.params[critic_name][mlp_model_critic_name].keys())
                            if "Dense" in elem
                        ],
                        reset_period=100000,#2560000 // updates_per_step,
                        # reset_period=1,
                        reset_start_step=0,
                        # recycle_rate=0.9,
                    )
                )

    def update_controller(self, episode_reward):
        if self.critic_regularization == 2:
            self.controller.update(episode_reward)
            pessimism_value = self.controller.sample()
            self.pessimism_value = jnp.array(pessimism_value).reshape(self.num_seeds, 1)

    def save_state(self, path: str):
        self.actor.save(os.path.join(path, 'actor'))
        self.critic.save(os.path.join(path, 'critic'))
        self.target_critic.save(os.path.join(path, 'target_critic'))
        self.temp.save(os.path.join(path, 'temp'))
        with open(os.path.join(path, 'step'), 'w') as f:
            f.write(str(self.step))
            
    def calculate_init_values(self, init_value, log_temp_min, log_temp_max):
        value = np.exp(np.arctanh((np.log(init_value) - log_temp_min)/((log_temp_max - log_temp_min) * 0.5) - 1))
        return value

    def load_state(self, path: str):
        self.actor = self.actor.load(os.path.join(path, 'actor'))
        self.critic = self.critic.load(os.path.join(path, 'critic'))
        self.target_critic = self.target_critic.load(os.path.join(path, 'target_critic'))
        self.temp = self.temp.load(os.path.join(path, 'temp'))
        # Restore the step counter
        with open(os.path.join(path, 'step'), 'r') as f:
            self.step = int(f.read())

    def sample_activations(
        self, observations: np.ndarray, temperature: float = 1.0
    ) -> jnp.ndarray:
        activations = policies.sample_activations(
            self.rng, self.actor.apply_fn, self.actor.params, observations, temperature
        )
        return activations

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observations, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def sample_actions_o(self,
                       observations: np.ndarray, 
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions_o(self.rng, self.actor_o.apply_fn, self.actor_o.params, 
                                                 self.actor.apply_fn, self.actor.params, observations,
                                                 self.std_multiplier, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def get_validation_td(self, validation_batch: Batch):
        rng, validation_td = validate_batch(self.rng, self.actor, self.critic, self.target_critic, self.temp,
                                            self.pessimism, validation_batch, self.discount, self.pessimism_value, self.critic_regularization)
        self.rng = rng
        return validation_td

    def get_qvals(self, observations: np.ndarray, actions: np.ndarray) -> jnp.ndarray:
        q_vals = sample_qvals(self.critic, observations, actions, self.network_regularization)
        return q_vals

    def get_temp(self) -> np.ndarray:
        temp_val = sample_temp(self.temp)
        return np.asarray(temp_val)

    def get_distribution_parameters(self, validation_batch: Batch):
        mu, std = policies.get_distribution_parameters(self.actor.apply_fn, self.actor.params, validation_batch.observations)
        return mu, std

    def calculate_churn_statistics(self, validation_batch: Batch, mu: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
        new_mu, new_std = self.get_distribution_parameters(validation_batch)
        kl_backward = (jnp.log(std/new_std) + (new_std ** 2 + (new_mu - mu) ** 2)/(2 * std ** 2) - 1/2).sum(-1).mean(-1)
        kl_forward = (jnp.log(new_std/std) + (std ** 2 + (mu - new_mu) ** 2)/(2 * new_std ** 2) - 1/2).sum(-1).mean(-1)
        mu_diff = np.abs(mu - new_mu).sum(-1).mean(-1)
        std_diff = np.abs(std - new_std).sum(-1).mean(-1)
        return kl_backward, kl_forward, mu_diff, std_diff

    def update(self, batch: Batch, num_updates: int, env_step: int) -> InfoDict:
        if self.use_reset:
            if env_step in self.reset_list:
                self.reset()
        if self.use_redo:
            # Second elem of batch.observation shape is replay ratio [seeds, replay_ratio, batch_size, obs_dim]
            # Actor
            intermediates = self.sample_activations(jax.lax.stop_gradient(batch.observations[:,0,::]), temperature=1)
            online_params, opt_state = self.weight_recycler_actor.maybe_update_weights(
                env_step, intermediates, self.actor.params, self.rng, self.actor.opt_state)
            self.actor = self.actor.replace(params=online_params, opt_state=opt_state)

            # Critics
            intermediates = critic_activations(
                jax.lax.stop_gradient(batch.observations[:, 0, ::]),
                jax.lax.stop_gradient(batch.actions[:, 0, ::]),
                self.target_critic.params,
                self.target_critic.apply_fn,
                self.target_critic.batch_stats if hasattr(self.target_critic, "batch_stats") else None,
            )
            critics_online_params = {}
            critics_opt_states = {}
            for critic_name, weight_recycler_critic in zip(self. critics_names, self.weight_recyclers_critic):
                online_params, opt_state = weight_recycler_critic.maybe_update_weights(
                    env_step, intermediates, self.critic.params, self.rng, self.critic.opt_state)

                online_params_cp = copy.deepcopy(online_params)
                opt_state_cp = copy.deepcopy(opt_state)
                for next_critic_name in self.critics_names:
                    if next_critic_name != critic_name:
                        online_params_cp.pop(next_critic_name)
                        # TODO: is this enough?
                        opt_state_cp[0].mu.pop(next_critic_name)
                        opt_state_cp[0].nu.pop(next_critic_name)

                critics_online_params[critic_name] = copy.deepcopy(online_params_cp)
                critics_opt_states[critic_name] = copy.deepcopy(opt_state_cp)

            online_params = critics_online_params[self.critics_names[0]]
            online_params.update(critics_online_params[self.critics_names[1]])

            opt_state = critics_opt_states[self.critics_names[0]]
            opt_state[0].mu.update(critics_opt_states[self.critics_names[1]][0].mu)
            opt_state[0].nu.update(critics_opt_states[self.critics_names[1]][0].nu)
            self.critic = self.critic.replace(params=online_params, opt_state=opt_state)

        step, rng, actor, critic, target_critic, temp, pessimism, actor_o, optimism, regularizer, info = _do_multiple_updates(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            self.pessimism,
            self.actor_o,
            self.optimism,
            self.regularizer,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.step,
            num_updates,
            self.pessimism_value,
            self.critic_regularization,
            self.network_regularization,
            self.use_sam,
            self.distributional,
            self.n_atoms,
            self.quantile_taus, 
            self.v_min, 
            self.v_max, 
            self.support,
            self.soft_critic,
            self.std_multiplier,
            self.action_dim,
            self.kl_target,
            self.pessimism_value_
        )
        self.step = step
        self.rng = rng
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.pessimism = pessimism
        self.actor_o = actor_o
        self.optimism = optimism
        self.regularizer = regularizer
        return info

    def reset(self):
        self.step = 1
        self.actor, self.critic, self.target_critic, self.temp, self.pessimism, self.actor_o, self.optimism, self.regularizer, self.rng = self.init_models(self.seeds)


@functools.partial(jax.jit, static_argnames=('network_regularization'))
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
def _sample_qvals(
        critic: Model,
        observations: np.ndarray,
        actions: np.ndarray,
        network_regularization: int) -> jnp.ndarray:
    if network_regularization == 2:
        (q1, q2), _ = critic(observations, actions, training=False)
    else:
        q1, q2 = critic(observations, actions)
    q_mean = (q1 + q2)/2
    return q_mean

def sample_qvals(
        critic: Model,
        observations: np.ndarray,
        actions: np.ndarray,
        network_regularization: int) -> jnp.ndarray:
    return _sample_qvals(critic, observations, actions, network_regularization)

@functools.partial(jax.jit, static_argnames=())
@functools.partial(jax.vmap, in_axes=(0))
def _sample_temp(temp: Model) -> jnp.ndarray:
    return temp()

def sample_temp(temp: Model) -> jnp.ndarray:
    return _sample_temp(temp)
