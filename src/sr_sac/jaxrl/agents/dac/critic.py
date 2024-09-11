from typing import Tuple
import functools

import jax
import jax.numpy as jnp
import numpy as np

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm
from flax import linen as nn

def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)

def huber_replace(td_errors, kappa: float = 1.0):
    return jnp.where(jnp.absolute(td_errors) <= kappa, 0.5 * td_errors ** 2, kappa * (jnp.absolute(td_errors) - 0.5 * kappa))

def calculate_quantile_huber_loss(td_errors, taus, kappa: float = 1.0):
    element_wise_huber_loss = huber_replace(td_errors, kappa)
    mask = jax.lax.stop_gradient(jnp.where(td_errors < 0, 1, 0)) # detach this
    element_wise_quantile_huber_loss = jnp.absolute(taus[..., None] - mask) * element_wise_huber_loss / kappa
    quantile_huber_loss = element_wise_quantile_huber_loss.sum(axis=1).mean()
    return quantile_huber_loss

def update(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, pessimism: Model, batch: Batch, discount: float, pessimism_value: jnp.ndarray,
           critic_regularization: int, network_regularization: int, soft_critic: bool, use_sam: bool) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    if network_regularization == 2:
        (next_q1, next_q2), _ = target_critic(batch.next_observations, next_actions)
    else:
        next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    if critic_regularization == 3:
        val_ = pessimism()
        next_q = (next_q1 + next_q2) / 2 - val_ * jnp.abs(next_q1 - next_q2)
    else:
        val_ = jnp.squeeze(pessimism_value)
        next_q = (next_q1 + next_q2) / 2 - val_ * jnp.abs(next_q1 - next_q2) / 2
    
    target_q = batch.rewards + discount * batch.masks * next_q
    if soft_critic:
        target_q -= discount * temp() * batch.masks * next_log_probs

    def critic_loss_fn(critic_params: Params, batch_stats=None) -> Tuple[jnp.ndarray, InfoDict]:
        if batch_stats is not None:
            critic_fn = lambda actions: critic.apply(
                {"params": critic_params, "batch_stats": batch_stats}, batch.observations, actions, mutable=["batch_stats"]
            )
        else:
            critic_fn = lambda actions: critic.apply({"params": critic_params}, batch.observations, actions)

        def _critic_fn(actions):
            if batch_stats is not None:
                (q1, q2), batch_stats_updates = critic_fn(actions)
                return 0.5 * (q1 + q2).mean(), (q1, q2, batch_stats_updates)
            else:
                q1, q2 = critic_fn(actions)
                return 0.5 * (q1 + q2).mean(), (q1, q2)

        if batch_stats is not None:
            (_, (q1, q2, batch_stats_updates)), action_grad = jax.value_and_grad(_critic_fn, has_aux=True)(batch.actions)
        else:
            (_, (q1, q2)), action_grad = jax.value_and_grad(_critic_fn, has_aux=True)(batch.actions)
        errors = ((q1 + q2) / 2 - target_q).mean()
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        if batch_stats is not None:
            return critic_loss, {
                "batch_stats_updates": batch_stats_updates,
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
                "r": batch.rewards.mean(),
                "errors": errors,
                "critic_pnorm": tree_norm(critic_params),
                "critic_agnorm": jnp.sqrt((action_grad**2).sum(-1)).mean(0),
            }
        else:
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
                "r": batch.rewards.mean(),
                "errors": errors,
                "critic_pnorm": tree_norm(critic_params),
                "critic_agnorm": jnp.sqrt((action_grad**2).sum(-1)).mean(0),
            }

    new_critic, info = critic.apply_gradient(critic_loss_fn, use_sam=use_sam)
    # jax.debug.print("{x}", x=jax.numpy.max(jax.numpy.linalg.eigvals(critic.params['CriticSN_1']['MLP_SN_0']['Dense_1']['kernel'])))
    # jax.debug.print("{x}", x=jax.numpy.max(jax.numpy.linalg.eigvals(critic.params['Critic_1']['MLP_0']['Dense_1']['kernel'])))
    info["critic_gnorm"] = info.pop("grad_norm")
    return new_critic, info

def update_categorical(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, pessimism: Model, batch: Batch, discount: float, pessimism_value: jnp.ndarray,
           critic_regularization: int, network_regularization: int, soft_critic: bool, use_sam: bool, n_atoms: int, v_min: float, v_max: float, support: jnp.ndarray) -> Tuple[Model, InfoDict]:
    
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    
    if network_regularization == 2:
        (next_dist1, next_dist2), _ = target_critic(batch.next_observations, next_actions)
    else:
        next_dist1, next_dist2 = target_critic(batch.next_observations, next_actions)
        
    next_pmfs = (next_dist1 + next_dist2) / 2
    masks = batch.masks[..., None] 
    next_atoms = batch.rewards[..., None] + discount * masks * support
    if soft_critic:
        next_atoms -= discount * temp().mean() * masks * next_log_probs[..., None]
    
    delta_z = support[0,1] - support[0,0]
    tz = jnp.clip(next_atoms, a_min=(v_min), a_max=(v_max))
    b = (tz - v_min) / delta_z
    l = jnp.clip(jnp.floor(b), a_min=0, a_max=n_atoms - 1)
    u = jnp.clip(jnp.ceil(b), a_min=0, a_max=n_atoms - 1)
    d_m_l = (u + (l == u).astype(jnp.float32) - b) * next_pmfs
    d_m_u = (b - l) * next_pmfs
    target_pmfs = jnp.zeros_like(next_pmfs)
    def project_to_bins(i, val):
        val = val.at[i, l[i].astype(jnp.int32)].add(d_m_l[i])
        val = val.at[i, u[i].astype(jnp.int32)].add(d_m_u[i])
        return val
    target_pmfs = jax.lax.fori_loop(0, target_pmfs.shape[0], project_to_bins, target_pmfs)

    def critic_loss_fn(critic_params: Params, batch_stats=None) -> Tuple[jnp.ndarray, InfoDict]:
        if batch_stats is not None:
            critic_fn = lambda actions: critic.apply(
                {"params": critic_params, "batch_stats": batch_stats}, batch.observations, actions, mutable=["batch_stats"]
            )
        else:
            critic_fn = lambda actions: critic.apply({"params": critic_params}, batch.observations, actions)

        def _critic_fn(actions):
            if batch_stats is not None:
                (old_pmfs1, old_pmfs2), batch_stats_updates = critic_fn(actions)
                return 0.5 * (old_pmfs1 + old_pmfs2).mean(), (old_pmfs1, old_pmfs2, batch_stats_updates)
            else:
                old_pmfs1, old_pmfs2 = critic_fn(actions)
                return 0.5 * (old_pmfs1 + old_pmfs2).mean(), (old_pmfs1, old_pmfs2)

        if batch_stats is not None:
            (_, (old_pmfs1, old_pmfs2, batch_stats_updates)), action_grad = jax.value_and_grad(_critic_fn, has_aux=True)(batch.actions)
        else:
            (_, (old_pmfs1, old_pmfs2)), action_grad = jax.value_and_grad(_critic_fn, has_aux=True)(batch.actions)
        
        old_pmfs1 = jnp.clip(old_pmfs1, a_min=1e-5, a_max=1 - 1e-5)
        old_pmfs2 = jnp.clip(old_pmfs2, a_min=1e-5, a_max=1 - 1e-5)
        loss1 = (-(target_pmfs * jnp.log(old_pmfs1)).sum(-1)).mean()
        loss2 = (-(target_pmfs * jnp.log(old_pmfs2)).sum(-1)).mean()
        q1 = (old_pmfs1 * support).sum(-1)
        q2 = (old_pmfs2 * support).sum(-1)
        errors = 0.0
        
        critic_loss = loss1 + loss2
        if batch_stats is not None:
            return critic_loss, {
                "batch_stats_updates": batch_stats_updates,
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
                "r": batch.rewards.mean(),
                "errors": errors,
                "critic_pnorm": tree_norm(critic_params),
                "critic_agnorm": jnp.sqrt((action_grad**2).sum(-1)).mean(0),
            }
        else:
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
                "r": batch.rewards.mean(),
                "errors": errors,
                "critic_pnorm": tree_norm(critic_params),
                "critic_agnorm": jnp.sqrt((action_grad**2).sum(-1)).mean(0),
            }

    new_critic, info = critic.apply_gradient(critic_loss_fn, use_sam=use_sam)
    # jax.debug.print("{x}", x=jax.numpy.max(jax.numpy.linalg.eigvals(critic.params['CriticSN_1']['MLP_SN_0']['Dense_1']['kernel'])))
    # jax.debug.print("{x}", x=jax.numpy.max(jax.numpy.linalg.eigvals(critic.params['Critic_1']['MLP_0']['Dense_1']['kernel'])))
    info["critic_gnorm"] = info.pop("grad_norm")
    return new_critic, info

def update_quantile(key: PRNGKey, actor: Model, quantile_critic: Model, target_quantile_critic: Model,
           temp: Model, pessimism: Model, batch: Batch, discount: float, pessimism_value: jnp.ndarray,
           critic_regularization: int, network_regularization: int, soft_critic: bool, use_sam: bool, n_atoms: int, taus: jnp.ndarray) -> Tuple[Model, InfoDict]:
    kappa = 1.0
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    if network_regularization == 2:
        (next_q1, next_q2), _ = target_quantile_critic(batch.next_observations, next_actions)
    else:
        next_q1, next_q2 = target_quantile_critic(batch.next_observations, next_actions)
    val_ = jnp.squeeze(pessimism_value)
    next_q = (next_q1 + next_q2) / 2 - val_ * jnp.abs(next_q1 - next_q2) / 2
    masks = batch.masks[..., None, None] 
    target_q = batch.rewards[..., None, None] + discount * masks * next_q[:, None, :]
    if soft_critic:
        target_q -= discount * temp().mean() * masks * next_log_probs[..., None, None]

    def critic_loss_fn(quantile_critic_params: Params, batch_stats=None) -> Tuple[jnp.ndarray, InfoDict]:
        if batch_stats is not None:
            critic_fn = lambda actions: quantile_critic.apply(
                {"params": quantile_critic_params, "batch_stats": batch_stats}, batch.observations, actions, mutable=["batch_stats"]
            )
        else:
            critic_fn = lambda actions: quantile_critic.apply({"params": quantile_critic_params}, batch.observations, actions)

        def _critic_fn(actions):
            if batch_stats is not None:
                (q1, q2), batch_stats_updates = critic_fn(actions)
                return 0.5 * (q1 + q2).mean(), (q1, q2, batch_stats_updates)
            else:
                q1, q2 = critic_fn(actions)
                return 0.5 * (q1 + q2).mean(), (q1, q2)

        if batch_stats is not None:
            (_, (q1, q2, batch_stats_updates)), action_grad = jax.value_and_grad(_critic_fn, has_aux=True)(batch.actions)
        else:
            (_, (q1, q2)), action_grad = jax.value_and_grad(_critic_fn, has_aux=True)(batch.actions)
        errors = ((q1 + q2) / 2 - target_q).mean()
        td_errors1 = target_q - q1[..., None]
        td_errors2 = target_q - q2[..., None] 
        loss_1 = calculate_quantile_huber_loss(td_errors1, taus, kappa=kappa)
        loss_2 = calculate_quantile_huber_loss(td_errors2, taus, kappa=kappa)
        critic_loss = loss_1 + loss_2
        if batch_stats is not None:
            return critic_loss, {
                "batch_stats_updates": batch_stats_updates,
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
                "r": batch.rewards.mean(),
                "errors": errors,
                "critic_pnorm": tree_norm(quantile_critic_params),
                "critic_agnorm": jnp.sqrt((action_grad**2).sum(-1)).mean(0),
            }
        else:
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
                "r": batch.rewards.mean(),
                "errors": errors,
                "critic_pnorm": tree_norm(quantile_critic_params),
                "critic_agnorm": jnp.sqrt((action_grad**2).sum(-1)).mean(0),
            }

    new_quantile_critic, info = quantile_critic.apply_gradient(critic_loss_fn, use_sam=use_sam)
    # jax.debug.print("{x}", x=jax.numpy.max(jax.numpy.linalg.eigvals(critic.params['CriticSN_1']['MLP_SN_0']['Dense_1']['kernel'])))
    # jax.debug.print("{x}", x=jax.numpy.max(jax.numpy.linalg.eigvals(critic.params['Critic_1']['MLP_0']['Dense_1']['kernel'])))
    info["critic_gnorm"] = info.pop("grad_norm")
    return new_quantile_critic, info

@functools.partial(jax.jit, static_argnames=('discount', 'critic_regularization'))
@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, None, 0, None))
def _validate_batch(rng: PRNGKey, actor: Model, critic: Model, target_critic: Model, temp: Model, pessimism: Model, 
                    validation_batch: Batch, discount: float, pessimism_value: jnp.ndarray, critic_regularization: int) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor(validation_batch.next_observations)
    rng, key = jax.random.split(rng)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    if critic.batch_stats is not None:
        (next_q1, next_q2), _ = target_critic(validation_batch.next_observations, next_actions)   
    else:
        next_q1, next_q2 = target_critic(validation_batch.next_observations, next_actions)        
    if critic_regularization == 3:
        val_ = pessimism()
        next_q = (next_q1 + next_q2) / 2 - val_ * jnp.abs(next_q1 - next_q2)
    else:
        val_ = jnp.squeeze(pessimism_value)
        next_q = (next_q1 + next_q2) / 2 - val_ * jnp.abs(next_q1 - next_q2) / 2
    target_q = validation_batch.rewards + discount * validation_batch.masks * next_q
    target_q -= discount * temp() * validation_batch.masks * next_log_probs
    if critic.batch_stats is not None:
        (q1, q2), _ = critic(validation_batch.observations, validation_batch.actions,**{"training":False})
    else:
        q1, q2 = critic(validation_batch.observations, validation_batch.actions)
    validation_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
    return rng, validation_loss

def validate_batch(rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, pessimism: Model, validation_batch: Batch, discount: float, pessimism_value: jnp.ndarray, critic_regularization: int) -> Tuple[PRNGKey, jnp.ndarray]:
    return _validate_batch(rng, actor, critic, target_critic, temp, pessimism, validation_batch, discount, pessimism_value, critic_regularization)


@functools.partial(jax.jit, static_argnames=("critic_def"))
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None, 0))
def critic_activations(observations: np.ndarray, actions: np.ndarray, critic_params: Params, critic_def:nn.Module, batch_stats=None):
    filter_rep = lambda l, _: l.name is not None and 'act' in l.name
    if batch_stats is not None:
        _, feat = critic_def.apply(
            {"params": critic_params, "batch_stats": batch_stats},
            observations,
            actions,
            mutable=['intermediates'],
            capture_intermediates=filter_rep,
            training=False
        )
    else:
        _, feat = critic_def.apply(
            {"params": critic_params},
            observations,
            actions,
            mutable=['intermediates'],
            capture_intermediates=filter_rep,
        )

    return feat['intermediates']
