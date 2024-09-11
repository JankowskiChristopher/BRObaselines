from typing import Tuple

import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm


def update(key: PRNGKey, actor: Model, critic: Model, temp: Model, pessimism: Model,
           batch: Batch, pessimism_value: jnp.ndarray, critic_regularization: int, use_sam:bool) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, mu, std = actor.apply({'params': actor_params}, batch.observations, return_params=True)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        if critic.batch_stats is not None:
            (q1, q2), _ = critic(batch.observations, actions, **{"training":False})
        else:
            q1, q2 = critic(batch.observations, actions)
        if critic_regularization == 3:
            val_ = pessimism()
            q = (q1 + q2)/2 - val_ * jnp.abs(q1 - q2)
        else:
            val_ = jnp.squeeze(pessimism_value)
            q = (q1 + q2)/2 - val_ * jnp.abs(q1 - q2) / 2
        actor_loss = (log_probs * temp().mean() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'entropy_dispersion': -log_probs.var(),
            'actor_pnorm': tree_norm(actor_params),
            'std': std.mean(),
            'std_var': std.mean(axis=-1).var(),
            'actor_action': jnp.mean(jnp.abs(actions)),
            'pessimism': val_
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn, use_sam=use_sam)
    info['actor_gnorm'] = info.pop('grad_norm')
    return new_actor, info

def update_optimistic(
        key: PRNGKey, actor_c: Model, actor_o: Model, critic: Model, optimism: Model, 
        regularizer: Model, batch: Batch, std_multiplier: float, action_dim: float, use_sam: bool,
        ) -> Tuple[Model, InfoDict]:
    def actor_o_loss_fn(actor_o_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        _, mu_c, std_c = actor_c(batch.observations, temperature=1.0, return_params=True)
        dist, mu_o, std_ox = actor_o.apply({'params': actor_o_params}, observations=batch.observations, means=mu_c, stds=std_c, std_multiplier=std_multiplier, return_params=True)
        std_o = std_ox / std_multiplier
        actions = dist.sample(seed=key)
        if critic.batch_stats is not None:
            (q1, q2), _ = critic(batch.observations, actions, **{"training":False})
        else:
            q1, q2 = critic(batch.observations, actions)
        kl = (jnp.log(std_c/std_o) + (std_o ** 2 + (mu_o - mu_c) ** 2)/(2 * std_c ** 2) - 1/2).sum(-1)
        q_ub = (q1 + q2) / 2 + optimism() * jnp.abs(q1 - q2) / 2
        actor_e_loss = (-q_ub).mean() + regularizer() * kl.mean()    
        return actor_e_loss, {
            'actor_o_loss': actor_e_loss,
            'kl': kl.mean()/action_dim,
            'actor_o_pnorm': tree_norm(actor_o_params),
            'std_c': std_c.mean(),
            'std_o': std_ox.mean(),
            'std_o_corr': std_o.mean(),
            'Q_mean': ((q1 + q2) / 2).mean(),
            'Q_std': (jnp.abs(q1 - q2) / 2).mean(),
            'Qloss': (-q_ub).mean(),
            'KLoss': kl.mean(),
        }
    new_actor_o, info = actor_o.apply_gradient(actor_o_loss_fn, use_sam=use_sam)
    info['actor_o_gnorm'] = info.pop('grad_norm')
    return new_actor_o, info

def update_quantile(key: PRNGKey, actor: Model, quantile_critic: Model, temp: Model, pessimism: Model,
           batch: Batch, pessimism_value: jnp.ndarray, critic_regularization: int, use_sam:bool) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, mu, std = actor.apply({'params': actor_params}, batch.observations, return_params=True)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        if quantile_critic.batch_stats is not None:
            (q1, q2), _ = quantile_critic(batch.observations, actions, **{"training":False})
        else:
            q1, q2 = quantile_critic(batch.observations, actions)
        if critic_regularization == 3:
            val_ = pessimism()
            q = (q1 + q2)/2 - val_ * jnp.abs(q1 - q2) / 2
        else:
            val_ = jnp.squeeze(pessimism_value)
            q = (q1 + q2)/2 - val_ * jnp.abs(q1 - q2) / 2
        q = q.mean(-1)
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'entropy_dispersion': -log_probs.var(),
            'actor_pnorm': tree_norm(actor_params),
            'std': std.mean(),
            'std_var': std.mean(axis=-1).var(),
            'actor_action': jnp.mean(jnp.abs(actions)),
            'pessimism': val_
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn, use_sam=use_sam)
    info['actor_gnorm'] = info.pop('grad_norm')
    return new_actor, info

def update_optimistic_quantile(
        key: PRNGKey, actor_c: Model, actor_o: Model, quantile_critic: Model, optimism: Model, 
        regularizer: Model, batch: Batch, std_multiplier: float, action_dim: float, use_sam: bool
        ) -> Tuple[Model, InfoDict]:
    def actor_o_loss_fn(actor_o_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        _, mu_c, std_c = actor_c(batch.observations, temperature=1.0, return_params=True)
        dist, mu_o, std_ox = actor_o.apply({'params': actor_o_params}, observations=batch.observations, means=mu_c, stds=std_c, std_multiplier=std_multiplier, return_params=True)
        std_o = std_ox / std_multiplier
        actions = dist.sample(seed=key)
        if quantile_critic.batch_stats is not None:
            (q1, q2), _ = quantile_critic(batch.observations, actions, **{"training":False})
        else:
            q1, q2 = quantile_critic(batch.observations, actions)
        kl = (jnp.log(std_c/std_o) + (std_o ** 2 + (mu_o - mu_c) ** 2)/(2 * std_c ** 2) - 1/2).sum(-1)
        q_ub = (q1 + q2) / 2 + optimism() * jnp.abs(q1 - q2) / 2
        q_ub = q_ub.mean(-1)
        actor_e_loss = (-q_ub).mean() + regularizer() * kl.mean()    
        return actor_e_loss, {
            'actor_o_loss': actor_e_loss,
            'kl': kl.mean()/action_dim,
            'actor_o_pnorm': tree_norm(actor_o_params),
            'std_c': std_c.mean(),
            'std_o': std_ox.mean(),
            'std_o_corr': std_o.mean(),
            'Q_mean': ((q1 + q2) / 2).mean(),
            'Q_std': (jnp.abs(q1 - q2) / 2).mean(),
            'Qloss': (-q_ub).mean(),
            'KLoss': kl.mean(),
        }
    new_actor_o, info = actor_o.apply_gradient(actor_o_loss_fn, use_sam=use_sam)
    info['actor_o_gnorm'] = info.pop('grad_norm')
    return new_actor_o, info


def update_categorical(key: PRNGKey, actor: Model, quantile_critic: Model, temp: Model, pessimism: Model,
           batch: Batch, pessimism_value: jnp.ndarray, critic_regularization: int, use_sam: bool, support: jnp.ndarray) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, mu, std = actor.apply({'params': actor_params}, batch.observations, return_params=True)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        if quantile_critic.batch_stats is not None:
            (probs1, probs2), _ = quantile_critic(batch.observations, actions, **{"training":False})
        else:
            probs1, probs2 = quantile_critic(batch.observations, actions)
        probs = (probs1 + probs2)/2
        q = (probs * support).sum(-1)
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'entropy_dispersion': -log_probs.var(),
            'actor_pnorm': tree_norm(actor_params),
            'std': std.mean(),
            'std_var': std.mean(axis=-1).var(),
            'actor_action': jnp.mean(jnp.abs(actions)),
            #'pessimism': val_
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn, use_sam=use_sam)
    info['actor_gnorm'] = info.pop('grad_norm')
    return new_actor, info


