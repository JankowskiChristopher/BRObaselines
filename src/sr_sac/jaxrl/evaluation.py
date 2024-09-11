from typing import Dict
import numpy as np
import gym

from jaxrl.networks.common import Model
import jax.numpy as jnp
from jax.random import KeyArray
from typing import Callable
import jax
from functools import partial


@partial(jax.jit, static_argnums=(1, 2, 3, 5))
@partial(jax.vmap, in_axes=(0, None, None, None, 0, None))
def evaluate_brax(
    actor: Model,
    eval_step_fn: Callable,
    eval_reset_fn: Callable,
    num_episodes: int,
    rng: KeyArray,
    episode_length: int = 1000,
) -> Dict[str, float]:
    rng, key = jax.random.split(rng)
    state = eval_reset_fn(rng=key)

    def one_step(i, state_ret):
        state, ret = state_ret
        dist = actor.apply({"params": actor.params}, state.obs, 0.0)
        try:
            action = dist.sample(seed=key)
        except AttributeError:
            action = dist
        next_state = eval_step_fn(state, action)
        ret = ret + next_state.reward
        return (next_state, ret)

    ret = jnp.zeros(num_episodes)
    last_state, ret = jax.lax.fori_loop(0, episode_length, one_step, (state, ret))
    eval_metrics = last_state.info["eval_metrics"]
    avg_episode_length = eval_metrics.completed_episodes_steps / eval_metrics.completed_episodes
    metrics = dict(
        dict(
            {
                f"eval_episode_{name}": value / eval_metrics.completed_episodes
                for name, value in eval_metrics.completed_episodes_metrics.items()
            }
        ),
        **dict(
            {"eval_completed_episodes": eval_metrics.completed_episodes, "eval_avg_episode_length": avg_episode_length}
        ),
    )
    metrics["return"] = metrics["eval_episode_reward"]
    return metrics


def evaluate(agent, env: gym.Env, num_episodes: int, episode_length: int) -> Dict[str, float]:
    if "brax" in str(type(env)).lower():
        return evaluate_brax(agent.actor, env.step, env.reset, num_episodes, agent.rng, episode_length)
    else:
        n_seeds = env.num_envs
        returns = []
        for _ in range(num_episodes):
            observations, dones = env.reset(), np.array([False] * n_seeds)
            rets, length = np.zeros(n_seeds), 0
            while not dones.all():
                actions = agent.sample_actions(observations, temperature=0.0)
                prev_dones = dones
                observations, rewards, dones, infos = env.step(actions)
                rets += rewards * (1 - prev_dones)
                length += 1
                if length >= episode_length:
                    break
            returns.append(rets)
        return {"return": np.array(returns).mean(axis=0)}


def evaluate_validate(
    agent, validation_buffer, env: gym.Env, num_episodes: int, episode_length: int
):
    n_seeds = env.num_envs
    returns = []
    diffs = []
    for _episode in range(num_episodes):
        observations, dones = env.reset(), np.array([False] * n_seeds)
        rets, length = np.zeros(n_seeds), 0
        implied_rews = np.zeros(n_seeds)
        i = 0
        while not dones.all():
            actions = agent.sample_actions(observations, temperature=0.0)
            if length == 0:
                q_vals = agent.get_qvals(observations, actions)
            prev_dones = dones
            next_observations, rewards, dones, infos = env.step(actions)
            implied_rews += 0.99 ** i * rewards
            i += 1
            if _episode == 0:
                masks = env.generate_masks(dones, infos)
                validation_buffer.insert(observations, actions, rewards, masks, dones, next_observations)
            observations = next_observations
            rets += rewards * (1 - prev_dones)
            length += 1
            if length >= episode_length:
                break
        temp_val = agent.get_temp()
        implied_q = implied_rews + (temp_val * agent.target_entropy * (1 / (1 - 0.99))) - temp_val * agent.target_entropy
        diff_ = q_vals - implied_q
        returns.append(rets)
        diffs.append(diff_)
    validation_batch = validation_buffer.sample_parallel(256)
    td_val = agent.get_validation_td(validation_batch)
    return {
        "return": np.array(returns).mean(axis=0),
        "q_overestimation": np.array(diffs).mean(axis=0),
        "validation_td": np.array(td_val),
    }, validation_buffer


def prefill_buffer(validation_buffer, env: gym.Env, num_episodes: int, episode_length: int):
    n_seeds = env.num_envs
    for _episode in range(num_episodes):
        observations, dones = env.reset(), np.array([False] * n_seeds)
        length = 0
        while not dones.all():
            actions = env.action_space.sample()
            next_observations, rewards, dones, infos = env.step(actions)
            if _episode == 0:
                masks = env.generate_masks(dones, infos)
                validation_buffer.insert(observations, actions, rewards, masks, dones, next_observations)
            observations = next_observations
            length += 1
            if length >= episode_length:
                break
    return validation_buffer

def evaluate_validate_mw(agent, validation_buffer, envs: gym.Env, num_episodes: int, episode_length: int, temperature: float = 0.0) -> Dict[str, float]:
    n_seeds = envs.num_envs
    goals_final = []
    goals_mean = []
    returns = []
    diffs = []
    for _episode in range(num_episodes):
        observations, terms, truns = envs.reset(), np.array([False] * n_seeds), np.array([False] * n_seeds)
        rets = np.zeros(n_seeds)
        goals_mean_ = 0.0
        implied_rews = np.zeros(n_seeds)
        idx = 0
        for i in range(episode_length): # CHANGE?
            actions = agent.sample_actions(observations, temperature=temperature)
            if i == 0:
                q_vals = agent.get_qvals(observations, actions)
            next_observations, rewards, terms, truns, goals = envs.step(actions)
            implied_rews += 0.99 ** idx * rewards
            idx += 1
            goals_mean_ += goals / episode_length
            rets += rewards
            masks = envs.generate_masks(terms, truns)
            terms_float = terms.astype(np.float32)            
            if _episode == 0:
                validation_buffer.insert(observations, actions, rewards, masks, terms_float, next_observations)    
            observations = next_observations            
        goals_final.append(goals)
        goals_mean_[goals_mean_ > 0] = 1.0
        goals_mean.append(goals_mean_)
        temp_val = agent.get_temp()
        implied_q = implied_rews + (temp_val * agent.target_entropy * (1 / (1 - 0.99))) - temp_val * agent.target_entropy
        diff_ = q_vals - implied_q
        returns.append(rets)
        diffs.append(diff_)
    validation_batch = validation_buffer.sample_parallel(256)
    td_val = agent.get_validation_td(validation_batch)
    error_parts = td_val * (1/(1-0.99))
    return {'goals': np.array(goals_final).mean(axis=0), 'goals_mean': np.array(goals_mean).mean(axis=0), 'return': np.array(returns).mean(axis=0), 'q_overestimation': np.array(diffs).mean(axis=0), 'error_parts': np.array(error_parts),  'validation_td': np.array(td_val)}, validation_buffer

def evaluate_mw(agent, envs: gym.Env, num_episodes: int, episode_length: int, temperature: float = 0.0) -> Dict[str, float]:
    n_seeds = envs.num_envs
    goals_final = []
    goals_mean = []
    returns = []
    for _episode in range(num_episodes):
        observations, terms, truns = envs.reset(), np.array([False] * n_seeds), np.array([False] * n_seeds)
        rets = np.zeros(n_seeds)
        goals_mean_ = 0.0
        idx = 0
        for i in range(episode_length): # CHANGE?
            actions = agent.sample_actions(observations, temperature=temperature)
            next_observations, rewards, terms, truns, goals = envs.step(actions)
            idx += 1
            goals_mean_ += goals / episode_length
            rets += rewards
            observations = next_observations            
        goals_final.append(goals)
        goals_mean_[goals_mean_ > 0] = 1.0
        goals_mean.append(goals_mean_)
        returns.append(rets)
    return {'goals': np.array(goals_final).mean(axis=0), 'goals_mean': np.array(goals_mean).mean(axis=0), 'return': np.array(returns).mean(axis=0)}
