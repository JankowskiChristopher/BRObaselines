from jaxrl.agents.sac.critic import critic_activations
import jax
import jax.numpy as jnp
import numpy as np
import flax
import wandb


def log_dict(seed, score, prefix="critic", dormant_thre: float = 0.1):
    total_neurons, total_deadneurons = 0.0, 0.0
    score_dict = flax.traverse_util.flatten_dict(score, sep="/")

    log_dict = {}
    prefix = f"seed{seed}/dorm_{prefix}-"
    for k, m in score_dict.items():
        m = m[0]
        layer_size = float(jnp.size(m))
        deadneurons_count = jnp.count_nonzero(m <= dormant_thre)
        total_neurons += layer_size
        total_deadneurons += deadneurons_count
        # tmp_tag = k[:-9].replace("/", "-")
        tmp_tag = k
        log_dict[f"{prefix}dead_percentage-{tmp_tag}"] = (float(deadneurons_count) / layer_size) * 100.0
    log_dict[f"{prefix}dead_percentage"] = (float(total_deadneurons) / total_neurons) * 100.0
    return log_dict


def estimate_neuron_score(activation):
    """Calculates neuron score based on absolute value of activation.

    The score of feature map is the normalized average score over
    the spatial dimension.

    Args:
        activation: intermediate activation of each layer

    Returns:
        element_score: score of each element in feature map in the spatial dim.
        neuron_score: score of feature map
    """
    reduce_axes = list(range(activation.ndim - 1))
    score = jnp.mean(jnp.abs(activation), axis=reduce_axes)
    # Normalize so that all scores sum to one.
    score /= jnp.mean(score) + 1e-9
    return score


def compute_singular_values(matrix, thresh=1e-5):
    """Compute srank(matrix) and other values."""
    ret_dict = dict()
    try:
        singular_vals = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        print("SVD failed")
        return {"singular_vals": np.nan, "nuclear_norm": np.nan, "condition_number": np.nan, "rank": np.nan}

    rank = max(np.sum(singular_vals >= thresh), 1)
    nuclear_norm = np.sum(singular_vals)
    condition_number = singular_vals[0] / (singular_vals[-1] + 1e-8)
    ret_dict["singular_vals"] = singular_vals
    ret_dict["nuclear_norm"] = nuclear_norm
    ret_dict["condition_number"] = condition_number
    ret_dict["rank"] = rank
    return ret_dict


def calculate_dormant_neurons(agent, i, replay_buffer, flags):
    if i % flags.dormant_calc_every == 0:
        # TODO: assert that batch_sizes_multiplier*flags.batch_size is at least bigger than the widest dense layer
        batch_sizes_multiplier = 4
        log_dict_wandb = {}

        batches = replay_buffer.sample_parallel_multibatch(flags.batch_size * batch_sizes_multiplier, 1)
        observations_bt = batches.observations[:, 0, :, :]
        actions_bt = batches.actions[:, 0, :, :]

        # Actor
        intermediates = agent.sample_activations(jax.lax.stop_gradient(observations_bt), temperature=1)
        flattened, _ = jax.tree_util.tree_flatten_with_path(intermediates)
        for seed in range(flags.num_seeds):
            # we use flattened[-1][0][-3].key to take the last layer of the MLP0
            features_dict = compute_singular_values(
                jax.tree_util.tree_map(lambda x: x[seed], intermediates)["MLP_0"][flattened[-1][0][-3].key]["__call__"][
                    0
                ]
            )
            log_dict_wandb.update({f"seed{seed}/actor-{flattened[-1][0][-3].key}-RANK": features_dict["rank"]})
            score_tree = jax.tree_map(estimate_neuron_score, jax.tree_util.tree_map(lambda x: x[seed], intermediates))
            neuron_score_dict = flax.traverse_util.flatten_dict(score_tree, sep="/")
            neuron_score_dict = {key.split("/")[-2]: element for key, element in neuron_score_dict.items()}
            log_dict_wandb.update(log_dict(seed, neuron_score_dict, prefix=f"actor", dormant_thre=flags.dormant_thre))

        # Critic
        # TODO: should we use target critic or critic?
        intermediates = critic_activations(
            jax.lax.stop_gradient(observations_bt),
            jax.lax.stop_gradient(actions_bt),
            agent.target_critic.params,
            agent.target_critic.apply_fn,
            agent.target_critic.batch_stats if hasattr(agent.target_critic, "batch_stats") else None
        )
        flattened, _ = jax.tree_util.tree_flatten_with_path(intermediates)

        # QUITE DIRTY HACK # TODO - fix
        critic_name = list(intermediates.keys())[0].split("_")[0]
        mlp_name = list(intermediates[f"{critic_name}_0"])[0][:-2]
        for seed in range(flags.num_seeds):
            for critic in range(2):
                features_dict = compute_singular_values(
                    jax.tree_util.tree_map(lambda x: x[seed], intermediates)[f"{critic_name}_{critic}"][f"{mlp_name}_0"][
                        flattened[-1][0][-3].key
                    ]["__call__"][0]
                )
                log_dict_wandb.update(
                    {
                        f"seed{seed}/critic_0-{flattened[-1][0][-3].key}-RANK": features_dict["rank"],
                    }
                )
            score_tree = jax.tree_map(estimate_neuron_score, jax.tree_util.tree_map(lambda x: x[seed], intermediates))
            neuron_score_dict = flax.traverse_util.flatten_dict(score_tree, sep="/")
            neuron_score_dict = {key.split("/")[-2]:element for key, element in neuron_score_dict.items()}
            log_dict_wandb.update(log_dict(seed, neuron_score_dict, prefix=f"critic", dormant_thre=flags.dormant_thre))

        log_dict_wandb.update({"timestep": i})
        wandb.log(log_dict_wandb)
        del batches, observations_bt, actions_bt, intermediates
