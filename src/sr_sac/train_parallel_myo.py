import os

from src.utils.memory_monitor import get_gpu_memory

# os.environ["WANDB_MODE"] = "disabled"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'



import random
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from jaxrl.agents import SACLearner, DACLearner, DACSimpleLearner
from jaxrl.datasets import ParallelReplayBuffer
from jaxrl.myosuite_gym import make_env_myo
import copy
import pickle

import wandb

import logging
import tensorflow_probability.substrates.numpy as tfp

FLAGS = flags.FLAGS

## DO NOT TOUCH
flags.DEFINE_string("exp", "", "Experiment description (not actually used).")
flags.DEFINE_string("suffix", "", "Experiment name suffix.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_string('algo_name', 'SAC', 'Tensorboard logging dir.')
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 100000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("checkpoint_freq", int(4e6), "Frequency at which to save agent and buffer.")
flags.DEFINE_integer('max_steps', int(105000), 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', int(105000), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(5000),'Number of training steps to start training.')
flags.DEFINE_integer('action_repeat', int(1),'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_boolean('anneal_target_beta', False, 'Use tqdm bar.')
flags.DEFINE_boolean('start_from_scratch', True, 'Avoid loading checkpoints.')
flags.DEFINE_integer('num_seeds', 1, 'Number of parallel seeds to run.')
##

### TYM RUSZAMY
flags.DEFINE_integer("critic_regularization", 1, "0 - None; 1 - CDQL; 2 - TOP; 3 - GPL")
flags.DEFINE_integer("network_regularization", 3, "0 - None; 1 - LayerNorm; 2 - SpectralNorm; 3 - WeightDecay")

flags.DEFINE_integer("updates_per_step", 32, "Number of updates per step.")
flags.DEFINE_integer("critic_depth", 1, "Number of updates per step.")
flags.DEFINE_integer("critic_size", 256, "Number of updates per step.")
flags.DEFINE_integer("sn_type", 0, "Number of updates per step.")
flags.DEFINE_string("env_name", "cheetah-run", "Environment name.")
flags.DEFINE_string("universe", "myo", "Universe name.")

flags.DEFINE_integer("actor_type", 3, "Number of updates per step.")
flags.DEFINE_integer("actor_depth", 1, "Number of updates per step.")
flags.DEFINE_integer("actor_size", 256, "Number of updates per step.")

flags.DEFINE_integer("n_atoms", 100, "Avoid loading checkpoints.")
flags.DEFINE_integer("distributional", 0, "Avoid loading checkpoints.")
flags.DEFINE_float('v_min_cat', 0, "Dormant neuron threshold")
flags.DEFINE_float('v_max_cat', 200, "Dormant neuron threshold")
flags.DEFINE_boolean("soft_critic", True, "Avoid loading checkpoints.")
flags.DEFINE_boolean("use_reset", True, "Avoid loading checkpoints.")

flags.DEFINE_float('kl_target', 0.05, "Dormant neuron threshold")
flags.DEFINE_float('std_multiplier', 0.75, "Dormant neuron threshold")        
        
flags.DEFINE_float('dormant_thre', 0, "Dormant neuron threshold")
flags.DEFINE_integer('dormant_calc_every', 2500000, "How often calculate dormant neurons")
flags.DEFINE_boolean('track_GPU_memory', True, "Track GPU usage")


config_flags.DEFINE_config_file(
    "config", "./src/sr_sac/configs/sac_default.py", "File path to the training hyperparameter configuration.", lock_config=False
)


def log_multiple_seeds_to_wandb(step, infos, suffix:str=""):
    dict_to_log = {"timestep": step}
    for info_key in infos:
        for seed, value in enumerate(infos[info_key]):
            dict_to_log[f'seed{seed}/{info_key}{suffix}'] = value
    wandb.log(dict_to_log, step=step)

            
def evaluate_if_time_to(i, agent, eval_env, eval_returns, info, seeds, save_dir):
    if i % FLAGS.eval_interval == 0:
        if i < FLAGS.start_training:
            eval_stats = {'return': np.zeros(eval_env.num_envs)}
        else:
            eval_stats = eval_env.evaluate(agent, num_episodes=FLAGS.eval_episodes, temperature=0.0)
            #eval_stats = evaluate_mw(agent, eval_env, FLAGS.eval_episodes, episode_length=100)

        for j, seed in enumerate(seeds):
            eval_returns[j].append(
                (i, eval_stats['return'][j]))
            np.savetxt(os.path.join(save_dir, f'{seed}.txt'),
                       eval_returns[j],
                       fmt=['%d', '%.1f'])
        log_multiple_seeds_to_wandb(i, eval_stats, suffix="_eval")

def restore_checkpoint_if_existing(path, agent, replay_buffer):
    if FLAGS.start_from_scratch:
        return 1, agent, replay_buffer, [[] for _ in range(FLAGS.num_seeds)], 0
    else:
        try:
            # Just to protect against agent/replay buffer failure.
            checkpoint_agent = copy.deepcopy(agent)
            checkpoint_agent.load_state(path)
            replay_buffer.load(path)
            with open(os.path.join(path, 'step'), 'r') as f:
                start_step = int(f.read())
            with open(os.path.join(path, 'update_count'), 'r') as f:
                update_count = int(f.read())
            # Load eval returns with pickle
            with open(os.path.join(path, 'eval_returns.pkl'), 'rb') as f:
                eval_returns = pickle.load(f)
            print(f'Loaded checkpoint from {path} at step {start_step}.')
            return start_step, checkpoint_agent, replay_buffer, eval_returns, update_count
        except:
            print("No valid checkpoint found. Starting from scratch.")
            return 1, agent, replay_buffer, [[] for _ in range(FLAGS.num_seeds)], 0


def save_checkpoint(path, step, agent, replay_buffer, eval_returns, update_count):
    #agent.save_state(path)
    replay_buffer.save(path)
    with open(os.path.join(path, 'step'), 'w') as f:
        f.write(str(step))
    with open(os.path.join(path, 'eval_returns.pkl'), 'wb') as f:
        pickle.dump(eval_returns, f)
    with open(os.path.join(path, 'update_count'), 'w') as f:
        f.write(str(update_count))
    print("Saved checkpoint to {} at step {}".format(path, step))

def main(_):
    save_dir = './' + FLAGS.algo_name + '/' + FLAGS.env_name + '_' + str(FLAGS.updates_per_step) + '/' + 'CR:' + str(FLAGS.critic_regularization) + '_' + 'NR:' + str(FLAGS.network_regularization) + '/'
    wandb.init(
        config=FLAGS,
        entity="krzysztofj",
        project="DeepNets_new",
        group=f"{FLAGS.env_name}_10",
        name="SR-SA-32"
    )  
    os.makedirs(save_dir, exist_ok=True)
    
    #env = make_env_gym(env_name="HalfCheetah-v4")
    #env.reset()
    env = make_env_myo(FLAGS.env_name, FLAGS.num_seeds, FLAGS.seed)
    eval_env = make_env_myo(FLAGS.env_name, FLAGS.num_seeds, FLAGS.seed+42)
    
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    
    tfp.distributions.TransformedDistribution(tfp.distributions.Normal(0.0, 1.0), tfp.bijectors.Identity())
    logger = logging.getLogger("root")
    class CheckTypesFilter(logging.Filter):
        def filter(self, record):
            return "check_types" not in record.getMessage()
    logger.addFilter(CheckTypesFilter())
    
    # Kwargs setup
    all_kwargs = FLAGS.flag_values_dict()
    all_kwargs.update(all_kwargs.pop('config'))
    kwargs = dict(FLAGS.config)
    
    if FLAGS.algo_name == 'SAC':
        Agent = SACLearner
    if FLAGS.algo_name == 'DAC':
        Agent = DACLearner
        kwargs["kl_target"] = FLAGS.kl_target
        kwargs["std_multiplier"] = FLAGS.std_multiplier
    if FLAGS.algo_name == 'DACSimple':
        Agent = DACSimpleLearner
        kwargs["kl_target"] = FLAGS.kl_target
        kwargs["std_multiplier"] = FLAGS.std_multiplier
        
    kwargs["critic_regularization"] = FLAGS.critic_regularization
    kwargs["network_regularization"] = FLAGS.network_regularization
    kwargs["updates_per_step"] = FLAGS.updates_per_step
    kwargs["critic_depth"] = FLAGS.critic_depth
    kwargs["critic_size"] = FLAGS.critic_size
    kwargs["sn_type"] = FLAGS.sn_type
    kwargs["distributional"] = FLAGS.distributional
    kwargs["n_atoms"] = FLAGS.n_atoms
    kwargs["v_min_cat"] = FLAGS.v_min_cat
    kwargs["v_max_cat"] = FLAGS.v_max_cat
    kwargs["soft_critic"] = FLAGS.soft_critic
    
    kwargs["actor_type"] = FLAGS.actor_type
    kwargs["actor_depth"] = FLAGS.actor_depth
    kwargs["actor_size"] = FLAGS.actor_size
    
    kwargs["use_reset"] = FLAGS.use_reset


    _ = kwargs.pop('algo')
    

    agent = Agent(FLAGS.seed,
                  env.observation_space.sample()[0, np.newaxis],
                  env.action_space.sample()[0, np.newaxis], num_seeds=FLAGS.num_seeds,
                  **kwargs)
        
    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1],
                                         FLAGS.replay_buffer_size,
                                         num_seeds=FLAGS.num_seeds)
    
    observations, terms, rewards, infos = env.reset(), False, 0.0, {}    
    start_step, agent, replay_buffer, eval_returns, update_count = restore_checkpoint_if_existing(save_dir,
                                                                                                  agent,replay_buffer)
    for i in tqdm.tqdm(range(start_step, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        
        if i < FLAGS.start_training:
            actions = env.action_space.sample()
        else:
            actions = agent.sample_actions(observations, temperature=1.0)
                       
        next_observations, rewards, terms, truns, goals = env.step(actions)
    
        masks = env.generate_masks(terms, truns)
        #terms = terms.astype(np.float32)
        replay_buffer.insert(observations, actions, rewards, masks, truns, next_observations)
        observations = next_observations
        observations, terms, truns, reward_mask = env.reset_where_done(observations, terms, truns)
            
            
        if i >= FLAGS.start_training:
            if i % FLAGS.eval_interval == 0:
                validation_batch = replay_buffer.sample_parallel(256)
                mu, std = agent.get_distribution_parameters(validation_batch)
            batches = replay_buffer.sample_parallel_multibatch(FLAGS.batch_size, FLAGS.updates_per_step)
            infos = agent.update(batches, FLAGS.updates_per_step, i)
            update_count += FLAGS.updates_per_step
            #calculate_dormant_neurons(agent, i, replay_buffer, flags=FLAGS)
            if (i % FLAGS.eval_interval == 0):
                kl_b, kl_f, mu_d, std_d = agent.calculate_churn_statistics(validation_batch, mu, std)
                infos_churn = {'churn_kl_b': kl_b,
                               'churn_kl_f': kl_f,
                               'churn_mu_diff': mu_d,
                               'churn_std_diff': std_d,
                               }
                infos = {**infos, **infos_churn}
            if i % FLAGS.log_interval == 0:
                log_multiple_seeds_to_wandb(i, infos)

        evaluate_if_time_to(i, agent, eval_env, eval_returns, infos, list(range(FLAGS.seed, FLAGS.seed+FLAGS.num_seeds)), save_dir)
        # check GPU memory
        if FLAGS.track_GPU_memory:
            memory = get_gpu_memory()
            os.makedirs("mem", exist_ok=True)
            with open(f'mem/sr-sac_{FLAGS.universe}_memory.csv', 'a') as f:
                f.write(f"sr-sac,{FLAGS.universe},{FLAGS.env_name},{memory}\n")

if __name__ == '__main__':
    app.run(main)
