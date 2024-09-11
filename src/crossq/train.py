import os

from src.utils.memory_monitor import get_gpu_memory

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['WANDB_DIR'] = '/tmp'

import argparse
import time

import gymnasium as gym
import jax
import numpy as np
from sbx import SAC
from stable_baselines3.common.callbacks import CallbackList
from sbx.sac.actor_critic_evaluation_callback import (CriticBiasCallback,
                                                      EvalCallback)
from sbx.sac.utils import *
from shimmy.registration import DM_CONTROL_SUITE_ENVS # important, do not delete
from stable_baselines3.common.env_util import make_vec_env  # Used in CrossQ code
from wandb.integration.sb3 import WandbCallback

import wandb
from src.gym_wrappers.wrapper_factory import make_env
from src.utils.csv_writer import CSVWriter

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=False, default="HumanoidStandup-v4", help="Set Environment.")
parser.add_argument('--universe', type=str, required=False, default='gym',
                    choices=['gym', 'dm_control', 'metaworld', 'myo', 'maniskill2', "shimmy_dm_control"], help='Set Universe')
parser.add_argument("--algo", type=str, required=False, default='crossq',
                    choices=['crossq', 'sac', 'redq', 'droq', 'td3'],
                    help="algorithm to use (essentially a named hyperparameter set for the base SAC algorithm)")
parser.add_argument("--seed", type=int, required=False, default=1, help="Set Seed.")
parser.add_argument("--log_freq", type=int, required=False, default=25000, help="how many times to log during training")

parser.add_argument('--wandb_entity', type=str, required=False, default='krzysztofj', help='your wandb entity name')
parser.add_argument('--wandb_project', type=str, required=False, default='crossQ', help='wandb project name')
parser.add_argument("--wandb_mode", type=str, required=False, default='online', choices=['disabled', 'online'],
                    help="enable/disable wandb logging")
parser.add_argument("--eval_qbias", type=int, required=False, default=0, choices=[0, 1],
                    help="enable/diasble q bias evaluation (expensive; experiments will run much slower)")

parser.add_argument("--adam_b1", type=float, required=False, default=0.5, help="adam b1 hyperparameter")
parser.add_argument("--bn", type=float, required=False, default=False, choices=[0, 1],
                    help="Use batch norm layers in the actor and critic networks")
parser.add_argument("--bn_momentum", type=float, required=False, default=0.99, help="batch norm momentum parameter")
parser.add_argument("--bn_mode", type=str, required=False, default='brn_actor',
                    help="batch norm mode (bn / brn / brn_actor). brn_actor also uses batch renorm in the actor network")
parser.add_argument("--critic_activation", type=str, required=False, default='relu', help="critic activation function")
parser.add_argument("--crossq_style", type=float, required=False, default=1, choices=[0, 1],
                    help="crossq style joint forward pass through critic network")
parser.add_argument("--dropout", type=int, required=False, default=0, choices=[0, 1],
                    help="whether to use dropout for SAC")
parser.add_argument("--ln", type=float, required=False, default=False, choices=[0, 1],
                    help="layernorm in critic network")
parser.add_argument("--lr", type=float, required=False, default=1e-3, help="actor and critic learning rate")
parser.add_argument("--n_critics", type=int, required=False, default=2, help="number of critics to use")
parser.add_argument("--n_neurons", type=int, required=False, default=256,
                    help="number of neurons for each critic layer")
parser.add_argument("--policy_delay", type=int, required=False, default=1,
                    help="policy is updated after this many critic updates")
parser.add_argument("--tau", type=float, required=False, default=0.005, help="target network averaging")
parser.add_argument("--utd", type=int, required=False, default=1, help="number of critic updates per env step")
parser.add_argument("--total_timesteps", type=int, required=False, default=int(1e6) + 100,
                    help="total number of training steps")
parser.add_argument("--max_timesteps", type=int, required=False, default=None,
                    help="alternative syntax for total number of training steps")

parser.add_argument("--bnstats_live_net", type=int, required=False, default=0, choices=[0, 1],
                    help="use bn running statistics from live network within the target network")

# time benchmarking
parser.add_argument("--eval_0_step", action='store_false')  # Whether to evaluate at step 0
parser.add_argument("--csv_writing_enabled", action='store_false')
parser.add_argument("--track_GPU_memory", action='store_true')

experiment_time = time.time()
args = parser.parse_args()

print(f"\n\n\nStarting CrossQ with time benchmark values eval_0_step: {args.eval_0_step} and csv writing enabled: {args.csv_writing_enabled}.\n\n\n")

seed = args.seed
args.algo = str.lower(args.algo)
args.bn = bool(args.bn)
args.crossq_style = bool(args.crossq_style)
args.tau = float(args.tau) if not args.crossq_style else 1.0
args.bn_momentum = float(args.bn_momentum) if args.bn else 0.0
dropout_rate, layer_norm = None, False
policy_q_reduce_fn = jax.numpy.min
net_arch = {'pi': [256, 256], 'qf': [args.n_neurons, args.n_neurons]}

total_timesteps = int(args.total_timesteps)
if args.max_timesteps is not None:
    print(f"Using alternative syntax for total number of training steps: {args.max_timesteps}.")
    total_timesteps = (int(args.max_timesteps) + 1) # alternative syntax. Add small number to have last eval
    args.total_timesteps = total_timesteps
eval_freq = args.log_freq

print(f'CrossQ will run with number of timesteps: {total_timesteps} and evaluation frequency: {eval_freq}.')

td3_mode = False

# Logging results
file_name = f"{args.algo}_{args.env}_{args.seed}"
file_name = file_name.replace("/", "_") # necessary as Shimmy uses / in env names.
env_str = args.env.replace("/", "_")
file_path = f"./csv/{args.algo}/{args.universe}/{env_str}/{file_name}.csv"
csv_writer = CSVWriter(file_path, ["step", "reward", "seed"], csv_writing_enabled=args.csv_writing_enabled)

if args.algo == 'droq':
    dropout_rate = 0.01
    layer_norm = True
    policy_q_reduce_fn = jax.numpy.mean
    args.n_critics = 2
    # args.adam_b1 = 0.9  # adam default
    args.adam_b2 = 0.999  # adam default
    args.policy_delay = 20
    args.utd = 20
    group = f'DroQ_{args.env}_bn({args.bn})_ln{(args.ln)}_xqstyle({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_Adam({args.adam_b1})_Q({net_arch["qf"][0]})'

elif args.algo == 'redq':
    policy_q_reduce_fn = jax.numpy.mean
    args.n_critics = 10
    # args.adam_b1 = 0.9  # adam default
    args.adam_b2 = 0.999  # adam default
    args.policy_delay = 20
    args.utd = 20
    group = f'REDQ_{args.env}_bn({args.bn})_ln{(args.ln)}_xqstyle({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_Adam({args.adam_b1})_Q({net_arch["qf"][0]})'

elif args.algo == 'td3':
    # With the right hyperparameters, this here can run all the above algorithms
    # and ablations.
    td3_mode = True
    layer_norm = args.ln
    if args.dropout:
        dropout_rate = 0.01
    group = f'TD3_{args.env}_bn({args.bn}/{args.bn_momentum}/{args.bn_mode})_ln{(args.ln)}_xq({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_A{args.adam_b1}_Q({net_arch["qf"][0]})_l{args.lr}'

elif args.algo == 'sac':
    # With the right hyperparameters, this here can run all the above algorithms
    # and ablations.
    layer_norm = args.ln
    if args.dropout:
        dropout_rate = 0.01
    group = f'SAC_{args.env}_bn({args.bn}/{args.bn_momentum}/{args.bn_mode})_ln{(args.ln)}_xq({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_A{args.adam_b1}_Q({net_arch["qf"][0]})_l{args.lr}'

elif args.algo == 'crossq':
    args.adam_b1 = 0.5
    args.policy_delay = 3
    args.n_critics = 2
    args.utd = 1  # nice
    net_arch["qf"] = [2048, 2048]  # wider critics
    args.bn = True  # use batch norm
    args.bn_momentum = 0.99
    args.crossq_style = True  # with a joint forward pass
    args.tau = 1.0  # without target networks
    group = f'CrossQ_{args.env}'

else:
    raise NotImplemented

args_dict = vars(args)
args_dict.update({
    "dropout_rate": dropout_rate,
    "layer_norm": layer_norm
})

with wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=f"{args.algo}_{args.universe}_{args.env}",
        name=f"{args.algo}_{args.env}_{args.seed}_{time.time()}",
        tags=[],
        sync_tensorboard=True,
        config=args_dict,
        settings=wandb.Settings(start_method="fork") if is_slurm_job() else None,
        mode=args.wandb_mode
) as wandb_run:
    # SLURM maintenance
    if is_slurm_job():
        print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
        wandb_run.summary['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID')

    if args.universe not in ['gym', 'shimmy_dm_control']:
        print(f"Using universe: {args.universe}")
        # Use custom wrappers not used in CrossQ.
        training_env = make_env(args.env, args.universe, seed=seed)
    else:
        # Use original gym environment as in CrossQ.
        print(f"Using original CrossQ gym environment: {args.env}")
        training_env = gym.make(args.env)

    if args.env == 'dm_control/humanoid-stand':
        training_env.observation_space['head_height'] = gym.spaces.Box(-np.inf, np.inf, (1,))
    if args.env == 'dm_control/fish-swim':
        training_env.observation_space['upright'] = gym.spaces.Box(-np.inf, np.inf, (1,))

    import optax

    if isinstance(training_env.observation_space, gym.spaces.Dict):
        print("Using MultiInputPolicy as type of observation space is a dict.")

    model = SAC(
        "MultiInputPolicy" if isinstance(training_env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        training_env,
        policy_kwargs=dict({
            'activation_fn': activation_fn[args.critic_activation],
            'layer_norm': layer_norm,
            'batch_norm': bool(args.bn),
            'batch_norm_momentum': float(args.bn_momentum),
            'batch_norm_mode': args.bn_mode,
            'dropout_rate': dropout_rate,
            'n_critics': args.n_critics,
            'net_arch': net_arch,
            'optimizer_class': optax.adam,
            'optimizer_kwargs': dict({
                'b1': args.adam_b1,
                'b2': 0.999  # default
            })
        }),
        gradient_steps=args.utd,
        policy_delay=args.policy_delay,
        crossq_style=bool(args.crossq_style),
        td3_mode=td3_mode,
        use_bnstats_from_live_net=bool(args.bnstats_live_net),
        policy_q_reduce_fn=policy_q_reduce_fn,
        learning_starts=5000,
        learning_rate=args.lr,
        qf_learning_rate=args.lr,
        tau=args.tau,
        gamma=0.99 if not args.env == 'Swimmer-v4' else 0.9999,
        verbose=0,
        buffer_size=1_000_000,
        seed=seed,
        stats_window_size=1,  # don't smooth the episode return stats over time
        tensorboard_log=f"logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/",
    )

    # Create log dir where evaluation results will be saved
    eval_log_dir = f"./eval_logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/eval/"
    qbias_log_dir = f"./eval_logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/qbias/"
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(qbias_log_dir, exist_ok=True)

    # Create callback that evaluates agent. Use original Gym wrapper or custom.
    eval_callback = EvalCallback(
        make_env(args.env, args.universe, seed=seed) if args.universe not in ['gym', 'shimmy_dm_control'] else make_vec_env(args.env, n_envs=1,
                                                                                                 seed=seed),
        jax_random_key_for_seeds=args.seed,
        best_model_save_path=None,
        log_path=eval_log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=10 if args.eval_0_step else 0,
        deterministic=True,
        render=False,
        csv_writer=csv_writer,
    )

    # Callback that evaluates q bias according to the REDQ paper.
    q_bias_callback = CriticBiasCallback(
        make_env(args.env, args.universe, seed=seed) if args.universe not in ['gym', 'shimmy_dm_control'] else make_vec_env(args.env, n_envs=1,
                                                                                                 seed=seed),
        jax_random_key_for_seeds=args.seed,
        best_model_save_path=None,
        log_path=qbias_log_dir, eval_freq=eval_freq,
        n_eval_episodes=1, render=False
    )

    callback_list = CallbackList(
        [eval_callback, q_bias_callback, WandbCallback(verbose=0, )] if args.eval_qbias else
        [eval_callback, WandbCallback(verbose=0, )]
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback_list)

    # flush the csv writer
    csv_writer.write()

    # check GPU memory
    if args.track_GPU_memory:
        memory = get_gpu_memory()
        os.makedirs("mem", exist_ok=True)
        with open(f'mem/crossq_{args.universe}_memory.csv', 'a') as f:
            f.write(f"crossq,{args.universe},{args.env},{memory}\n")
