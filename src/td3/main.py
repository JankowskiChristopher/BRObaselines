import argparse
import os
import time
from typing import Tuple, Optional

import numpy as np
import TD3
import torch

import utils
import wandb
from src.gym_wrappers.wrapper_factory import make_env, get_goal_value
from src.utils.csv_writer import CSVWriter
import logging

from src.utils.memory_monitor import get_gpu_memory

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy,
                env_name: str,
                universe_name: str,
                seed: int,
                return_goal: bool,
                eval_episodes: int = 10) -> Tuple[float, float, Optional[float]]:
    eval_env = make_env(env_name, universe_name, seed=seed)
    avg_reward = 0.
    avg_goal = 0. if return_goal else None
    rewards = []
    random_seed = np.random.randint(0, int(1e9))

    for eval_seed in range(eval_episodes):
        state, info = eval_env.reset(seed=eval_seed + random_seed)
        done = False
        goal = False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, dw, tr, info = eval_env.step(action)
            done = (dw or tr)  # added
            avg_reward += reward
            rewards.append(reward)
            if (avg_goal is not None) and ('success' in info.keys()):
                goal = goal or (info['success'] > 0)

        if avg_goal is not None:
            avg_goal += float(goal)

    avg_reward /= eval_episodes
    if avg_goal is not None:
        avg_goal /= eval_episodes
    std_reward = np.std(np.array(rewards))

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward, std_reward, avg_goal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v4")  # OpenAI gym environment name
    parser.add_argument("--universe", default="gym")  # Universe of env. gym, dm_control or myo.
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=25000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--wandb_entity", default="krzysztofj")  # wandb_username
    parser.add_argument("--csv_writing_enabled", action='store_false')  # In time benchmarking do not save to CSV
    parser.add_argument("--eval_0_step", action='store_false')  # In time benchmarking do not evaluate at 0 step
    parser.add_argument("--track_GPU_memory", action='store_true')  # Track GPU memory usage

    args = parser.parse_args()
    logger.info(f"Starting TD3 with time benchmark variables csv writing enabled: {args.csv_writing_enabled} "
                f"and eval_0_step: {args.eval_0_step}.")

    wandb_username = args.wandb_entity

    wandb.init(
        config=dict(vars(args)),
        entity=wandb_username,
        project="BaselinesBBFDAC",
        group=f"td3_{args.universe}_{args.env}",
        name=f"td3_{args.env}_{args.seed}_{time.time()}"
    )

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    csv_writer = CSVWriter(f"./csv/td3/{args.universe}/{args.env}/{file_name}.csv",
                           ["step", "reward", "seed"],
                           csv_writing_enabled=args.csv_writing_enabled)
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    os.makedirs("results", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    env = make_env(args.env, args.universe, seed=args.seed)

    # Set seeds
    # env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    log_goal = get_goal_value(args.universe)
    if args.eval_0_step:
        eval_reward, eval_reward_std, eval_goal = eval_policy(policy, args.env, args.universe, args.seed, log_goal)
        if log_goal:
            # goals will be logged. Myosuite and Metaworld.
            csv_writer.add_row({"step": 0, "reward": eval_goal, "seed": args.seed})
        else:
            # rewards will be logged. Gymnasium and DeepMind Control.
            csv_writer.add_row({"step": 0, "reward": eval_reward, "seed": args.seed})

    state, info = env.reset(seed=args.seed + 100)
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    start_time = time.perf_counter()
    LOG_TRAIN_FREQ = 15000
    last_train_log = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, term, trun, _ = env.step(action)
        done = (term or trun)
        done_float = float(not term or trun)  # this is already mask = (1 - done) in algorithm.

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_float)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # Log training info
            if (t - last_train_log) >= LOG_TRAIN_FREQ:
                print(f"Total T: {t + 1} "
                      f"Episode Num: {episode_num} "
                      f"Timesteps: {episode_timesteps} "
                      f"Reward: {episode_reward:.3f}")

                wandb.log({"train/reward": episode_reward,
                           "train/episode_timesteps": episode_timesteps,
                           "train/episode_num": episode_num}, step=t)
                last_train_log = t

            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            # Reset environment
            state, info = env.reset(seed=args.seed + 100)
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t > 0) and (t % args.eval_freq == 0):
            evaluation_result, eval_result_std, goal_result = eval_policy(policy, args.env, args.universe, args.seed,
                                                                          log_goal)

            if log_goal:
                csv_writer.add_row({"step": t, "reward": goal_result, "seed": args.seed})
                wandb.log({"eval/goal_avg": goal_result}, step=t)
            else:
                csv_writer.add_row({"step": t, "reward": evaluation_result, "seed": args.seed})

            wandb.log({"eval/reward_avg": evaluation_result,
                       "eval/reward_std": eval_result_std}, step=t)

            if args.save_model:
                policy.save(f"./models/{file_name}")

        if (t > 0) and (t % LOG_TRAIN_FREQ == 0):
            elapsed_time = time.perf_counter() - start_time
            fps = round(LOG_TRAIN_FREQ / elapsed_time, 2)
            start_time = time.perf_counter()
            wandb.log({"train/fps": fps}, step=t)

    # Write if some data is still present in the writer.
    csv_writer.write()

    # check GPU memory
    if args.track_GPU_memory:
        memory = get_gpu_memory()
        os.makedirs("mem", exist_ok=True)
        with open(f'mem/td3_{args.universe}_memory.csv', 'a') as f:
            f.write(f"td3,{args.universe},{args.env},{memory}\n")
