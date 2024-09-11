#!/usr/bin/env python3
import logging
import os
import time

import hydra
import numpy as np
import torch
from logger import Logger
from replay_buffer import ReplayBuffer
from src.utils.memory_monitor import get_gpu_memory
from video import VideoRecorder

import utils
import wandb
from src.gym_wrappers.wrapper_factory import make_env, get_goal_value
from src.utils.csv_writer import CSVWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from agent import Agent

    logger.info("Agent imported")
    agent_imported = True
except ImportError:
    logger.warning("Import agent not found")
    agent_imported = False


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        logger.info(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        wandb.init(
            config=dict(cfg),
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            group=f"sac_{cfg.universe}_{cfg.wandb_group}",
            name=f"sac_{cfg.env}_{cfg.seed}_{time.time()}"
        )

        file_name = f"sac_{cfg.env}_{cfg.seed}"
        self.csv_writer = CSVWriter(f"./csv/sac/{cfg.universe}/{cfg.env}/{file_name}.csv",
                                    ["step", "reward", "seed"],
                                    csv_writing_enabled=cfg.csv_writing_enabled)

        utils.set_seed_everywhere(cfg.seed)
        if cfg.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device(cfg.device)
        else:
            self.device = torch.device('cpu')
        logger.info(f"Device: {self.device}")
        self.env = make_env(cfg.env, cfg.universe, cfg.seed)
        self.TRAIN_LOG_FREQ = 15000
        logger.info(f"Instantiating Agent")

        critic = hydra.utils.instantiate(cfg.critic_cfg,
                                         obs_dim=self.env.observation_space.shape[0],
                                         action_dim=self.env.action_space.shape[0])

        critic_target = hydra.utils.instantiate(cfg.critic_cfg,
                                                obs_dim=self.env.observation_space.shape[0],
                                                action_dim=self.env.action_space.shape[0])

        actor = hydra.utils.instantiate(cfg.actor_cfg,
                                        obs_dim=self.env.observation_space.shape[0],
                                        action_dim=self.env.action_space.shape[0])

        self.agent = hydra.utils.instantiate(cfg.agent,
                                             obs_dim=self.env.observation_space.shape[0],
                                             action_dim=self.env.action_space.shape[0],
                                             action_range=[
                                                 float(self.env.action_space.low.min()),
                                                 float(self.env.action_space.high.max()),
                                             ],
                                             actor=actor.to(self.device),
                                             critic=critic.to(self.device),
                                             critic_target=critic_target.to(self.device), )
        logger.info(f"End instantiating Agent")

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

        self.return_goal = get_goal_value(cfg.universe)

    def evaluate(self):
        average_episode_reward = 0
        avg_goal = 0. if self.return_goal else None
        random_seed = np.random.randint(0, int(1e9))

        for episode in range(self.cfg.num_eval_episodes):

            obs, info = self.env.reset(seed=episode + random_seed)
            if agent_imported:
                self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            goal = False

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)

                obs, reward, term, trun, info = self.env.step(action)
                done = (term or trun)
                self.video_recorder.record(self.env)
                episode_reward += reward
                if (avg_goal is not None) and ('success' in info.keys()):
                    goal = goal or (info['success'] > 0)

            if avg_goal is not None:
                avg_goal += float(goal)
            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        if avg_goal is not None:
            avg_goal /= self.cfg.num_eval_episodes

        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)
        wandb.log({'eval/episode_reward': average_episode_reward}, step=self.step)

        if self.return_goal:
            # goals will be logged. Myosuite and Metaworld.
            self.csv_writer.add_row({"step": self.step, "reward": avg_goal, "seed": self.cfg.seed})
            wandb.log({'eval/goal': avg_goal}, step=self.step)
        else:
            # rewards will be logged. Gymnasium and DeepMind Control.
            self.csv_writer.add_row({"step": self.step, "reward": average_episode_reward, "seed": self.cfg.seed})

    def run(self):
        episode, episode_reward = 0, 0
        done = True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if self.step > 0 and (self.step % self.TRAIN_LOG_FREQ) == 0:
                fps = round(self.TRAIN_LOG_FREQ / (time.time() - start_time), 2)
                self.logger.log('train/fps', fps, self.step)
                wandb.log({'train/fps': fps}, step=self.step)
                start_time = time.time()
                wandb.log({'train/episode_reward': episode_reward,
                           "train/episode": episode}, step=self.step)

            # evaluate agent periodically
            if self.step % self.cfg.eval_frequency == 0:
                if self.cfg.eval_0_step or self.step > 0:
                    self.logger.log('eval/episode', episode, self.step)
                    wandb.log({'eval/episode': episode}, step=self.step)
                    self.evaluate()

            self.logger.log('train/episode_reward', episode_reward, self.step)
            self.logger.log('train/episode', episode, self.step)

            if done:
                obs, info = self.env.reset(seed=self.cfg.seed)
                if agent_imported:
                    self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, term, trun, _ = self.env.step(action)

            # allow infinite bootstrap
            done = (term or trun)
            mask = not term or trun  # 0 only if term == 1 and trun == 0
            # done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, float(done), mask)

            obs = next_obs
            episode_step += 1
            self.step += 1

        # Last write to dump context
        self.csv_writer.write()

        # check GPU memory
        if self.cfg.track_GPU_memory:
            memory = get_gpu_memory()
            os.makedirs("mem", exist_ok=True)
            with open(f'mem/sac_{self.cfg.universe}_memory.csv', 'a') as f:
                f.write(f"sac,{self.cfg.universe},{self.cfg.env},{memory}\n")


@hydra.main(config_path='config', config_name='train', version_base=None)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
