from copy import deepcopy
import warnings

import gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from envs.dmcontrol import make_env as make_dm_control_env
except Exception as e:
	print(e)
	make_dm_control_env = missing_dependencies
try:
	from envs.maniskill import make_env as make_maniskill_env
except Exception as e:
	print(e)
	make_maniskill_env = missing_dependencies
try:
	from envs.metaworld import make_env as make_metaworld_env
except Exception as e:
	print("STH BAD IN METAWORLD ENVIRONMENT")
	print(e)
	make_metaworld_env = missing_dependencies
try:
	from envs.myosuite import make_env as make_myosuite_env
except Exception as e:
	print(e)
	make_myosuite_env = missing_dependencies


warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env
	

def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)
	universe = cfg.universe

	if cfg.multitask:
		env = make_multitask_env(cfg)

	else:
		env = None
		for fn, _universe in zip([make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env],
								['dm_control', 'maniskill2', 'metaworld', 'myo']):
			try:
				env = fn(cfg)
				universe = _universe
			except ValueError as e:
				print('Trying next environment...')
				print(e)
				pass
		if env is None:
			raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
		env = TensorWrapper(env)
	if cfg.get('obs', 'state') == 'rgb':
		env = PixelWrapper(cfg, env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	# Was changed for benchmarking
	if cfg.is_time_benchmark:
		cfg.seed_steps = 5000 if universe == 'myo' else 2500  # myo is the only env without action repeat
		print(f'Setting seed steps to {cfg.seed_steps} for benchmarking...')

		if universe == 'myo':
			cfg.steps = cfg.steps * 2
			print(f'Multiplying steps by 2 for myo environment: {cfg.steps}')
	else:
		cfg.seed_steps = max(1000, 5*cfg.episode_length)
		print('Automatic seed steps:', cfg.seed_steps)
	return env
