import logging
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_env(env_name: str, universe: str = "gym", seed: Optional[int] = None):
    # OpenAI Gym - Gymnasium version
    if universe == "gym":
        import gymnasium as gym
        from gymnasium.wrappers import RescaleAction

        from .gymnasium_wrapper import GymnasiumWrapper

        logger.info(f"Creating Gymnasium environment with name: {env_name}")
        env = gym.make(env_name)
        env = GymnasiumWrapper(env)
        env = RescaleAction(env, min_action=-1, max_action=1)
        return env
    # DeepMind Control Suite
    elif universe == "dm_control":
        from dm_control import suite

        from .dmc_wrapper import DMCWrapper

        if env_name == 'ball_in_cup_catch':
            domain_name = 'ball_in_cup'
            task_name = 'catch'
        else:
            domain_name = env_name.split('_')[0]
            task_name = '_'.join(env_name.split('_')[1:])
        if (domain_name, task_name) not in suite.ALL_TASKS:
            raise ValueError('Unknown task:', task_name)

        assert seed is not None, "Seed must be provided for dm_control environments"

        logger.info(f"Creating dm_control environment with domain: {domain_name} and task: {task_name}")
        env = DMCWrapper(domain_name, task_name, seed)
        return env
    # MyoSuite
    elif universe == "myo":
        import gym
        import myosuite  # IMPORTANT. Necessary import to register environments even though IDE does not see it

        from .myo_wrapper import MYOSUITE_TASKS, MyoSuiteWrapper

        assert env_name in MYOSUITE_TASKS, f"Unknown MyoSuite task: {env_name}"
        logger.info(f"Creating MyoSuite environment with task: {env_name}")
        env = gym.make(MYOSUITE_TASKS[env_name])
        env.seed(seed)
        # Do not rescale actions here as such action wrapper is not available in the Gym version supported by MyoSuite.
        env = MyoSuiteWrapper(env)
        return env
    # MetaWorld
    elif universe == "metaworld":
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

        from .metaworld_wrapper import MetaWorldWrapper

        logger.info(f"Creating MetaWorld environment with task: {env_name}")

        if not env_name.endswith("goal-observable"):
            env_name += "-goal-observable"

        env_constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
        env = env_constructor(seed=seed)
        env = MetaWorldWrapper(env)

        return env
    # ManiSkill2
    elif universe == "maniskill2":
        import gymnasium as gym
        import mani_skill2.envs  # tutorial states that important when multiprocessing

        from .maniskill2_wrapper import MANISKILL_TASKS, Maniskill2Wrapper

        logger.info(f"Creating ManiSkill2 environment with task: {env_name}")

        obs_mode = "state"
        control_mode = Maniskill2Wrapper.get_camera_pos(env_name)

        if env_name not in MANISKILL_TASKS.keys():
            raise ValueError(f"Unknown ManiSkill2 task: {env_name}. Supported {list(MANISKILL_TASKS.keys())}")

        env = gym.make(MANISKILL_TASKS[env_name],
                       obs_mode=obs_mode,
                       control_mode=control_mode,
                       max_episode_steps=200,
                       render_camera_cfgs=dict(width=384, height=384))
        env = Maniskill2Wrapper(env, max_t=200)
        return env

    else:
        raise ValueError(f"Universe {universe} not recognized and not supported")


def get_goal_value(universe: str) -> bool:
    if universe in ["myo", "metaworld", "maniskill2"]:
        return True
    if universe in ["dm_control", "gym"]:
        return False
    raise ValueError(f"Universe {universe} not recognized and not supported")