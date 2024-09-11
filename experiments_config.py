from typing import Dict, List, Union

Config = Dict[str, Union[str, int]]
all_experiments: Dict[str, List[Config]] = {
    # OpenAI gym environments. Memory when 5 seeds running. Runs on GPU.
    "gym": [
        {"env_name": "HalfCheetah-v4"},  # <1.5GB in TD3 and SAC
        {"env_name": "Hopper-v4"},  # <1.5GB in TD3 and SAC
        {"env_name": "Humanoid-v4"},  # <4.1GB in SAC, 6.8GB in CrossQ
        {"env_name": "Ant-v4"},  # <1.5GB in TD3
        {"env_name": "Pendulum-v1"},  # <1GB in TD3, 1.25GB in SAC
    ],
    # Shimmy wrapper for DeepMind control environments. Mem set for 2 seeds, comments for 1 seed.
    "shimmy_dm_control": [
        # {"env_name": "dm_control/acrobot-swingup", "mem": 9500},  # 0.6 GB in TD3, 0.5GB in SAC, 1.5GB in CrossQ.
        # {"env_name": "dm_control/cheetah-run", "mem": 11000},  # 1.6 GB in CrossQ
        {"env_name": "dm_control/dog-trot", "mem": 25000},  # 4.5 GB in TD3, 2.5GB in SAC, 3.5GB in CrossQ.
        {"env_name": "dm_control/dog-run", "mem": 25000},  # 4.5 GB in TD3, 2.5GB in SAC, 3.5GB in CrossQ.
        {"env_name": "dm_control/dog-walk", "mem": 25000},  # 4.5 GB in TD3, 2.5GB in SAC, 3.5GB in CrossQ.
        {"env_name": "dm_control/dog-stand", "mem": 25000},  # 4.5 GB in TD3, 2.5GB in SAC, 3.5GB in CrossQ. # TODO was 21000
        # {"env_name": "dm_control/fish-swim", "mem": 9500},  # 0.8 GB in TD3, 0.6GB in SAC, 1.6GB in CrossQ.
        # {"env_name": "dm_control/hopper-hop", "mem": 9500},  # 0.8 GB in TD3, 0.6GB in SAC, 1.6GB in CrossQ.
        # {"env_name": "dm_control/humanoid-run", "mem": 12500},  # 2GB in CrossQ
        {"env_name": "dm_control/humanoid-stand", "mem": 25000},  # 1.5 GB in TD3, 1GB in SAC, 2GB in CrossQ.
        # {"env_name": "dm_control/humanoid-walk", "mem": 12500},  # 1.5 GB in TD3, 1GB in SAC, 2GB in CrossQ.
        # {"env_name": "dm_control/pendulum-swingup", "mem": 9500},  # 1.4 GB in CrossQ
        # {"env_name": "dm_control/quadruped-run", "mem": 12500},  # 1.5 GB in TD3, 1GB in SAC, 2.2GB in CrossQ.
        # {"env_name": "dm_control/finger-turn_hard", "mem": 9500},  # 0.6 GB in TD3, 0.5GB in SAC, 1.5GB in CrossQ.
        # {"env_name": "dm_control/walker-run", "mem": 11000},  # 0.8 GB in TD3, 0.6GB in SAC, 1.6GB in CrossQ.
    ],
    # DeepMind control environments (memory in sufficient when run on Ares)
    # Experiments were run on CPU, so memory usage is higher than on GPU.
    # Mem set for 4 seeds, default where not set.
    "dm_control": [
        {"env_name": "acrobot_swingup"},  # 0.6 GB in TD3, 0.5GB in SAC, 1.5GB in CrossQ.
        {"env_name": "cheetah_run", "mem": 11000},  # 1.6 GB in CrossQ
        {"env_name": "dog_trot", "mem": 19000},  # 4.5 GB in TD3, 2.5GB in SAC, 3.5GB in CrossQ.
        {"env_name": "dog_run", "mem": 19000},  # 4.5 GB in TD3, 2.5GB in SAC, 3.5GB in CrossQ.
        {"env_name": "dog_walk", "mem": 19000},  # 4.5 GB in TD3, 2.5GB in SAC, 3.5GB in CrossQ.
        {"env_name": "dog_stand", "mem": 19000},  # 4.5 GB in TD3, 2.5GB in SAC, 3.5GB in CrossQ.
        {"env_name": "fish_swim"},  # 0.8 GB in TD3, 0.6GB in SAC, 1.6GB in CrossQ.
        {"env_name": "hopper_hop"},  # 0.8 GB in TD3, 0.6GB in SAC, 1.6GB in CrossQ.
        {"env_name": "humanoid_run", "mem": 12000},  # 2GB in CrossQ
        {"env_name": "humanoid_stand", "mem": 12000},  # 1.5 GB in TD3, 1GB in SAC, 2GB in CrossQ.
        {"env_name": "humanoid_walk", "mem": 12000},  # 1.5 GB in TD3, 1GB in SAC, 2GB in CrossQ.
        {"env_name": "pendulum_swingup"},  # 1.4 GB in CrossQ
        {"env_name": "quadruped_run", "mem": 12000},  # 1.5 GB in TD3, 1GB in SAC, 2.2GB in CrossQ.
        {"env_name": "walker_run"},  # 0.8 GB in TD3, 0.6GB in SAC, 1.6GB in CrossQ.
        {"env_name": "finger_turn_hard"},  # 0.6 GB in TD3, 0.5GB in SAC, 1.5GB in CrossQ.
    ],
    # MyoSuite environments
    "myo": [
        {"env_name": "myo-reach"},
        {"env_name": "myo-reach-hard"},
        {"env_name": "myo-pose"},
        {"env_name": "myo-pose-hard"},
        {"env_name": "myo-obj-hold"},
        {"env_name": "myo-obj-hold-hard"},
        {"env_name": "myo-key-turn"},
        {"env_name": "myo-key-turn-hard"},
        {"env_name": "myo-pen-twirl"},
        {"env_name": "myo-pen-twirl-hard"},
    ],
    # MetaWorld environments
    "metaworld": [
        {"env_name": "push-v2"},
        {"env_name": "sweep-v2"},
        {"env_name": "stick-pull-v2"},
        {"env_name": "hand-insert-v2"},
        {"env_name": "assembly-v2"},
        {"env_name": "reach-v2"},
        {"env_name": "coffee-push-v2"},
        {"env_name": "coffee-pull-v2"},
        {"env_name": "basketball-v2"},
        {"env_name": "hammer-v2"},
        {"env_name": "push-back-v2"},
        {"env_name": "pick-place-wall-v2"},
        {"env_name": "lever-pull-v2"},
        {"env_name": "disassemble-v2"},
        {"env_name": "button-press-v2"},
        # {"env_name": "box-close-v2"}, # Additional environments
        # {"env_name": "drawer-open-v2"},
        # {"env_name": "stick-push-v2"},
    ],
}

