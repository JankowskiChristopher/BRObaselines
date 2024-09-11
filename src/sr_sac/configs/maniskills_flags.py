from ml_collections import config_flags
from absl import flags


def set_flags():
    # DO NOT MODIFY
    flags.DEFINE_string("exp", "", "Experiment description (not actually used).")
    flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
    flags.DEFINE_string("algo_name", "SAC", "Tensorboard logging dir.")
    flags.DEFINE_integer("seed", 0, "Random seed.")
    flags.DEFINE_integer("eval_episodes", 5, "Number of episodes used for evaluation.")
    flags.DEFINE_integer("log_interval", 25000, "Logging interval.")
    flags.DEFINE_integer("eval_interval", 25000, "Eval interval.")
    flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
    flags.DEFINE_integer("checkpoint_freq", int(3e6), "Frequency at which to save agent and buffer.")
    flags.DEFINE_integer("max_steps", int(1000000), "Number of training steps.")
    flags.DEFINE_integer("replay_buffer_size", int(1000000), "Number of training steps.")
    flags.DEFINE_integer("start_training", int(10000), "Number of training steps to start training.")
    flags.DEFINE_integer("action_repeat", int(1), "Number of training steps to start training.")
    flags.DEFINE_boolean("tqdm", False, "Use tqdm progress bar.")
    flags.DEFINE_boolean("anneal_target_beta", False, "Use tqdm bar.")
    flags.DEFINE_boolean("start_from_scratch", True, "Avoid loading checkpoints.")
    flags.DEFINE_integer("num_seeds", 10, "Number of parallel seeds to run.")

    # CAN BE MODIFIED
    flags.DEFINE_integer("critic_regularization", 0, "0 - None; 1 - CDQL; 2 - TOP; 3 - GPL")
    flags.DEFINE_integer(
        "network_regularization", 1, "0 - None; 1 - LayerNorm; 2 - SpectralNorm; 3 - WeightDecay"
    )

    flags.DEFINE_integer("updates_per_step", 10, "Number of updates per step.")
    flags.DEFINE_integer("critic_depth", 2, "Number of updates per step.")
    flags.DEFINE_integer("critic_size", 512, "Number of updates per step.")
    flags.DEFINE_integer("sn_type", 0, "Number of updates per step.")
    flags.DEFINE_string("env_name", "cheetah-run", "Environment name.")

    flags.DEFINE_integer("n_atoms", 100, "Avoid loading checkpoints.")
    flags.DEFINE_integer("distributional", 1, "Avoid loading checkpoints.")
    flags.DEFINE_float("v_min_cat", 0, "Dormant neuron threshold")
    flags.DEFINE_float("v_max_cat", 200, "Dormant neuron threshold")
    flags.DEFINE_integer("actor_type", 1, "Number of updates per step.")
    flags.DEFINE_boolean("soft_critic", True, "Avoid loading checkpoints.")

    flags.DEFINE_float("dormant_thre", 0, "Dormant neuron threshold")
    flags.DEFINE_integer("dormant_calc_every", 25000, "How often calculate dormant neurons")

    config_flags.DEFINE_config_file(
        "config",
        "configs/sac_default.py",
        "File path to the training hyperparameter configuration.",
        lock_config=False,
    )
