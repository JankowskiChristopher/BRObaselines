import argparse
import os
from pathlib import Path

from experiments_config import all_experiments

sbatch_script_template = """#!/bin/bash
#
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --output=./outputs/{outputs_name}

source $SCRATCH/miniconda3/bin/activate base
conda init bash
source ~/.bashrc

export PYTHONPATH=$(pwd)

touch {csv_filename}
echo -e "algo,universe,env,seed,time" >> {csv_filename}

{commands_to_run}

"""

ALGOS_TO_KWARGS = {
    'sac': {'main': 'src/pytorch_sac/train.py',
            'hydra': 1,
            'run_args_str': 'num_train_steps=105000 eval_frequency=1000000 num_seed_steps=5000 eval_0_step=False csv_writing_enabled=False'},
    'td3': {'main': 'src/td3/main.py',
            'hydra': 0,
            'run_args_str': '--max_timesteps 105000 --eval_freq 1000000 --start_timesteps 5000 --eval_0_step --csv_writing_enabled'},
    'crossq': {'main': 'src/crossq/train.py',
               'hydra': 0,
               'run_args_str': '--total_timesteps 105000 --log_freq 1000000 --eval_0_step --csv_writing_enabled --wandb_mode disabled'},
    'tdmpc2': {'main': 'src/tdmpc2/tdmpc2/train.py',
               'hydra': 1,
               'task_syntax': 1,
               'run_args_str': 'steps=52500 disable_wandb=true eval_freq=1000000 is_time_benchmark=true'},
    'bbf_dac': {'main': 'src/bbf_dac/train_parallel.py',
                'hydra': 0,
                'task_syntax': 2,
                'run_args_str': ''},
    'sr_sac': {'main': 'src/sr_sac/train_parallel.py',
               'hydra': 0,
               'task_syntax': 2,
               'run_args_str': ''},
}

ALGO_UNIVERSE_TO_CONDA = {
    ('sac', 'gym'): 'sac_td3_dm_control',
    ('td3', 'gym'): 'sac_td3_dm_control',
    ('sac', 'dm_control'): 'sac_td3_dm_control',
    ('td3', 'dm_control'): 'sac_td3_dm_control',
    ('crossq', 'dm_control'): 'crossq',  # same name but incompatible with gym
    ('tdmpc2', 'dm_control'): 'tdmpc2',
    ('bbf_dac', 'dm_control'): 'bbf_sac',
    ('sr_sac', 'dm_control'): 'bbf_sac',
    ('sac', 'metaworld'): 'sac_td3_metaworld',
    ('td3', 'metaworld'): 'sac_td3_metaworld',
    ('crossq', 'metaworld'): 'crossq_metaworld',
    ('tdmpc2', 'metaworld'): 'tdmpc2_metaworld',
    ('bbf_dac', 'metaworld'): 'bbf_dac_mw',
    ('sr_sac', 'metaworld'): 'bbf_dac_mw',
    ('sac', 'myo'): 'sac_td3_myo',
    ('td3', 'myo'): 'sac_td3_myo',
    ('crossq', 'myo'): 'crossq_myo',
    ('tdmpc2', 'myo'): 'tdmpc2_myo',
    ('bbf_dac', 'myo'): 'bbf_dac_myo',
    ('sr_sac', 'myo'): 'bbf_dac_myo',
    ('crossq', 'shimmy_dm_control'): 'crossq',
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters for running sbatch scripts.")

    # Add arguments
    parser.add_argument("--universes", nargs='+', type=str, default=['dm_control'],
                        choices=['gym', 'dm_control', 'metaworld', 'myo', 'shimmy_dm_control'],
                        help="Universe the task is in")
    parser.add_argument("--algos", nargs='+', type=str, default=["sac"],
                        choices=['sac', 'td3', 'crossq', 'tdmpc2', 'bbf_dac', 'sr_sac'],
                        help="Universe the task is in")
    parser.add_argument("--envs", nargs='+', type=str, default=None, help="Task to run. None means all.")
    parser.add_argument("--time", type=str, default="2-23:59:59", help="Max job duration.")
    parser.add_argument("--t", type=int, default=2800, help="Max job duration in minutes..")
    parser.add_argument("--gres", type=str, default="gpu:1", help="Gres for the job.")
    parser.add_argument("--mem", type=str, default=8000, help="Mam memory for the job in MB.")
    parser.add_argument("--seed", type=int, default=0, help="Start seed for the job.")
    parser.add_argument('--run_args', type=str, default='', help='Additional arguments for the run command.')
    parser.add_argument('--partition', type=str, default='plgrid-gpu-a100', help='Partition for the job.')

    return parser.parse_args()


def env_name_to_tdmpc2_env_name(env_name: str, universe: str) -> str:
    if universe == 'dm_control':
        if env_name == "finger_turn_hard":
            return "finger-turn_hard"
        return env_name.replace('_', '-')
    if universe == 'metaworld':
        # removes -v2 characters
        return 'mw-' + env_name[:-3]
    if universe == 'myo':
        return env_name
    raise ValueError(f"Universe {universe} not supported by tdmpc2 benchmark.")


def env_name_to_bbf_dac_name(env_name: str, universe: str) -> str:
    if universe == 'dm_control':
        if env_name == "finger_turn_hard":
            return "finger-turn_hard"
        return env_name.replace('_', '-')
    if universe == 'metaworld':
        return env_name + '-goal-observable'
    if universe == 'myo':
        return env_name
    raise ValueError(f"Universe {universe} not supported by bbf_sac benchmark.")


def bbf_dac_main_from_universe(universe: str, algo_name: str) -> str:
    """
    Returns the main file for bbf_dac algorithm depending on the universe.
    Works also for sr_sac as is has a very similar structure.
    :param universe: str: Universe the task is in.
    :param algo_name: str: Name of the algorithm.
    :return: str: Path to the main file for the algorithm.
    """
    assert algo_name in ['bbf_dac', 'sr_sac'], f"Function works only for bbf_dac and sr_sac, not {algo_name}."
    if universe == 'dm_control':
        return f'src/{algo_name}/train_parallel.py'
    if universe == 'metaworld':
        return f'src/{algo_name}/train_parallel_mw.py'
    if universe == 'myo':
        return f'src/{algo_name}/train_parallel_myo.py'
    raise ValueError(f"Universe {universe} not supported by bbf_sac benchmark.")


if __name__ == '__main__':
    default_params_dict = vars(parse_arguments())

    scripts_dir = "./sbatch_scripts"
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs("./time", exist_ok=True)  # just in case to not get any errors in sbatch

    print(f"Script will run with parameters {default_params_dict}\n")

    for universe in default_params_dict['universes']:
        assert universe in all_experiments.keys(), \
            f"Universe {universe} not found in experiments_config"

    script_name = ('time_' +
                   '_'.join(default_params_dict['algos'])
                   + '_'
                   + '_'.join(default_params_dict['universes'])
                   + f"_seed_{default_params_dict['seed']}"
                   + default_params_dict['run_args'].replace(' ', '').replace('--', '_')
                   + ".sh")
    output_name = script_name.replace(".sh", ".out")
    csv_name = str(Path('./time') / script_name.replace(".sh", ".csv"))

    partial_commands_to_run = []
    all_commands_to_run = []
    for universe in default_params_dict['universes']:
        # prepare names of the tasks to run on
        envs = [env['env_name'] for env in all_experiments[universe]]
        if default_params_dict['envs'] is not None:
            envs = default_params_dict['envs']
            assert len(envs) > 0, f"No experiments found for {default_params_dict['envs']}."
            print(f'Running experiments only for envs: {default_params_dict["envs"]}.')

        # testing all algos
        for algo in default_params_dict['algos']:
            algo_kwargs = ALGOS_TO_KWARGS[algo]
            use_hydra_syntax = (algo_kwargs['hydra'] == 1)
            seed = default_params_dict['seed']
            # some algos use 'task' instead of 'env'
            task_syntax = algo_kwargs.get('task_syntax', 0)
            if task_syntax == 0:
                env_or_task_str = 'env'
            elif task_syntax == 1:
                env_or_task_str = 'task'
            else:
                env_or_task_str = 'env_name'
            for env in envs:
                # convert the name of the env to syntax use in tdmpc2
                if algo == 'tdmpc2':
                    env = env_name_to_tdmpc2_env_name(env, universe)
                if algo in ['bbf_dac', 'sr_sac']:
                    env = env_name_to_bbf_dac_name(env, universe)
                    algo_kwargs['main'] = bbf_dac_main_from_universe(universe, algo)

                # prepare command to run with different syntax depending whether hydra is used ot not
                if use_hydra_syntax:
                    command_to_run = f"python {algo_kwargs['main']} {env_or_task_str}={env} universe={universe} seed={seed}"
                else:
                    command_to_run = f"python {algo_kwargs['main']} --{env_or_task_str} {env} --universe {universe} --seed {seed}"

                command_to_run = (f"start=$(date +%s)\n"
                                  f"{command_to_run} {algo_kwargs['run_args_str'] + default_params_dict['run_args']}\n"
                                  f"elapsed=$(( $(date +%s) - start ))\n"
                                  f"echo -e \"{algo},{universe},{env},{seed},$elapsed\" >> {csv_name}")
                partial_commands_to_run.append(command_to_run)

            # Handle activating and deactivating conda environments
            conda_env = ALGO_UNIVERSE_TO_CONDA[(algo, universe)]
            big_command_to_run = (f'conda activate {conda_env}\n' +
                                  f'python -m wandb disabled\n\n' +
                                  '\n'.join(partial_commands_to_run) + f'\nconda deactivate\n')

            all_commands_to_run.append(big_command_to_run)
            partial_commands_to_run = []

    print(f"{len(all_commands_to_run)} big commands to run.")

    # save script
    sbatch_dict = {**default_params_dict,
                   'commands_to_run': '\n'.join(all_commands_to_run),
                   'outputs_name': output_name,
                   'csv_filename': csv_name}  # for the csv file
    with open(f"{scripts_dir}/{script_name}", "w") as f:
        f.write(sbatch_script_template.format(**sbatch_dict))

    print(f"Running sbatch with script {script_name}\n")
    exit_code = os.system(
        f"sbatch -A plgplgplasticityrl-gpu-a100 -c 8 -t {default_params_dict['t']} --mem 10G {scripts_dir}/{script_name}")
    if exit_code != 0:
        print(f"Error in sbatch, Exit code: {exit_code}")
