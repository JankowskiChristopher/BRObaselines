import argparse
import os
from pathlib import Path

from experiments_config import all_experiments

sbatch_script_template = """#!/bin/bash
#
#SBATCH --partition=common
#SBATCH --qos=16gpu3d
#SBATCH --gres={gres}
#SBATCH --output=./outputs/{outputs_name}
#SBATCH --time={time}
#SBATCH --mem={mem}

source ~/miniconda3/bin/activate base
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
    # TODO is order not to eval set eval to 1000000
}

ALGO_UNIVERSE_TO_CONDA = {
    ('sac', 'gym'): 'sac_td3_dm_control',
    ('td3', 'gym'): 'sac_td3_dm_control',
    ('crossq', 'gym'): 'crossq_gym',  # same name but incompatible with dm_control
    ('sac', 'dm_control'): 'sac_td3_dm_control',
    ('td3', 'dm_control'): 'sac_td3_dm_control',
    ('crossq', 'dm_control'): 'crossq_gym',  # same name but incompatible with gym
    ('tdmpc2', 'dm_control'): 'tdmpc2',
    ('sac', 'metaworld'): 'sac_td3_metaworld',
    ('td3', 'metaworld'): 'sac_td3_metaworld',
    ('crossq', 'metaworld'): 'crossq',
    ('tdmpc2', 'metaworld'): 'tdmpc2',  # same name but incompatible other (pip install as whole env is 11GB)
    ('sac', 'myo'): 'sac_td3_myo',
    ('td3', 'myo'): 'sac_td3_myo',
    ('crossq', 'myo'): 'crossq_myo',
    ('tdmpc2', 'myo'): 'tdmpc2',  # same name but incompatible other (pip install as whole env is 11GB)
    ('crossq', 'shimmy_dm_control'): 'crossq',
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters for running sbatch scripts.")

    # Add arguments
    parser.add_argument("--universes", nargs='+', type=str, default=['dm_control'],
                        choices=['gym', 'dm_control', 'metaworld', 'myo', 'shimmy_dm_control'],
                        help="Universe the task is in")
    parser.add_argument("--algos", nargs='+', type=str, default=["sac"],
                        choices=['sac', 'td3', 'crossq', 'tdmpc2', 'bbf_dac'],
                        help="Universe the task is in")
    parser.add_argument("--envs", nargs='+', type=str, default=None, help="Task to run. None means all.")
    parser.add_argument("--time", type=str, default="2-23:59:59", help="Max job duration.")
    parser.add_argument("--gres", type=str, default="gpu:titanv:1", help="Gres for the job.")
    parser.add_argument("--mem", type=str, default=8000, help="Mam memory for the job in MB.")
    parser.add_argument("--seed", type=int, default=0, help="Start seed for the job.")

    return parser.parse_args()


def env_name_to_tdmpc2_env_name(env_name: str, universe: str) -> str:
    if universe == 'dm_control':
        return env_name.replace('_', '-')
    if universe == 'metaworld':
        # removes -v2 characters
        return env_name[:-3]
    if universe == 'myo':
        return env_name
    raise ValueError(f"Universe {universe} not supported by tdmpc2 benchmark.")


if __name__ == '__main__':
    default_params_dict = vars(parse_arguments())

    scripts_dir = "./sbatch_scripts"
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs("./time", exist_ok=True)  # just in case to not get any errors in sbatch

    print(f"Script will run with parameters {default_params_dict}\n")

    for universe in default_params_dict['universes']:
        assert universe in all_experiments.keys(), \
            f"Universe {universe} not found in experiments_config"

    script_name = 'time_' + '_'.join(default_params_dict['algos']) + '_' + '_'.join(
        default_params_dict['universes']) + f"_seed_{default_params_dict['seed']}" + ".sh"
    output_name = script_name.replace(".sh", ".out")
    csv_name = str(Path('./time') / script_name.replace(".sh", ".csv"))

    partial_commands_to_run = []
    all_commands_to_run = []
    for universe in default_params_dict['universes']:
        # prepare names of the tasks to run on
        envs = [env['env_name'] for env in all_experiments[universe]]
        if default_params_dict['envs'] is not None:
            envs = default_params_dict['env']
            assert len(envs) > 0, f"No experiments found for {default_params_dict['envs']}."
            print(f'Running experiments only for envs: {default_params_dict["envs"]}.')

        # testing all algos
        for algo in default_params_dict['algos']:
            algo_kwargs = ALGOS_TO_KWARGS[algo]
            use_hydra_syntax = (algo_kwargs['hydra'] == 1)
            seed = default_params_dict['seed']
            # some algos use 'task' instead of 'env'
            task_syntax = algo_kwargs.get('task_syntax', 0)
            env_or_task_str = 'task' if task_syntax else 'env'
            for env in envs:
                # convert the name of the env to syntax use in tdmpc2
                if algo == 'tdmpc2':
                    env = env_name_to_tdmpc2_env_name(env, universe)

                # prepare command to run with different syntax depending whether hydra is used ot not
                if use_hydra_syntax:
                    command_to_run = f"python {algo_kwargs['main']} {env_or_task_str}={env} universe={universe} seed={seed}"
                else:
                    command_to_run = f"python {algo_kwargs['main']} --{env_or_task_str} {env} --universe {universe} --seed {seed}"

                command_to_run = (f"start=$(date +%s)\n"
                                  f"{command_to_run} {algo_kwargs['run_args_str']}\n"
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
    exit_code = os.system(f"sbatch {scripts_dir}/{script_name}")
    if exit_code != 0:
        print(f"Error in sbatch, Exit code: {exit_code}")
