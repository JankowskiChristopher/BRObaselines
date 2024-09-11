import argparse
import os
import time
from pathlib import Path
from experiments_config import all_experiments

sbatch_script_template = """#!/bin/bash
#
#SBATCH --partition=common
#SBATCH --qos=16gpu3d
#SBATCH --gres=gpu:1
#SBATCH --output=./outputs/last_run_{env_name}_{agent_name}.out
#SBATCH --time={time}
#SBATCH --mem={mem}

source ~/miniconda3/bin/activate base
conda init bash
source ~/.bashrc
conda activate {conda_env}

export PYTHONPATH=$(pwd)

hydra={hydra}

if [ "$hydra" -eq 1 ]; then
    for ((i = {start_seed}; i < {end_seed}; i++)); do
        python {main} env={env_name} universe={universe} seed="$i" num_train_steps={timesteps} &
    done
else
    for ((i = {start_seed}; i < {end_seed}; i++)); do
        python {main} --env {env_name} --universe {universe} --seed "$i" --max_timesteps {timesteps} &
    done
fi    


wait
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters for running sbatch scripts.")

    # Add arguments
    parser.add_argument("--universe", type=str, default="metaworld", help="Universe the task is in")
    parser.add_argument("--timesteps", type=int, default=int(1e6), help="Number of timesteps to run for.")
    parser.add_argument("--env", type=str, default="all", help="Task to run (defaults to all)")
    parser.add_argument("--time", type=str, default="18:00:00", help="Max job duration")
    parser.add_argument("--mem", type=str, default=14000, help="Mam memory for the job in MB.")
    parser.add_argument("--start_seed", type=int, default=0, help="Start seed for the job.")
    parser.add_argument("--end_seed", type=int, default=5, help="End seed for the job.")
    parser.add_argument("--conda_env", type=str, default="metaworld",
                        help="Conda environment to activate.")
    parser.add_argument("--main", type=str, default="src/pytorch_sac/train.py",
                        help="Path of main file  to run.")
    parser.add_argument("--hydra", type=int, default=1, help="Whether hydra is used (1) or not (0).")

    return parser.parse_args()


def prepare_agent_name_from_path(main_file_dir: str):
    parent = str(Path(main_file_dir).parent).replace("/", "_").replace(".", "_")
    filename = str(Path(main_file_dir).stem).replace("/", "_").replace(".", "_")
    return f"{parent}_{filename}"


if __name__ == '__main__':
    default_params_dict = vars(parse_arguments())

    scripts_dir = "./sbatch_scripts"
    os.makedirs(scripts_dir, exist_ok=True)
    SLEEP_TIME = 5

    assert default_params_dict['universe'] in all_experiments.keys(), \
        f"Universe {default_params_dict['universe']} not found in experiments_config"
    experiments = all_experiments[default_params_dict['universe']]
    if default_params_dict['env'] != "all":
        experiments = [e for e in experiments if e['env_name'] == default_params_dict['env']]
        assert len(experiments) > 0, f"No experiments found for {default_params_dict['env']}."
        print(f'Running experiments only for for {default_params_dict["env"]}.')

    # Preprocess args
    # Check if main file exists
    assert Path(default_params_dict['main']).exists(), \
        f"Path to main file {default_params_dict['main']} does not exist"

    if str(Path(default_params_dict['main'])) == "src/td3/main.py":
        print("Setting default parameters for TD3 in case user forgets\n")
        default_params_dict['hydra'] = 0  # in case user forgets

    print(f"Script will run with parameters {default_params_dict}\n")

    # Run experiments
    for exp in experiments:
        print(f"Running experiment for {exp['env_name']}")
        sbatch_dict = {**default_params_dict, **exp}

        sorted_keys = sorted(sbatch_dict.keys(), key=lambda x: (x != 'env_name', x))
        sorted_keys = [key for key in sorted_keys if key not in ['main', 'conda_env']]

        agent_name = prepare_agent_name_from_path(sbatch_dict['main'])
        sbatch_dict['agent_name'] = agent_name
        script_name = ("_".join([str(sbatch_dict[key]) for key in sorted_keys])) + f"_{agent_name}" + ".sh"
        script_name = script_name.replace("/", "_").replace(".", "_")

        print(f"Saving script to {scripts_dir}/{script_name}")
        with open(f"{scripts_dir}/{script_name}", "w") as f:
            f.write(sbatch_script_template.format(**sbatch_dict))

        print(f"Running sbatch with script {script_name}\n")
        exit_code = os.system(f"sbatch {scripts_dir}/{script_name}")
        if exit_code != 0:
            print(f"Error in sbatch, Exit code: {exit_code}")

        # wandb crashes if too many jobs are submitted at once
        time.sleep(SLEEP_TIME)
