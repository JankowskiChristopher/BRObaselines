#!/bin/bash
#
#SBATCH --partition=common
#SBATCH --qos=16gpu3d
#SBATCH --gres=gpu:1
#SBATCH --output=./outputs/last_run_hopper_pendulum.out
#SBATCH --time=1-16:00:00

#source ~/miniconda3/bin/activate
#conda init bash
#source ~/.bashrc
#conda activate td3

NUM_SEEDS=10
TASKS=("Hopper-v4" "Pendulum-v1")

for task in "${TASKS[@]}"; do
    for ((i = 0; i < NUM_SEEDS; i++)); do
      python main.py --env="$task" --seed="$i" &
    done
done

wait