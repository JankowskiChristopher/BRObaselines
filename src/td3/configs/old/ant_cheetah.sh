#!/bin/bash
#
#SBATCH --partition=common
#SBATCH --qos=16gpu3d
#SBATCH --gres=gpu:1
#SBATCH --output=./outputs/last_run.out
#SBATCH --time=1-16:00:00

#source ~/miniconda3/bin/activate
#conda init bash
#source ~/.bashrc
#conda activate td3

NUM_SEEDS=10
TASK_NAME="Ant-v4"

for ((i = 0; i < NUM_SEEDS; i++)); do
    python main.py --env="$TASK_NAME" --seed="$i" &
done

wait