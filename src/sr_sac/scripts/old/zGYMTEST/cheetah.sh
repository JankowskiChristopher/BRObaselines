#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate gym

module load CUDA/12.0.0

python3 train_parallel_gym.py --env_name="HalfCheetah-v4" --num_seeds=5

wait
