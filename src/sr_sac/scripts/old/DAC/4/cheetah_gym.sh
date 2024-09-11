#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate gym

module load CUDA/12.0.0

python3 train_parallel_gym.py --env_name="HalfCheetah-v4" --algo_name=DACSimple --kl_target=0.1 --std_multiplier=0.5

wait
