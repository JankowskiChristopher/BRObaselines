#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate jax

module load CUDA/12.0.0

python3 train_parallel.py --env_name=cheetah-run --network_regularization=2 --critic_depth=1 --sn_type=1

wait
