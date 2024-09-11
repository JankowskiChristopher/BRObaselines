#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate jax

module load CUDA/12.0.0

python3 train_parallel.py --env_name=cheetah-run --network_regularization=0 --critic_depth=2

wait
