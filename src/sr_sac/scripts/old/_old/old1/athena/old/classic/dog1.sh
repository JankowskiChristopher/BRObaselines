#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate jax

module load CUDA/12.0.0

python3 train_parallel.py --env_name=dog-trot --network_regularization=3 --critic_depth=1

wait
