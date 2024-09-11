#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate jax

python3 train_parallel.py --env_name=humanoid-stand --network_regularization=1 --critic_depth=1

wait
