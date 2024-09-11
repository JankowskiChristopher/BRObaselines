#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate jax

python3 train_parallel.py --env_name=dog-trot --network_regularization=1 --critic_depth=1 --n_atoms=50

wait
