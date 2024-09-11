#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate jax

module load CUDA/12.0.0

python3 train_parallel_maniskill.py --env_name=LiftCube-v0 --network_regularization=1 --critic_depth=1 --tqdm=True --num_seeds=2

wait
