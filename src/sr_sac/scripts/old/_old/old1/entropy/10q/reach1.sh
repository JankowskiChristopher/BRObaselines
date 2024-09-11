#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate mw

python3 train_parallel_mw.py --env_name=reach-v2-goal-observable --network_regularization=1 --critic_depth=1 --n_atoms=10

wait
