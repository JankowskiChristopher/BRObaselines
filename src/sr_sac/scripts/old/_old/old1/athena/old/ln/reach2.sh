#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate mw

module load CUDA/12.0.0

python3 train_parallel_mw.py --env_name=reach-v2-goal-observable --network_regularization=0 --critic_depth=2

wait
