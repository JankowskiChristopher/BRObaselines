#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate mw

python3 train_parallel_mw.py --env_name=sweep-v2-goal-observable --network_regularization=1 --critic_depth=1 --distributional=2

wait
