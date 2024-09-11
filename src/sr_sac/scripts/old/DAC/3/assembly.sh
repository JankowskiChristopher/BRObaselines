#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate mw

module load CUDA/12.0.0

python3 train_parallel_mw.py --env_name=assembly-v2-goal-observable --algo_name=DACSimple --kl_target=0.15 --std_multiplier=0.75

wait
