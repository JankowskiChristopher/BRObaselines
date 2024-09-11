#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate jax

module load CUDA/12.0.0

python3 train_parallel.py --env_name=humanoid-stand --algo_name=DACSimple --kl_target=0.1 --std_multiplier=1.25

wait
