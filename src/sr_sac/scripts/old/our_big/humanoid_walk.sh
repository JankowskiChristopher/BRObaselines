#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate jax

module load CUDA/12.0.0

python3 train_parallel.py --env_name=humanoid-walk --actor_type=1 --actor_size=512 --actor_depth=2

wait
