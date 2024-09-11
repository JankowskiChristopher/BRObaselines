#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate jax

python train_parallel.py --env_name humanoid-run --algo_name DAC --save_dir ./DAC/humanoid_run/ 

wait
