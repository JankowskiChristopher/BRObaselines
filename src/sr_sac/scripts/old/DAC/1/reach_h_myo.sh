#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate myo

module load CUDA/12.0.0

python3 train_parallel_myo.py --env_name="myo-reach-hard" --algo_name=DACSimple --kl_target=0.25 --std_multiplier=0.5

wait
