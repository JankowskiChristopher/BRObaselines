#!/bin/bash
#
#SBATCH --partition=common
#SBATCH --qos=16gpu3d
#SBATCH --gres=gpu:1
#SBATCH --output=./outputs/last_run.out
#SBATCH --time=1-16:00:00

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate td3

python main.py --env=HalfCheetah-v4 &
python main.py --env=Humanoid-v4 &
python main.py --env=Pendulum-v1 &
python main.py --env=Ant-v4 &
python main.py --env=Hopper-v4 &

wait