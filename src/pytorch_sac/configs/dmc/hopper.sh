#!/bin/bash
#
#SBATCH --partition=common
#SBATCH --qos=16gpu3d
#SBATCH --gres=gpu:1
#SBATCH --output=./outputs/last_run_hopper.out
#SBATCH --time=1-0:00:00
#SBATCH --mem=12000

# ANSI color codes
RED='\033[0;31m' # Red
G='\033[0;32m' # Green
NC='\033[0m' # Reset color

# Display script usage
function show_usage {
  echo "Script to run task"
  echo "Usage: $0 [-s] [-e]"
  echo "  -s: start seed"
  echo "  -e: end seed"
  echo "  -h: Display this help message"
  exit 1
}

TASK_NAME="hopper_hop"
START_SEED=5
END_SEED=10

# Parse command-line options
while getopts ":s:e:h" opt; do
  case $opt in
    s)
      START_SEED="$OPTARG"
      echo -e "${G}Start seed changed to $START_SEED${NC}"
      ;;
    e)
      END_SEED="$OPTARG"
      echo -e "${G}End seed changed to $END_SEED${NC}"
      ;;
    h)
      show_usage
      ;;
    \?)
      echo -e "${RED}Error: Invalid option: -${OPTARG}${NC}"
      show_usage
      ;;
  esac
done

# Activate virtual envs
source ~/miniconda3/bin/activate base
conda init bash
source ~/.bashrc
conda activate sac_pytorch


for ((i = START_SEED; i < END_SEED; i++)); do
    python train.py env="$TASK_NAME" seed="$i" num_train_steps=3000000 &
done

wait