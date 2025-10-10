#!/bin/bash

#SBATCH --partition=GPU-a40s       # select a partition i.e. "GPU-a100"
#SBATCH --gres=gpu:1                # Use GPU
#SBATCH --nodes=1                   # select number of nodes
#SBATCH --ntasks-per-node=12        # select number of tasks per node
#SBATCH --time=7-00:00:00           # Maximum time

# Load any required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-fennel

sol="wat"

python gas.py --model_size small --model_type mace --sol $sol

# for using a GRACE model trained on a_wpS dataset
#python gas.py --model_size small --model_type grace --sol $sol --dataset a_wpS