#!/bin/bash
#SBATCH -p gpu


# Load any required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-fennel

sol="wat"

python ase_npt.py --model_size small --model_type mace --sol $sol

# for using a GRACE model trained on a_wpS dataset
#python ase_npt.py --model_size small --model_type grace --sol $sol --dataset a_wpS