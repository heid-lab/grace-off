#!/bin/bash
#SBATCH -p gpu


# Load any required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-fennel

sol="wat"

python ase_npt.py --model_size small --model_type mace --sol $sol