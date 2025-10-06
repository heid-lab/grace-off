#!/bin/bash
#SBATCH -p gpu


# Load any required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-fennel

sol="wat"

python3 ase_gas_mace.py --sol $sol