#!/bin/bash
#SBATCH -p 4090


# Load any required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grace-off

layers="2l"
model="a_wpS_medium" #a_wpS_small
sol="wat"

python3 ase_gas.py --layers $layers --model $model --sol $sol