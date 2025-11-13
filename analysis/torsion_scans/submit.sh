#!/bin/bash

#SBATCH -p 4090


# Load any required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-fennel_2

python get_torsion_scans.py