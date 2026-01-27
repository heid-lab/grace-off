#!/bin/bash

#SBATCH -p ADA


# Load any required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grace-off
# conda activate ai-fennel_2

python get_torsion_scans.py