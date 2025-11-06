#!/bin/bash

#SBATCH -p gpu


# Load any required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-fennel_2

python get_torsion_scans.py