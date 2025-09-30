#!/bin/bash

#SBATCH --partition=GPU-l40s       # select a partition i.e. "GPU-a100"
#SBATCH --gres=gpu:1                # Use GPU
#SBATCH --nodes=1                   # select number of nodes
#SBATCH --ntasks-per-node=12        # select number of tasks per node
#SBATCH --time=7-00:00:00           # Maximum time

source /home/johannes.karwounopoulos/micromamba/etc/profile.d/mamba.sh
micromamba activate /home/johannes.karwounopoulos/micromamba/envs/grace-linus

gracemaker

gracemaker -r -s